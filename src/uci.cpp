#include "uci.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "nnue.h"
#include "position.h"
#include "tables.h"
#include "types.h"

namespace {

#ifndef ASKAIG_VERSION
#define ASKAIG_VERSION "dev" // set by CMake (the build date YYYYMMDD); see CMakeLists.txt
#endif
  constexpr auto ENGINE_NAME   = "Askaig " ASKAIG_VERSION;
  constexpr auto ENGINE_AUTHOR = "the Askaig developers (see AUTHORS file)";

  // Number of perft worker threads (the "Threads" UCI option).
  int g_threads = 1;

  // Largest sensible thread count to advertise/accept.
  unsigned max_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
  }

  // Converts an internal Move to its UCI string. Two special cases:
  //  - castling is stored king-to-rook (e1h1 / e1a1) but UCI expects king-two-squares
  //    (e1g1 / e1c1), so the destination is recomputed from the king square;
  //  - promotions get a trailing piece letter (low 2 flag bits: 0=n 1=b 2=r 3=q).
  std::string move_to_uci(Move m) {
    Square    from = m.from();
    Square    to   = m.to();
    MoveFlags f    = m.flags();

    if (f == OO)
      to = Square(from + 2);
    else if (f == OOO)
      to = Square(from - 2);

    std::string s = SQSTR[from];
    s += SQSTR[to];

    if ((f >= PR_KNIGHT && f <= PR_QUEEN) || (f >= PC_KNIGHT && f <= PC_QUEEN)) {
      constexpr char promo[] = {'n', 'b', 'r', 'q'};
      s += promo[f & 0b11];
    }
    return s;
  }

  // Plays the legal move whose UCI string matches `mstr`. Returns false if none matches.
  template<Color Us>
  bool try_play(Position &pos, const std::string &mstr) {
    MoveList<Us> list(pos);
    for (Move m: list)
      if (move_to_uci(m) == mstr) {
        pos.play<Us>(m);
        return true;
      }
    return false;
  }

  bool play_uci_move(Position &pos, const std::string &mstr) {
    return pos.turn() == WHITE ? try_play<WHITE>(pos, mstr) : try_play<BLACK>(pos, mstr);
  }

  // ---- Exact perft hash (a pure-function memoisation) --------------------------------------------
  //
  // perft(position, depth) is a *pure function*: the node count for a position at a depth never
  // changes, so cached entries are never stale and the table is never cleared.
  //
  // The Chess Programming Wiki notes perft hashing gives "a small chance for inaccurate results".
  // That inaccuracy has two sources: (1) two different positions colliding on the same 64-bit
  // Zobrist key, and — specific to this engine — (2) our Zobrist hash encodes only piece placement +
  // side to move, NOT castling rights or the en-passant square, both of which change the legal-move
  // count. We avoid BOTH by making each entry store the *full* move-generation state and accepting a
  // hit only on an exact, bit-for-bit match (not merely a matching hash). A hit therefore means the
  // position is provably identical, so the cached count is exact — zero chance of a wrong result.
  //
  // The table is thread_local, so the parallel perft workers never share it: no concurrent writes,
  // hence no torn reads, hence exactness holds without any locking.
  struct PerftEntry {
    uint64_t board[4]; // the 64 squares packed 4 bits each — the exact piece placement
    Bitboard castle; // history entry bitboard: distinguishes positions with different castling rights
    uint64_t count; // the perft node count (valid only at `depth`)
    int16_t  epsq; // en-passant square (or NO_SQUARE) — also affects the move count
    int8_t   side; // side to move
    int8_t   depth; // depth this count is for; -1 marks an empty slot
  };

  constexpr size_t PERFT_TT_ENTRIES     = 1u << 20; // per thread (~59 MB); only allocated when used
  constexpr int    PERFT_HASH_MIN_DEPTH = 4; // memoise only where a subtree is big enough to pay

  // Per-thread cache. Each perft worker is its own thread, so this is private to it (no races).
  thread_local std::vector<PerftEntry> t_perft_tt;

  // Allocate (once) the calling thread's table. Pure-function cache: if it already exists we keep it
  // (it can never go stale), so repeat perfts on the main thread reuse prior work.
  void ensure_perft_tt() {
    if (t_perft_tt.size() != PERFT_TT_ENTRIES)
      t_perft_tt.assign(PERFT_TT_ENTRIES, PerftEntry{{0, 0, 0, 0}, 0, 0, NO_SQUARE, 0, -1});
  }

  // Packs the full move-generation state of `p` into a PerftEntry fingerprint (count/depth unset).
  PerftEntry perft_fingerprint(const Position &p) {
    PerftEntry e{{0, 0, 0, 0}, 0, 0, NO_SQUARE, 0, -1};
    for (int s = 0; s < NSQUARES; ++s)
      e.board[s >> 4] |= static_cast<uint64_t>(p.at(Square(s)) & 0xF) << ((s & 15) * 4);
    e.castle = p.history[p.ply()].entry;
    e.epsq   = static_cast<int16_t>(p.history[p.ply()].epsq);
    e.side   = static_cast<int8_t>(p.turn());
    return e;
  }

  // True iff two fingerprints describe the bit-for-bit identical move-generation state.
  bool same_position(const PerftEntry &a, const PerftEntry &b) {
    return a.board[0] == b.board[0] && a.board[1] == b.board[1] && a.board[2] == b.board[2] &&
           a.board[3] == b.board[3] && a.castle == b.castle && a.epsq == b.epsq && a.side == b.side;
  }

  size_t perft_index(uint64_t hash, const PerftEntry &k) {
    // Mix the Zobrist hash with the ep/castling state (which the hash omits) so positions differing
    // only in those spread to different slots; the exact compare still guarantees correctness.
    const uint64_t h = hash ^ (static_cast<uint64_t>(static_cast<uint16_t>(k.epsq)) * 0x9E3779B97F4A7C15ull) ^ k.castle;
    return h & (PERFT_TT_ENTRIES - 1);
  }

  // perft (move-generation node count) — used by "go perft <depth>". Memoised by the exact perft
  // hash above when the calling thread has a table and the depth is worth caching.
  template<Color Us>
  uint64_t perft(Position &p, int depth, bool use_cache) {
    // Bulk counting (always on): at the last ply return the legal-move count directly, skipping a
    // make/unmake of every leaf (the bulk of the tree) — far faster, identical total.
    if (depth <= 1) {
      MoveList<Us> list(p);
      return static_cast<uint64_t>(list.size());
    }

    // The perft hash is a memoisation of subtree counts; `go perft <d> noncache` skips it so every
    // subtree is recomputed (the tree is still bulk-counted at the leaves).
    const bool use_hash = use_cache && !t_perft_tt.empty() && depth >= PERFT_HASH_MIN_DEPTH;
    PerftEntry key{{0, 0, 0, 0}, 0, 0, NO_SQUARE, 0, -1};
    size_t     idx = 0;
    if (use_hash) {
      key                 = perft_fingerprint(p);
      idx                 = perft_index(p.get_hash(), key);
      const PerftEntry &e = t_perft_tt[idx];
      if (e.depth == depth && same_position(e, key)) // exact match — provably the same position
        return e.count;
    }

    uint64_t     nodes = 0;
    MoveList<Us> list(p);
    for (Move m: list) {
      p.play<Us>(m);
      nodes += perft<~Us>(p, depth - 1, use_cache);
      p.undo<Us>(m);
    }

    if (use_hash) {
      PerftEntry &e = t_perft_tt[idx]; // depth-preferred: keep the costlier (deeper) result on a clash
      if (e.depth <= depth) {
        key.count = nodes;
        key.depth = static_cast<int8_t>(depth);
        e         = key;
      }
    }
    return nodes;
  }

  // Byte-clones a Position so each perft worker can make/unmake on its own copy. A plain copy would
  // invoke UndoInfo's copy constructor (which resets epsq/captured) and corrupt the history stack;
  // every Position member is plain data, so the bit pattern is a faithful, independent copy.
  Position clone_position(const Position &src) {
    Position dst;
    std::memcpy(static_cast<void *>(&dst), static_cast<const void *>(&src), sizeof(Position));
    return dst;
  }

  // Divides the perft node count by root move (the standard "divide" output), parallelised across
  // the "Threads" option. Each root move's subtree is an independent computation, so workers grab
  // root moves from a shared atomic counter and accumulate into per-move slots on their own board
  // clone; the divide lines are then printed in move-generation order.
  template<Color Us>
  void perft_divide(Position &p, int depth, bool use_cache) {
    std::vector<Move> moves;
    {
      MoveList<Us> list(p);
      moves.assign(list.begin(), list.end());
    }
    const size_t          n = moves.size();
    std::vector<uint64_t> counts(n, 1); // depth 1: every root move is itself one leaf

    int nthreads = g_threads < 1 ? 1 : g_threads;
    if (nthreads > static_cast<int>(n))
      nthreads = static_cast<int>(n == 0 ? 1 : n);

    // The exact perft hash pays off only for deep subtrees; allocate it then (per worker thread).
    // `noncache` skips the hash on purpose — every subtree is recomputed.
    const bool hash = use_cache && depth - 1 >= PERFT_HASH_MIN_DEPTH;

    const auto start = std::chrono::steady_clock::now();

    if (depth > 1) {
      if (nthreads <= 1) {
        if (hash)
          ensure_perft_tt();
        for (size_t i = 0; i < n; ++i) {
          p.play<Us>(moves[i]);
          counts[i] = perft<~Us>(p, depth - 1, use_cache);
          p.undo<Us>(moves[i]);
        }
      } else {
        std::atomic<size_t> next{0};
        auto                worker = [&]() {
          if (hash)
            ensure_perft_tt(); // this worker thread's private (thread_local) table
          Position local = clone_position(p);
          size_t   i;
          while ((i = next.fetch_add(1, std::memory_order_relaxed)) < n) {
            local.play<Us>(moves[i]);
            counts[i] = perft<~Us>(local, depth - 1, use_cache);
            local.undo<Us>(moves[i]);
          }
        };
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(nthreads - 1));
        for (int t = 1; t < nthreads; ++t)
          pool.emplace_back(worker);
        worker(); // this thread is a worker too
        for (auto &th: pool)
          th.join();
      }
    }

    uint64_t total = 0;
    for (size_t i = 0; i < n; ++i) {
      std::cout << move_to_uci(moves[i]) << ": " << counts[i] << "\n";
      total += counts[i];
    }
    const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

    std::cout << "\nNodes searched: " << total << "\n";
    std::cout << "Speed: "
              << (us > 0 ? static_cast<uint64_t>(static_cast<double>(total) * 1'000'000.0 / static_cast<double>(us))
                         : 0)
              << " nodes/s\n";
  }

  void run_perft(Position &pos, int depth, bool use_cache) {
    if (pos.turn() == WHITE)
      perft_divide<WHITE>(pos, depth, use_cache);
    else
      perft_divide<BLACK>(pos, depth, use_cache);
  }

  // --- selftest nnue -----------------------------------------------------------------------------
  // Plays seeded-random legal games and asserts at every checked ply that the incremental
  // accumulator evaluation (push/pop + lazy walk-back) is bit-identical to a full refresh.
  // Coverage by construction: all 11 MoveFlags cases (the start FENs force castling, en passant
  // and promotions), multi-ply lazy chains (evaluation is skipped randomly), and mixed
  // undo/pop back-off sequences. The scalar kernels are the reference; when the SIMD kernels
  // land, this same test pins them to the scalar results (all paths are exact integer math).

  void play_any_color(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.play<WHITE>(m);
    else
      p.play<BLACK>(m);
  }
  // undo<C> takes the color that MADE the move — the opposite of the side to move afterwards.
  void undo_any_color(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.undo<BLACK>(m);
    else
      p.undo<WHITE>(m);
  }
  size_t legal_moves(Position &p, Move *out) {
    if (p.turn() == WHITE) {
      MoveList<WHITE> l(p);
      std::copy(l.begin(), l.end(), out);
      return l.size();
    }
    MoveList<BLACK> l(p);
    std::copy(l.begin(), l.end(), out);
    return l.size();
  }

  void selftest_nnue(int games, int maxply) {
    if (!nnue::loaded()) {
      std::cout << "selftest nnue FAIL: no net loaded\n";
      return;
    }
    // Start positions chosen so random play quickly reaches every move type: castling rights
    // both sides (kiwipete), en-passant-rich pawn endings, and promotion storms (pos4 and the
    // 8-passers race, which also drives accumulator values toward their extremes).
    constexpr const char *FENS[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", // kiwipete
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", // perft pos4 (promotions)
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // perft pos3 (en passant)
            "8/PPPP4/8/8/8/2k5/4pppp/K7 w - - 0 1", // promotion race (extreme accumulator values)
    };
    PRNG            rng(0xA5CA16u);
    nnue::Evaluator ev;
    uint64_t        checks = 0;

    const auto check = [&](const Position &pos) {
      const int inc = ev.evaluate(pos);
      const int ref = nnue::evaluate_refresh(pos);
      ++checks;
      if (inc != ref) {
        std::cout << "selftest nnue FAIL: incremental " << inc << " != refresh " << ref << "\n  fen " << pos.fen()
                  << "\n";
        return false;
      }
      return true;
    };

    for (int g = 0; g < games; ++g) {
      Position pos;
      Position::set(FENS[g % std::size(FENS)], pos);
      ev.reset(pos);
      std::vector<Move> made;

      for (int ply = 0; ply < maxply; ++ply) {
        // Occasionally back off a few plies: pop() must land back on already-computed (or
        // still-lazy) ancestors and the next pushes must rebuild correctly from there.
        if (!made.empty() && rng.rand<uint64_t>() % 8 == 0) {
          int k = 1 + static_cast<int>(rng.rand<uint64_t>() % std::min<size_t>(made.size(), 6));
          while (k--) {
            undo_any_color(pos, made.back());
            made.pop_back();
            ev.pop();
          }
          if (!check(pos))
            return;
          continue;
        }

        Move         moves[218];
        const size_t n = legal_moves(pos, moves);
        if (n == 0)
          break; // mate/stalemate — start the next game
        const Move m = moves[rng.rand<uint64_t>() % n];
        ev.push(pos, m);
        play_any_color(pos, m);
        made.push_back(m);

        // Evaluate only half the time so lazy multi-ply walk-back chains get exercised too.
        if (rng.rand<uint64_t>() % 2 == 0 && !check(pos))
          return;
      }
      if (!check(pos)) // end-of-game: one final check on whatever the lazy state is
        return;
    }
    std::cout << "selftest nnue PASS: " << games << " games, " << checks
              << " eval checks, incremental == refresh\n";
  }

  // --- bench evalnps -----------------------------------------------------------------------------
  // Micro-benchmark of the NNUE evaluation path a search will drive: cycles of push -> evaluate
  // (one incremental apply + output dot) then unwind, on a fixed reversible knight-shuffle line
  // from the start position, plus a separate full-refresh loop. Deterministic; movegen excluded.
  void bench_evalnps() {
    if (!nnue::loaded()) {
      std::cout << "bench evalnps FAIL: no net loaded\n";
      return;
    }
    Position pos;
    Position::set(DEFAULT_FEN, pos);
    nnue::Evaluator ev;
    ev.reset(pos);

    // A 4-ply reversible cycle (knights out and back): the position returns to startpos, so the
    // loop can run forever without growing the stack beyond 4 plies.
    const Move cyc[4] = {Move(g1, f3, QUIET), Move(g8, f6, QUIET), Move(f3, g1, QUIET), Move(f6, g8, QUIET)};

    // volatile: LTO can prove the eval functions pure and hoist them out of the loops otherwise
    // (observed with the refresh loop) — a forced store per eval is noise at these sizes.
    volatile int64_t sink = 0;

    constexpr int ITERS = 500'000; // x4 evals per iteration
    auto          t0    = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
      for (const Move m: cyc) {
        ev.push(pos, m);
        play_any_color(pos, m);
        sink = sink + ev.evaluate(pos);
      }
      for (int k = 0; k < 4; ++k) {
        undo_any_color(pos, cyc[3 - k]);
        ev.pop();
      }
    }
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t evals = uint64_t(ITERS) * 4;
    std::cout << "incremental: " << evals << " evals in " << us / 1000 << " ms = "
              << (us > 0 ? evals * 1'000'000 / uint64_t(us) : 0) << " evals/s\n";

    // Refresh loop: walk the same 2-ply cycle so every call sees a different position.
    constexpr int RITERS = 50'000; // x4 evals per iteration
    t0                   = std::chrono::steady_clock::now();
    for (int it = 0; it < RITERS; ++it) {
      play_any_color(pos, cyc[0]);
      sink = sink + nnue::evaluate_refresh(pos);
      play_any_color(pos, cyc[1]);
      sink = sink + nnue::evaluate_refresh(pos);
      undo_any_color(pos, cyc[1]);
      sink = sink + nnue::evaluate_refresh(pos);
      undo_any_color(pos, cyc[0]);
      sink = sink + nnue::evaluate_refresh(pos);
    }
    us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t revals = uint64_t(RITERS) * 4;
    std::cout << "refresh:     " << revals << " evals in " << us / 1000 << " ms = "
              << (us > 0 ? revals * 1'000'000 / uint64_t(us) : 0) << " evals/s\n";
    std::cout << "checksum " << sink << "\n"; // also a cross-build (NEON/AVX2/scalar) invariant
  }

  // --- d / display ---------------------------------------------------------------------------
  // Pretty board preview: colored piece grid, the position facts, and the current static NNUE
  // eval from the observer's (White's) point of view: + = better for White, - = better for
  // Black, plus the raw centipawn value.
  void display_cmd(const Position &pos) {
    constexpr const char *LBL = "\033[38;5;208m"; // orange labels
    constexpr const char *WPC = "\033[1;97m"; // white pieces: bold bright white
    constexpr const char *BPC = "\033[1;94m"; // black pieces: bright blue
    constexpr const char *DIM = "\033[2m"; // frame / empty squares
    constexpr const char *RST = "\033[0m";

    // A castling right is lost once its king/rook "entry" squares have been touched.
    const Bitboard entry = pos.castle_entry();
    std::string    castles;
    if (!(entry & WHITE_OO_MASK))
      castles += 'K';
    if (!(entry & WHITE_OOO_MASK))
      castles += 'Q';
    if (!(entry & BLACK_OO_MASK))
      castles += 'k';
    if (!(entry & BLACK_OOO_MASK))
      castles += 'q';
    if (castles.empty())
      castles = "-";

    const Square ep    = pos.history[pos.ply()].epsq;
    const bool   check = pos.turn() == WHITE ? pos.in_check<WHITE>() : pos.in_check<BLACK>();

    std::ostringstream zob;
    zob << "0x" << std::hex << std::uppercase << pos.get_hash();

    const std::string L = LBL, R = RST;
    const std::string info[] = {
            L + "FEN: " + R + pos.fen() + " " + std::to_string(pos.fifty()) + " " +
                    std::to_string(pos.ply() / 2 + 1),
            L + "Zobrist Key: " + R + zob.str(),
            L + "Castle Rights: " + R + castles,
            L + "Side To Move: " + R + (pos.turn() == WHITE ? "White" : "Black"),
            L + "En Passant: " + R + (ep == NO_SQUARE ? "NULL" : SQSTR[ep]),
            L + "Half Moves: " + R + std::to_string(pos.fifty()),
            L + "In Check: " + R + (check ? "true" : "false"),
    };

    std::cout << "   " << DIM << "-----------------" << RST << "\n";
    for (int r = 7; r >= 0; --r) {
      std::cout << " " << LBL << r + 1 << RST << " " << DIM << "|" << RST;
      for (int f = 0; f < 8; ++f) {
        const Piece pc = pos.at(create_square(File(f), Rank(r)));
        if (pc == NO_PIECE)
          std::cout << " " << DIM << "." << RST;
        else
          std::cout << " " << (color_of(pc) == WHITE ? WPC : BPC) << PIECE_STR[pc] << RST;
      }
      std::cout << " " << DIM << "|" << RST;
      if (const size_t i = size_t(7 - r); i < std::size(info))
        std::cout << " " << info[i];
      std::cout << "\n";
    }
    std::cout << "   " << DIM << "-----------------" << RST << "\n";
    std::cout << "     " << LBL << "A B C D E F G H" << RST << "\n\n";

    if (nnue::loaded()) {
      const int   stm_cp = nnue::evaluate_refresh(pos);
      const int   cp     = pos.turn() == WHITE ? stm_cp : -stm_cp; // observer = White's POV
      const char *col    = cp > 0 ? "\033[92m" : (cp < 0 ? "\033[91m" : RST);
      char        pawns[16];
      std::snprintf(pawns, sizeof pawns, "%+.2f", cp / 100.0);
      std::cout << " " << LBL << "Eval: " << RST << col << pawns << RST << " (cp " << cp << ")\n";
    } else
      std::cout << " " << LBL << "Eval: " << RST << "no net loaded\n";
  }

  // Handles "position [startpos | fen <fen>] [moves <m1> ...]".
  void position_cmd(std::optional<Position> &pos, std::istringstream &is) {
    std::string token;
    std::string fen;
    is >> token;

    if (token == "startpos") {
      fen = DEFAULT_FEN;
      is >> token; // consume "moves" if present
    } else if (token == "fen") {
      while (is >> token && token != "moves") {
        if (!fen.empty())
          fen += ' ';
        fen += token;
      }
    } else {
      return; // malformed
    }

    pos.emplace(); // set() assumes a freshly-constructed Position
    Position::set(fen, *pos);

    if (token == "moves") {
      std::string mv;
      while (is >> mv)
        if (!play_uci_move(*pos, mv)) {
          std::cout << "info string ignoring illegal/unknown move " << mv << "\n";
          break;
        }
    }
  }

  // Handles "go ...". Only "go perft <depth> [noncache]" is implemented (a move-generation node
  // count); the search forms (depth/movetime/wtime/...) are gone — search and evaluation are being
  // rebuilt from scratch.
  void go_cmd(Position &pos, std::istringstream &is) {
    std::string token;
    while (is >> token) {
      if (token == "perft") {
        int d = 1;
        is >> d;
        // Optional trailing "noncache": skip the perft-hash memoisation so every subtree is
        // recomputed (the whole tree is walked, still bulk-counted at the leaves). Default uses the cache.
        std::string opt;
        const bool  use_cache = !(is >> opt && opt == "noncache");
        run_perft(pos, d, use_cache);
        return;
      }
    }
    std::cout << "info string search not implemented yet (only 'go perft <depth> [noncache]')\n";
  }

} // namespace

void uci::loop() {
  std::optional<Position> pos;
  pos.emplace();
  Position::set(DEFAULT_FEN, *pos);

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream is(line);
    std::string        cmd;
    is >> cmd;

    if (cmd == "uci") {
      std::cout << "id name " << ENGINE_NAME << "\n";
      std::cout << "id author " << ENGINE_AUTHOR << "\n";
      std::cout << "option name Threads type spin default 1 min 1 max " << max_threads() << "\n";
      std::cout << "option name EvalFile type string default <embedded>\n";
      std::cout << "uciok\n";
    } else if (cmd == "isready") {
      std::cout << "readyok\n";
    } else if (cmd == "ucinewgame") {
      pos.emplace();
      Position::set(DEFAULT_FEN, *pos);
    } else if (cmd == "position") {
      position_cmd(pos, is);
    } else if (cmd == "go") {
      go_cmd(*pos, is);
    } else if (cmd == "d" || cmd == "display") {
      display_cmd(*pos);
    } else if (cmd == "eval") {
      // Debug helper (like Stockfish's "eval"): the raw static NNUE evaluation of the current
      // position via a full accumulator refresh. No search — for net testing (see parity.py).
      if (!nnue::loaded())
        std::cout << "info string no NNUE net loaded (use setoption name EvalFile value <path>)\n";
      else {
        const int cp = nnue::evaluate_refresh(*pos);
        std::cout << "eval " << cp << " cp (side to move), " << (pos->turn() == WHITE ? cp : -cp)
                  << " cp (white)\n";
      }
    } else if (cmd == "setoption") {
      // setoption name <id> [value <x>]
      std::string token;
      std::string name;
      is >> token >> name >> token; // "name", <id>, "value"
      if (name == "Threads") {
        int t = 0;
        if (is >> t)
          g_threads = t < 1 ? 1 : (t > 1024 ? 1024 : t);
      } else if (name == "EvalFile") {
        // The value is the rest of the line (paths can contain spaces). A bad file keeps the
        // currently loaded net (the embedded default) and reports why.
        std::string path;
        std::getline(is, path);
        const size_t start = path.find_first_not_of(' ');
        path = start == std::string::npos ? std::string{} : path.substr(start);
        std::string err;
        if (path.empty() || !nnue::load_file(path, &err))
          std::cout << "info string EvalFile '" << path << "' rejected (" << (path.empty() ? "empty path" : err)
                    << "), keeping the current net\n";
        else
          std::cout << "info string EvalFile loaded: " << path << "\n";
      }
      // (unknown options: silently ignored, per the UCI spec)
    } else if (cmd == "bench") {
      // Hidden benchmark commands: "bench evalnps".
      std::string what;
      is >> what;
      if (what == "evalnps")
        bench_evalnps();
    } else if (cmd == "selftest") {
      // Hidden test commands (not advertised): "selftest nnue [games] [maxply]".
      std::string what;
      is >> what;
      if (what == "nnue") {
        int g = 0, p = 0;
        is >> g >> p;
        selftest_nnue(g > 0 ? g : 100, p > 0 ? p : 80);
      }
    } else if (cmd == "register" || cmd.empty()) {
      // accepted but no-op
    } else if (cmd == "quit" || cmd == "exit") {
      break;
    }
    // unknown commands are silently ignored, as the UCI spec requires

    std::cout.flush();
  }
}
