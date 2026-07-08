#include "uci.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "see.h"
#include "tables.h"
#include "tt.h"
#include "types.h"

namespace {

#ifndef ASKAIG_VERSION
#define ASKAIG_VERSION "dev" // set by CMake (the build date YYYYMMDD); see CMakeLists.txt
#endif
  constexpr auto ENGINE_NAME   = "Askaig " ASKAIG_VERSION;
  constexpr auto ENGINE_AUTHOR = "the Askaig developers (see AUTHORS file)";

  // Number of perft worker threads (the "Threads" UCI option; the search is single-threaded).
  int g_threads = 1;

  // The search runs on this background thread so the UCI loop stays responsive (stop/isready
  // work mid-search). g_out serialises stdout so info/bestmove lines never interleave.
  std::thread g_search;
  std::mutex  g_out;

  // Stops any running search and joins its thread. Called before any command that mutates
  // engine state, so a search never runs concurrently with a position/option change.
  void stop_search() {
    search::request_stop();
    if (g_search.joinable())
      g_search.join();
  }

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

  bool selftest_nnue(int games, int maxply) {
    if (!nnue::loaded()) {
      std::cout << "selftest nnue FAIL: no net loaded\n";
      return false;
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
            return false;
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
          return false;
      }
      if (!check(pos)) // end-of-game: one final check on whatever the lazy state is
        return false;
    }
    std::cout << "selftest nnue PASS: " << games << " games, " << checks
              << " eval checks, incremental == refresh\n";
    return true;
  }

  // --- selftest perft ------------------------------------------------------------------------
  // Fast movegen regression check: the classic Chess Programming Wiki reference positions
  // (the same six used for `bench`/NNUE-selftest FENs elsewhere in this file), at depths chosen
  // to run in well under a second combined while still exercising every special-move class
  // (castling, en passant, promotions). The expected counts are the well-known published
  // values, cross-checked against this engine's own (already independently perft-verified —
  // see the README's `go perft 6` = 119060324 check) output before being hardcoded here.
  bool selftest_perft() {
    struct Case {
      const char *fen;
      int         depth;
      uint64_t    expected;
    };
    constexpr Case CASES[] = {
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4'865'609}, // startpos
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4'085'603}, // kiwipete
            {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674'624}, // pos3 (en passant)
            {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422'333}, // pos4 (promotions)
            {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2'103'487}, // pos5
            {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3'894'594}, // pos6
    };
    bool ok = true;
    for (const Case &c: CASES) {
      Position pos;
      Position::set(c.fen, pos);
      const uint64_t got = pos.turn() == WHITE ? perft<WHITE>(pos, c.depth, true) : perft<BLACK>(pos, c.depth, true);
      if (got != c.expected) {
        std::cout << "selftest perft FAIL: depth " << c.depth << " got " << got << " expected " << c.expected
                  << " fen " << c.fen << "\n";
        ok = false;
      }
    }
    if (ok)
      std::cout << "selftest perft PASS: " << std::size(CASES) << " known positions match exactly\n";
    return ok;
  }

  // --- selftest see --------------------------------------------------------------------------
  // Static exchange evaluation correctness: a handful of hand-traced absolute-value cases
  // (values follow search::PIECE_VAL: P=100 N=320 B=330 R=500 Q=900) chosen to each exercise a
  // different branch of the swap algorithm, plus a monotonicity/bounds fuzz pass over random
  // captures from real playouts — a correct see_ge must be true for every threshold at or below
  // the exchange's true value and false for every threshold above it (never true-after-false as
  // the threshold rises), and must saturate to true/false far outside any realistic exchange.
  bool selftest_see() {
    bool ok = true;
    const auto expect = [&](bool cond, const char *what) {
      if (!cond) {
        std::cout << "selftest see FAIL: " << what << "\n";
        ok = false;
      }
    };

    { // Undefended capture: Rd1xd4 wins a pawn outright (net +100), nothing else attacks d4.
      // Exercises the first early-return (threshold > target value: false immediately) and the
      // full loop with an empty attacker set on the first iteration (true).
      Position pos;
      Position::set("4k3/8/8/8/3p4/8/8/3R3K w - - 0 1", pos);
      const Move m(d1, d4, CAPTURE);
      expect(search::see_ge(pos, m, 100), "undefended pawn capture: expected SEE >= 100");
      expect(!search::see_ge(pos, m, 101), "undefended pawn capture: expected SEE < 101");
    }
    { // Pawn-defended capture: Rd1xd4 wins a knight (+320) but a black pawn on c5 recaptures the
      // rook (-500): net -180. Exercises the second early-return (recapture-for-free check).
      Position pos;
      Position::set("4k3/8/8/2p5/3n4/8/8/3R3K w - - 0 1", pos);
      const Move m(d1, d4, CAPTURE);
      expect(search::see_ge(pos, m, -180), "pawn-defended knight capture: expected SEE >= -180");
      expect(!search::see_ge(pos, m, -179), "pawn-defended knight capture: expected SEE < -179");
    }
    { // Equal trade: Rd1xd4 (a black rook) is recaptured by a second black rook on d8 (open
      // file, no blockers) — net exactly 0. Exercises the full loop running one extra ply deep.
      Position pos;
      Position::set("3rk3/8/8/8/3r4/8/8/3R3K w - - 0 1", pos);
      const Move m(d1, d4, CAPTURE);
      expect(search::see_ge(pos, m, 0), "even rook trade: expected SEE >= 0");
      expect(!search::see_ge(pos, m, 1), "even rook trade: expected SEE < 1");
    }

    { // Fuzz: every capture seen across a few plies of random legal play in three structurally
      // different openings, checked for monotonicity and sane saturation at extreme thresholds.
      PRNG                  rng(0xC0FFEEu);
      constexpr const char *FENS[] = {
              "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
              "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", // kiwipete
              "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", // perft pos4
      };
      constexpr int THRESH[] = {-2000, -900, -500, -320, -100, -1, 0, 1, 100, 320, 500, 900, 2000};
      int           checked  = 0;
      for (const char *fen: FENS) {
        Position pos;
        Position::set(fen, pos);
        for (int ply = 0; ply < 60 && checked < 300; ++ply) {
          Move         moves[218];
          const size_t n = legal_moves(pos, moves);
          if (n == 0)
            break;
          for (size_t i = 0; i < n && checked < 300; ++i) {
            if (!moves[i].is_capture())
              continue;
            bool prev = true; // see_ge at -infinity is always true
            for (int th: THRESH) {
              const bool cur = search::see_ge(pos, moves[i], th);
              if (cur && !prev) {
                std::cout << "selftest see FAIL: non-monotonic at threshold " << th << " fen " << pos.fen() << "\n";
                return false;
              }
              prev = cur;
            }
            expect(search::see_ge(pos, moves[i], -100'000), "see_ge must hold far below any real exchange");
            expect(!search::see_ge(pos, moves[i], 100'000), "see_ge must fail far above any real exchange");
            ++checked;
          }
          const Move m = moves[rng.rand<uint64_t>() % n];
          play_any_color(pos, m);
        }
      }
      if (ok)
        std::cout << "  (" << checked << " random captures fuzzed for monotonicity/bounds)\n";
    }

    if (ok)
      std::cout << "selftest see PASS: hand-traced cases and the fuzz pass all correct\n";
    return ok;
  }

  // --- selftest draw ---------------------------------------------------------------------------
  // Position::is_draw() flags a draw on the SECOND occurrence of a position (a repeat), not the
  // literal FIDE threefold — the search treats any repeated line as drawish rather than waiting
  // for a third occurrence that a hypothetical search line will rarely reach — plus the
  // fifty-move rule and dead material (bare kings + at most one minor). Verified directly
  // against Position, independent of search/NNUE. The shuffle/fifty cases keep a rook per side
  // on the board precisely so the dead-material rule cannot fire before the rule under test.
  bool selftest_draw() {
    bool ok = true;
    const auto expect = [&](bool cond, const char *what) {
      if (!cond) {
        std::cout << "selftest draw FAIL: " << what << "\n";
        ok = false;
      }
    };

    { // Negative control: a fresh position must never be flagged a draw.
      Position pos;
      Position::set(DEFAULT_FEN, pos);
      expect(!pos.is_draw(), "startpos incorrectly flagged as a draw");
    }
    { // A king-shuffle round trip (4 plies) returns to the exact starting position — its SECOND
      // occurrence — and must flip is_draw() to true exactly there, not on the 3 plies before it.
      Position pos;
      Position::set("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", pos);
      constexpr const char *SHUFFLE[] = {"e1d1", "e8d8", "d1e1", "d8e8"};
      for (int i = 0; i < 4; ++i) {
        if (!play_uci_move(pos, SHUFFLE[i])) {
          std::cout << "selftest draw FAIL: illegal shuffle move " << SHUFFLE[i] << "\n";
          return false;
        }
        const bool should_be_draw = (i == 3);
        expect(pos.is_draw() == should_be_draw,
               should_be_draw ? "repetition not detected after the round trip"
                              : "false positive: draw flagged before the position actually repeated");
      }
    }
    { // Fifty-move rule: one halfmove short of the limit, then one quiet move over it.
      Position pos;
      Position::set("r3k3/8/8/8/8/8/8/R3K3 w - - 99 1", pos);
      expect(!pos.is_draw(), "fifty-move rule fired one halfmove early");
      if (!play_uci_move(pos, "e1d1")) {
        std::cout << "selftest draw FAIL: illegal fifty-move test move\n";
        return false;
      }
      expect(pos.is_draw(), "fifty-move rule did not fire at halfmove 100");
    }
    { // Dead material: KvK, KBvK and KNvK can never mate — drawn on the spot, whoever moves.
      // KRvK, KPvK and KNNvK all still contain legal mates and must NOT be flagged.
      const auto draws = [](const char *fen) {
        Position pos;
        Position::set(fen, pos);
        return pos.is_draw();
      };
      expect(draws("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), "K vs K not flagged as a dead draw");
      expect(draws("4k3/8/8/8/8/8/2B5/4K3 b - - 0 1"), "KB vs K not flagged as a dead draw");
      expect(draws("4k3/8/2n5/8/8/8/8/4K3 w - - 0 1"), "KN vs K not flagged as a dead draw");
      expect(!draws("4k3/8/8/8/8/8/8/2R1K3 w - - 0 1"), "KR vs K wrongly flagged as a draw");
      expect(!draws("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"), "KP vs K wrongly flagged as a draw");
      expect(!draws("4k3/8/8/8/8/8/1NN5/4K3 b - - 0 1"), "KNN vs K wrongly flagged as a draw");
    }

    if (ok)
      std::cout << "selftest draw PASS: negative control, repetition, fifty-move rule, and dead material all correct\n";
    return ok;
  }

  // --- selftest search -------------------------------------------------------------------------
  // Runs a small, varied position set through the real search at Threads in {1, 2, 4} and
  // checks, independent of the search's own move generator, that: the reported bestmove is
  // legal, the FULL reported PV replays as legal moves one by one, and the score is within the
  // representable mate range. This is the in-engine version of the ad-hoc PV-legality checks
  // used throughout development — threading bugs (in the YBWC era: a stale split snapshot or
  // child PV; under lazy SMP: any cross-thread state leaking into the main thread's PV) show
  // up exactly as illegal PV moves at Threads > 1, the most direct regression guard there is.
  bool selftest_search() {
    if (!nnue::loaded()) {
      std::cout << "selftest search FAIL: no net loaded\n";
      return false;
    }
    search::clear_stop(); // see bench_cmd's comment — think() no longer resets this itself
    struct Case {
      const char *fen;
      int         depth;
    };
    constexpr Case CASES[] = {
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 9}, // quiet startpos
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 8}, // kiwipete
            {"2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", 10}, // WAC.001, mate in 2
            {"8/PPPP4/8/8/8/2k5/4pppp/K7 w - - 0 1", 8}, // promotion race
    };

    const int  saved_threads = g_threads;
    const auto legal_replay  = [](const char *fen, const std::vector<Move> &pv) {
      Position rp;
      Position::set(fen, rp);
      for (Move m: pv) {
        Move         moves[218];
        const size_t n     = legal_moves(rp, moves);
        bool         found = false;
        for (size_t i = 0; i < n; ++i)
          if (moves[i].to_from() == m.to_from()) {
            found = true;
            break;
          }
        if (!found)
          return false;
        play_any_color(rp, m);
      }
      return true;
    };

    bool ok = true;
    for (const int threads: {1, 2, 4}) {
      g_threads = threads;
      search::set_threads(threads);
      for (const Case &c: CASES) {
        Position pos;
        Position::set(c.fen, pos);
        tt::clear();
        search::new_game();
        const search::Result r = search::think(pos, c.depth, nullptr, 0, 0);

        Move         legal[218];
        const size_t n     = legal_moves(pos, legal);
        bool         found = r.best.to_from() == 0;
        for (size_t i = 0; i < n && !found; ++i)
          found = legal[i].to_from() == r.best.to_from();
        if (!found) {
          std::cout << "selftest search FAIL: T" << threads << " illegal bestmove on " << c.fen << "\n";
          ok = false;
        }
        if (!legal_replay(c.fen, r.pv)) {
          std::cout << "selftest search FAIL: T" << threads << " illegal PV move on " << c.fen << "\n";
          ok = false;
        }
        if (std::abs(r.score) > search::MATE) {
          std::cout << "selftest search FAIL: T" << threads << " score " << r.score << " out of range on " << c.fen
                    << "\n";
          ok = false;
        }
      }
    }
    g_threads = saved_threads;
    search::set_threads(saved_threads);
    tt::clear();
    search::new_game();

    if (ok)
      std::cout << "selftest search PASS: " << std::size(CASES) << " positions x {1,2,4} threads, legal PV + sane score\n";
    return ok;
  }

  // --- selftest stop ---------------------------------------------------------------------------
  // Regression test for a real bug found while stress-testing "go" immediately followed by
  // "stop": think() used to unconditionally reset g_stop at its own start. A stop requested on
  // the UCI thread BEFORE the newly-spawned search thread's think() call actually started
  // running could have that request silently wiped out by think()'s own reset, so the search
  // ran to full (clamped) depth instead of stopping — observed as a multi-second delay where an
  // instant response was expected. The fix moved the reset to search::clear_stop(), called
  // synchronously by the spawner before the thread starts (go_cmd) or once up front by
  // synchronous callers (bench_cmd, selftest_search here) — think() itself must never touch it.
  //
  // A real timing race is inherently flaky to test for directly, so this reproduces the bug
  // deterministically instead: pre-set g_stop BEFORE calling think() (simulating "the stop
  // request already landed"), with no depth/time limit at all. If think() were still resetting
  // g_stop internally, it would search for real (many nodes, real time); since it must not, the
  // very first time_up()/aborted() check has to bail immediately.
  bool selftest_stop() {
    if (!nnue::loaded()) {
      std::cout << "selftest stop FAIL: no net loaded\n";
      return false;
    }
    Position pos;
    Position::set(DEFAULT_FEN, pos);
    tt::clear();
    search::new_game();

    search::clear_stop();
    search::request_stop(); // simulate: the stop request already landed before think() runs
    const auto           t0 = std::chrono::steady_clock::now();
    const search::Result r  = search::think(pos, search::MAX_PLY - 1, nullptr, 0, 0);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    search::clear_stop(); // leave global state clean for whatever runs next

    bool ok = true;
    if (r.nodes > 100) {
      std::cout << "selftest stop FAIL: " << r.nodes << " nodes searched despite g_stop being set before think() ran\n";
      ok = false;
    }
    if (ms > 500) {
      std::cout << "selftest stop FAIL: think() took " << ms << "ms despite g_stop being set before it ran\n";
      ok = false;
    }
    if (r.best.to_from() == 0) {
      std::cout << "selftest stop FAIL: no fallback bestmove returned\n";
      ok = false;
    }
    if (ok)
      std::cout << "selftest stop PASS: a pre-set stop is honoured immediately (" << r.nodes << " nodes, " << ms
                << "ms)\n";
    return ok;
  }

  // --- selftest contempt -----------------------------------------------------------------------
  // Regression test for the "Contempt" option: with it at 0, draw_score() must reproduce the old
  // unconditional draw==0 exactly, so a position where the honest best line is worse than a
  // drawing resource (repetition/perpetual) gets searched to a ~0 score. With Contempt clearly
  // positive, that same resource must score noticeably *worse* than 0 from the root side's own
  // perspective, since draw_score() now charges it -Contempt instead of handing it out for free —
  // pushing the search to prefer fighting on unless every alternative is even worse. Compares the
  // two scores rather than pinning either to an exact value: robust to future search tuning
  // shifting the honest evaluation, while still catching a broken/reverted contempt wire-up (the
  // two scores would then be equal).
  bool selftest_contempt() {
    if (!nnue::loaded()) {
      std::cout << "selftest contempt FAIL: no net loaded\n";
      return false;
    }
    // Q+K vs Q+K+a-pawn: White (to move) is down a pawn but has real perpetual-check/repetition
    // resources, so at moderate depth the "honest" line is worse than the drawing one.
    constexpr const char *FEN   = "3qk3/p7/8/8/8/8/8/3QK3 w - - 0 1";
    constexpr int         DEPTH = 12;

    const int saved_threads  = g_threads;
    g_threads                = 1; // determinism: score comparisons need a bit-reproducible search
    search::set_threads(1);

    Position pos;
    Position::set(FEN, pos);
    tt::clear();
    search::new_game();
    search::set_contempt(0);
    const int score_c0 = search::think(pos, DEPTH, nullptr, 0, 0).score;

    Position::set(FEN, pos);
    tt::clear();
    search::new_game();
    search::set_contempt(80);
    const int score_c80 = search::think(pos, DEPTH, nullptr, 0, 0).score;

    search::set_contempt(0); // leave global state at the actual default for whatever runs next
    g_threads = saved_threads;
    search::set_threads(saved_threads);
    tt::clear();
    search::new_game();

    bool ok = true;
    if (score_c0 > 5) {
      std::cout << "selftest contempt FAIL: Contempt=0 score " << score_c0 << " (expected ~0, a free draw taken)\n";
      ok = false;
    }
    if (score_c80 > score_c0 - 30) {
      std::cout << "selftest contempt FAIL: Contempt=80 score " << score_c80 << " not clearly below Contempt=0 score "
                << score_c0 << "\n";
      ok = false;
    }
    if (ok)
      std::cout << "selftest contempt PASS: Contempt=0 score " << score_c0 << ", Contempt=80 score " << score_c80
                << "\n";
    return ok;
  }

  // --- selftest all ----------------------------------------------------------------------------
  // Runs every selftest and reports a combined verdict. Uses &= (not &&) deliberately: every
  // suite runs regardless of an earlier failure, so a single pass reports everything that's
  // broken instead of stopping at the first one.
  void selftest_all() {
    bool ok = true;
    ok &= selftest_perft();
    ok &= selftest_see();
    ok &= selftest_draw();
    ok &= selftest_nnue(500, 80);
    ok &= selftest_search();
    ok &= selftest_contempt();
    ok &= selftest_stop();
    std::cout << (ok ? "selftest all: ALL PASS\n" : "selftest all: SOME FAILED (see above)\n");
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

  // Formats a search score as a UCI "score ..." field: mate scores as "mate <n>" in moves.
  std::string format_score(int score) {
    if (score >= search::MATE_IN_MAX)
      return "mate " + std::to_string((search::MATE - score + 1) / 2);
    if (score <= -search::MATE_IN_MAX)
      return "mate " + std::to_string(-((search::MATE + score + 1) / 2));
    return "cp " + std::to_string(score);
  }

  // --- bench (search signature) ------------------------------------------------------------------
  // Fixed positions searched at a fixed depth with a fixed-size, cleared TT: the summed node
  // count is a deterministic signature of the search — any functional change moves it, a pure
  // speedup does not (the `./askaig bench` OpenBench-style convention).
  constexpr const char *BENCH_FENS[] = {
          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // startpos
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // kiwipete
          "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // perft pos3 (rook endgame)
          "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", // perft pos4 (promotions)
          "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", // perft pos5
          "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", // perft pos6
          "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 1", // dragon yugoslav
          "r2q1rk1/p1pnbppp/1p2pn2/8/2pP4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 1", // QGD
          "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 w - - 0 1", // hedgehog
          "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", // WAC.001 (mating attack)
          "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 b - - 0 1", // bishop endgame
          "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", // K+P vs K
  };
  constexpr int BENCH_DEPTH = 12;

  void bench_cmd(int depth) {
    // Runs synchronously on this thread (no concurrent stop-sender to race), but g_stop may
    // still be true from an earlier "go" + "stop" — think() no longer resets it itself (see
    // search::clear_stop()'s comment), so every non-backgrounded caller must.
    search::clear_stop();
    search::set_node_limit(0); // a prior "go nodes" must not cap the signature runs
    const size_t prev_mb = tt::size_mb(); // restore the user's Hash afterwards
    tt::resize(16); // fixed size: the signature must not depend on the Hash setting
    uint64_t   total_nodes = 0;
    const auto t0          = std::chrono::steady_clock::now();
    int        i           = 0;
    for (const char *fen: BENCH_FENS) {
      tt::clear();
      search::new_game(); // fresh heuristics per position -> bit-reproducible
      Position bp;
      Position::set(fen, bp);
      search::Result r = search::think(bp, depth, nullptr, 0, 0);
      total_nodes += r.nodes;
      std::cout << "position " << ++i << "/" << std::size(BENCH_FENS) << " bestmove " << move_to_uci(r.best)
                << " nodes " << r.nodes << "\n";
    }
    const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\nTotal time (ms) : " << ms << "\nNodes searched  : " << total_nodes
              << "\nNodes/second    : " << (ms > 0 ? total_nodes * 1000 / static_cast<uint64_t>(ms) : total_nodes)
              << "\n"
              << std::flush;
    tt::resize(prev_mb);
    tt::clear();
    search::new_game(); // don't leak bench heuristics into a real game
  }

  // --- d / display ---------------------------------------------------------------------------
  // Board preview (plain text, no ANSI codes): piece grid, the position facts, and the current
  // static NNUE eval from the observer's (White's) point of view: + = better for White,
  // - = better for Black, plus the raw centipawn value.
  void display_cmd(const Position &pos) {
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

    // The static NNUE eval, observer's (White's) POV: + = better for White, - for Black.
    std::string eval_str = "no net loaded";
    if (nnue::loaded()) {
      const int stm_cp = nnue::evaluate_refresh(pos);
      const int cp     = pos.turn() == WHITE ? stm_cp : -stm_cp;
      char      pawns[16];
      std::snprintf(pawns, sizeof pawns, "%+.2f", cp / 100.0);
      eval_str = std::string(pawns) + " (cp " + std::to_string(cp) + ")";
    }

    const std::string info[] = {
            "FEN: " + pos.fen() + " " + std::to_string(pos.fifty()) + " " + std::to_string(pos.ply() / 2 + 1),
            "Zobrist Key: " + zob.str(),
            "Castle Rights: " + castles,
            std::string("Side To Move: ") + (pos.turn() == WHITE ? "White" : "Black"),
            std::string("En Passant: ") + (ep == NO_SQUARE ? "NULL" : SQSTR[ep]),
            "Half Moves: " + std::to_string(pos.fifty()),
            std::string("In Check: ") + (check ? "true" : "false"),
            "Eval: " + eval_str,
    };

    std::cout << "   -------------------\n"; // 19 dashes: spans pipe-to-pipe of the 22-char rows
    for (int r = 7; r >= 0; --r) {
      std::cout << " " << r + 1 << " |";
      for (int f = 0; f < 8; ++f) {
        const Piece pc = pos.at(create_square(File(f), Rank(r)));
        std::cout << " " << (pc == NO_PIECE ? '.' : PIECE_STR[pc]);
      }
      std::cout << " |";
      if (const size_t i = size_t(7 - r); i < std::size(info))
        std::cout << " " << info[i];
      std::cout << "\n";
    }
    std::cout << "   -------------------\n"; // 19 dashes: spans pipe-to-pipe of the 22-char rows
    std::cout << "     A B C D E F G H\n";
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
    if (!Position::set(fen, *pos)) {
      // A malformed FEN: `*pos` is now a partially-built garbage position (set() stops the
      // instant it finds the problem, without undoing what came before) — reset to a
      // known-good position rather than handing `go`/`eval`/`d` something that could itself
      // misbehave on it (e.g. a missing king breaking bsf() on an empty bitboard).
      std::cout << "info string ignoring malformed FEN (reset to startpos): " << fen << "\n";
      pos.emplace();
      Position::set(DEFAULT_FEN, *pos);
      return;
    }

    if (token == "moves") {
      std::string mv;
      while (is >> mv)
        if (!play_uci_move(*pos, mv)) {
          std::cout << "info string ignoring illegal/unknown move " << mv << "\n";
          break;
        }
    }
  }

  constexpr int     DEFAULT_DEPTH    = 12;
  constexpr int64_t MOVE_OVERHEAD_MS = 30; // safety buffer for GUI/transport lag
  constexpr int64_t URGENT_MS        = 200; // panic floor: never plan to leave less than this on the clock

  // Handles "go ...". "go perft <depth> [noncache]" counts move-generation nodes; otherwise
  // it runs the iterative-deepening search. Recognised limits: depth <n>, infinite,
  // movetime <ms>, wtime/btime/winc/binc/movestogo. (searchmoves/ponder/nodes/mate ignored.)
  void go_cmd(Position &pos, std::istringstream &is) {
    stop_search(); // never run two searches at once, nor search while pos can change

    int         depth    = 0;
    int64_t     movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0;
    int64_t     nodes     = 0;
    int         movestogo = 0;
    bool        infinite  = false;
    std::string token;
    while (is >> token) {
      if (token == "perft") {
        int d = 1;
        is >> d;
        // Optional trailing "noncache": skip the perft-hash memoisation so every subtree is
        // recomputed (still bulk-counted at the leaves). Default uses the cache.
        std::string opt;
        const bool  use_cache = !(is >> opt && opt == "noncache");
        run_perft(pos, d, use_cache); // synchronous: perft is a debug node count, not a game move
        return;
      }
      if (token == "depth")
        is >> depth;
      else if (token == "movetime")
        is >> movetime;
      else if (token == "wtime")
        is >> wtime;
      else if (token == "btime")
        is >> btime;
      else if (token == "winc")
        is >> winc;
      else if (token == "binc")
        is >> binc;
      else if (token == "movestogo")
        is >> movestogo;
      else if (token == "nodes")
        is >> nodes;
      else if (token == "infinite")
        infinite = true;
    }

    // Soft node cap: set (or cleared) for EVERY go, so a previous "go nodes" can't leak.
    search::set_node_limit(nodes > 0 ? uint64_t(nodes) : 0);

    // Resolve the limits into (max_depth, soft_ms, hard_ms). Time limits drive the search to
    // the ply ceiling and let the clock stop it; a bare depth caps the depth instead.
    int     max_depth = DEFAULT_DEPTH;
    int64_t soft_ms = 0, hard_ms = 0;
    if (nodes > 0)
      max_depth = search::MAX_PLY; // node-capped search: run until the cap stops it
    if (movetime > 0) {
      max_depth = search::MAX_PLY;
      hard_ms   = std::max<int64_t>(movetime - MOVE_OVERHEAD_MS, 1);
      soft_ms   = 0; // fixed move time: run the whole budget
    } else if (wtime > 0 || btime > 0) {
      max_depth         = search::MAX_PLY;
      const int64_t t   = pos.turn() == WHITE ? wtime : btime;
      const int64_t inc = pos.turn() == WHITE ? winc : binc;
      const int     mtg = movestogo > 0 ? movestogo : 40;
      const int64_t avail = std::max<int64_t>(t - MOVE_OVERHEAD_MS, 1);
      // A share of the clock plus most of the increment, hard-capped at 3x that (and never
      // more than half the remaining time, with a small reserve so lag can't flag us).
      const int64_t opt     = std::min<int64_t>(avail / mtg + 3 * inc / 4, std::max<int64_t>(avail / 2, 1));
      const int64_t reserve = std::clamp<int64_t>(t / 10, MOVE_OVERHEAD_MS, 500);
      soft_ms               = std::max<int64_t>(opt, 1);
      hard_ms = std::max<int64_t>(1, std::min<int64_t>({3 * opt, std::max<int64_t>(avail / 2, 1), t - reserve}));
      // Urgent floor: whatever the budget formula above says, never let the hard deadline plan
      // to leave less than URGENT_MS on the clock. Since the deadline is polled every 2048
      // nodes and after every root move (see time_up() in search.cpp), this also catches the
      // position dynamically running long mid-search — not just a `go` that already starts
      // with little time left — and returns whatever bestmove the search has by then.
      hard_ms = std::min(hard_ms, std::max<int64_t>(t - URGENT_MS, 1));
      soft_ms = std::min(soft_ms, hard_ms);
    } else if (infinite) {
      max_depth = search::MAX_PLY; // effectively until "stop"
    } else if (depth > 0)
      max_depth = depth;
    if (depth > 0) // an explicit depth is always an upper bound, even alongside a time limit
      max_depth = std::min(max_depth, depth);

    // clear_stop() MUST happen here, synchronously on this (the UCI) thread, before the search
    // thread is spawned — not inside think() on the new thread, which could start AFTER this
    // thread has already read and processed a "stop" for it, silently discarding that request
    // (found by testing "go depth 99999" immediately followed by "stop": the search ran to
    // completion instead of stopping, because think()'s old internal reset raced the stop).
    search::clear_stop();

    // Search on a background thread; the loop keeps reading stdin so "stop"/"isready" work.
    // `pos` is safe to capture by pointer: every pos-mutating command calls stop_search() first.
    Position *pp = &pos;
    g_search     = std::thread([pp, max_depth, soft_ms, hard_ms]() {
      search::Result r = search::think(
              *pp, max_depth,
              [](int d, const search::Result &res, uint64_t nodes, long long ms) {
                const uint64_t              nps = ms > 0 ? nodes * 1000 / uint64_t(ms) : nodes * 1000;
                std::lock_guard<std::mutex> lk(g_out);
                std::cout << "info depth " << d << " seldepth " << res.seldepth << " score "
                          << format_score(res.score) << " nodes " << nodes << " nps " << nps << " hashfull "
                          << tt::hashfull() << " time " << ms;
                if (!res.pv.empty()) {
                  std::cout << " pv";
                  for (Move m: res.pv)
                    std::cout << " " << move_to_uci(m);
                }
                std::cout << "\n" << std::flush;
              },
              soft_ms, hard_ms);

      std::lock_guard<std::mutex> lk(g_out);
      std::cout << "bestmove " << (r.best.to_from() != 0 ? move_to_uci(r.best) : "0000") << "\n" << std::flush;
    });
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
      std::cout << "option name Hash type spin default " << tt::DEFAULT_HASH_MB << " min 1 max 65536\n";
      std::cout << "option name Threads type spin default 1 min 1 max " << max_threads() << "\n";
      std::cout << "option name Contempt type spin default 0 min -100 max 100\n"; // cp cost of a draw
      std::cout << "option name EvalFile type string default <embedded>\n";
      std::cout << "uciok\n";
    } else if (cmd == "isready") {
      // Must answer even mid-search, so this never touches engine state.
      std::lock_guard<std::mutex> lk(g_out);
      std::cout << "readyok\n" << std::flush;
    } else if (cmd == "ucinewgame") {
      stop_search();
      pos.emplace();
      Position::set(DEFAULT_FEN, *pos);
      tt::clear();
      search::new_game(); // history tables persist across "go"s, but not across games
    } else if (cmd == "position") {
      stop_search();
      position_cmd(pos, is);
    } else if (cmd == "go") {
      go_cmd(*pos, is);
    } else if (cmd == "stop") {
      search::request_stop(); // the search thread finishes promptly and prints its bestmove
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
      stop_search(); // resizing the TT under a running search is unsafe
      // setoption name <id> [value <x>]
      std::string token;
      std::string name;
      is >> token >> name >> token; // "name", <id>, "value"
      if (name == "Hash") {
        int mb = 0;
        if (is >> mb) {
          mb = std::clamp(mb, 1, 65536);
          tt::resize(size_t(mb));
        }
      } else if (name == "Threads") {
        // Drives both the perft workers and the lazy-SMP helper pool (Threads-1 helpers).
        int t = 0;
        if (is >> t) {
          g_threads = t < 1 ? 1 : (t > 1024 ? 1024 : t);
          search::set_threads(g_threads);
        }
      } else if (name == "Contempt") {
        // Centipawns the engine's own side pays to avoid a draw (negative: seeks draws). 0
        // reproduces the old unconditional draw==0 exactly.
        int cp = 0;
        if (is >> cp)
          search::set_contempt(std::clamp(cp, -100, 100));
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
      // "bench [depth]" = the search signature; "bench evalnps" = the NNUE micro-benchmark.
      stop_search(); // bench owns the TT/heuristics while it runs
      std::string what;
      is >> what;
      if (what == "evalnps")
        bench_evalnps();
      else {
        const int d = what.empty() ? 0 : std::atoi(what.c_str());
        bench_cmd(d > 0 ? d : BENCH_DEPTH);
      }
    } else if (cmd == "selftest") {
      // Hidden test commands (not advertised): "selftest {nnue [games] [maxply] | perft | see |
      // draw | search | contempt | stop | all}". selftest_search drives the shared TT/history/
      // SMP pool directly, so stop any running search first, exactly like bench.
      stop_search();
      std::string what;
      is >> what;
      if (what == "nnue") {
        int g = 0, p = 0;
        is >> g >> p;
        selftest_nnue(g > 0 ? g : 100, p > 0 ? p : 80);
      } else if (what == "perft")
        selftest_perft();
      else if (what == "see")
        selftest_see();
      else if (what == "draw")
        selftest_draw();
      else if (what == "search")
        selftest_search();
      else if (what == "contempt")
        selftest_contempt();
      else if (what == "stop")
        selftest_stop();
      else if (what == "all")
        selftest_all();
    } else if (cmd == "register" || cmd.empty()) {
      // accepted but no-op
    } else if (cmd == "quit" || cmd == "exit") {
      stop_search();
      break;
    }
    // unknown commands are silently ignored, as the UCI spec requires

    std::cout.flush();
  }

  stop_search(); // joins the search thread on EOF/quit so it isn't destroyed while joinable
}
