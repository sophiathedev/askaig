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
#include "eval.h"
#include "position.h"
#include "search.h"
#include "tables.h"
#include "tt.h"
#include "types.h"

namespace {

  constexpr auto ENGINE_NAME   = "Askaig 20260608";
  constexpr auto ENGINE_AUTHOR = "the Askaig developers (see AUTHORS file)";

  // Number of search threads (the "Threads" UCI option).
  int g_threads = 1;

  // Time budget for the current "go" (computed when the command is parsed). On a "ponderhit" these
  // are handed to the search so the clock starts then. Touched only on the UCI thread.
  int64_t g_ponder_soft = 0, g_ponder_hard = 0;

  // The search runs on this background thread so the UCI loop stays responsive (stop/isready work
  // mid-search). `g_out` serialises stdout so the search thread's "info"/"bestmove" lines never
  // interleave with the main thread's replies.
  std::thread g_search;
  std::mutex  g_out;

  // Stops any running search and joins its thread. Called before any command that mutates engine
  // state, so a search never runs concurrently with a position/option change.
  void stop_search() {
    search::request_stop();
    if (g_search.joinable())
      g_search.join();
  }

  std::string move_to_uci(Move m); // defined below (bench prints bestmoves)

  // --- bench ---------------------------------------------------------------------------------
  // Fixed positions for "bench": a spread of openings/middlegames/endgames (the perft-suite
  // classics plus the node-A/B suite this engine is routinely measured on). Searched at a fixed
  // depth with Threads=1 and a fixed-size, cleared TT, the summed node count is a deterministic
  // SIGNATURE of the search: any change to search/eval semantics moves it, a pure speedup does
  // not — the standard way to tell "functional" from "non-functional" patches at a glance (and
  // the `./askaig bench` convention OpenBench-style testers expect).
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
  constexpr int BENCH_DEPTH = 12; // default; "bench <depth>" overrides

  void bench_cmd(int depth) {
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
      search::Result r = search::think(
              bp, depth, /*threads=*/1, [](int, const search::Result &, uint64_t, long long) {}, 0, 0, false);
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
    search::new_game(); // don't leak bench heuristics into a real game
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

  // Formats a search score as a UCI "score ..." field. Mate scores (within MAX_MATE_PLY of
  // MATE) are reported as "mate <n>" in moves, with the sign giving who is mating.
  std::string format_score(int score) {
    constexpr int MAX_MATE_PLY = 256;
    if (score > search::MATE - MAX_MATE_PLY)
      return "mate " + std::to_string((search::MATE - score + 1) / 2);
    if (score < -(search::MATE - MAX_MATE_PLY))
      return "mate " + std::to_string(-((search::MATE + score + 1) / 2));
    return "cp " + std::to_string(score);
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
  uint64_t perft(Position &p, int depth, bool bulk) {
    // Bulk counting: at the last ply return the legal-move count directly, skipping a make/unmake of
    // every leaf (the bulk of the tree) — far faster, identical total. Non-bulk recurses to depth 0
    // and counts 1 per leaf, so play/undo runs on every move; it is the "go perft <d> nonbulk" mode
    // (a full make/unmake stress test, and the node count other engines report when not bulk-counting).
    if (bulk) {
      if (depth <= 1) {
        MoveList<Us> list(p);
        return static_cast<uint64_t>(list.size());
      }
    } else if (depth == 0)
      return 1;

    // The perft hash is a memoisation; non-bulk deliberately skips it so the whole tree is walked.
    const bool use_hash = bulk && !t_perft_tt.empty() && depth >= PERFT_HASH_MIN_DEPTH;
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
      nodes += perft<~Us>(p, depth - 1, bulk);
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
  void perft_divide(Position &p, int depth, bool bulk) {
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
    // Non-bulk skips the hash (and bulk counting) on purpose — it walks every node.
    const bool hash = bulk && depth - 1 >= PERFT_HASH_MIN_DEPTH;

    const auto start = std::chrono::steady_clock::now();

    if (depth > 1) {
      if (nthreads <= 1) {
        if (hash)
          ensure_perft_tt();
        for (size_t i = 0; i < n; ++i) {
          p.play<Us>(moves[i]);
          counts[i] = perft<~Us>(p, depth - 1, bulk);
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
            counts[i] = perft<~Us>(local, depth - 1, bulk);
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

  void run_perft(Position &pos, int depth, bool bulk) {
    if (pos.turn() == WHITE)
      perft_divide<WHITE>(pos, depth, bulk);
    else
      perft_divide<BLACK>(pos, depth, bulk);
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

  constexpr int     DEFAULT_DEPTH    = 8;
  constexpr int64_t MOVE_OVERHEAD_MS = 30; // safety buffer for GUI/transport lag, reserved from the clock

  // Plays a Move on `p`, dispatching on the side to move (Position::play is templated on Color).
  void play_any(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.play<WHITE>(m);
    else
      p.play<BLACK>(m);
  }

  // Truncates a reported principal variation so it never continues past a draw by the fifty-move rule
  // or repetition: replaying `pv` from `root`, a continuation move that would reach a draw is dropped
  // (along with the rest), so the last reported position is never a claimed draw. The first move (the
  // bestmove) is always kept, even if it itself reaches the draw. This only changes the `info ... pv`
  // / ponder OUTPUT — not the search — and silences match-runner "PV continues after fifty-move rule"
  // warnings, which the draw-seeking eval terms (fifty-move damping, drawish scaling) made frequent.
  std::vector<Move> clamp_pv_to_draw(const Position &root, const std::vector<Move> &pv) {
    Position          p = clone_position(root);
    std::vector<Move> out;
    for (Move m: pv) {
      Position nxt = clone_position(p);
      play_any(nxt, m);
      const bool draw = nxt.is_draw();
      if (draw && !out.empty())
        break; // a later move reaches a draw -> stop before it (no PV move follows a drawn position)
      play_any(p, m);
      out.push_back(m);
      if (draw)
        break; // even the first move reaches the draw: keep it (it is the bestmove) but go no further
    }
    return out;
  }

  // Handles "go ...". "go perft <depth>" counts move-generation nodes; otherwise it runs an
  // iterative-deepening, multi-threaded (Lazy SMP) negamax/alpha-beta search. Recognised limits:
  //   depth <n>            — fixed depth (default DEFAULT_DEPTH when no limit is given)
  //   infinite             — until "stop"
  //   movetime <ms>        — fixed time per move
  //   wtime/btime <ms>, winc/binc <ms>, movestogo <n> — clock-based time management
  // (searchmoves/ponder/nodes/mate are not implemented and are ignored.)
  void go_cmd(Position &pos, std::istringstream &is) {
    stop_search(); // never run two searches at once, nor search while pos can change

    int         depth    = 0; // 0 => not specified
    int64_t     movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0;
    int         movestogo = 0;
    bool        infinite  = false;
    bool        ponder    = false;
    std::string token;
    while (is >> token) {
      if (token == "perft") {
        int d = 1;
        is >> d;
        // Optional trailing "nonbulk": disable bulk counting (recurse to depth 0, make/unmake every
        // leaf) — slower, same total, exercises the full play/undo path. Default is bulk counting.
        std::string opt;
        const bool  bulk = !(is >> opt && opt == "nonbulk");
        run_perft(pos, d, bulk); // synchronous: perft is a debug node count, not a game move
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
      else if (token == "infinite")
        infinite = true;
      else if (token == "ponder")
        ponder = true;
    }

    // Resolve the limits into (max_depth, soft_ms, hard_ms). Time limits (<=0 means none) drive the
    // search to the ply ceiling and let the clock stop it; a bare depth caps it instead.
    int     max_depth = DEFAULT_DEPTH;
    int64_t soft_ms = 0, hard_ms = 0;
    if (movetime > 0) {
      max_depth = search::MAX_DEPTH;
      hard_ms   = std::max<int64_t>(movetime - MOVE_OVERHEAD_MS, 1);
      soft_ms   = 0; // a fixed move time: run the whole budget, no adaptive early stop
    } else if (wtime > 0 || btime > 0) {
      max_depth           = search::MAX_DEPTH;
      const int64_t t     = pos.turn() == WHITE ? wtime : btime; // our remaining time
      const int64_t inc   = pos.turn() == WHITE ? winc : binc; // our increment
      const int     mtg   = movestogo > 0 ? movestogo : 40; // assume 40 moves to go under sudden death
      const int64_t avail = std::max<int64_t>(t - MOVE_OVERHEAD_MS, 1);
      // Optimum slice of the clock for this move (a share of the remaining time plus the increment);
      // the search scales it by best-move stability. The hard cap bounds a single move to 3x the
      // optimum and never more than HALF of what we have left — a long final iteration can't drain
      // the bank and leave us scraping by 2s in the middlegame. (Was 5x / 75%, which front-loaded.)
      const int64_t opt = std::min(avail / mtg + inc, avail);
      soft_ms           = std::max<int64_t>(opt, 1);
      hard_ms           = std::max<int64_t>(soft_ms, std::min<int64_t>(opt * 3, std::max<int64_t>(avail / 2, 1)));
    } else if (infinite) {
      max_depth = search::MAX_DEPTH; // search to the ply ceiling — effectively until "stop"
    } else if (depth > 0) {
      max_depth = depth;
    }
    if (ponder) // ponder: think deep on the opponent's clock until ponderhit/stop arms the limit
      max_depth = search::MAX_DEPTH;
    if (depth > 0) // an explicit depth is always an upper bound, even alongside a time limit
      max_depth = std::min(max_depth, depth);

    // Remember the budget so a later "ponderhit" can start the clock (the search ignores it until then).
    g_ponder_soft = soft_ms;
    g_ponder_hard = hard_ms;

    // Search on a background thread; the loop keeps reading stdin so "stop"/"isready" stay live.
    // `pos` is safe to capture by pointer: every pos-mutating command calls stop_search() first.
    Position *pp = &pos;
    g_search     = std::thread([pp, max_depth, soft_ms, hard_ms, ponder]() {
      search::Result r = search::think(
              *pp, max_depth, g_threads,
              [pp](int d, const search::Result &res, uint64_t nodes, long long ms) {
                const uint64_t              nps = ms > 0 ? nodes * 1000 / static_cast<uint64_t>(ms) : nodes * 1000;
                const std::vector<Move>     pv  = clamp_pv_to_draw(*pp, res.pv); // don't report past a draw
                std::lock_guard<std::mutex> lk(g_out);
                std::cout << "info depth " << d << " seldepth " << res.seldepth << " score " << format_score(res.score)
                          << " nodes " << nodes << " nps " << nps << " time " << ms;
                if (!pv.empty()) {
                  std::cout << " pv";
                  for (Move m: pv)
                    std::cout << " " << move_to_uci(m);
                } else if (res.best.to_from() != 0)
                  std::cout << " pv " << move_to_uci(res.best);
                std::cout << "\n" << std::flush;
              },
              soft_ms, hard_ms, ponder);

      const std::vector<Move>     pv = clamp_pv_to_draw(*pp, r.pv); // don't ponder into a draw either
      std::lock_guard<std::mutex> lk(g_out);
      std::cout << "bestmove " << (r.best.to_from() != 0 ? move_to_uci(r.best) : "0000");
      if (pv.size() >= 2) // the move we expect the opponent to reply with -> the GUI can ponder on it
        std::cout << " ponder " << move_to_uci(pv[1]);
      std::cout << "\n" << std::flush;
    });
  }

} // namespace

void uci::bench(int depth) { bench_cmd(depth > 0 ? depth : BENCH_DEPTH); }

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
      std::cout << "option name Ponder type check default false\n"; // enables the GUI to send "go ponder"
      // SPSA-tunable search constants (pruning margins, depth gates) — advertised so a tuner can set
      // them; harmless to a normal GUI (it just ignores options it doesn't use).
      for (const search::Tunable &t: search::tunables())
        std::cout << "option name " << t.name << " type spin default " << *t.ptr << " min " << t.min << " max " << t.max
                  << "\n";
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
      search::new_game(); // the move-ordering statistics persist across "go"s, but not across games
    } else if (cmd == "position") {
      stop_search();
      position_cmd(pos, is);
    } else if (cmd == "go") {
      go_cmd(*pos, is);
    } else if (cmd == "d" || cmd == "display") {
      std::cout << *pos;
    } else if (cmd == "eval") {
      // Debug helper (like Stockfish's "eval"): the raw static evaluation of the current position,
      // in centipawns from the side to move's perspective. No search — for eval-term testing/tuning.
      std::cout << "eval " << eval::evaluate(*pos) << " cp (side to move)\n" << std::flush;
    } else if (cmd == "setoption") {
      stop_search(); // resizing the TT (or changing threads) under a running search is unsafe
      // setoption name <id> [value <x>]
      std::string token;
      std::string name;
      is >> token >> name >> token; // "name", <id>, "value"
      if (name == "Hash") {
        int mb = 0;
        if (is >> mb) {
          mb = mb < 1 ? 1 : (mb > 65536 ? 65536 : mb);
          tt::resize(static_cast<size_t>(mb));
        }
      } else if (name == "Threads") {
        int t = 0;
        if (is >> t)
          g_threads = t < 1 ? 1 : (t > 1024 ? 1024 : t);
      } else {
        // A search tunable (SPSA): setoption name <X> value <n>. set_tunable clamps to the option's
        // [min,max] and returns false for an unknown name (silently ignored, per the UCI spec).
        int v = 0;
        if (is >> v)
          search::set_tunable(name, v);
      }
    } else if (cmd == "stop") {
      search::request_stop(); // the search thread finishes promptly and prints its bestmove
    } else if (cmd == "ponderhit") {
      // The predicted move was played: start our clock now (no-op if we aren't pondering).
      search::request_ponderhit(g_ponder_soft, g_ponder_hard);
    } else if (cmd == "bench") {
      stop_search(); // bench owns the TT/heuristics while it runs
      int d = 0;
      is >> d;
      bench_cmd(d > 0 ? d : BENCH_DEPTH);
    } else if (cmd == "register" || cmd.empty()) {
      // accepted but no-op
    } else if (cmd == "quit" || cmd == "exit") {
      stop_search();
      break;
    }
    // unknown commands are silently ignored, as the UCI spec requires

    std::cout.flush();
  }

  stop_search(); // joins the search thread on EOF/quit so its std::thread isn't destroyed while joinable
}
