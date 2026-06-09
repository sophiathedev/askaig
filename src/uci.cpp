#include "uci.h"
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
  uint64_t perft(Position &p, int depth) {
    if (depth <= 1) {
      MoveList<Us> list(p);
      return static_cast<uint64_t>(list.size());
    }

    const bool use_hash = !t_perft_tt.empty() && depth >= PERFT_HASH_MIN_DEPTH;
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
      nodes += perft<~Us>(p, depth - 1);
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
  void perft_divide(Position &p, int depth) {
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
    const bool hash = depth - 1 >= PERFT_HASH_MIN_DEPTH;

    const auto start = std::chrono::steady_clock::now();

    if (depth > 1) {
      if (nthreads <= 1) {
        if (hash)
          ensure_perft_tt();
        for (size_t i = 0; i < n; ++i) {
          p.play<Us>(moves[i]);
          counts[i] = perft<~Us>(p, depth - 1);
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
            counts[i] = perft<~Us>(local, depth - 1);
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

  void run_perft(Position &pos, int depth) {
    if (pos.turn() == WHITE)
      perft_divide<WHITE>(pos, depth);
    else
      perft_divide<BLACK>(pos, depth);
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

  constexpr int DEFAULT_DEPTH = 8;

  // Handles "go ...". "go perft <depth>" counts move-generation nodes; otherwise it runs an
  // iterative-deepening, multi-threaded (Lazy SMP) negamax/alpha-beta search ("go depth <n>",
  // default DEFAULT_DEPTH), printing an info line per depth. Time-control args are ignored.
  void go_cmd(Position &pos, std::istringstream &is) {
    stop_search(); // never run two searches at once, nor search while pos can change

    int         depth = DEFAULT_DEPTH;
    std::string token;
    while (is >> token) {
      if (token == "perft") {
        int d = 1;
        is >> d;
        run_perft(pos, d); // synchronous: perft is a debug node count, not a game move
        return;
      }
      if (token == "depth")
        is >> depth;
    }

    // Search on a background thread; the loop keeps reading stdin so "stop"/"isready" stay live.
    // `pos` is safe to capture by pointer: every pos-mutating command calls stop_search() first.
    Position *pp = &pos;
    g_search     = std::thread([pp, depth]() {
      search::Result r =
              search::think(*pp, depth, g_threads, [](int d, const search::Result &res, uint64_t nodes, long long ms) {
                const uint64_t              nps = ms > 0 ? nodes * 1000 / static_cast<uint64_t>(ms) : nodes * 1000;
                std::lock_guard<std::mutex> lk(g_out);
                std::cout << "info depth " << d << " score " << format_score(res.score) << " nodes " << nodes << " nps "
                          << nps << " time " << ms;
                if (!res.pv.empty()) {
                  std::cout << " pv";
                  for (Move m: res.pv)
                    std::cout << " " << move_to_uci(m);
                } else if (res.best.to_from() != 0)
                  std::cout << " pv " << move_to_uci(res.best);
                std::cout << "\n" << std::flush;
              });

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
    } else if (cmd == "position") {
      stop_search();
      position_cmd(pos, is);
    } else if (cmd == "go") {
      go_cmd(*pos, is);
    } else if (cmd == "d" || cmd == "display") {
      std::cout << *pos;
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
      }
    } else if (cmd == "stop") {
      search::request_stop(); // the search thread finishes promptly and prints its bestmove
    } else if (cmd == "ponderhit" || cmd == "register" || cmd.empty()) {
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
