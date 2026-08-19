#include "uci.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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
#include "datagen.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "see.h"
#include "syzygy.h"
#include "tables.h"
#include "tt.h"
#include "types.h"

namespace {

#ifndef ASKAIG_VERSION
#define ASKAIG_VERSION "dev" // set by cmake
#endif
  constexpr auto ENGINE_NAME   = "Askaig " ASKAIG_VERSION;
  constexpr auto ENGINE_AUTHOR = "the Askaig developers (see AUTHORS file)";

  int g_threads = 1;

  std::thread g_search;
  std::mutex  g_out;

  void stop_search() {
    search::request_stop();
    if (g_search.joinable())
      g_search.join();
  }

  unsigned max_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
  }

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
      constexpr std::array<char, 4> promo = {'n', 'b', 'r', 'q'};
      s += promo[f & 0b11];
    }
    return s;
  }

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

  struct PerftEntry {
    uint64_t board[4];
    Bitboard castle;
    uint64_t count;
    int16_t  epsq;
    int8_t   side;
    int8_t   depth; // -1 = empty
  };

  constexpr size_t PERFT_TT_ENTRIES     = 1u << 20;
  constexpr int    PERFT_HASH_MIN_DEPTH = 4;

  thread_local std::vector<PerftEntry> t_perft_tt;

  void ensure_perft_tt() {
    if (t_perft_tt.size() != PERFT_TT_ENTRIES)
      t_perft_tt.assign(PERFT_TT_ENTRIES, PerftEntry{{0, 0, 0, 0}, 0, 0, NO_SQUARE, 0, -1});
  }

  PerftEntry perft_fingerprint(const Position &p) {
    PerftEntry e{{0, 0, 0, 0}, 0, 0, NO_SQUARE, 0, -1};
    for (int s = 0; s < int(NSQUARES); ++s)
      e.board[s >> 4] |= static_cast<uint64_t>(p.at(Square(s)) & 0xF) << ((s & 15) * 4);
    e.castle = p.history[p.ply()].entry;
    e.epsq   = static_cast<int16_t>(p.history[p.ply()].epsq);
    e.side   = static_cast<int8_t>(p.turn());
    return e;
  }

  bool same_position(const PerftEntry &a, const PerftEntry &b) {
    return a.board[0] == b.board[0] && a.board[1] == b.board[1] && a.board[2] == b.board[2] &&
           a.board[3] == b.board[3] && a.castle == b.castle && a.epsq == b.epsq && a.side == b.side;
  }

  size_t perft_index(uint64_t hash, const PerftEntry &k) {
    const uint64_t h = hash ^ (static_cast<uint64_t>(static_cast<uint16_t>(k.epsq)) * 0x9E3779B97F4A7C15ull) ^ k.castle;
    return h & (PERFT_TT_ENTRIES - 1);
  }

  template<Color Us>
  uint64_t perft(Position &p, int depth, bool use_cache) {
    if (depth <= 1) {
      MoveList<Us> list(p);
      return static_cast<uint64_t>(list.size());
    }

    const bool use_hash = use_cache && !t_perft_tt.empty() && depth >= PERFT_HASH_MIN_DEPTH;
    PerftEntry key{{0, 0, 0, 0}, 0, 0, NO_SQUARE, 0, -1};
    size_t     idx = 0;
    if (use_hash) {
      key                 = perft_fingerprint(p);
      idx                 = perft_index(p.get_hash(), key);
      const PerftEntry &e = t_perft_tt[idx];
      if (e.depth == depth && same_position(e, key))
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
      PerftEntry &e = t_perft_tt[idx]; // depth preferred
      if (e.depth <= depth) {
        key.count = nodes;
        key.depth = static_cast<int8_t>(depth);
        e         = key;
      }
    }
    return nodes;
  }

  Position clone_position(const Position &src) {
    Position dst;
    std::memcpy(static_cast<void *>(&dst), static_cast<const void *>(&src), sizeof(Position));
    return dst;
  }

  template<Color Us>
  void perft_divide(Position &p, int depth, bool use_cache) {
    std::vector<Move> moves;
    {
      MoveList<Us> list(p);
      moves.assign(list.begin(), list.end());
    }
    const size_t          n = moves.size();
    std::vector<uint64_t> counts(n, 1);

    int nthreads = g_threads < 1 ? 1 : g_threads;
    if (nthreads > static_cast<int>(n))
      nthreads = static_cast<int>(n == 0 ? 1 : n);

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
            ensure_perft_tt();
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
        worker();
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


  void play_any_color(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.play<WHITE>(m);
    else
      p.play<BLACK>(m);
  }
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
          break;
        const Move m = moves[rng.rand<uint64_t>() % n];
        ev.push(pos, m);
        play_any_color(pos, m);
        made.push_back(m);

        if (rng.rand<uint64_t>() % 2 == 0 && !check(pos))
          return false;
      }
      if (!check(pos))
        return false;
    }
    std::cout << "selftest nnue PASS: " << games << " games, " << checks << " eval checks, incremental == refresh\n";
    return true;
  }

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
        std::cout << "selftest perft FAIL: depth " << c.depth << " got " << got << " expected " << c.expected << " fen "
                  << c.fen << "\n";
        ok = false;
      }
    }
    if (ok)
      std::cout << "selftest perft PASS: " << std::size(CASES) << " known positions match exactly\n";
    return ok;
  }

  bool selftest_see() {
    bool       ok     = true;
    const auto expect = [&](bool cond, const char *what) {
      if (!cond) {
        std::cout << "selftest see FAIL: " << what << "\n";
        ok = false;
      }
    };

    {
      Position pos;
      Position::set("4k3/8/8/8/3p4/8/8/3R3K w - - 0 1", pos);
      const Move m(d1, d4, CAPTURE);
      expect(search::see_ge(pos, m, 100), "undefended pawn capture: expected SEE >= 100");
      expect(!search::see_ge(pos, m, 101), "undefended pawn capture: expected SEE < 101");
    }
    {
      Position pos;
      Position::set("4k3/8/8/2p5/3n4/8/8/3R3K w - - 0 1", pos);
      const Move m(d1, d4, CAPTURE);
      expect(search::see_ge(pos, m, -180), "pawn-defended knight capture: expected SEE >= -180");
      expect(!search::see_ge(pos, m, -179), "pawn-defended knight capture: expected SEE < -179");
    }
    {
      Position pos;
      Position::set("3rk3/8/8/8/3r4/8/8/3R3K w - - 0 1", pos);
      const Move m(d1, d4, CAPTURE);
      expect(search::see_ge(pos, m, 0), "even rook trade: expected SEE >= 0");
      expect(!search::see_ge(pos, m, 1), "even rook trade: expected SEE < 1");
    }

    {
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
            bool prev = true;
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

  bool selftest_draw() {
    bool       ok     = true;
    const auto expect = [&](bool cond, const char *what) {
      if (!cond) {
        std::cout << "selftest draw FAIL: " << what << "\n";
        ok = false;
      }
    };

    {
      Position pos;
      Position::set(DEFAULT_FEN, pos);
      expect(!pos.is_draw(), "startpos incorrectly flagged as a draw");
    }
    {
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
    {
      Position pos;
      Position::set("r3k3/8/8/8/8/8/8/R3K3 w - - 99 1", pos);
      expect(!pos.is_draw(), "fifty-move rule fired one halfmove early");
      if (!play_uci_move(pos, "e1d1")) {
        std::cout << "selftest draw FAIL: illegal fifty-move test move\n";
        return false;
      }
      expect(pos.is_draw(), "fifty-move rule did not fire at halfmove 100");
    }
    {
      Position pos;
      Position::set("5k2/8/8/8/8/8/8/4K2R w K - 99 1", pos);
      if (!play_uci_move(pos, "e1g1")) {
        std::cout << "selftest draw FAIL: illegal castling fifty-move test\n";
        return false;
      }
      expect(pos.is_draw(), "castling did not advance the fifty-move clock");
    }
    {
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

  bool selftest_search() {
    if (!nnue::loaded()) {
      std::cout << "selftest search FAIL: no net loaded\n";
      return false;
    }
    search::clear_stop();
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
      std::cout << "selftest search PASS: " << std::size(CASES)
                << " positions x {1,2,4} threads, legal PV + sane score\n";
    return ok;
  }

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
    search::request_stop();
    const auto           t0 = std::chrono::steady_clock::now();
    const search::Result r  = search::think(pos, search::MAX_PLY - 1, nullptr, 0, 0);
    const auto           ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    search::clear_stop();

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

  bool selftest_contempt() {
    if (!nnue::loaded()) {
      std::cout << "selftest contempt FAIL: no net loaded\n";
      return false;
    }
    constexpr const char *FEN   = "3qk3/p7/8/8/8/8/8/3QK3 w - - 0 1";
    constexpr int         DEPTH = 12;

    const int saved_threads = g_threads;
    g_threads               = 1; // deterministic scores
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

    search::set_contempt(0);
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

  void bench_evalnps() {
    if (!nnue::loaded()) {
      std::cout << "bench evalnps FAIL: no net loaded\n";
      return;
    }
    Position pos;
    Position::set(DEFAULT_FEN, pos);
    nnue::Evaluator ev;
    ev.reset(pos);

    const std::array<Move, 4> cyc = {Move(g1, f3, QUIET), Move(g8, f6, QUIET), Move(f3, g1, QUIET),
                                     Move(f6, g8, QUIET)};

    volatile int64_t sink = 0;

    constexpr int ITERS = 500'000;
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
    std::cout << "incremental: " << evals << " evals in " << us / 1000
              << " ms = " << (us > 0 ? evals * 1'000'000 / uint64_t(us) : 0) << " evals/s\n";

    constexpr int RITERS = 50'000;
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
    std::cout << "refresh:     " << revals << " evals in " << us / 1000
              << " ms = " << (us > 0 ? revals * 1'000'000 / uint64_t(us) : 0) << " evals/s\n";
    std::cout << "checksum " << sink << "\n"; // cross-build invariant
  }

  std::string format_score(int score) {
    if (score >= search::MATE_IN_MAX)
      return "mate " + std::to_string((search::MATE - score + 1) / 2);
    if (score <= -search::MATE_IN_MAX)
      return "mate " + std::to_string(-((search::MATE + score + 1) / 2));
    return "cp " + std::to_string(score);
  }

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
    search::clear_stop();
    search::set_node_limit(0);
    const size_t prev_mb = tt::size_mb();
    tt::resize(16); // deterministic signature
    uint64_t   total_nodes = 0;
    const auto t0          = std::chrono::steady_clock::now();
    int        i           = 0;
    for (const char *fen: BENCH_FENS) {
      tt::clear();
      search::new_game();
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
    search::new_game();
  }

  void display_cmd(const Position &pos) {
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

    std::cout << "   -------------------\n";
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
    std::cout << "   -------------------\n";
    std::cout << "     A B C D E F G H\n";
  }

  void position_cmd(std::optional<Position> &pos, std::istringstream &is) {
    std::string token;
    std::string fen;
    is >> token;

    if (token == "startpos") {
      fen = DEFAULT_FEN;
      is >> token;
    } else if (token == "fen") {
      while (is >> token && token != "moves") {
        if (!fen.empty())
          fen += ' ';
        fen += token;
      }
    } else {
      return;
    }

    pos.emplace(); // set needs a fresh position
    if (!Position::set(fen, *pos)) {
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
  constexpr int64_t MOVE_OVERHEAD_MS = 30;
  constexpr int64_t URGENT_MS        = 200;

  bool parse_int_option(const std::string &value, int &out) {
    std::istringstream input(value);
    char               trailing;
    return bool(input >> out) && !(input >> trailing);
  }

  bool parse_bool_option(std::string value, bool &out) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (value == "true" || value == "1" || value == "on") {
      out = true;
      return true;
    }
    if (value == "false" || value == "0" || value == "off") {
      out = false;
      return true;
    }
    return false;
  }

  void go_cmd(Position &pos, std::istringstream &is) {
    stop_search(); // serialize searches

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
        std::string opt;
        const bool  use_cache = !(is >> opt && opt == "noncache");
        run_perft(pos, d, use_cache);
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

    search::set_node_limit(nodes > 0 ? uint64_t(nodes) : 0);

    int     max_depth = DEFAULT_DEPTH;
    int64_t soft_ms = 0, hard_ms = 0;
    if (nodes > 0)
      max_depth = search::MAX_PLY;
    if (movetime > 0) {
      max_depth = search::MAX_PLY;
      hard_ms   = std::max<int64_t>(movetime - MOVE_OVERHEAD_MS, 1);
      soft_ms   = 0;
    } else if (wtime > 0 || btime > 0) {
      max_depth             = search::MAX_PLY;
      const int64_t t       = pos.turn() == WHITE ? wtime : btime;
      const int64_t inc     = pos.turn() == WHITE ? winc : binc;
      const int     mtg     = movestogo > 0 ? movestogo : 40;
      const int64_t avail   = std::max<int64_t>(t - MOVE_OVERHEAD_MS, 1);
      const int64_t opt     = std::min<int64_t>(avail / mtg + 3 * inc / 4, std::max<int64_t>(avail / 2, 1));
      const int64_t reserve = std::clamp<int64_t>(t / 10, MOVE_OVERHEAD_MS, 500);
      soft_ms               = std::max<int64_t>(opt, 1);
      hard_ms = std::max<int64_t>(1, std::min<int64_t>({3 * opt, std::max<int64_t>(avail / 2, 1), t - reserve}));
      hard_ms = std::min(hard_ms, std::max<int64_t>(t - URGENT_MS, 1));
      soft_ms = std::min(soft_ms, hard_ms);
    } else if (infinite) {
      max_depth = search::MAX_PLY;
    } else if (depth > 0)
      max_depth = depth;
    if (depth > 0)
      max_depth = std::min(max_depth, depth);

    search::clear_stop();

    Position *pp = &pos;
    g_search     = std::thread([pp, max_depth, soft_ms, hard_ms]() {
      search::Result r = search::think(
              *pp, max_depth,
              [](int d, const search::Result &res, uint64_t nodes, long long ms) {
                const uint64_t              nps = ms > 0 ? nodes * 1000 / uint64_t(ms) : nodes * 1000;
                std::lock_guard<std::mutex> lk(g_out);
                std::cout << "info depth " << d << " seldepth " << res.seldepth << " score " << format_score(res.score)
                          << " nodes " << nodes << " nps " << nps << " hashfull " << tt::hashfull() << " tbhits "
                          << res.tbhits << " time " << ms;
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

void uci::loop(bool tune) {
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
      std::cout << "option name SyzygyPath type string default <empty>\n";
      std::cout << "option name SyzygyProbeDepth type spin default 1 min 1 max 100\n";
      std::cout << "option name Syzygy50MoveRule type check default true\n";
      std::cout << "option name SyzygyProbeLimit type spin default 7 min 0 max 7\n";
      if (tune) // hidden spsa options
        for (const auto &p: search::tunables())
          std::cout << "option name " << p.name << " type spin default " << p.def << " min " << p.lo << " max " << p.hi
                    << "\n";
      std::cout << "uciok\n";
    } else if (cmd == "isready") {
      std::lock_guard<std::mutex> lk(g_out);
      std::cout << "readyok\n" << std::flush;
    } else if (cmd == "ucinewgame") {
      stop_search();
      pos.emplace();
      Position::set(DEFAULT_FEN, *pos);
      tt::clear();
      search::new_game();
    } else if (cmd == "position") {
      stop_search();
      position_cmd(pos, is);
    } else if (cmd == "go") {
      go_cmd(*pos, is);
    } else if (cmd == "stop") {
      search::request_stop();
    } else if (cmd == "d" || cmd == "display") {
      display_cmd(*pos);
    } else if (cmd == "eval") {
      if (!nnue::loaded())
        std::cout << "info string no NNUE net loaded (use setoption name EvalFile value <path>)\n";
      else {
        const int cp = nnue::evaluate_refresh(*pos);
        std::cout << "eval " << cp << " cp (side to move), " << (pos->turn() == WHITE ? cp : -cp) << " cp (white)\n";
      }
    } else if (cmd == "setoption") {
      stop_search(); // stop before resize
      std::string token;
      std::string name;
      std::string value;
      is >> token;
      if (token != "name") {
        std::cout << "info string setoption rejected: expected 'name'\n";
        std::cout.flush();
        continue;
      }
      while (is >> token && token != "value") {
        if (!name.empty())
          name += ' ';
        name += token;
      }
      if (token == "value") {
        std::getline(is, value);
        const size_t start = value.find_first_not_of(' ');
        value              = start == std::string::npos ? std::string{} : value.substr(start);
      }

      if (name == "Hash") {
        int mb = 0;
        if (parse_int_option(value, mb)) {
          mb = std::clamp(mb, 1, 65536);
          tt::resize(size_t(mb));
          std::cout << "info string option Hash set to " << tt::size_mb() << "\n";
        } else
          std::cout << "info string option Hash rejected: invalid integer '" << value << "'\n";
      } else if (name == "Threads") {
        int t = 0;
        if (parse_int_option(value, t)) {
          g_threads = t < 1 ? 1 : (t > 1024 ? 1024 : t);
          search::set_threads(g_threads);
          std::cout << "info string option Threads set to " << g_threads << "\n";
        } else
          std::cout << "info string option Threads rejected: invalid integer '" << value << "'\n";
      } else if (name == "Contempt") {
        int cp = 0;
        if (parse_int_option(value, cp)) {
          cp = std::clamp(cp, -100, 100);
          search::set_contempt(cp);
          std::cout << "info string option Contempt set to " << cp << "\n";
        } else
          std::cout << "info string option Contempt rejected: invalid integer '" << value << "'\n";
      } else if (name == "EvalFile") {
        std::string err;
        if (value.empty() || !nnue::load_file(value, &err))
          std::cout << "info string option EvalFile rejected: '" << value << "' ("
                    << (value.empty() ? "empty path" : err) << "), keeping the current net\n";
        else
          std::cout << "info string option EvalFile set to '" << value << "'\n";
      } else if (name == "SyzygyPath") {
        if (syzygy::set_path(value)) {
          tt::clear();
          std::cout << "info string option SyzygyPath set to '" << syzygy::path() << "' (max "
                    << syzygy::max_cardinality() << " pieces)\n";
        } else
          std::cout << "info string option SyzygyPath rejected: '" << value << "'\n";
      } else if (name == "SyzygyProbeDepth") {
        int depth = 0;
        if (parse_int_option(value, depth)) {
          syzygy::set_probe_depth(depth);
          tt::clear();
          std::cout << "info string option SyzygyProbeDepth set to " << syzygy::probe_depth << "\n";
        } else
          std::cout << "info string option SyzygyProbeDepth rejected: invalid integer '" << value << "'\n";
      } else if (name == "Syzygy50MoveRule") {
        bool enabled = false;
        if (parse_bool_option(value, enabled)) {
          syzygy::set_50_move_rule(enabled);
          tt::clear();
          std::cout << "info string option Syzygy50MoveRule set to " << (enabled ? "true" : "false") << "\n";
        } else
          std::cout << "info string option Syzygy50MoveRule rejected: invalid boolean '" << value << "'\n";
      } else if (name == "SyzygyProbeLimit") {
        int limit = 0;
        if (parse_int_option(value, limit)) {
          syzygy::set_probe_limit(limit);
          tt::clear();
          std::cout << "info string option SyzygyProbeLimit set to " << syzygy::probe_limit << " (effective "
                    << syzygy::cardinality << ")\n";
        } else
          std::cout << "info string option SyzygyProbeLimit rejected: invalid integer '" << value << "'\n";
      } else if (tune) {
        bool matched = false;
        for (const auto &p: search::tunables())
          if (name == p.name) {
            int v   = 0;
            matched = true;
            if (parse_int_option(value, v)) {
              *p.p = std::clamp(v, p.lo, p.hi);
              search::params_dirty();
              std::cout << "info string option " << p.name << " set to " << *p.p << "\n";
            } else
              std::cout << "info string option " << p.name << " rejected: invalid integer '" << value << "'\n";
            break;
          }
        if (!matched)
          std::cout << "info string option '" << name << "' ignored: unknown option\n";
      } else {
        std::cout << "info string option '" << name << "' ignored: unknown option\n";
      }
    } else if (cmd == "bench") {
      stop_search();
      std::string what;
      is >> what;
      if (what == "evalnps")
        bench_evalnps();
      else {
        const int d = what.empty() ? 0 : std::atoi(what.c_str());
        bench_cmd(d > 0 ? d : BENCH_DEPTH);
      }
    } else if (cmd == "datagen") {
      stop_search();
      uint64_t    count = 0, nodes = 5000, seed = 1, tmp = 0;
      std::string out;
      is >> count >> out;
      if (is >> tmp)
        nodes = tmp;
      if (is >> tmp)
        seed = tmp;
      if (count == 0 || out.empty())
        std::cout << "usage: datagen <count> <out.bf> [nodes] [seed]\n";
      else
        datagen::run(count, out, nodes, seed);
    } else if (cmd == "selftest") {
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
    } else if (cmd == "quit" || cmd == "exit") {
      stop_search();
      break;
    }

    std::cout.flush();
  }

  stop_search(); // join on eof
}
