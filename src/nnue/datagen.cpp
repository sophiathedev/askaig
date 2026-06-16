#include "nnue.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "position.h"
#include "search.h"
#include "types.h"

// Self-play training-data generation for the NNUE trainer (bullet). Each game starts from a few random
// plies (for opening diversity), then both sides play the engine's own fixed-depth search; every quiet
// position (side-to-move not in check, best move not a capture/promotion, non-mate score) is recorded
// with the search score (side-to-move cp) and, once the game finishes, the game result (WDL, side-to-
// move view). Output is one text line per position: `<FEN> | <score> | <wdl>`, which bullet ingests via
// its text->binary converter. The labels come from the engine's HCE search (NNUE eval is not routed
// yet), which is exactly the bootstrap data for training the first net.
namespace nnue {

  namespace {

    constexpr int OPENING_RANDOM = 8; // random plies at the start of each game (diversity)
    constexpr int MAX_PLY        = 320; // hard cap on game length
    constexpr int RESIGN_SCORE   = 2000; // |white-cp| at/above which a side is considered winning
    constexpr int RESIGN_PLIES   = 6; // consecutive such plies -> adjudicate the win
    constexpr int MATE_BOUND     = 20000; // skip recording mate-coded scores (not centipawns)
    constexpr int ADJ_DRAW_BAND  = 300; // |white-cp| below this at the ply cap -> call it a draw
    constexpr int MOVE_HARD_MS   = 5000; // per-move search cap (ms): a runaway position can't stall a game

    // Runtime colour dispatch over the colour-templated Position methods.
    int legal_moves(Position &p, Move *list) {
      return p.turn() == WHITE ? int(p.generate_legals<WHITE>(list) - list)
                               : int(p.generate_legals<BLACK>(list) - list);
    }
    void do_play(Position &p, Move m) {
      if (p.turn() == WHITE)
        p.play<WHITE>(m);
      else
        p.play<BLACK>(m);
    }
    bool in_check_now(Position &p) { return p.turn() == WHITE ? p.in_check<WHITE>() : p.in_check<BLACK>(); }

    struct Rec {
      std::string fen;
      int         score; // side-to-move cp
      Color       stm;
    };

    // Compact human-readable count (1234 -> "1.2K", 2.08e6 -> "2.08M", 1.5e9 -> "1.50B").
    std::string human_count(uint64_t n) {
      char buf[32];
      if (n >= 1000000000ULL)
        std::snprintf(buf, sizeof(buf), "%.2fB", double(n) / 1e9);
      else if (n >= 1000000ULL)
        std::snprintf(buf, sizeof(buf), "%.2fM", double(n) / 1e6);
      else if (n >= 1000ULL)
        std::snprintf(buf, sizeof(buf), "%.1fK", double(n) / 1e3);
      else
        std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(n));
      return buf;
    }

    // Seconds -> "H:MM:SS" (or "M:SS" under an hour).
    std::string fmt_time(double sec) {
      if (sec < 0 || sec > 1e8)
        sec = 0;
      const int t = int(sec);
      char      buf[32];
      if (t >= 3600)
        std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", t / 3600, (t % 3600) / 60, t % 60);
      else
        std::snprintf(buf, sizeof(buf), "%d:%02d", t / 60, t % 60);
      return buf;
    }

  } // namespace

  void datagen(const char *out, int games, int depth, uint64_t seed) {
    std::ofstream os(out);
    if (!os) {
      std::fprintf(stderr, "datagen: cannot open %s\n", out);
      return;
    }
    PRNG       rng(seed); // distinct seeds -> distinct games (run parallel shards with different seeds)
    const auto nullcb  = [](int, const search::Result &, uint64_t, long long) {};
    uint64_t   written = 0;

    using clock            = std::chrono::steady_clock;
    const auto t0          = clock::now();
    auto       last_print  = t0;
    uint64_t   total_nodes = 0;
    const auto secs_since  = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double>(b - a).count();
    };

    std::fprintf(stderr, "datagen: %d games, depth %d, seed %llu -> %s\n", games, depth,
                 static_cast<unsigned long long>(seed), out);

    // Live progress redrawn in place ('\r'), throttled to ~5 updates/sec. Called BOTH per ply (so a
    // long game never freezes the line) and at the end; `done_games` is games fully completed so far.
    const auto print_progress = [&](int done_games) {
      const auto now = clock::now();
      if (done_games != games && secs_since(last_print, now) < 0.2)
        return;
      last_print             = now;
      const double elapsed   = secs_since(t0, now);
      const double done      = done_games;
      const double frac      = games > 0 ? done / games : 0;
      const double pos_per_s = elapsed > 0 ? written / elapsed : 0;
      const double g_per_s   = elapsed > 0 ? done / elapsed : 0;
      const double nps       = elapsed > 0 ? total_nodes / elapsed : 0;
      const double eta       = g_per_s > 0 ? (games - done) / g_per_s : 0;
      os.flush();
      std::fprintf(stderr, "\r[%5.1f%%] %d/%d games | %s pos | %s pos/s | %.1f g/s | %s nps | up %s | eta %s    ",
                   frac * 100.0, done_games, games, human_count(written).c_str(),
                   human_count(uint64_t(pos_per_s)).c_str(), g_per_s, human_count(uint64_t(nps)).c_str(),
                   fmt_time(elapsed).c_str(), fmt_time(eta).c_str());
      std::fflush(stderr);
    };

    for (int g = 0; g < games; ++g) {
      Position pos;
      Position::set(DEFAULT_FEN, pos);
      search::new_game();

      // Random opening for diversity; abandon the game if a random line dead-ends.
      bool ok = true;
      for (int i = 0; i < OPENING_RANDOM; ++i) {
        Move      list[256];
        const int n = legal_moves(pos, list);
        if (n == 0) {
          ok = false;
          break;
        }
        do_play(pos, list[rng.rand<uint64_t>() % unsigned(n)]);
      }
      if (!ok)
        continue;

      std::vector<Rec> recs;
      double           result_white = 0.5; // 1 = white win, 0 = black win, 0.5 = draw
      bool             decided      = false;
      int              resign_count = 0;
      int              last_white   = 0;

      for (int ply = 0; ply < MAX_PLY; ++ply) {
        if (pos.is_draw()) {
          result_white = 0.5;
          decided      = true;
          break;
        }
        Move      list[256];
        const int n = legal_moves(pos, list);
        if (n == 0) { // checkmate or stalemate
          result_white = in_check_now(pos) ? (pos.turn() == WHITE ? 0.0 : 1.0) : 0.5;
          decided      = true;
          break;
        }

        const search::Result r  = search::think(pos, depth, 1, nullcb, 0, MOVE_HARD_MS, false);
        const int            ws = pos.turn() == WHITE ? r.score : -r.score;
        last_white              = ws;
        total_nodes += r.nodes;
        print_progress(g); // keep the line alive within a long game

        const bool noisy = r.best.is_capture() || (r.best.flags() >= PR_KNIGHT && r.best.flags() <= PR_QUEEN);
        if (!in_check_now(pos) && !noisy && std::abs(r.score) < MATE_BOUND)
          recs.push_back({pos.fen(), r.score, pos.turn()});

        if (std::abs(ws) >= RESIGN_SCORE) {
          if (++resign_count >= RESIGN_PLIES) {
            result_white = ws > 0 ? 1.0 : 0.0;
            decided      = true;
            break;
          }
        } else {
          resign_count = 0;
        }

        do_play(pos, r.best);
      }

      if (!decided) // hit the ply cap: adjudicate from the last score
        result_white = last_white > ADJ_DRAW_BAND ? 1.0 : (last_white < -ADJ_DRAW_BAND ? 0.0 : 0.5);

      for (const Rec &rec: recs) {
        const double wdl = rec.stm == WHITE ? result_white : 1.0 - result_white;
        os << rec.fen << " | " << rec.score << " | " << wdl << '\n';
        ++written;
      }

      print_progress(g + 1); // game finished
    }

    os.flush();
    const double elapsed = secs_since(t0, clock::now());
    std::fprintf(stderr, "\ndatagen done: %s positions in %s (%s pos/s) -> %s\n", human_count(written).c_str(),
                 fmt_time(elapsed).c_str(), human_count(uint64_t(elapsed > 0 ? written / elapsed : 0)).c_str(), out);
  }

} // namespace nnue
