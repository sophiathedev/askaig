#include "nnue.h"

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

        const search::Result r  = search::think(pos, depth, 1, nullcb, 0, 0, false);
        const int            ws = pos.turn() == WHITE ? r.score : -r.score;
        last_white              = ws;

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

      if ((g + 1) % 50 == 0) {
        os.flush();
        std::fprintf(stderr, "datagen: %d/%d games, %llu positions\n", g + 1, games,
                     static_cast<unsigned long long>(written));
      }
    }

    os.flush();
    std::fprintf(stderr, "datagen done: %llu positions -> %s\n", static_cast<unsigned long long>(written), out);
  }

} // namespace nnue
