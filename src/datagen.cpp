#include "datagen.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include "position.h"
#include "search.h"
#include "smp.h"
#include "tt.h"
#include "types.h"

// Self-play generator. Per game: a random opening (8 or 9 plies, alternating so both colors
// lead), a balance gate, then node-limited self-play to the end; every quiet position is
// buffered and the finished game's result is stamped into the records before they hit disk.
namespace {

  using search::do_move;
  using search::is_quiet;
  using search::stm_in_check;
  using Clock = std::chrono::steady_clock;

  constexpr int OPENING_GATE_CP = 400; // openings this lopsided are thrown away
  constexpr int SCORE_CAP_CP    = 1000; // records with |score| past this are dropped
  constexpr int WIN_ADJ_PLIES   = 4; // consecutive |score| >= SCORE_CAP_CP plies to adjudicate
  constexpr int DRAW_ADJ_PLIES  = 8; // consecutive |score| <= 10 plies (after ply 80)
  constexpr int MAX_PLIES       = 400;

  // bulletformat "ChessBoard": see tools/nnue/convert_fen.py for the authoritative layout.
  struct Record {
    uint64_t occ;
    uint8_t  pcs[16];
    int16_t  score; // side-to-move relative centipawns
    uint8_t  result; // 0 = stm loss, 1 = draw, 2 = stm win
    uint8_t  ksq; // stm king square (normalized)
    uint8_t  opp_ksq; // opponent king square (normalized) ^ 56
    uint8_t  pad[3];
  };
  static_assert(sizeof(Record) == 32);

  // Encodes `pos` stm-normalized: black to move mirrors squares (^56) and swaps colors, so
  // nibble color 0 always means "the side to move". `score` is already stm-relative.
  Record encode(const Position &pos, int score) {
    Record     r{};
    const bool black = pos.turn() == BLACK;
    r.score          = int16_t(score);

    int idx = 0;
    for (int sq = 0; sq < NSQUARES; ++sq) {
      const Square orig = Square(black ? sq ^ 56 : sq);
      const Piece  pc   = pos.at(orig);
      if (pc == NO_PIECE)
        continue;
      const int type = type_of(pc);
      const int col  = (color_of(pc) == pos.turn()) ? 0 : 1; // 0 = stm
      r.occ |= 1ull << sq;
      r.pcs[idx / 2] |= uint8_t(((col << 3) | type) << (4 * (idx % 2)));
      if (type == KING) {
        if (col == 0)
          r.ksq = uint8_t(sq);
        else
          r.opp_ksq = uint8_t(sq ^ 56);
      }
      ++idx;
    }
    return r;
  }

  size_t count_legals(Position &pos) {
    Move buf[218];
    return pos.turn() == WHITE ? size_t(pos.generate_legals<WHITE>(buf) - buf)
                               : size_t(pos.generate_legals<BLACK>(buf) - buf);
  }

  Move random_legal(Position &pos, PRNG &rng) {
    Move         buf[218];
    const size_t n = pos.turn() == WHITE ? size_t(pos.generate_legals<WHITE>(buf) - buf)
                                         : size_t(pos.generate_legals<BLACK>(buf) - buf);
    return n == 0 ? Move() : buf[rng.rand<uint64_t>() % n];
  }

  int search_once(Position &pos, uint64_t nodes, Move *best) {
    search::clear_stop(); // the node cap left g_stop raised after the previous search
    const search::Result r = search::think(pos, search::MAX_PLY - 1, nullptr, 0, 0);
    if (best)
      *best = r.best;
    return r.score;
  }

} // namespace

void datagen::run(uint64_t count, const std::string &out, uint64_t nodes, uint64_t seed) {
  std::FILE *f = std::fopen(out.c_str(), "ab");
  if (!f) {
    std::printf("datagen: cannot open %s\n", out.c_str());
    return;
  }
  tt::resize(16);
  search::set_node_limit(nodes ? nodes : 5000);
  PRNG rng(seed ? seed : 0x9E3779B97F4A7C15ull);

  struct Pending {
    Record rec;
    bool   black; // stm of the record (to invert the white-POV result)
  };
  std::vector<Pending> game;
  game.reserve(MAX_PLIES);

  uint64_t   written = 0, games = 0;
  const auto t0 = Clock::now();

  while (written < count) {
    tt::clear();
    search::new_game();
    game.clear();

    Position pos;
    Position::set(DEFAULT_FEN, pos);

    // Random opening, alternating length so both colors get the first "real" move.
    const int plies = 8 + int(games & 1);
    bool      ok    = true;
    for (int i = 0; i < plies; ++i) {
      const Move m = random_legal(pos, rng);
      if (m.to_from() == 0) {
        ok = false;
        break;
      }
      do_move(pos, m);
    }
    ++games;
    if (!ok || count_legals(pos) == 0)
      continue;
    if (std::abs(search_once(pos, nodes, nullptr)) > OPENING_GATE_CP)
      continue; // lopsided opening: labels would echo the imbalance, not the play

    // Play the game out.
    int wr         = -1; // white-POV result: 0 loss, 1 draw, 2 win (-1 = still running)
    int win_plies = 0, draw_plies = 0;
    for (int ply = 0; ply < MAX_PLIES; ++ply) {
      if (count_legals(pos) == 0) {
        wr = stm_in_check(pos) ? (pos.turn() == WHITE ? 0 : 2) : 1; // mate / stalemate
        break;
      }
      if (pos.is_draw()) {
        wr = 1;
        break;
      }

      Move      best;
      const int score = search_once(pos, nodes, &best);
      if (best.to_from() == 0)
        break; // aborted search: discard the game tail

      if (!stm_in_check(pos) && is_quiet(best) && std::abs(score) < SCORE_CAP_CP)
        game.push_back({encode(pos, score), pos.turn() == BLACK});

      // Adjudication: a held decisive score ends the game; a long dead-equal stretch draws it.
      win_plies  = std::abs(score) >= SCORE_CAP_CP ? win_plies + 1 : 0;
      draw_plies = std::abs(score) <= 10 && ply >= 80 ? draw_plies + 1 : 0;
      if (win_plies >= WIN_ADJ_PLIES) {
        const bool stm_wins = score > 0;
        wr                  = (pos.turn() == WHITE) == stm_wins ? 2 : 0;
        break;
      }
      if (draw_plies >= DRAW_ADJ_PLIES) {
        wr = 1;
        break;
      }

      do_move(pos, best);
    }
    if (wr < 0)
      wr = 1; // ply cap / aborted: call it a draw rather than invent a winner

    for (const Pending &p: game) {
      Record rec = p.rec;
      rec.result = uint8_t(p.black ? 2 - wr : wr);
      std::fwrite(&rec, sizeof(rec), 1, f);
      if (++written % 10000 == 0) {
        const double el = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("datagen: %llu positions, %llu games, %.0f pos/s\n", (unsigned long long) written,
                    (unsigned long long) games, double(written) / el);
        std::fflush(stdout);
      }
      if (written >= count)
        break;
    }
    std::fflush(f);
  }

  const double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("datagen done: %llu positions from %llu games in %.0fs (%.0f pos/s) -> %s\n",
              (unsigned long long) written, (unsigned long long) games, el, double(written) / el, out.c_str());
  std::fflush(stdout);
  std::fclose(f);
  search::set_node_limit(0);
}
