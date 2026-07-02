#pragma once

#include <cstdint>
#include <utility>
#include "history.h"
#include "position.h"
#include "see.h"
#include "types.h"

// Move ordering. The movegen produces all legal moves at once (there is no staged generator),
// so the picker scores everything up front into bands and yields moves by selection-sort —
// O(n) per pick, and typically only a few picks happen before a beta cutoff.
//
// Bands (high to low): TT move, winning/equal captures (SEE >= 0, by MVV + capture history),
// killers, quiets (butterfly + continuation history), losing captures.
namespace search {

  class MovePicker {
  public:
    // ch1/ch2: continuation-history slices of the previous and the move before it (nullable).
    MovePicker(Position &pos, const Histories &hist, Move ttm, const Move *killers, const ContTable *ch1,
               const ContTable *ch2, bool quiescence)
        : n(0), cur(0) {
      Move buf[218];
      const size_t cnt =
              pos.turn() == WHITE ? size_t(pos.generate_legals<WHITE>(buf) - buf) : size_t(pos.generate_legals<BLACK>(buf) - buf);
      const bool in_check = pos.turn() == WHITE ? pos.in_check<WHITE>() : pos.in_check<BLACK>();

      for (size_t i = 0; i < cnt; ++i) {
        const Move m = buf[i];
        // Quiescence outside check: captures and queen promotions only.
        if (quiescence && !in_check && !m.is_capture() && m.flags() != PR_QUEEN)
          continue;

        int score;
        if (ttm.to_from() != 0 && m.to_from() == ttm.to_from())
          score = 4'000'000; // the TT move goes first, always
        else if (m.is_capture()) {
          const PieceType captured =
                  m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
          const Piece pc = pos.at(m.from());
          score          = 32 * PIECE_VAL[captured] + hist.capture[pc][m.to()][captured] +
                  (m.flags() == PC_QUEEN || m.flags() == PR_QUEEN ? PIECE_VAL[QUEEN] : 0);
          // Winning/equal captures above killers and quiets; losing captures below everything.
          score += see_ge(pos, m, 0) ? 2'000'000 : -2'000'000;
        } else if (killers && (m.to_from() == killers[0].to_from() || m.to_from() == killers[1].to_from()))
          score = 1'000'000;
        else {
          const Piece pc = pos.at(m.from());
          score          = hist.butterfly[pos.turn()][m.from()][m.to()];
          if (ch1)
            score += (*ch1)[pc][m.to()];
          if (ch2)
            score += (*ch2)[pc][m.to()];
        }
        moves[n]  = m;
        scores[n] = score;
        ++n;
      }
    }

    // Yields the next-best move, or a null move (to_from() == 0) when exhausted.
    Move next() {
      if (cur >= n)
        return Move();
      size_t best = cur;
      for (size_t i = cur + 1; i < n; ++i)
        if (scores[i] > scores[best])
          best = i;
      std::swap(moves[cur], moves[best]);
      std::swap(scores[cur], scores[best]);
      return moves[cur++];
    }

    [[nodiscard]] size_t total() const { return n; }
    [[nodiscard]] size_t remaining() const { return n - cur; }

  private:
    Move   moves[218];
    int    scores[218];
    size_t n, cur;
  };

} // namespace search
