#pragma once

#include <cstdint>
#include <utility>
#include "history.h"
#include "position.h"
#include "see.h"
#include "types.h"

// Move ordering. Legal moves are generated in one shot into the picker's array (quiescence
// outside check uses the CAPTURES_ONLY movegen) and scored up front into bands:
//
//   TT move > winning/equal captures (MVV + capture history) > killers >
//   quiets (butterfly + continuation history) > losing captures.
//
// Yielding is hybrid: the first SORT_AFTER picks use a linear max-scan (most nodes cut within
// a few moves and never pay more); a node still picking after that sorts the remaining tail
// once (stable insertion sort, ties keep generation order) and walks it sequentially.
//
// SEE is lazy: captures are scored into the winning band unverified, and see_ge runs only
// when a capture actually surfaces — a failure demotes it into the losing band (in place
// while scanning, re-inserted at its sorted rank after) and the next candidate surfaces.
namespace search {

  class MovePicker {
  public:
    // Band bases. A capture's MVV/history term stays within +-47000 and a quiet's within
    // +-49152, far below the band spacing, so the +-500'000 window tests below are safe.
    static constexpr int SCORE_TT      = 4'000'000;
    static constexpr int SCORE_CAPTURE = 2'000'000;
    static constexpr int SCORE_KILLER  = 1'000'000;
    static constexpr int SCORE_COUNTER = 800'000; // countermove: just below the killers
    static constexpr int DEMOTION      = 4'000'000; // capture band -> losing band
    static constexpr int SORT_AFTER    = 4; // max-scan picks before the tail gets sorted

    // What next() proved about the move it just yielded (valid until the next call).
    enum SeeBand : int8_t {
      SEE_UNKNOWN, // TT move or a non-capture: never verified here
      SEE_WINNING, // capture verified see_ge(m, 0)
      SEE_LOSING, // capture that failed see_ge(m, 0)
    };

    // ch1/ch2: continuation-history slices of the previous two plies (nullable).
    [[gnu::hot]] MovePicker(Position &pos, const Histories &hist, Move ttm, const Move *killers, Move counter,
                            const ContTable *ch1, const ContTable *ch2, bool quiescence)
        : pos(&pos), cur(0) {
      const bool in_check  = pos.turn() == WHITE ? pos.in_check<WHITE>() : pos.in_check<BLACK>();
      const bool caps_only = quiescence && !in_check; // in check QS searches every evasion
      Move      *end = pos.turn() == WHITE
                               ? (caps_only ? pos.generate_legals<WHITE, true>(moves) : pos.generate_legals<WHITE>(moves))
                               : (caps_only ? pos.generate_legals<BLACK, true>(moves) : pos.generate_legals<BLACK>(moves));
      n = size_t(end - moves);

      for (size_t i = 0; i < n; ++i) {
        const Move m = moves[i];

        int score;
        if (ttm.to_from() != 0 && m.to_from() == ttm.to_from())
          score = SCORE_TT;
        else if (m.is_capture()) {
          const PieceType captured =
                  m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
          const Piece pc = pos.at(m.from());
          score          = SCORE_CAPTURE + 32 * PIECE_VAL[captured] + hist.capture[pc][m.to()][captured] +
                  (m.flags() == PC_QUEEN ? PIECE_VAL[QUEEN] : 0);
        } else if (killers && (m.to_from() == killers[0].to_from() || m.to_from() == killers[1].to_from()))
          score = SCORE_KILLER;
        else if (counter.to_from() != 0 && m.to_from() == counter.to_from())
          score = SCORE_COUNTER;
        else {
          const Piece pc = pos.at(m.from());
          score          = hist.butterfly[pos.turn()][m.from()][m.to()];
          if (ch1)
            score += (*ch1)[pc][m.to()];
          if (ch2)
            score += (*ch2)[pc][m.to()];
        }
        scores[i] = score;
      }
    }

    // Yields the next-best move, or a null move (to_from() == 0) when exhausted. Requires the
    // Position to be at the node this picker was built for (callers make/unmake between calls).
    [[gnu::hot, nodiscard]] Move next() {
      if (!sorted) {
        if (yields < SORT_AFTER) {
          while (cur < n) {
            size_t best = cur;
            for (size_t i = cur + 1; i < n; ++i)
              if (scores[i] > scores[best])
                best = i;
            // Lazy SEE: verify before the yield swap; a failure demotes in place, no yield.
            if (scores[best] > SCORE_CAPTURE - 500'000 && scores[best] < SCORE_CAPTURE + 500'000) {
              if (!see_ge(*pos, moves[best], 0)) {
                scores[best] -= DEMOTION;
                continue;
              }
              band = SEE_WINNING;
            } else
              band = scores[best] < -(SCORE_CAPTURE - 500'000) ? SEE_LOSING : SEE_UNKNOWN;
            std::swap(moves[cur], moves[best]);
            std::swap(scores[cur], scores[best]);
            ++yields;
            return moves[cur++];
          }
          return Move();
        }
        // Likely an ALL-node: sort the tail once, then walk it.
        for (size_t i = cur + 1; i < n; ++i) {
          const Move m = moves[i];
          const int  s = scores[i];
          size_t     j = i;
          for (; j > cur && scores[j - 1] < s; --j) {
            moves[j]  = moves[j - 1];
            scores[j] = scores[j - 1];
          }
          moves[j]  = m;
          scores[j] = s;
        }
        sorted = true;
      }

      while (cur < n) {
        const int s = scores[cur];
        // Lazy SEE, sorted flavour: a failure re-inserts the capture at its demoted rank.
        if (s > SCORE_CAPTURE - 500'000 && s < SCORE_CAPTURE + 500'000) {
          if (!see_ge(*pos, moves[cur], 0)) {
            const Move m  = moves[cur];
            const int  ds = s - DEMOTION;
            size_t     j  = cur + 1;
            while (j < n && scores[j] > ds)
              ++j;
            for (size_t k = cur; k + 1 < j; ++k) {
              moves[k]  = moves[k + 1];
              scores[k] = scores[k + 1];
            }
            moves[j - 1]  = m;
            scores[j - 1] = ds;
            continue;
          }
          band = SEE_WINNING;
        } else
          band = s < -(SCORE_CAPTURE - 500'000) ? SEE_LOSING : SEE_UNKNOWN;
        return moves[cur++];
      }
      return Move();
    }

    // SEE verdict of the last yielded move, so the search doesn't re-run see_ge on it.
    [[nodiscard]] SeeBand yielded_see() const { return band; }

    [[nodiscard]] size_t total() const { return n; }

  private:
    const Position *pos;
    Move            moves[218];
    int             scores[218];
    size_t          n, cur;
    int             yields = 0;
    bool            sorted = false;
    SeeBand         band   = SEE_UNKNOWN;
  };

} // namespace search
