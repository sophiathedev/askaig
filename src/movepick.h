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
//
// SEE is LAZY: the constructor scores every capture into the winning band without calling
// see_ge, and next() verifies a capture only when it has actually reached the front of the
// queue — demoting it a full DEMOTION down into the losing band when it fails, then picking
// again. At the typical node the TT move (or the first capture) cuts immediately and no other
// capture is ever verified. The yielded sequence is IDENTICAL to eager SEE scoring: unverified
// captures rank exactly where verified winning captures would, so whichever move is on top
// after a demotion is the same move eager ordering would have yielded there.
namespace search {

  class MovePicker {
  public:
    // Band bases. A capture's history/MVV term is bounded by |32*900 + 16384 + 900| < 47000,
    // a quiet's by 3*16384 — every band below is separated by far more than either, so bands
    // can never overlap and the +-500'000 window tests in next()/yielded_see() are safe.
    static constexpr int SCORE_TT      = 4'000'000;
    static constexpr int SCORE_CAPTURE = 2'000'000; // unverified/winning captures
    static constexpr int SCORE_KILLER  = 1'000'000;
    static constexpr int DEMOTION      = 4'000'000; // capture band -> losing band (-2'000'000)

    // What next() proved about the move it just yielded (valid until the next call).
    enum SeeBand : int8_t {
      SEE_UNKNOWN, // TT move or a non-capture: never verified here
      SEE_WINNING, // capture verified see_ge(m, 0)
      SEE_LOSING, // capture that failed see_ge(m, 0) and was yielded from the losing band
    };

    // ch1/ch2: continuation-history slices of the previous and the move before it (nullable).
    // Constructed at every node (main search) or pruning probe (QS/ProbCut) — hot, but not
    // `pure`: it writes the picker's own move/score arrays, a real (if purely local) effect.
    [[gnu::hot]] MovePicker(Position &pos, const Histories &hist, Move ttm, const Move *killers,
                            const ContTable *ch1, const ContTable *ch2, bool quiescence)
        : pos(&pos), n(0), cur(0) {
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
          score = SCORE_TT; // the TT move goes first, always
        else if (m.is_capture()) {
          const PieceType captured =
                  m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
          const Piece pc = pos.at(m.from());
          score          = SCORE_CAPTURE + 32 * PIECE_VAL[captured] + hist.capture[pc][m.to()][captured] +
                  (m.flags() == PC_QUEEN ? PIECE_VAL[QUEEN] : 0);
        } else if (killers && (m.to_from() == killers[0].to_from() || m.to_from() == killers[1].to_from()))
          score = SCORE_KILLER;
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

    // Yields the next-best move, or a null move (to_from() == 0) when exhausted. Called in a
    // tight loop at every node; discarding the result would silently skip a move. Requires the
    // Position to be at the node this picker was built for (every caller makes/unmakes between
    // calls, so it always is).
    [[gnu::hot, nodiscard]] Move next() {
      while (cur < n) {
        size_t best = cur;
        for (size_t i = cur + 1; i < n; ++i)
          if (scores[i] > scores[best])
            best = i;
        // Lazy SEE: a capture surfacing from the (so far only assumed) winning band must
        // prove itself now; failures drop into the losing band and something else surfaces.
        // Verified BEFORE the yield swap, demoted IN PLACE on failure: yields (and their
        // array swaps) then happen in exactly the sequence eager scoring would produce —
        // including score ties, which selection-sort breaks by array position.
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
        return moves[cur++];
      }
      return Move();
    }

    // The SEE verdict of the move the last next() call yielded — lets the search reuse the
    // picker's own verification instead of re-running see_ge on the same move.
    [[nodiscard]] SeeBand yielded_see() const { return band; }

    [[nodiscard]] size_t total() const { return n; }
    [[nodiscard]] size_t remaining() const { return n - cur; }

  private:
    const Position *pos;
    Move            moves[218];
    int             scores[218];
    size_t          n, cur;
    SeeBand         band = SEE_UNKNOWN;
  };

} // namespace search
