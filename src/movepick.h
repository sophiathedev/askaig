#pragma once

#include <cstdint>
#include <utility>
#include "history.h"
#include "position.h"
#include "see.h"
#include "types.h"

// Move ordering. The movegen produces its legal moves at once (there is no staged generator;
// quiescence outside check uses the CAPTURES_ONLY variant), directly into the picker's own
// array, and the constructor scores everything up front into bands.
//
// Bands (high to low): TT move, winning/equal captures (SEE >= 0, by MVV + capture history),
// killers, quiets (butterfly + continuation history), losing captures.
//
// Yielding is HYBRID, driven by the profile: most nodes cut after 1-3 picks, where a linear
// max-scan of the unsorted array is the cheapest possible pick — but at ALL-nodes that
// re-scan per pick was ~21% of total engine time. So the first SORT_AFTER yields use the
// scan; a node still picking after that is treated as an ALL-node and the remaining tail is
// insertion-sorted ONCE (stable: ties keep generation order), after which next() is a plain
// sequential walk.
//
// SEE is LAZY: the constructor scores every capture into the winning band without calling
// see_ge, and next() verifies a capture only when it actually surfaces — a failure demotes
// it a full DEMOTION down into the losing band (in place while scanning; re-inserted at its
// sorted rank once sorted) and the next candidate surfaces instead. At the typical node the
// TT move (or the first capture) cuts immediately and no other capture is ever verified.
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
    // Yields served by max-scan before the tail is sorted. Covers the overwhelmingly common
    // beta-cutoff-in-a-few-moves case; only likely ALL-nodes ever pay for the sort.
    static constexpr int SORT_AFTER = 4;

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
        : pos(&pos), cur(0) {
      // Quiescence outside check generates captures + quiet queen promotions ONLY (the
      // CAPTURES_ONLY movegen) instead of generating everything and discarding the quiets —
      // same move set, in the same generation order, without emitting what QS never scores.
      // In check the full generator runs: QS searches every evasion there.
      const bool in_check  = pos.turn() == WHITE ? pos.in_check<WHITE>() : pos.in_check<BLACK>();
      const bool caps_only = quiescence && !in_check;
      Move      *end = pos.turn() == WHITE
                               ? (caps_only ? pos.generate_legals<WHITE, true>(moves) : pos.generate_legals<WHITE>(moves))
                               : (caps_only ? pos.generate_legals<BLACK, true>(moves) : pos.generate_legals<BLACK>(moves));
      n = size_t(end - moves);

      for (size_t i = 0; i < n; ++i) {
        const Move m = moves[i];

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
        scores[i] = score;
      }
    }

    // Yields the next-best move, or a null move (to_from() == 0) when exhausted. Called in a
    // tight loop at every node; discarding the result would silently skip a move. Requires the
    // Position to be at the node this picker was built for (every caller makes/unmakes between
    // calls, so it always is).
    [[gnu::hot, nodiscard]] Move next() {
      if (!sorted) {
        if (yields < SORT_AFTER) {
          // Cheap phase: one linear max-scan per pick over the unsorted remainder.
          while (cur < n) {
            size_t best = cur;
            for (size_t i = cur + 1; i < n; ++i)
              if (scores[i] > scores[best])
                best = i;
            // Lazy SEE, scan flavour: verify before the yield swap; a failure demotes the
            // capture IN PLACE and the scan re-runs — no yield happened.
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
        // Still picking after SORT_AFTER yields: likely an ALL-node. Sort the remaining tail
        // once (stable insertion sort, score-descending; ties keep their generation order)
        // and fall through to the sequential walk below.
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
        // Lazy SEE, sorted flavour: a failure re-inserts the capture at its demoted rank in
        // the (sorted) losing tail — shift the block above it left one slot — and the next
        // candidate is already sitting at cur.
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

    // The SEE verdict of the move the last next() call yielded — lets the search reuse the
    // picker's own verification instead of re-running see_ge on the same move.
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
