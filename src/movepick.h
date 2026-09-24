#pragma once

#include <cassert>
#include <cstdint>
#include <utility>
#include "history.h"
#include "position.h"
#include "see.h"
#include "types.h"

namespace search {

  class MovePicker {
  public:
    static constexpr int  SCORE_TT                = 4'000'000;
    static constexpr int  SCORE_CAPTURE           = 2'000'000;
    static constexpr int  SCORE_KILLER            = 1'000'000;
    static constexpr int  SCORE_COUNTER           = 800'000;
    static constexpr int  DEMOTION                = 4'000'000;
    static constexpr int  SORT_AFTER              = 4;
    static constexpr bool ENABLE_TT_STAGE         = true;
    static constexpr bool ENABLE_CAPTURE_STAGE    = true;
    static constexpr bool ENABLE_REFUTATION_STAGE = true;
    static constexpr bool ENABLE_LAZY_BAD_SORT    = true;

    enum SeeBand : int8_t {
      SEE_UNKNOWN, // TT move or a non-capture: never verified here
      SEE_WINNING, // capture verified see_ge(m, 0)
      SEE_LOSING, // capture that failed see_ge(m, 0)
    };

    [[gnu::hot]] MovePicker(Position &pos, const Histories &hist, Move ttm, const Move *killers, Move counter,
                            const ContTable *ch1, const ContTable *ch2, bool quiescence, bool staged = ENABLE_TT_STAGE,
                            bool split = ENABLE_CAPTURE_STAGE, bool refutations = ENABLE_REFUTATION_STAGE,
                            bool lazy_bad_sort = ENABLE_LAZY_BAD_SORT) :
        pos(&pos), hist(&hist), ttm(ttm), counter(counter), ch1(ch1), ch2(ch2) {
      const bool in_check = pos.turn() == WHITE ? pos.in_check<WHITE>() : pos.in_check<BLACK>();
      caps_only           = quiescence && !in_check;
      split_captures      = staged && split && !quiescence && !in_check;
      stage_refutations   = split_captures && refutations;
      defer_bad_sort      = split_captures && lazy_bad_sort;
      if (killers) {
        this->killers[0] = killers[0];
        this->killers[1] = killers[1];
      }
      if (staged && (!caps_only || ttm.is_capture()) && pos.legal_tt_move(ttm))
        stage = TT;
      else if (split_captures && !ttm.to_from())
        stage = REST;
      else
        generate();
    }

  private:
    int move_score(Move m) const {
      const Position  &pos  = *this->pos;
      const Histories &hist = *this->hist;
      if (ttm.to_from() != 0 && m == ttm)
        return SCORE_TT;
      const Piece pc = pos.at(m.from());
      if (m.is_capture()) {
        const PieceType captured = m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
        return SCORE_CAPTURE + 32 * PIECE_VAL[captured] + hist.capture[pc][m.to()][captured] +
               (m.flags() == PC_QUEEN ? PIECE_VAL[QUEEN] : 0);
      }
      if (m == killers[0] || m == killers[1])
        return SCORE_KILLER;
      if (counter.to_from() != 0 && m == counter)
        return SCORE_COUNTER;
      int score = hist.butterfly[pos.turn()][m.from()][m.to()];
      if (ch1)
        score += (*ch1)[pc][m.to()];
      if (ch2)
        score += (*ch2)[pc][m.to()];
      return score;
    }

    template<MoveGen Mode>
    void append_moves() {
      Move        *end   = pos->turn() == WHITE ? pos->generate_legals<WHITE, Mode>(moves + n, &context)
                                                : pos->generate_legals<BLACK, Mode>(moves + n, &context);
      const size_t count = size_t(end - moves);
      for (size_t i = n; i < count; ++i) {
        const Move m = moves[i];
        if (tt_emitted && m == ttm)
          continue;
        bool emitted = false;
        for (size_t j = 0; j < refutation_index; ++j)
          emitted |= m == refutation_moves[j];
        if (emitted)
          continue;
        moves[n]    = m;
        scores[n++] = move_score(m);
      }
      assert(n <= std::size(moves));
    }

    void prepare_rest() {
      if (split_captures) {
        append_moves<MoveGen::CAPTURES>();
        yields = tt_emitted ? 1 : 0;
        stage  = CAPTURES;
      } else
        generate();
    }

    void prepare_quiets() {
      append_moves<MoveGen::QUIETS>();
      stage = READY;
    }

    void prepare_refutations() {
      if (stage_refutations) {
        const Move candidates[] = {killers[0], killers[1], counter};
        // special quiets keep their original ordering in the full quiet list
        for (Move m: candidates)
          if (m.to_from() && !m.is_capture() && m.flags() != QUIET && m.flags() != DOUBLE_PUSH) {
            prepare_quiets();
            return;
          }
        stage = REFUTATIONS;
        return;
      }
      prepare_quiets();
    }

    Move peek_refutation() {
      if (refutation_index < refutation_count)
        return refutation_moves[refutation_index];
      while (refutation_candidate < 3) {
        const Move m = refutation_candidate < 2 ? killers[refutation_candidate] : counter;
        ++refutation_candidate;
        if (!m.to_from() || m.is_capture() || (tt_emitted && m == ttm))
          continue;
        bool duplicate = false;
        for (size_t i = 0; i < refutation_count; ++i)
          duplicate |= m == refutation_moves[i];
        if (!duplicate && pos->legal_tt_move(m)) {
          refutation_moves[refutation_count++] = m;
          return m;
        }
      }
      return Move();
    }

    void generate() {
      defer_bad_sort = false;
      Position &pos  = *this->pos;
      Move     *end  = pos.turn() == WHITE ? (caps_only ? pos.generate_legals<WHITE, MoveGen::QUIESCENCE>(moves)
                                                        : pos.generate_legals<WHITE>(moves))
                                           : (caps_only ? pos.generate_legals<BLACK, MoveGen::QUIESCENCE>(moves)
                                                        : pos.generate_legals<BLACK>(moves));
      n              = size_t(end - moves);

      for (size_t i = 0; i < n; ++i)
        scores[i] = move_score(moves[i]);
      if (stage == REST) {
        // reproduce the eager picker's first swap, including equal-score order
        size_t i = 0;
        while (i < n && moves[i] != ttm)
          ++i;
        assert(i < n);
        std::swap(moves[0], moves[i]);
        std::swap(scores[0], scores[i]);
        cur = yields = 1;
      }
      stage = READY;
    }

    void sort_remaining(bool skip_bad) {
      for (size_t i = cur + 1; i < n; ++i) {
        const Move m = moves[i];
        const int  s = scores[i];
        if (skip_bad && s < -(SCORE_CAPTURE - 500'000))
          continue;
        size_t j = i;
        for (; j > cur && scores[j - 1] < s; --j) {
          moves[j]  = moves[j - 1];
          scores[j] = scores[j - 1];
        }
        moves[j]  = m;
        scores[j] = s;
      }
    }

  public:
    [[nodiscard]] bool has_moves() {
      if (stage == TT)
        return true;
      if (stage == REST)
        prepare_rest();
      if (stage == CAPTURES && cur == n)
        prepare_refutations();
      if (stage == REFUTATIONS) {
        if (peek_refutation().to_from())
          return true;
        prepare_quiets();
      }
      return cur < n;
    }

    // lazy SEE; failed captures move to the losing band
    [[gnu::hot, nodiscard]] Move next() {
      if (stage == TT) {
        stage      = REST;
        tt_emitted = true;
        band       = SEE_UNKNOWN;
        return ttm;
      }
      if (stage == REST)
        prepare_rest();
      if (stage == CAPTURES) {
        while (cur < n) {
          size_t best = cur;
          for (size_t i = cur + 1; i < n; ++i)
            if (scores[i] > scores[best])
              best = i;
          if (scores[best] < SCORE_CAPTURE - 500'000)
            break;
          if (!see_ge(*pos, moves[best], 0)) {
            scores[best] -= DEMOTION;
            continue;
          }
          std::swap(moves[cur], moves[best]);
          std::swap(scores[cur], scores[best]);
          band = SEE_WINNING;
          ++yields;
          return moves[cur++];
        }
        prepare_refutations();
      }
      if (stage == REFUTATIONS) {
        if (const Move m = peek_refutation(); m.to_from()) {
          band = SEE_UNKNOWN;
          ++yields;
          ++refutation_index;
          return m;
        }
        prepare_quiets();
      }
      if (!sorted) {
        if (yields < SORT_AFTER) {
          while (cur < n) {
            size_t best = cur;
            for (size_t i = cur + 1; i < n; ++i)
              if (scores[i] > scores[best])
                best = i;
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
        sort_remaining(defer_bad_sort);
        sorted = true;
      }

      if (defer_bad_sort && cur < n && scores[cur] < -(SCORE_CAPTURE - 500'000)) {
        sort_remaining(false);
        defer_bad_sort = false;
      }
      while (cur < n) {
        const int s = scores[cur];
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

    [[nodiscard]] SeeBand yielded_see() const { return band; }

  private:
    enum Stage { TT, REST, CAPTURES, REFUTATIONS, READY };
    Position        *pos;
    const Histories *hist;
    Move             ttm, counter, killers[2]{};
    const ContTable *ch1, *ch2;
    bool             caps_only;
    bool             split_captures, tt_emitted = false;
    bool             stage_refutations;
    bool             defer_bad_sort;
    Move             refutation_moves[3];
    uint8_t          refutation_count = 0, refutation_index = 0, refutation_candidate = 0;
    LegalContext     context;
    Stage            stage = READY;
    Move             moves[218];
    int              scores[218];
    size_t           n = 0, cur = 0;
    int              yields = 0;
    bool             sorted = false;
    SeeBand          band   = SEE_UNKNOWN;
  };

} // namespace search
