#pragma once

#include "types.h"

class Position;

namespace eval {

  // Static evaluation (material + PST + pawn structure + king safety + mobility + pins) from the
  // side-to-move's perspective (negamax convention: positive means the side to move is better).
  [[nodiscard]] int evaluate(const Position &pos) noexcept;

  // True if `c`'s pawn on `sq` is passed (no enemy pawn can block or capture it on its way to
  // promotion). Used by the search's passed-pawn extension.
  [[nodiscard, gnu::pure]] bool is_passed_pawn(const Position &pos, Color c, Square sq) noexcept;

} // namespace eval
