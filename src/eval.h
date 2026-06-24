#pragma once

#include <string>
#include <vector>
#include "types.h"

class Position;

namespace eval {

  // --- Texel tuning registry ---------------------------------------------------------------
  // Every "unproven starting value" evaluation constant, exposed by name for the tuner
  // (`askaig tune`). The defaults reproduce the bench signature exactly; the tuner mutates the
  // values through the pointers. Structural constants (phase weights, king-attack scale curve,
  // PeSTO material/PST — already tuned upstream) are deliberately NOT registered.
  struct Param {
    std::string name;
    int        *value;
  };
  [[nodiscard]] const std::vector<Param> &params() noexcept;

  // Static evaluation (material + PST + pawn structure + king safety + mobility + pins) from the
  // side-to-move's perspective (negamax convention: positive means the side to move is better).
  [[nodiscard]] int evaluate(const Position &pos) noexcept;

  // True if `c`'s pawn on `sq` is passed (no enemy pawn can block or capture it on its way to
  // promotion). Used by the search's passed-pawn extension.
  [[nodiscard, gnu::pure]] bool is_passed_pawn(const Position &pos, Color c, Square sq) noexcept;

} // namespace eval
