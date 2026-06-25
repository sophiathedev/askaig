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
    int         min, max; // [min, max] range, exposed as the UCI spin bounds (for tools/spsa.py)
  };
  [[nodiscard]] const std::vector<Param> &params() noexcept;

  // Set a registered eval parameter by name (clamped to its [min, max]); returns true if matched.
  // Mirrors search::set_tunable — lets tools/spsa.py drive the eval constants (e.g. king-safety
  // shelter/storm) through `setoption` in debug mode. A normal game never calls this.
  bool set_param(const std::string &name, int value) noexcept;

  // Tuning mode: bypass the thread-local pawn-structure cache so the eval re-reads the (mutated)
  // weights every call. ONLY the gradient tuner sets this (it perturbs weights and re-evaluates the
  // same position, which would otherwise hit a stale cache entry). Default off → normal play and the
  // bench node signature are byte-for-byte unchanged.
  void set_tuning(bool on) noexcept;

  // Static evaluation (material + PST + pawn structure + king safety + mobility + pins) from the
  // side-to-move's perspective (negamax convention: positive means the side to move is better).
  [[nodiscard]] int evaluate(const Position &pos) noexcept;

  // True if `c`'s pawn on `sq` is passed (no enemy pawn can block or capture it on its way to
  // promotion). Used by the search's passed-pawn extension.
  [[nodiscard, gnu::pure]] bool is_passed_pawn(const Position &pos, Color c, Square sq) noexcept;

} // namespace eval
