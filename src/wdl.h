#pragma once

#include <string>

// Win/Draw/Loss estimation for the UCI_ShowWDL option: an approximate mapping from a centipawn
// search score to win/draw/loss probabilities, used only for the optional `wdl W D L` info field.
// Kept out of uci.cpp so the model (and any future calibration) lives on its own.
namespace wdl {

  // Probabilities in permille (each in [0, 1000], summing to exactly 1000).
  struct WDL {
    int win, draw, loss;
  };

  // Approximate W/D/L for a centipawn score, from the side-to-move's perspective.
  [[nodiscard]] WDL from_cp(int cp);

  // The full UCI `wdl W D L` field for a search score. A proven mate (|score| near MATE) is
  // reported as a certain result (1000 0 0 / 0 0 1000).
  [[nodiscard]] std::string format(int score);

} // namespace wdl
