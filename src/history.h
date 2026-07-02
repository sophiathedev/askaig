#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "types.h"

// Move-ordering / pruning statistics ("history"), shared by movepick.h and search.cpp.
// All tables use the standard gravity update: h += bonus - h*|bonus|/HIST_MAX, which decays
// toward zero and saturates at +-HIST_MAX without explicit clamping.
namespace search {

  constexpr int HIST_MAX = 16384;

  inline void hist_update(int16_t &h, int bonus) {
    h = static_cast<int16_t>(h + bonus - int(h) * std::abs(bonus) / HIST_MAX);
  }

  // One continuation-history slice: indexed by the CURRENT move's [piece][to], selected by the
  // PREVIOUS move's (piece, to) — countermove history at 1 ply, follow-up history at 2 plies.
  using ContTable = int16_t[NPIECES][NSQUARES];

  struct Histories {
    // Butterfly history: [stm][from][to] of quiet moves.
    int16_t butterfly[NCOLORS][NSQUARES][NSQUARES];
    // Capture history: [moving piece][to][captured piece type].
    int16_t capture[NPIECES][NSQUARES][NPIECE_TYPES];
    // Continuation history: [prev piece][prev to] -> ContTable over the current move. (~3.7 MB)
    ContTable cont[NPIECES][NSQUARES];
    // Static-eval correction history, keyed by (stm, pawn-structure key): learns a per-structure
    // offset between the static eval and the search result. Applied and updated in search.cpp.
    static constexpr size_t CORR_SIZE = 16384;
    int16_t                 corr[NCOLORS][CORR_SIZE];

    void clear() { std::memset(this, 0, sizeof(*this)); }
  };

} // namespace search
