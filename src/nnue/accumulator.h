#pragma once

#include <cstdint>
#include "features.h"
#include "network.h"
#include "types.h"

// The NNUE accumulator: two perspective vectors (WHITE, BLACK) of int16, one running sum of the active
// feature columns plus the FT bias. v1 keeps the add/sub helpers here (used by the from-scratch refresh
// now, and by the incremental Position primitives in M2); they reference the loaded network `g_net`.
//
// This header depends ONLY on types.h/network.h/features.h (NOT position.h), so position.h can include it
// without a cycle.
namespace nnue {

  extern const Network *g_net; // loaded network (nullptr until init()); defined in nnue.cpp

  struct Accumulator {
    alignas(64) int16_t v[NCOLORS][L1];
  };

  // acc[P] += / -= the feature column `f` (one contiguous L1 vector of FT weights).
  inline void add_feature(Accumulator &a, Color P, int f) noexcept {
    const int16_t *w = g_net->ft_weights + size_t(f) * L1;
    for (int i = 0; i < L1; ++i)
      a.v[P][i] += w[i];
  }
  inline void sub_feature(Accumulator &a, Color P, int f) noexcept {
    const int16_t *w = g_net->ft_weights + size_t(f) * L1;
    for (int i = 0; i < L1; ++i)
      a.v[P][i] -= w[i];
  }

} // namespace nnue
