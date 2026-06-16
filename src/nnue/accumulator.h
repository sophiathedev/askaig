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

  // acc[P] += / -= the feature column `f` (one contiguous L1 vector of FT weights). int16 wraps
  // identically (2's complement) in scalar and SIMD, so the three branches are bit-identical. L1 is a
  // multiple of 16, so no scalar tail is needed.
  inline void add_feature(Accumulator &a, Color P, int f) noexcept {
    const int16_t *w   = g_net->ft_weights + size_t(f) * L1;
    int16_t       *acc = a.v[P];
#if defined(SIMD) && defined(ARCH_AVX2)
    for (int i = 0; i < L1; i += 16) {
      const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(acc + i));
      const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + i));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(acc + i), _mm256_add_epi16(s, x));
    }
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
    for (int i = 0; i < L1; i += 8)
      vst1q_s16(acc + i, vaddq_s16(vld1q_s16(acc + i), vld1q_s16(w + i)));
#else
    for (int i = 0; i < L1; ++i)
      acc[i] += w[i];
#endif
  }
  inline void sub_feature(Accumulator &a, Color P, int f) noexcept {
    const int16_t *w   = g_net->ft_weights + size_t(f) * L1;
    int16_t       *acc = a.v[P];
#if defined(SIMD) && defined(ARCH_AVX2)
    for (int i = 0; i < L1; i += 16) {
      const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(acc + i));
      const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + i));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(acc + i), _mm256_sub_epi16(s, x));
    }
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
    for (int i = 0; i < L1; i += 8)
      vst1q_s16(acc + i, vsubq_s16(vld1q_s16(acc + i), vld1q_s16(w + i)));
#else
    for (int i = 0; i < L1; ++i)
      acc[i] -= w[i];
#endif
  }

} // namespace nnue
