#include "nnue.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>

#if defined(SIMD) && defined(ARCH_AVX2)
  #include <immintrin.h>
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
  #include <arm_neon.h>
#endif

// The loader memcpys little-endian int16/int32 weight blocks straight into memory; a
// big-endian port would need byte swaps here, so fail loudly at compile time instead.
static_assert(std::endian::native == std::endian::little, "NNUE net loading assumes a little-endian host");

namespace {

  using namespace nnue;

  // The quantized network, in the exact layout the kernels read. One global copy (~6.3 MB,
  // zero-initialised BSS; nets are immutable after load, every thread reads the same weights).
  struct Network {
    alignas(64) int16_t ft_w[FEATURES * HL]; // feature-major: row f = ft_w[f*HL .. f*HL+HL)
    alignas(64) int16_t ft_b[HL];
    alignas(64) int16_t out_w[OUT_BUCKETS][2 * HL]; // [bucket][stm half | other half]
    int32_t out_b[OUT_BUCKETS];
  };

  Network g_net;
  bool    g_loaded = false;

  // King context of a perspective given its PERSPECTIVE-ORIENTED king square (black: sq^56).
  [[gnu::pure, gnu::always_inline]] inline KingCtx king_ctx(Square oriented_ksq) {
    const bool mir = file_of(oriented_ksq) >= EFILE;
    const int  sq  = mir ? (oriented_ksq ^ 7) : oriented_ksq;
    return {int8_t(KING_BUCKET[sq]), mir};
  }

  [[gnu::pure, gnu::always_inline]] inline KingCtx king_ctx_of(const Position &pos, Color persp) {
    const Square k = bsf(pos.bitboard_of(persp, KING));
    return king_ctx(persp == WHITE ? k : Square(k ^ 56));
  }

  // Feature index of (pc, s) from `persp`'s point of view under king context `c`.
  [[gnu::const, gnu::always_inline]] inline int feature_index(Color persp, Piece pc, Square s, KingCtx c) {
    const int rel = color_of(pc) == persp ? 0 : 1;
    int       sq  = persp == WHITE ? s : (s ^ 56);
    if (c.mirror)
      sq ^= 7;
    return 768 * c.bucket + 64 * (6 * rel + type_of(pc)) + sq;
  }

  [[gnu::pure, gnu::always_inline]] inline const int16_t *ft_row(Color persp, Piece pc, Square s, KingCtx c) {
    return &g_net.ft_w[feature_index(persp, pc, s, c) * HL];
  }

  // The material output bucket: 2..32 pieces -> 0..7.
  [[gnu::pure, gnu::always_inline]] inline int out_bucket(const Position &pos) {
    return (pop_count(pos.all_pieces<WHITE>() | pos.all_pieces<BLACK>()) - 2) / 4;
  }

  // --- Kernels ---------------------------------------------------------------------------------
  // Three implementations selected at compile time: AVX2, NEON, scalar. All are exact integer
  // math (int16 adds wrap identically, int32 sums are order-independent mod 2^32), so the paths
  // are bit-identical — `selftest nnue` pins the vector paths to the scalar reference.
  //
  // Alignment: Network/Accumulator arrays are alignas(64) and one feature row is HL*2 bytes,
  // so every row and half is 64-byte aligned — aligned vector loads throughout, no tail loops.

#if defined(SIMD) && defined(ARCH_AVX2)

  [[gnu::hot]] void add1_sub1(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *s0) {
    for (int i = 0; i < HL; i += 16) {
      __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(src + i));
      v         = _mm256_add_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(a0 + i)));
      v         = _mm256_sub_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(s0 + i)));
      _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i), v);
    }
  }
  [[gnu::hot]] void add1_sub2(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *s0, const int16_t *s1) {
    for (int i = 0; i < HL; i += 16) {
      __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(src + i));
      v         = _mm256_add_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(a0 + i)));
      v         = _mm256_sub_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(s0 + i)));
      v         = _mm256_sub_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(s1 + i)));
      _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i), v);
    }
  }
  [[gnu::hot]] void add2_sub2(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *a1, const int16_t *s0,
                 const int16_t *s1) {
    for (int i = 0; i < HL; i += 16) {
      __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(src + i));
      v         = _mm256_add_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(a0 + i)));
      v         = _mm256_add_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(a1 + i)));
      v         = _mm256_sub_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(s0 + i)));
      v         = _mm256_sub_epi16(v, _mm256_load_si256(reinterpret_cast<const __m256i *>(s1 + i)));
      _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i), v);
    }
  }

  // Refresh: tiled so the accumulator tile stays in registers across all piece rows.
  [[gnu::hot]] void refresh_kernel(int16_t *dst, const int16_t *const *rows, int n) {
    for (int i = 0; i < HL; i += 64) { // 4 x 16-lane vectors per tile
      __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(g_net.ft_b + i));
      __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(g_net.ft_b + i + 16));
      __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(g_net.ft_b + i + 32));
      __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(g_net.ft_b + i + 48));
      for (int k = 0; k < n; ++k) {
        const int16_t *r = rows[k] + i;
        v0               = _mm256_add_epi16(v0, _mm256_load_si256(reinterpret_cast<const __m256i *>(r)));
        v1               = _mm256_add_epi16(v1, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 16)));
        v2               = _mm256_add_epi16(v2, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 32)));
        v3               = _mm256_add_epi16(v3, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 48)));
      }
      _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i), v0);
      _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i + 16), v1);
      _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i + 32), v2);
      _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i + 48), v3);
    }
  }

  // In-place diff apply for the refresh cache: acc += sum(add rows) - sum(sub rows), tiled
  // like refresh_kernel so each accumulator tile is loaded and stored exactly once.
  [[gnu::hot]] void apply_rows_kernel(int16_t *acc, const int16_t *const *add, int na, const int16_t *const *sub,
                                      int ns) {
    for (int i = 0; i < HL; i += 64) { // 4 x 16-lane vectors per tile
      __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc + i));
      __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc + i + 16));
      __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc + i + 32));
      __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc + i + 48));
      for (int k = 0; k < na; ++k) {
        const int16_t *r = add[k] + i;
        v0               = _mm256_add_epi16(v0, _mm256_load_si256(reinterpret_cast<const __m256i *>(r)));
        v1               = _mm256_add_epi16(v1, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 16)));
        v2               = _mm256_add_epi16(v2, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 32)));
        v3               = _mm256_add_epi16(v3, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 48)));
      }
      for (int k = 0; k < ns; ++k) {
        const int16_t *r = sub[k] + i;
        v0               = _mm256_sub_epi16(v0, _mm256_load_si256(reinterpret_cast<const __m256i *>(r)));
        v1               = _mm256_sub_epi16(v1, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 16)));
        v2               = _mm256_sub_epi16(v2, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 32)));
        v3               = _mm256_sub_epi16(v3, _mm256_load_si256(reinterpret_cast<const __m256i *>(r + 48)));
      }
      _mm256_store_si256(reinterpret_cast<__m256i *>(acc + i), v0);
      _mm256_store_si256(reinterpret_cast<__m256i *>(acc + i + 16), v1);
      _mm256_store_si256(reinterpret_cast<__m256i *>(acc + i + 32), v2);
      _mm256_store_si256(reinterpret_cast<__m256i *>(acc + i + 48), v3);
    }
  }

  // One half of the SCReLU dot: sum clamp(a,0,QA) * (clamp(a,0,QA) * w) in int32 lanes.
  [[gnu::pure, gnu::hot]] __m256i dot_half(const int16_t *a, const int16_t *w) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(QA);
    __m256i       s0 = zero, s1 = zero;
    for (int i = 0; i < HL; i += 32) {
      __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(a + i));
      __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(a + i + 16));
      v0         = _mm256_min_epi16(_mm256_max_epi16(v0, zero), qa);
      v1         = _mm256_min_epi16(_mm256_max_epi16(v1, zero), qa);
      const __m256i p0 = _mm256_mullo_epi16(v0, _mm256_load_si256(reinterpret_cast<const __m256i *>(w + i)));
      const __m256i p1 = _mm256_mullo_epi16(v1, _mm256_load_si256(reinterpret_cast<const __m256i *>(w + i + 16)));
      s0               = _mm256_add_epi32(s0, _mm256_madd_epi16(v0, p0));
      s1               = _mm256_add_epi32(s1, _mm256_madd_epi16(v1, p1));
    }
    return _mm256_add_epi32(s0, s1);
  }

  [[gnu::pure, gnu::hot]] int32_t dot_reduce(const int16_t *stm_a, const int16_t *opp_a, const int16_t *w) {
    const __m256i s  = _mm256_add_epi32(dot_half(stm_a, w), dot_half(opp_a, w + HL));
    __m128i       s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    s4               = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    s4               = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    return _mm_cvtsi128_si32(s4);
  }

#elif defined(SIMD) && defined(ARCH_ARM_NEON)

  // The add/sub update kernels process an 8-vector (64 int16) tile per iteration — eight
  // independent dependency chains keep the four SIMD pipes filled across the multi-cycle
  // load latencies, and 1/8 the loop-branch overhead.
  [[gnu::hot]] void add1_sub1(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *s0) {
    for (int i = 0; i < HL; i += 64) {
      int16x8_t v0 = vsubq_s16(vaddq_s16(vld1q_s16(src + i), vld1q_s16(a0 + i)), vld1q_s16(s0 + i));
      int16x8_t v1 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 8), vld1q_s16(a0 + i + 8)), vld1q_s16(s0 + i + 8));
      int16x8_t v2 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 16), vld1q_s16(a0 + i + 16)), vld1q_s16(s0 + i + 16));
      int16x8_t v3 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 24), vld1q_s16(a0 + i + 24)), vld1q_s16(s0 + i + 24));
      int16x8_t v4 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 32), vld1q_s16(a0 + i + 32)), vld1q_s16(s0 + i + 32));
      int16x8_t v5 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 40), vld1q_s16(a0 + i + 40)), vld1q_s16(s0 + i + 40));
      int16x8_t v6 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 48), vld1q_s16(a0 + i + 48)), vld1q_s16(s0 + i + 48));
      int16x8_t v7 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 56), vld1q_s16(a0 + i + 56)), vld1q_s16(s0 + i + 56));
      vst1q_s16(dst + i, v0);
      vst1q_s16(dst + i + 8, v1);
      vst1q_s16(dst + i + 16, v2);
      vst1q_s16(dst + i + 24, v3);
      vst1q_s16(dst + i + 32, v4);
      vst1q_s16(dst + i + 40, v5);
      vst1q_s16(dst + i + 48, v6);
      vst1q_s16(dst + i + 56, v7);
    }
  }
  [[gnu::hot]] void add1_sub2(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *s0, const int16_t *s1) {
    for (int i = 0; i < HL; i += 64) {
#define AS12(off) \
  vsubq_s16(vsubq_s16(vaddq_s16(vld1q_s16(src + i + (off)), vld1q_s16(a0 + i + (off))), vld1q_s16(s0 + i + (off))), \
            vld1q_s16(s1 + i + (off)))
      int16x8_t v0 = AS12(0);
      int16x8_t v1 = AS12(8);
      int16x8_t v2 = AS12(16);
      int16x8_t v3 = AS12(24);
      int16x8_t v4 = AS12(32);
      int16x8_t v5 = AS12(40);
      int16x8_t v6 = AS12(48);
      int16x8_t v7 = AS12(56);
#undef AS12
      vst1q_s16(dst + i, v0);
      vst1q_s16(dst + i + 8, v1);
      vst1q_s16(dst + i + 16, v2);
      vst1q_s16(dst + i + 24, v3);
      vst1q_s16(dst + i + 32, v4);
      vst1q_s16(dst + i + 40, v5);
      vst1q_s16(dst + i + 48, v6);
      vst1q_s16(dst + i + 56, v7);
    }
  }
  [[gnu::hot]] void add2_sub2(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *a1, const int16_t *s0,
                 const int16_t *s1) {
    for (int i = 0; i < HL; i += 64) {
#define AS22(off) \
  vsubq_s16(vsubq_s16(vaddq_s16(vaddq_s16(vld1q_s16(src + i + (off)), vld1q_s16(a0 + i + (off))), \
                                vld1q_s16(a1 + i + (off))), \
                      vld1q_s16(s0 + i + (off))), \
            vld1q_s16(s1 + i + (off)))
      int16x8_t v0 = AS22(0);
      int16x8_t v1 = AS22(8);
      int16x8_t v2 = AS22(16);
      int16x8_t v3 = AS22(24);
      int16x8_t v4 = AS22(32);
      int16x8_t v5 = AS22(40);
      int16x8_t v6 = AS22(48);
      int16x8_t v7 = AS22(56);
#undef AS22
      vst1q_s16(dst + i, v0);
      vst1q_s16(dst + i + 8, v1);
      vst1q_s16(dst + i + 16, v2);
      vst1q_s16(dst + i + 24, v3);
      vst1q_s16(dst + i + 32, v4);
      vst1q_s16(dst + i + 40, v5);
      vst1q_s16(dst + i + 48, v6);
      vst1q_s16(dst + i + 56, v7);
    }
  }

  [[gnu::hot]] void refresh_kernel(int16_t *dst, const int16_t *const *rows, int n) {
    for (int i = 0; i < HL; i += 64) { // 8 x 8-lane vectors per tile
      int16x8_t v0 = vld1q_s16(g_net.ft_b + i);
      int16x8_t v1 = vld1q_s16(g_net.ft_b + i + 8);
      int16x8_t v2 = vld1q_s16(g_net.ft_b + i + 16);
      int16x8_t v3 = vld1q_s16(g_net.ft_b + i + 24);
      int16x8_t v4 = vld1q_s16(g_net.ft_b + i + 32);
      int16x8_t v5 = vld1q_s16(g_net.ft_b + i + 40);
      int16x8_t v6 = vld1q_s16(g_net.ft_b + i + 48);
      int16x8_t v7 = vld1q_s16(g_net.ft_b + i + 56);
      for (int k = 0; k < n; ++k) {
        const int16_t *r = rows[k] + i;
        v0               = vaddq_s16(v0, vld1q_s16(r));
        v1               = vaddq_s16(v1, vld1q_s16(r + 8));
        v2               = vaddq_s16(v2, vld1q_s16(r + 16));
        v3               = vaddq_s16(v3, vld1q_s16(r + 24));
        v4               = vaddq_s16(v4, vld1q_s16(r + 32));
        v5               = vaddq_s16(v5, vld1q_s16(r + 40));
        v6               = vaddq_s16(v6, vld1q_s16(r + 48));
        v7               = vaddq_s16(v7, vld1q_s16(r + 56));
      }
      vst1q_s16(dst + i, v0);
      vst1q_s16(dst + i + 8, v1);
      vst1q_s16(dst + i + 16, v2);
      vst1q_s16(dst + i + 24, v3);
      vst1q_s16(dst + i + 32, v4);
      vst1q_s16(dst + i + 40, v5);
      vst1q_s16(dst + i + 48, v6);
      vst1q_s16(dst + i + 56, v7);
    }
  }

  // In-place diff apply for the refresh cache — see the AVX2 twin above.
  [[gnu::hot]] void apply_rows_kernel(int16_t *acc, const int16_t *const *add, int na, const int16_t *const *sub,
                                      int ns) {
    for (int i = 0; i < HL; i += 64) { // 8 x 8-lane vectors per tile
      int16x8_t v0 = vld1q_s16(acc + i);
      int16x8_t v1 = vld1q_s16(acc + i + 8);
      int16x8_t v2 = vld1q_s16(acc + i + 16);
      int16x8_t v3 = vld1q_s16(acc + i + 24);
      int16x8_t v4 = vld1q_s16(acc + i + 32);
      int16x8_t v5 = vld1q_s16(acc + i + 40);
      int16x8_t v6 = vld1q_s16(acc + i + 48);
      int16x8_t v7 = vld1q_s16(acc + i + 56);
      for (int k = 0; k < na; ++k) {
        const int16_t *r = add[k] + i;
        v0               = vaddq_s16(v0, vld1q_s16(r));
        v1               = vaddq_s16(v1, vld1q_s16(r + 8));
        v2               = vaddq_s16(v2, vld1q_s16(r + 16));
        v3               = vaddq_s16(v3, vld1q_s16(r + 24));
        v4               = vaddq_s16(v4, vld1q_s16(r + 32));
        v5               = vaddq_s16(v5, vld1q_s16(r + 40));
        v6               = vaddq_s16(v6, vld1q_s16(r + 48));
        v7               = vaddq_s16(v7, vld1q_s16(r + 56));
      }
      for (int k = 0; k < ns; ++k) {
        const int16_t *r = sub[k] + i;
        v0               = vsubq_s16(v0, vld1q_s16(r));
        v1               = vsubq_s16(v1, vld1q_s16(r + 8));
        v2               = vsubq_s16(v2, vld1q_s16(r + 16));
        v3               = vsubq_s16(v3, vld1q_s16(r + 24));
        v4               = vsubq_s16(v4, vld1q_s16(r + 32));
        v5               = vsubq_s16(v5, vld1q_s16(r + 40));
        v6               = vsubq_s16(v6, vld1q_s16(r + 48));
        v7               = vsubq_s16(v7, vld1q_s16(r + 56));
      }
      vst1q_s16(acc + i, v0);
      vst1q_s16(acc + i + 8, v1);
      vst1q_s16(acc + i + 16, v2);
      vst1q_s16(acc + i + 24, v3);
      vst1q_s16(acc + i + 32, v4);
      vst1q_s16(acc + i + 40, v5);
      vst1q_s16(acc + i + 48, v6);
      vst1q_s16(acc + i + 56, v7);
    }
  }

  [[gnu::pure, gnu::hot]] int32_t dot_reduce(const int16_t *stm_a, const int16_t *opp_a, const int16_t *w) {
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t qa   = vdupq_n_s16(QA);
    const int16_t  *wo   = w + HL; // opponent half of the bucket's weights
    // Both perspective halves in ONE loop with EIGHT independent accumulators. vmlal has
    // multi-cycle latency and Apple cores run four SIMD pipes: four chains per half (the old
    // dot_half) leaves the pipes under-filled between dependent iterations — interleaving the
    // two halves doubles the independent work in flight and halves loop overhead. Each
    // accumulator sees exactly the same additions in the same order as before, and the final
    // reduction keeps the old (s0+s1)+(s2+s3), stm-half + opp-half structure: int32 wraparound
    // addition makes the whole thing bit-identical to the two-call version.
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t b0 = vdupq_n_s32(0), b1 = vdupq_n_s32(0), b2 = vdupq_n_s32(0), b3 = vdupq_n_s32(0);
    for (int i = 0; i < HL; i += 16) {
      int16x8_t v0 = vminq_s16(vmaxq_s16(vld1q_s16(stm_a + i), zero), qa);
      int16x8_t v1 = vminq_s16(vmaxq_s16(vld1q_s16(stm_a + i + 8), zero), qa);
      int16x8_t u0 = vminq_s16(vmaxq_s16(vld1q_s16(opp_a + i), zero), qa);
      int16x8_t u1 = vminq_s16(vmaxq_s16(vld1q_s16(opp_a + i + 8), zero), qa);
      // v*w fits int16 exactly (|w_q| <= 127) — the NEON equivalent of AVX2's mullo+madd.
      const int16x8_t p0 = vmulq_s16(v0, vld1q_s16(w + i));
      const int16x8_t p1 = vmulq_s16(v1, vld1q_s16(w + i + 8));
      const int16x8_t q0 = vmulq_s16(u0, vld1q_s16(wo + i));
      const int16x8_t q1 = vmulq_s16(u1, vld1q_s16(wo + i + 8));
      a0                 = vmlal_s16(a0, vget_low_s16(v0), vget_low_s16(p0));
      a1                 = vmlal_high_s16(a1, v0, p0);
      a2                 = vmlal_s16(a2, vget_low_s16(v1), vget_low_s16(p1));
      a3                 = vmlal_high_s16(a3, v1, p1);
      b0                 = vmlal_s16(b0, vget_low_s16(u0), vget_low_s16(q0));
      b1                 = vmlal_high_s16(b1, u0, q0);
      b2                 = vmlal_s16(b2, vget_low_s16(u1), vget_low_s16(q1));
      b3                 = vmlal_high_s16(b3, u1, q1);
    }
    const int32x4_t stm_sum = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
    const int32x4_t opp_sum = vaddq_s32(vaddq_s32(b0, b1), vaddq_s32(b2, b3));
    return vaddvq_s32(vaddq_s32(stm_sum, opp_sum));
  }

#else // scalar reference (SIMD off / unknown arch)

  [[gnu::hot]] void add1_sub1(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *s0) {
    for (int i = 0; i < HL; ++i)
      dst[i] = static_cast<int16_t>(src[i] + a0[i] - s0[i]);
  }
  [[gnu::hot]] void add1_sub2(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *s0, const int16_t *s1) {
    for (int i = 0; i < HL; ++i)
      dst[i] = static_cast<int16_t>(src[i] + a0[i] - s0[i] - s1[i]);
  }
  [[gnu::hot]] void add2_sub2(int16_t *dst, const int16_t *src, const int16_t *a0, const int16_t *a1, const int16_t *s0,
                 const int16_t *s1) {
    for (int i = 0; i < HL; ++i)
      dst[i] = static_cast<int16_t>(src[i] + a0[i] + a1[i] - s0[i] - s1[i]);
  }

  [[gnu::hot]] void refresh_kernel(int16_t *dst, const int16_t *const *rows, int n) {
    std::memcpy(dst, g_net.ft_b, sizeof(g_net.ft_b));
    for (int k = 0; k < n; ++k) {
      const int16_t *r = rows[k];
      for (int i = 0; i < HL; ++i)
        dst[i] = static_cast<int16_t>(dst[i] + r[i]);
    }
  }

  // In-place diff apply for the refresh cache — see the AVX2/NEON twins above. All three are
  // exact int16 wraparound arithmetic, so the paths stay bit-identical.
  [[gnu::hot]] void apply_rows_kernel(int16_t *acc, const int16_t *const *add, int na, const int16_t *const *sub,
                                      int ns) {
    for (int k = 0; k < na; ++k) {
      const int16_t *r = add[k];
      for (int i = 0; i < HL; ++i)
        acc[i] = static_cast<int16_t>(acc[i] + r[i]);
    }
    for (int k = 0; k < ns; ++k) {
      const int16_t *r = sub[k];
      for (int i = 0; i < HL; ++i)
        acc[i] = static_cast<int16_t>(acc[i] - r[i]);
    }
  }

  [[gnu::pure, gnu::hot]] int32_t dot_reduce(const int16_t *stm_a, const int16_t *opp_a, const int16_t *w) {
    int32_t        sum     = 0;
    const int16_t *half[2] = {stm_a, opp_a};
    for (int h = 0; h < 2; ++h) {
      const int16_t *a  = half[h];
      const int16_t *wh = w + h * HL;
      for (int i = 0; i < HL; ++i) {
        const int32_t v = std::clamp<int32_t>(a[i], 0, QA);
        sum += v * v * static_cast<int32_t>(wh[i]); // exact: |v*v*w| <= 255*255*127 < 2^31
      }
    }
    return sum;
  }

#endif

  // Rebuilds one perspective half from scratch (bias + every piece). Reference path for
  // evaluate_refresh only; the search refreshes through the cache (refresh_half_cached),
  // which selftest nnue pins bit-exactly against this.
  [[gnu::hot]] void refresh_half(Accumulator &acc, Color persp, const Position &pos) {
    const KingCtx  c = king_ctx_of(pos, persp);
    const int16_t *rows[40]; // 32 pieces max in a legal game; headroom for weird FEN input
    int            n = 0;
    for (int pc = WHITE_PAWN; pc <= BLACK_KING; ++pc) {
      if (pc > WHITE_KING && pc < BLACK_PAWN)
        continue; // the gap in the Piece enum (6, 7)
      Bitboard bb = pos.bitboard_of(Piece(pc));
      while (bb && n < 40)
        rows[n++] = ft_row(persp, Piece(pc), pop_lsb(&bb), c);
    }
    refresh_kernel(acc.v[persp], rows, n);
    acc.ctx[persp]      = c;
    acc.computed[persp] = true;
  }

  // Builds one perspective half of `child` from `parent` by applying child.dp under a king
  // context that is KNOWN not to change (the caller checked). Kings are ordinary features.
  [[gnu::hot]] void apply_half(Accumulator &child, const Accumulator &parent, Color persp) {
    const DirtyPiece &dp  = child.dp;
    const KingCtx     c   = parent.ctx[persp];
    const int16_t    *src = parent.v[persp];
    int16_t          *dst = child.v[persp];

    if (dp.n_add == 2) // castling: 2 adds, 2 subs
      add2_sub2(dst, src, ft_row(persp, dp.add_pc[0], dp.add_sq[0], c), ft_row(persp, dp.add_pc[1], dp.add_sq[1], c),
                ft_row(persp, dp.sub_pc[0], dp.sub_sq[0], c), ft_row(persp, dp.sub_pc[1], dp.sub_sq[1], c));
    else if (dp.n_sub == 2) // capture / en passant: 1 add, 2 subs
      add1_sub2(dst, src, ft_row(persp, dp.add_pc[0], dp.add_sq[0], c), ft_row(persp, dp.sub_pc[0], dp.sub_sq[0], c),
                ft_row(persp, dp.sub_pc[1], dp.sub_sq[1], c));
    else if (dp.n_add == 1) // quiet / promotion: 1 add, 1 sub
      add1_sub1(dst, src, ft_row(persp, dp.add_pc[0], dp.add_sq[0], c), ft_row(persp, dp.sub_pc[0], dp.sub_sq[0], c));
    else // null move: no feature changes
      std::memcpy(dst, src, sizeof(int16_t) * HL);

    child.ctx[persp]      = c;
    child.computed[persp] = true;
  }

  // True when `dp` moves `persp`'s own king (adds always contain the mover).
  [[gnu::pure, gnu::always_inline]] inline bool moves_own_king(const DirtyPiece &dp, Color persp) {
    for (int i = 0; i < dp.n_add; ++i)
      if (dp.add_pc[i] == make_piece(persp, KING))
        return true;
    return false;
  }

  // New king context after `dp` (call only when moves_own_king): from the king's add square.
  [[gnu::pure, gnu::always_inline]] inline KingCtx ctx_after(const DirtyPiece &dp, Color persp) {
    for (int i = 0; i < dp.n_add; ++i)
      if (dp.add_pc[i] == make_piece(persp, KING))
        return king_ctx(persp == WHITE ? dp.add_sq[i] : Square(dp.add_sq[i] ^ 56));
    return {0, false}; // unreachable
  }

  // SCReLU output layer through the position's material bucket, stm half first.
  [[gnu::pure, gnu::hot]] int output_dot(const Accumulator &acc, Color stm, int bucket) {
    const int32_t sum = dot_reduce(acc.v[stm], acc.v[~stm], g_net.out_w[bucket]);
    // sum is at scale QA^2*QB; one /QA plus the bias (at QA*QB) then rescale to centipawns.
    return ((sum / QA) + g_net.out_b[bucket]) * SCALE / (QA * QB);
  }

} // namespace

// --- Loading -------------------------------------------------------------------------------

bool nnue::load_buffer(const unsigned char *data, size_t size, std::string *err) {
  const auto fail = [err](const char *m) {
    if (err)
      *err = m;
    return false;
  };

  NetHeader h{};
  if (size < sizeof(NetHeader))
    return fail("file smaller than the header");
  std::memcpy(&h, data, sizeof(NetHeader));

  if (std::memcmp(h.magic, "AKNN", 4) != 0)
    return fail("bad magic (not an askaig .nnue file)");
  if (h.version != 2)
    return fail("unsupported version (this engine reads v2: king-bucketed 768, output buckets)");
  if (h.features != FEATURES || h.hl != HL || h.buckets != OUT_BUCKETS)
    return fail("architecture mismatch (features/hl/buckets)");
  if (h.qa != QA || h.qb != QB || h.scale != SCALE || h.activation != 1)
    return fail("quantization/activation mismatch");

  constexpr size_t FT_W  = sizeof(g_net.ft_w);
  constexpr size_t FT_B  = sizeof(g_net.ft_b);
  constexpr size_t OUT_W = sizeof(g_net.out_w);
  constexpr size_t OUT_B = sizeof(g_net.out_b);
  if (size != sizeof(NetHeader) + FT_W + FT_B + OUT_W + OUT_B)
    return fail("file size does not match the architecture");

  const unsigned char *p = data + sizeof(NetHeader);
  std::memcpy(g_net.ft_w, p, FT_W);
  p += FT_W;
  std::memcpy(g_net.ft_b, p, FT_B);
  p += FT_B;
  std::memcpy(g_net.out_w, p, OUT_W);
  p += OUT_W;
  std::memcpy(g_net.out_b, p, OUT_B);

  g_loaded = true;
  return true;
}

bool nnue::load_file(const std::string &path, std::string *err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err)
      *err = "cannot open file";
    return false;
  }
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return load_buffer(buf.data(), buf.size(), err);
}

// The build-time default net (a byte copy of networks/default.nnue), generated into the
// build tree by cmake/embed_file.cmake.
extern const unsigned char g_default_net[];
extern const size_t        g_default_net_size;

bool nnue::load_embedded(std::string *err) { return load_buffer(g_default_net, g_default_net_size, err); }

bool nnue::loaded() { return g_loaded; }

// --- Evaluator -----------------------------------------------------------------------------

nnue::Evaluator::Evaluator() : stack(new Accumulator[MAX_PLY + 8]), finny(new RefreshTable), top(0) {
  stack[0].computed[WHITE] = stack[0].computed[BLACK] = false;
}

void nnue::Evaluator::reset(const Position &pos) {
  top           = 0;
  finny->inited = false; // net weights may have changed since the last search (EvalFile)
  refresh_half_cached(WHITE, pos);
  refresh_half_cached(BLACK, pos);
}

// Rebuilds stack[top]'s `persp` half through the refresh cache: diff the position's piece
// placement against the cached one for this (perspective, mirror, bucket) context and apply
// just the difference — a handful of rows versus the ~32 a from-scratch rebuild pays.
[[gnu::hot]] void nnue::Evaluator::refresh_half_cached(Color persp, const Position &pos) {
  const KingCtx c = king_ctx_of(pos, persp);
  if (!finny->inited) [[unlikely]] {
    // First use after reset(): every cached context restarts from the empty board.
    for (auto &per_persp: finny->e)
      for (auto &per_mirror: per_persp)
        for (RefreshEntry &e: per_mirror) {
          std::memcpy(e.v, g_net.ft_b, sizeof(e.v));
          std::memset(e.bb, 0, sizeof(e.bb));
        }
    finny->inited = true;
  }
  RefreshEntry &e = finny->e[persp][c.mirror][c.bucket];

  const int16_t *add[40], *sub[40]; // 32 pieces max in a legal game; headroom for weird FENs
  int            na = 0, ns = 0;
  bool           over = false;
  for (int pc = WHITE_PAWN; pc <= BLACK_KING && !over; ++pc) {
    if (pc > WHITE_KING && pc < BLACK_PAWN)
      continue; // the gap in the Piece enum (6, 7)
    const Bitboard now = pos.bitboard_of(Piece(pc));
    Bitboard       a = now & ~e.bb[pc], s = e.bb[pc] & ~now;
    while (a && !(over = na == 40))
      add[na++] = ft_row(persp, Piece(pc), pop_lsb(&a), c);
    while (s && !(over = ns == 40))
      sub[ns++] = ft_row(persp, Piece(pc), pop_lsb(&s), c);
    e.bb[pc] = now;
  }
  if (over) [[unlikely]] {
    // A non-game FEN blew the diff headroom: rebuild the entry from scratch instead.
    const int16_t *rows[40];
    int            n = 0;
    for (int pc = WHITE_PAWN; pc <= BLACK_KING; ++pc) {
      if (pc > WHITE_KING && pc < BLACK_PAWN)
        continue;
      Bitboard bb = pos.bitboard_of(Piece(pc));
      e.bb[pc]    = bb;
      while (bb && n < 40)
        rows[n++] = ft_row(persp, Piece(pc), pop_lsb(&bb), c);
    }
    refresh_kernel(e.v, rows, n);
  } else
    apply_rows_kernel(e.v, add, na, sub, ns);

  Accumulator &acc = stack[top];
  std::memcpy(acc.v[persp], e.v, sizeof(e.v));
  acc.ctx[persp]      = c;
  acc.computed[persp] = true;
}

// Records the feature changes of `m` (to be played on `before`) into a new stack entry.
// Mirrors the play<C>() switch exactly; castling squares are hardcoded per color and the
// captured piece is read from `before`.
[[gnu::hot]] void nnue::Evaluator::push(const Position &before, Move m) {
  assert(top + 1 < MAX_PLY + 8);
  Accumulator &a     = stack[++top];
  a.computed[WHITE]  = false;
  a.computed[BLACK]  = false;
  DirtyPiece &dp     = a.dp;
  dp.n_add = dp.n_sub = 0;

  const auto add = [&dp](Piece pc, Square s) {
    dp.add_pc[dp.n_add] = pc;
    dp.add_sq[dp.n_add] = s;
    ++dp.n_add;
  };
  const auto sub = [&dp](Piece pc, Square s) {
    dp.sub_pc[dp.n_sub] = pc;
    dp.sub_sq[dp.n_sub] = s;
    ++dp.n_sub;
  };

  const Color c     = before.turn();
  const Piece mover = before.at(m.from());

  switch (m.flags()) {
    case QUIET:
    case DOUBLE_PUSH:
      sub(mover, m.from());
      add(mover, m.to());
      break;
    case OO:
      if (c == WHITE) {
        sub(WHITE_KING, e1), sub(WHITE_ROOK, h1);
        add(WHITE_KING, g1), add(WHITE_ROOK, f1);
      } else {
        sub(BLACK_KING, e8), sub(BLACK_ROOK, h8);
        add(BLACK_KING, g8), add(BLACK_ROOK, f8);
      }
      break;
    case OOO:
      if (c == WHITE) {
        sub(WHITE_KING, e1), sub(WHITE_ROOK, a1);
        add(WHITE_KING, c1), add(WHITE_ROOK, d1);
      } else {
        sub(BLACK_KING, e8), sub(BLACK_ROOK, a8);
        add(BLACK_KING, c8), add(BLACK_ROOK, d8);
      }
      break;
    case EN_PASSANT:
      sub(mover, m.from());
      sub(make_piece(~c, PAWN), m.to() + (c == WHITE ? SOUTH : NORTH));
      add(mover, m.to());
      break;
    case PR_KNIGHT:
    case PR_BISHOP:
    case PR_ROOK:
    case PR_QUEEN:
      sub(mover, m.from());
      add(make_piece(c, PieceType(KNIGHT + (m.flags() & 0b11))), m.to());
      break;
    case PC_KNIGHT:
    case PC_BISHOP:
    case PC_ROOK:
    case PC_QUEEN:
      sub(mover, m.from());
      sub(before.at(m.to()), m.to());
      add(make_piece(c, PieceType(KNIGHT + (m.flags() & 0b11))), m.to());
      break;
    case CAPTURE:
      sub(mover, m.from());
      sub(before.at(m.to()), m.to());
      add(mover, m.to());
      break;
    default:
      break;
  }
}

void nnue::Evaluator::push_null() {
  assert(top + 1 < MAX_PLY + 8);
  Accumulator &a    = stack[++top];
  a.computed[WHITE] = false;
  a.computed[BLACK] = false;
  a.dp.n_add = a.dp.n_sub = 0; // no feature changes; applied as a copy
}

[[gnu::hot]] void nnue::Evaluator::pop() {
  assert(top > 0);
  --top;
}

// Makes stack[top]'s `persp` half valid: walk back to the nearest computed ancestor and apply
// the recorded updates forward — unless an own-king move changes the (bucket, mirror) context
// anywhere in the chain (or there is no usable ancestor), in which case the half is rebuilt
// through the refresh cache (refresh_half_cached).
[[gnu::hot]] void nnue::Evaluator::ensure_half(Color persp, const Position &pos) {
  if (stack[top].computed[persp])
    return;

  constexpr int MAX_BACKTRACK = 24;
  int           j             = top;
  while (j > 0 && !stack[j].computed[persp] && top - j < MAX_BACKTRACK)
    --j;
  if (!stack[j].computed[persp]) {
    refresh_half_cached(persp, pos);
    return;
  }

  // Scan forward: any context-crossing king move forces a refresh instead.
  KingCtx c = stack[j].ctx[persp];
  for (int k = j + 1; k <= top; ++k) {
    if (moves_own_king(stack[k].dp, persp)) {
      const KingCtx nc = ctx_after(stack[k].dp, persp);
      if (!(nc == c)) {
        refresh_half_cached(persp, pos);
        return;
      }
      c = nc;
    }
  }
  for (int k = j + 1; k <= top; ++k)
    apply_half(stack[k], stack[k - 1], persp);
}

// The main search's static-eval entry point (lazy: pays for the walk-back only when called).
[[gnu::hot, nodiscard]] int nnue::Evaluator::evaluate(const Position &pos) {
  assert(g_loaded);
  ensure_half(WHITE, pos);
  ensure_half(BLACK, pos);
  return output_dot(stack[top], pos.turn(), out_bucket(pos));
}

// One-shot eval via a full LOCAL from-scratch refresh: the reference path for the UCI
// `eval`/`d` commands and the parity selftests.
[[gnu::pure, nodiscard]] int nnue::evaluate_refresh(const Position &pos) {
  assert(g_loaded);
  Accumulator acc;
  refresh_half(acc, WHITE, pos);
  refresh_half(acc, BLACK, pos);
  return output_dot(acc, pos.turn(), out_bucket(pos));
}
