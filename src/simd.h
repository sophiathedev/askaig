#pragma once

// SIMD / hardware-intrinsic implementations of the bit primitives.
//
// Enabled with the CMake option SIMD=ON, which also picks an ARCH and defines one of:
//   ARCH_AVX2      -> x86-64 hardware bit instructions (POPCNT / BMI TZCNT) + AVX2 vectors
//   ARCH_ARM_NEON  -> AArch64 NEON (vcnt / rbit+clz / vrbit)
// When SIMD is off (or no ARCH matches) every function falls back to the portable
// scalar implementation, so results are identical across all build configurations.
//
// This header is included from types.h *after* the Bitboard and Square types are declared.
// The scalar fallback uses <bit> (std::popcount / std::countr_zero), which the compiler
// lowers to the best available instruction (POPCNT/TZCNT/CNT/rbit+clz) on its own.

#include <bit>
#include <cstdint>

#if defined(SIMD) && defined(ARCH_AVX2)
#include <immintrin.h>
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
#include <arm_neon.h>
#endif

// Returns the number of set bits in the bitboard
[[gnu::const, gnu::always_inline]] inline int pop_count(Bitboard b) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  return static_cast<int>(_mm_popcnt_u64(b));
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
  return vaddv_u8(vcnt_u8(vcreate_u8(b)));
#else
  return std::popcount(b); // lowers to POPCNT/CNT when the target has it
#endif
}

// Returns the number of set bits in the bitboard. Kept as a distinct entry point for
// readability at call-sites; std::popcount is optimal regardless of bit density.
[[gnu::const, gnu::always_inline]] inline int sparse_pop_count(Bitboard b) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  return static_cast<int>(_mm_popcnt_u64(b));
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
  return vaddv_u8(vcnt_u8(vcreate_u8(b)));
#else
  return std::popcount(b);
#endif
}

// Returns the index of the least significant set bit. For b == 0 the SIMD branches are
// undefined (callers always guard with a non-zero bitboard, e.g. while (b) loops); the
// std::countr_zero fallback is well-defined (returns 64).
[[gnu::const, gnu::always_inline]] inline Square bsf(Bitboard b) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  return Square(_tzcnt_u64(b));
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
  // Lowers to rbit + clz. Deliberately the builtin, NOT inline asm: a hand-written
  // "rbit; clz" asm block was A/B-measured 2.4% SLOWER overall — the opaque asm
  // boundary blocks the compiler from scheduling/fusing around the two instructions.
  return Square(__builtin_ctzll(b));
#else
  return Square(std::countr_zero(b)); // lowers to TZCNT/rbit+clz when available
#endif
}

// Returns the index of the least significant set bit and clears it from the bitboard.
// On x86-64 the clear is an explicit BLSR (one uop, no flags dependency on the subtract);
// elsewhere the and-with-decrement lowers to the same two ALU ops the target offers.
[[gnu::always_inline]] inline Square pop_lsb(Bitboard *b) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  const Square lsb = Square(_tzcnt_u64(*b));
  *b               = _blsr_u64(*b);
  return lsb;
#else
  Square lsb = bsf(*b);
  *b &= *b - 1;
  return lsb;
#endif
}

// ---------------------------------------------------------------------------
// Vectorised bitboard accumulation helpers (used by Position::all_pieces and
// Position::attackers_from). These operate on small fixed groups of bitboards.
// ---------------------------------------------------------------------------

// OR-reduce six contiguous bitboards p[0..5] into one. On AArch64 this is deliberately
// scalar: the result is needed in a general-purpose register, and each NEON->GPR move
// (fmov) costs several cycles of latency on Apple cores — a pure ldp+orr chain beats
// vector ORs followed by two lane extractions.
[[gnu::pure, gnu::always_inline]] inline Bitboard or_reduce6(const Bitboard *p) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  __m256i v   = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)); // p[0..3]
  __m128i hi  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 4)); // p[4..5]
  __m128i lo4 = _mm256_castsi256_si128(v);
  __m128i up4 = _mm256_extracti128_si256(v, 1);
  __m128i r   = _mm_or_si128(_mm_or_si128(lo4, up4), hi);
  return static_cast<Bitboard>(_mm_extract_epi64(r, 0)) | static_cast<Bitboard>(_mm_extract_epi64(r, 1));
#else
  return (p[0] | p[1]) | (p[2] | p[3]) | (p[4] | p[5]);
#endif
}

// Computes (att[0] & msk[0]) | ... | (att[3] & msk[3]). Scalar on AArch64 for the same
// reason as or_reduce6: the operands are fresh magic-lookup results already in GPRs, so a
// NEON detour pays stack spills plus two fmov round-trips.
[[gnu::pure, gnu::always_inline]] inline Bitboard and_or_reduce4(const Bitboard *att, const Bitboard *msk) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(att));
  __m256i m = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(msk));
  __m256i x = _mm256_and_si256(a, m);
  __m128i r = _mm_or_si128(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  return static_cast<Bitboard>(_mm_extract_epi64(r, 0)) | static_cast<Bitboard>(_mm_extract_epi64(r, 1));
#else
  return ((att[0] & msk[0]) | (att[1] & msk[1])) | ((att[2] & msk[2]) | (att[3] & msk[3]));
#endif
}
