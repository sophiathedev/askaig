#pragma once


#include <bit>
#include <cstdint>

#if defined(SIMD) && defined(ARCH_AVX2)
#include <immintrin.h>
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
#include <arm_neon.h>
#endif

[[gnu::const, gnu::always_inline]] inline int pop_count(Bitboard b) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  return static_cast<int>(_mm_popcnt_u64(b));
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
  return vaddv_u8(vcnt_u8(vcreate_u8(b)));
#else
  return std::popcount(b);
#endif
}

[[gnu::const, gnu::always_inline]] inline int sparse_pop_count(Bitboard b) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  return static_cast<int>(_mm_popcnt_u64(b));
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
  return vaddv_u8(vcnt_u8(vcreate_u8(b)));
#else
  return std::popcount(b);
#endif
}

[[gnu::const, gnu::always_inline]] inline Square bsf(Bitboard b) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
  return Square(_tzcnt_u64(b));
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
  return Square(__builtin_ctzll(b));
#else
  return Square(std::countr_zero(b));
#endif
}

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
