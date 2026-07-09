#pragma once

#include "types.h"

extern const Bitboard KING_ATTACKS[NSQUARES];
extern const Bitboard KNIGHT_ATTACKS[NSQUARES];
extern const Bitboard WHITE_PAWN_ATTACKS[NSQUARES];
extern const Bitboard BLACK_PAWN_ATTACKS[NSQUARES];

extern Bitboard reverse(Bitboard b);
extern Bitboard sliding_attacks(Square square, Bitboard occ, Bitboard mask);

// Per-square "fancy magic" descriptor, packed so one slider lookup touches a single
// 32-byte struct (one cache line holds two squares) instead of four separate arrays
// (mask/magic/shift/base-pointer on four different lines). `shift` doubles as padding.
struct alignas(32) Magic {
  Bitboard  mask; // relevant-occupancy mask
  Bitboard  magic; // magic multiplier (unused on the PEXT path)
  Bitboard *ptr; // base of this square's slice in the flat attack table
  uint64_t  shift; // 64 - popcount(mask) (unused on the PEXT path)
};

extern Magic ROOK_MAGIC[NSQUARES];
extern Magic BISHOP_MAGIC[NSQUARES];

extern Bitboard get_rook_attacks_for_init(Square square, Bitboard occ);
extern void     initialise_rook_attacks();

extern Bitboard get_bishop_attacks_for_init(Square square, Bitboard occ);
extern void     initialise_bishop_attacks();

// Slice index for a slider with occupancy `occ`. On x86 with BMI2 (PEXT) this extracts the
// relevant occupancy bits directly — one instruction, no multiply; everywhere else (incl. ARM
// NEON) it is the classic magic multiply-shift. Both yield an index in [0, 2^bits), and BOTH
// the table init and the lookups go through this function, so the flat table is always filled
// at the slots the lookup will read — switching indexing methods can never desynchronise them.
[[gnu::const, gnu::always_inline]] inline uint64_t magic_index(const Magic &m, Bitboard occ) noexcept {
#if defined(ARCH_AVX2) && defined(__BMI2__)
  return _pext_u64(occ, m.mask);
#else
  return ((occ & m.mask) * m.magic) >> m.shift;
#endif
}

// Flat attack-table lookups, inline so they fold into the movegen loops.
[[gnu::pure, gnu::always_inline]] inline Bitboard get_rook_attacks(Square square, Bitboard occ) {
  const Magic &m = ROOK_MAGIC[square];
  return m.ptr[magic_index(m, occ)];
}

[[gnu::pure, gnu::always_inline]] inline Bitboard get_bishop_attacks(Square square, Bitboard occ) {
  const Magic &m = BISHOP_MAGIC[square];
  return m.ptr[magic_index(m, occ)];
}

extern Bitboard get_xray_rook_attacks(Square square, Bitboard occ, Bitboard blockers);
extern Bitboard get_xray_bishop_attacks(Square square, Bitboard occ, Bitboard blockers);

extern Bitboard SQUARES_BETWEEN_BB[NSQUARES][NSQUARES];
extern Bitboard LINE[NSQUARES][NSQUARES];
extern Bitboard PAWN_ATTACKS[NCOLORS][NSQUARES];
extern Bitboard PSEUDO_LEGAL_ATTACKS[NPIECE_TYPES][NSQUARES];

extern void initialise_squares_between();
extern void initialise_line();
extern void initialise_pseudo_legal();
extern void initialise_all_databases();


// Returns a bitboard containing all squares that a piece on a square can move to, in the given position
template<PieceType P>
constexpr Bitboard attacks(Square s, Bitboard occ) {
  static_assert(P != PAWN, "The piece type may not be a pawn; use pawn_attacks instead");
  return P == ROOK     ? get_rook_attacks(s, occ)
         : P == BISHOP ? get_bishop_attacks(s, occ)
         : P == QUEEN  ? attacks<ROOK>(s, occ) | attacks<BISHOP>(s, occ)
                       : PSEUDO_LEGAL_ATTACKS[P][s];
}

// Returns a bitboard containing all squares that a piece on a square can move to, in the given position
// Used when the piece type is not known at compile-time
constexpr Bitboard attacks(PieceType pt, Square s, Bitboard occ) {
  switch (pt) {
    case ROOK:
      return attacks<ROOK>(s, occ);
    case BISHOP:
      return attacks<BISHOP>(s, occ);
    case QUEEN:
      return attacks<QUEEN>(s, occ);
    default:
      return PSEUDO_LEGAL_ATTACKS[pt][s];
  }
}

// Returns a bitboard containing pawn attacks from all pawns in the given bitboard
template<Color C>
constexpr Bitboard pawn_attacks(Bitboard p) {
  return C == WHITE ? shift<NORTH_WEST>(p) | shift<NORTH_EAST>(p) : shift<SOUTH_WEST>(p) | shift<SOUTH_EAST>(p);
}

// Attacks of ALL knights in `b` at once (the classic setwise knight fill, CPW): a pure
// shift/mask ALU tree — no per-knight table load, no loop whose trip count (0..2 knights,
// varying node to node) the branch predictor must guess.
constexpr Bitboard knight_attacks(Bitboard b) {
  const Bitboard l1 = (b >> 1) & 0x7f7f7f7f7f7f7f7f;
  const Bitboard l2 = (b >> 2) & 0x3f3f3f3f3f3f3f3f;
  const Bitboard r1 = (b << 1) & 0xfefefefefefefefe;
  const Bitboard r2 = (b << 2) & 0xfcfcfcfcfcfcfcfc;
  const Bitboard h1 = l1 | r1;
  const Bitboard h2 = l2 | r2;
  return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

// Returns a bitboard containing pawn attacks from the pawn on the given square
template<Color C>
constexpr Bitboard pawn_attacks(Square s) {
  return PAWN_ATTACKS[C][s];
}
