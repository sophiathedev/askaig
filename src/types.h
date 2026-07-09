#pragma once
#ifdef _MSC_VER
  #pragma warning(disable : 26812) // unscoped enum (MoveFlags, PieceType, ...): intentional, matches Stockfish/Surge style
#endif

#include <array>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

constexpr size_t NCOLORS = 2;
enum Color : int { WHITE, BLACK };

// Inverts the color (WHITE -> BLACK) and (BLACK -> WHITE)
constexpr Color operator~(Color c) { return Color(c ^ BLACK); }

constexpr size_t NDIRS = 8;
enum Direction : int {
  NORTH       = 8,
  NORTH_EAST  = 9,
  EAST        = 1,
  SOUTH_EAST  = -7,
  SOUTH       = -8,
  SOUTH_WEST  = -9,
  WEST        = -1,
  NORTH_WEST  = 7,
  NORTH_NORTH = 16,
  SOUTH_SOUTH = -16
};

constexpr size_t NPIECE_TYPES = 6;
enum PieceType : int { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

// PIECE_STR[piece] is the algebraic chess representation of that piece
const std::string PIECE_STR = "PNBRQK~>pnbrqk.";

// The FEN of the starting position
const std::string DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";

// The Kiwipete position, used for perft debugging
const std::string KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";


constexpr size_t NPIECES = 15;
enum Piece : int {
  WHITE_PAWN,
  WHITE_KNIGHT,
  WHITE_BISHOP,
  WHITE_ROOK,
  WHITE_QUEEN,
  WHITE_KING,
  BLACK_PAWN = 8,
  BLACK_KNIGHT,
  BLACK_BISHOP,
  BLACK_ROOK,
  BLACK_QUEEN,
  BLACK_KING,
  NO_PIECE
};

constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) + pt); }

constexpr PieceType type_of(Piece pc) { return PieceType(pc & 0b111); }

constexpr Color color_of(Piece pc) { return Color((pc & 0b1000) >> 3); }


using Bitboard = uint64_t;

constexpr size_t NSQUARES = 64;
enum Square : int {
  a1,
  b1,
  c1,
  d1,
  e1,
  f1,
  g1,
  h1,
  a2,
  b2,
  c2,
  d2,
  e2,
  f2,
  g2,
  h2,
  a3,
  b3,
  c3,
  d3,
  e3,
  f3,
  g3,
  h3,
  a4,
  b4,
  c4,
  d4,
  e4,
  f4,
  g4,
  h4,
  a5,
  b5,
  c5,
  d5,
  e5,
  f5,
  g5,
  h5,
  a6,
  b6,
  c6,
  d6,
  e6,
  f6,
  g6,
  h6,
  a7,
  b7,
  c7,
  d7,
  e7,
  f7,
  g7,
  h7,
  a8,
  b8,
  c8,
  d8,
  e8,
  f8,
  g8,
  h8,
  NO_SQUARE
};

inline Square   &operator++(Square &s) { return s = Square(int(s) + 1); }
constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
inline Square   &operator+=(Square &s, Direction d) { return s = s + d; }
inline Square   &operator-=(Square &s, Direction d) { return s = s - d; }

enum File : int { AFILE, BFILE, CFILE, DFILE, EFILE, FFILE, GFILE, HFILE };

enum Rank : int { RANK1, RANK2, RANK3, RANK4, RANK5, RANK6, RANK7, RANK8 };

extern const char *SQSTR[65];

// Formulaic constant tables, constexpr in the header so the compiler SEES the values:
// constant-index accesses fold to immediates (castling squares, shift<>'s file masks become
// AND-with-immediate), instead of the opaque ldr that an extern table forces on every access.
inline constexpr Bitboard MASK_FILE[8] = {
        0x101010101010101,  0x202020202020202,  0x404040404040404,  0x808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
};

inline constexpr Bitboard MASK_RANK[8] = {0xff,         0xff00,         0xff0000,         0xff000000,
                                          0xff00000000, 0xff0000000000, 0xff000000000000, 0xff00000000000000};

inline constexpr Bitboard MASK_DIAGONAL[15] = {
        0x80,           0x8040,           0x804020,           0x80402010,        0x8040201008,
        0x804020100804, 0x80402010080402, 0x8040201008040201, 0x4020100804020100, 0x2010080402010000,
        0x1008040201000000, 0x804020100000000, 0x402010000000000, 0x201000000000000, 0x100000000000000,
};

inline constexpr Bitboard MASK_ANTI_DIAGONAL[15] = {
        0x1,           0x102,           0x10204,           0x1020408,          0x102040810,
        0x10204081020, 0x1020408102040, 0x102040810204080, 0x204081020408000,  0x408102040800000,
        0x810204080000000, 0x1020408000000000, 0x2040800000000000, 0x4080000000000000, 0x8000000000000000,
};

// SQUARE_BB[NO_SQUARE] == 0 is load-bearing: en-passant code indexes with a possibly-NO_SQUARE
// square and relies on getting an empty bitboard back. For indices that are PROVABLY a real
// square, prefer sq_bb() below (two ALU ops, no table access at all).
inline constexpr auto SQUARE_BB = [] {
  std::array<Bitboard, 65> t{};
  for (int i = 0; i < 64; ++i)
    t[i] = Bitboard(1) << i;
  t[64] = 0; // NO_SQUARE
  return t;
}();

// 1-bit bitboard of a real square (s < 64 REQUIRED — no NO_SQUARE sentinel handling).
constexpr Bitboard sq_bb(Square s) { return Bitboard(1) << s; }

extern void print_bitboard(Bitboard b);

// SIMD/hardware-intrinsic bit primitives (pop_count, sparse_pop_count, bsf, pop_lsb).
// Declared inline here so they inline at the call-site; needs Bitboard and Square to be
// visible, hence included at this point.
#include "simd.h"

constexpr Rank   rank_of(Square s) { return Rank(s >> 3); }
constexpr File   file_of(Square s) { return File(s & 0b111); }
constexpr int    diagonal_of(Square s) { return 7 + rank_of(s) - file_of(s); }
constexpr int    anti_diagonal_of(Square s) { return int(rank_of(s)) + int(file_of(s)); }
constexpr Square create_square(File f, Rank r) { return Square(r << 3 | f); }

// Shifts a bitboard in a particular direction. There is no wrapping, so bits that are shifted of the edge are lost
template<Direction D>
constexpr Bitboard shift(Bitboard b) {
  return D == NORTH           ? b << 8
         : D == SOUTH         ? b >> 8
         : D == NORTH + NORTH ? b << 16
         : D == SOUTH + SOUTH ? b >> 16
         : D == EAST          ? (b & ~MASK_FILE[HFILE]) << 1
         : D == WEST          ? (b & ~MASK_FILE[AFILE]) >> 1
         : D == NORTH_EAST    ? (b & ~MASK_FILE[HFILE]) << 9
         : D == NORTH_WEST    ? (b & ~MASK_FILE[AFILE]) << 7
         : D == SOUTH_EAST    ? (b & ~MASK_FILE[HFILE]) >> 7
         : D == SOUTH_WEST    ? (b & ~MASK_FILE[AFILE]) >> 9
                              : 0;
}

// Returns the actual rank from a given side's perspective (e.g. rank 1 is rank 8 from Black's perspective)
template<Color C>
constexpr Rank relative_rank(Rank r) {
  return C == WHITE ? r : Rank(RANK8 - r);
}

// Returns the actual direction from a given side's perspective (e.g. North is South from Black's perspective)
template<Color C>
constexpr Direction relative_dir(Direction d) {
  return Direction(C == WHITE ? d : -d);
}

// The type of the move
enum MoveFlags : int {
  QUIET              = 0b0000,
  DOUBLE_PUSH        = 0b0001,
  OO                 = 0b0010,
  OOO                = 0b0011,
  CAPTURE            = 0b1000,
  CAPTURES           = 0b1111,
  EN_PASSANT         = 0b1010,
  PROMOTIONS         = 0b0111,
  PROMOTION_CAPTURES = 0b1100,
  PR_KNIGHT          = 0b0100,
  PR_BISHOP          = 0b0101,
  PR_ROOK            = 0b0110,
  PR_QUEEN           = 0b0111,
  PC_KNIGHT          = 0b1100,
  PC_BISHOP          = 0b1101,
  PC_ROOK            = 0b1110,
  PC_QUEEN           = 0b1111,
};


class Move {
private:
  // The internal representation of the move
  uint16_t move;

public:
  // Defaults to a null move (a1a1)
  inline Move() : move(0) {}

  inline Move(uint16_t m) { move = m; }

  inline Move(Square from, Square to) : move(0) { move = (from << 6) | to; }

  inline Move(Square from, Square to, MoveFlags flags) : move(0) { move = (flags << 12) | (from << 6) | to; }

  Move(const std::string &move) {
    this->move = (create_square(File(move[0] - 'a'), Rank(move[1] - '1')) << 6) |
                 create_square(File(move[2] - 'a'), Rank(move[3] - '1'));
  }

  [[nodiscard]] inline Square    to() const { return Square(move & 0x3f); }
  [[nodiscard]] inline Square    from() const { return Square((move >> 6) & 0x3f); }
  [[nodiscard]] inline int       to_from() const { return move & 0xffff; }
  [[nodiscard]] inline MoveFlags flags() const { return MoveFlags((move >> 12) & 0xf); }

  [[nodiscard]] inline bool is_capture() const { return ((move >> 12) & CAPTURE) != 0; }

  // No user-declared copy assignment: it did nothing but the implicit memberwise copy already
  // does (move is a single uint16_t), and dropping it lets the implicit copy ctor/assignment
  // stay implicit instead of triggering the rule-of-three deprecation.
  bool operator==(Move a) const { return to_from() == a.to_from(); }
  bool operator!=(Move a) const { return to_from() != a.to_from(); }
};

// Adds, to the move pointer all moves of the form (from, s), where s is a square in the bitboard to
template<MoveFlags F = QUIET>
inline Move *make(Square from, Bitboard to, Move *list) {
  while (to)
    *list++ = Move(from, pop_lsb(&to), F);
  return list;
}

// Adds, to the move pointer all quiet promotion moves of the form (from, s), where s is a square in the bitboard to
template<>
inline Move *make<PROMOTIONS>(Square from, Bitboard to, Move *list) {
  Square p;
  while (to) {
    p       = pop_lsb(&to);
    *list++ = Move(from, p, PR_KNIGHT);
    *list++ = Move(from, p, PR_BISHOP);
    *list++ = Move(from, p, PR_ROOK);
    *list++ = Move(from, p, PR_QUEEN);
  }
  return list;
}

// Adds, to the move pointer all capture promotion moves of the form (from, s), where s is a square in the bitboard to
template<>
inline Move *make<PROMOTION_CAPTURES>(Square from, Bitboard to, Move *list) {
  Square p;
  while (to) {
    p       = pop_lsb(&to);
    *list++ = Move(from, p, PC_KNIGHT);
    *list++ = Move(from, p, PC_BISHOP);
    *list++ = Move(from, p, PC_ROOK);
    *list++ = Move(from, p, PC_QUEEN);
  }
  return list;
}

extern std::ostream &operator<<(std::ostream &os, const Move &m);

// The white king and kingside rook
constexpr Bitboard WHITE_OO_MASK = 0x90;
// The white king and queenside rook
constexpr Bitboard WHITE_OOO_MASK = 0x11;

// Squares between the white king and kingside rook
constexpr Bitboard WHITE_OO_BLOCKERS_AND_ATTACKERS_MASK = 0x60;
// Squares between the white king and queenside rook
constexpr Bitboard WHITE_OOO_BLOCKERS_AND_ATTACKERS_MASK = 0xe;

// The black king and kingside rook
constexpr Bitboard BLACK_OO_MASK = 0x9000000000000000;
// The black king and queenside rook
constexpr Bitboard BLACK_OOO_MASK = 0x1100000000000000;
// Squares between the black king and kingside rook
constexpr Bitboard BLACK_OO_BLOCKERS_AND_ATTACKERS_MASK = 0x6000000000000000;
// Squares between the black king and queenside rook
constexpr Bitboard BLACK_OOO_BLOCKERS_AND_ATTACKERS_MASK = 0xE00000000000000;

// The white king, white rooks, black king and black rooks
constexpr Bitboard ALL_CASTLING_MASK = 0x9100000000000091;

template<Color C>
constexpr Bitboard oo_mask() {
  return C == WHITE ? WHITE_OO_MASK : BLACK_OO_MASK;
}
template<Color C>
constexpr Bitboard ooo_mask() {
  return C == WHITE ? WHITE_OOO_MASK : BLACK_OOO_MASK;
}

template<Color C>
constexpr Bitboard oo_blockers_mask() {
  return C == WHITE ? WHITE_OO_BLOCKERS_AND_ATTACKERS_MASK : BLACK_OO_BLOCKERS_AND_ATTACKERS_MASK;
}

template<Color C>
constexpr Bitboard ooo_blockers_mask() {
  return C == WHITE ? WHITE_OOO_BLOCKERS_AND_ATTACKERS_MASK : BLACK_OOO_BLOCKERS_AND_ATTACKERS_MASK;
}

template<Color C>
constexpr Bitboard ignore_ooo_danger() {
  return C == WHITE ? 0x2 : 0x200000000000000;
}
