#include "types.h"
#include <iostream>

// Lookup tables of square names in algebraic chess notation
const char *SQSTR[65] = {"a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2",
                         "f2", "g2", "h2", "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4",
                         "c4", "d4", "e4", "f4", "g4", "h4", "a5", "b5", "c5", "d5", "e5", "f5", "g5",
                         "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6", "a7", "b7", "c7", "d7",
                         "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8", "None"};

// All masks have been generated from a Java program

// Precomputed file masks
// (MASK_FILE / MASK_RANK / MASK_DIAGONAL / MASK_ANTI_DIAGONAL / SQUARE_BB moved to types.h
// as constexpr so the compiler can fold them at the call-sites.)

// Prints the bitboard, little-endian format
void print_bitboard(Bitboard b) {
  for (int i = 56; i >= 0; i -= 8) {
    for (int j = 0; j < 8; j++)
      std::cout << static_cast<char>(((b >> (i + j)) & 1) + '0') << " ";
    std::cout << "\n";
  }
  std::cout << "\n";
}

// Returns the representation of the move type in algebraic chess notation. (capture) is used for debugging
const char *MOVE_TYPESTR[16] = {"",           "", " O-O",  " O-O-O", "N", "B", "R", "Q",
                                " (capture)", "", " e.p.", "",       "N", "B", "R", "Q"};

// Prints the move
// For example: e5d6 (capture); a7a8R; O-O
std::ostream &operator<<(std::ostream &os, const Move &m) {
  os << SQSTR[m.from()] << SQSTR[m.to()] << MOVE_TYPESTR[m.flags()];
  return os;
}
