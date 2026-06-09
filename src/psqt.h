#pragma once

#include "types.h"

// Piece values and piece-square tables, plus the per-piece score used by both the static
// evaluation and Position's incremental material+PST accumulator (`psqt_score`).
//
// PST are written from White's view in the usual textbook layout (index 0 = a8 ... 63 = h1), so a
// White piece on Square sq reads TABLE[pt][sq ^ 56] and a Black piece reads TABLE[pt][sq] (the
// table mirrored vertically). Values are Tomasz Michniewski's "simplified evaluation" tables.
namespace psqt {

  constexpr int VALUE[NPIECE_TYPES] = {100, 320, 330, 500, 900, 0};

  constexpr int TABLE[NPIECE_TYPES][NSQUARES] = {
          // PAWN
          {0,  0,   0,  0, 0,  0,  0,  0,   50,  50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30,  20,
           10, 10,  5,  5, 10, 25, 25, 10,  5,   5,  0,  0,  0,  20, 20, 0,  0,  0,  5,  -5, -10, 0,
           0,  -10, -5, 5, 5,  10, 10, -20, -20, 10, 10, 5,  0,  0,  0,  0,  0,  0,  0,  0},
          // KNIGHT
          {-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   0,   0,   0,   -20, -40, -30, 0,   10,  15, 15, 10,
           0,   -30, -30, 5,   15,  20,  20,  15,  5,   -30, -30, 0,   15,  20,  20,  15,  0,   -30, -30, 5,  10, 15,
           15,  10,  5,   -30, -40, -20, 0,   5,   5,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50},
          // BISHOP
          {-20, -10, -10, -10, -10, -10, -10, -20, -10, 0,   0,   0,   0,   0,   0,   -10, -10, 0,   5,   10, 10, 5,
           0,   -10, -10, 5,   5,   10,  10,  5,   5,   -10, -10, 0,   10,  10,  10,  10,  0,   -10, -10, 10, 10, 10,
           10,  10,  10,  -10, -10, 5,   0,   0,   0,   0,   5,   -10, -20, -10, -10, -10, -10, -10, -10, -20},
          // ROOK
          {0,  0, 0, 0, 0, 0, 0, 0,  5,  10, 10, 10, 10, 10, 10, 5,  -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
           -5, 0, 0, 0, 0, 0, 0, -5, -5, 0,  0,  0,  0,  0,  0,  -5, -5, 0, 0, 0, 0, 0, 0, -5, 0,  0, 0, 5, 5, 0, 0, 0},
          // QUEEN
          {-20, -10, -10, -5,  -5,  -10, -10, -20, -10, 0,  0, 0,   0,   0,   0,   -10, -10, 0,   5,   5,  5, 5,
           0,   -10, -5,  0,   5,   5,   5,   5,   0,   -5, 0, 0,   5,   5,   5,   5,   0,   -5,  -10, 5,  5, 5,
           5,   5,   0,   -10, -10, 0,   5,   0,   0,   0,  0, -10, -20, -10, -10, -5,  -5,  -10, -10, -20},
          // KING (middlegame)
          {-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40,
           -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20,
           -20, -20, -20, -10, 20,  20,  0,   0,   0,   0,   20,  20,  20,  30,  10,  0,   0,   10,  30,  20},
  };

  // Endgame king table: unlike the middlegame table (TABLE[KING], which makes the king hide on the
  // back rank), in the endgame the king should march to the centre, so this rewards central squares.
  constexpr int KING_EG[NSQUARES] = {
          -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0,   0,   -10, -20, -30, -30, -10, 20,  30,  30, 20,
          -10, -30, -30, -10, 30,  40,  40,  30,  -10, -30, -30, -10, 30,  40,  40,  30,  -10, -30, -30, -10, 20, 30,
          30,  20,  -10, -30, -30, -30, 0,   0,   0,   0,   -30, -30, -50, -30, -30, -30, -30, -30, -30, -50,
  };

  // Game phase: PHASE_MAX with all pieces on the board, decreasing as they are traded. Used to
  // interpolate ("taper") the score between the middlegame and endgame piece-square tables.
  constexpr int PHASE_MAX = 24; // 4*knight + 4*bishop + 4*2*rook + 2*4*queen

  // The square-table value of piece type `pt` (a8-first index `idx`) in the middlegame / endgame.
  // Only the king differs between phases; the other tables are shared.
  [[gnu::always_inline]] constexpr int table_mg(int pt, int idx) { return TABLE[pt][idx]; }
  [[gnu::always_inline]] constexpr int table_eg(int pt, int idx) { return pt == KING ? KING_EG[idx] : TABLE[pt][idx]; }

  // White-perspective contribution (material + PST) of piece `pc` on square `s`, in the middlegame
  // / endgame. `pc` must be a real piece (not NO_PIECE). always_inline: called from the make/unmake
  // primitives (put_piece/remove_piece/move_piece), the hottest path in the engine.
  [[gnu::always_inline]] constexpr int score_mg(Piece pc, Square s) {
    const int pt  = type_of(pc);
    const int idx = color_of(pc) == WHITE ? (s ^ 56) : static_cast<int>(s);
    const int v   = VALUE[pt] + table_mg(pt, idx);
    return color_of(pc) == WHITE ? v : -v;
  }
  [[gnu::always_inline]] constexpr int score_eg(Piece pc, Square s) {
    const int pt  = type_of(pc);
    const int idx = color_of(pc) == WHITE ? (s ^ 56) : static_cast<int>(s);
    const int v   = VALUE[pt] + table_eg(pt, idx);
    return color_of(pc) == WHITE ? v : -v;
  }

} // namespace psqt
