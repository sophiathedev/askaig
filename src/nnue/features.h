#pragma once

#include "network.h"
#include "types.h"

// HalfKA + horizontal-mirror feature indexing. A feature for perspective P is
// (king_bucket, perspective-relative piece, perspective-relative square):
//  - squares are oriented to P's view (vertical flip `^56` for BLACK so each side "sees" the board
//    from its own back rank),
//  - then horizontally mirrored (`^7`) when P's king sits on files e-h, so the king is always on the
//    a-d half (this is the "_hm" / horizontal-mirror trick that halves the king-relative input),
//  - pieces are split into "us" (color == P) and "them" planes.
namespace nnue {

  // King bucket for a perspective-relative (already WHITE-oriented + mirrored) king square. v1: a single
  // bucket. Growing buckets later swaps this for a 32-entry lookup over the mirrored king squares — a
  // net-only change.
  [[gnu::const]] inline int king_bucket(Square /*rel_king*/) noexcept { return 0; }

  // Feature index of piece `pc` on square `sq`, from perspective `P` whose king is on `ksq`.
  [[gnu::const]] inline int feature_index(Color P, Piece pc, Square sq, Square ksq) noexcept {
    int rel_sq   = P == WHITE ? int(sq) : int(sq) ^ 56;
    int rel_king = P == WHITE ? int(ksq) : int(ksq) ^ 56;
    if ((rel_king & 7) >= 4) { // king on files e-h -> mirror onto the a-d half
      rel_sq ^= 7;
      rel_king ^= 7;
    }
    const int rel_color = color_of(pc) == P ? 0 : 1; // 0 = our piece, 1 = their piece
    const int plane     = rel_color * 6 + int(type_of(pc));
    return king_bucket(Square(rel_king)) * (PIECE_PLANES * 64) + plane * 64 + rel_sq;
  }

} // namespace nnue
