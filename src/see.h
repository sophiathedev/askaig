#pragma once

#include "position.h"
#include "types.h"

namespace search {

  // Piece values for SEE / MVV-LVA / pruning margins. Indexed by PieceType; the two extra
  // slots make PIECE_VAL[type_of(NO_PIECE)] (= index 6) a safe zero for empty squares.
  constexpr int PIECE_VAL[8] = {100, 320, 330, 500, 900, 0, 0, 0};

  // Static exchange evaluation: true iff the capture sequence started by `m` on its target
  // square wins at least `threshold` centipawns (standard swap algorithm, both colors'
  // attackers including kings, with x-ray reveals).
  [[gnu::pure, gnu::hot, nodiscard]] bool see_ge(const Position &pos, Move m, int threshold);

} // namespace search
