#pragma once

#include "position.h"
#include "types.h"

namespace search {

  constexpr int PIECE_VAL[8] = {100, 320, 330, 500, 900, 0, 0, 0};

  [[gnu::pure, gnu::hot, nodiscard]] bool see_ge(const Position &pos, Move m, int threshold);

} // namespace search
