#pragma once

#include "types.h"

// KPK (king + pawn vs king) bitbase: an EXACT win/draw verdict for every K+P-vs-K position, built by
// retrograde analysis at startup (~24 KB, ported from Stockfish). The eval's drawish-endgame scaling
// uses it so the engine never mis-judges a drawn KPK as winnable (wrong-rook-pawn-with-king-in-front,
// the defender holding the corner/queening square, etc.) nor a won one as drawn. Ported convention:
// the pawn is WHITE and on files a-d; callers mirror colour/file before probing.
namespace kpk {

  // Build the bitbase. Call ONCE at startup, AFTER initialise_all_databases() (needs the attack tables).
  void init();

  // True iff the position is WON for the side with the pawn (White, pawn on files a-d). `stm` is the
  // side to move in that White-pawn frame.
  [[nodiscard]] bool probe(Square wksq, Square wpsq, Square bksq, Color stm);

} // namespace kpk
