#include "see.h"
#include "tables.h"

namespace {

  using search::PIECE_VAL;

  // All pieces of both colors attacking `s` under occupancy `occ` (kings included — the swap
  // loop needs them; Position::attackers_from omits kings and is single-color).
  Bitboard attackers_to(const Position &pos, Square s, Bitboard occ) {
    return (pawn_attacks<WHITE>(s) & pos.bitboard_of(BLACK_PAWN)) |
           (pawn_attacks<BLACK>(s) & pos.bitboard_of(WHITE_PAWN)) |
           (attacks<KNIGHT>(s, occ) & (pos.bitboard_of(WHITE_KNIGHT) | pos.bitboard_of(BLACK_KNIGHT))) |
           (attacks<BISHOP>(s, occ) &
            (pos.diagonal_sliders<WHITE>() | pos.diagonal_sliders<BLACK>())) |
           (attacks<ROOK>(s, occ) & (pos.orthogonal_sliders<WHITE>() | pos.orthogonal_sliders<BLACK>())) |
           (KING_ATTACKS[s] & (pos.bitboard_of(WHITE_KING) | pos.bitboard_of(BLACK_KING)));
  }

  template<Color C>
  Bitboard color_bb(const Position &pos) {
    return pos.all_pieces<C>();
  }

} // namespace

bool search::see_ge(const Position &pos, Move m, int threshold) {
  const MoveFlags f = m.flags();
  if (f == OO || f == OOO) // castling never wins or loses material
    return 0 >= threshold;

  const Square from = m.from(), to = m.to();

  // Balance after we make the capture (quiet moves capture nothing; EP captures a pawn that is
  // not on `to`). Promotion gains are ignored — conservative, and keeps the loop simple.
  int swap = (f == EN_PASSANT ? PIECE_VAL[PAWN] : PIECE_VAL[type_of(pos.at(to))]) - threshold;
  if (swap < 0) // even capturing for free is not enough
    return false;

  // ... and after the opponent recaptures our moving piece for free.
  swap = PIECE_VAL[type_of(pos.at(from))] - swap;
  if (swap <= 0) // we still meet the threshold even if recaptured
    return true;

  Bitboard occ = (pos.all_pieces<WHITE>() | pos.all_pieces<BLACK>()) ^ SQUARE_BB[from];
  occ |= SQUARE_BB[to];
  if (f == EN_PASSANT)
    occ ^= SQUARE_BB[pos.turn() == WHITE ? Square(to + SOUTH) : Square(to + NORTH)];

  const Bitboard diag = pos.diagonal_sliders<WHITE>() | pos.diagonal_sliders<BLACK>();
  const Bitboard orth = pos.orthogonal_sliders<WHITE>() | pos.orthogonal_sliders<BLACK>();

  Bitboard attackers = attackers_to(pos, to, occ);
  Color    stm       = ~pos.turn();
  bool     result    = true;

  while (true) {
    attackers &= occ;
    const Bitboard own = attackers & (stm == WHITE ? color_bb<WHITE>(pos) : color_bb<BLACK>(pos));
    if (!own)
      break; // side to move has no attacker left -> current `result` stands

    // Least valuable attacker first. The king is special: capturing with it is only legal if
    // the opponent has no attacker left — otherwise the flip is undone.
    Bitboard bb = 0;
    int      pt = PAWN;
    while (pt < KING && !(bb = own & pos.bitboard_of(make_piece(stm, PieceType(pt)))))
      ++pt;

    result = !result;
    if (pt == KING) {
      const Bitboard opp = stm == WHITE ? color_bb<BLACK>(pos) : color_bb<WHITE>(pos);
      return (attackers & opp) ? !result : result;
    }

    swap = PIECE_VAL[pt] - swap;
    if (swap < int(result)) // not enough even winning the exchange from here on
      break;

    occ ^= SQUARE_BB[bsf(bb)]; // remove the attacker; reveal x-rays through its square
    if (pt == PAWN || pt == BISHOP || pt == QUEEN)
      attackers |= attacks<BISHOP>(to, occ) & diag;
    if (pt == ROOK || pt == QUEEN)
      attackers |= attacks<ROOK>(to, occ) & orth;
    stm = ~stm;
  }
  return result;
}
