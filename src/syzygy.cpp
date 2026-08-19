#include "syzygy.h"
#include <algorithm>
#include <tbprobe.h>

namespace {

  std::string g_path = "<empty>";

  unsigned ep_square(const Position &pos) {
    const Square ep = pos.history[pos.ply()].epsq;
    return ep == NO_SQUARE ? 0u : static_cast<unsigned>(ep);
  }

  unsigned raw_probe_wdl(const Position &pos) {
    const Bitboard white = pos.all_pieces<WHITE>();
    const Bitboard black = pos.all_pieces<BLACK>();
    return tb_probe_wdl(white, black, pos.bitboard_of(WHITE_KING) | pos.bitboard_of(BLACK_KING),
                        pos.bitboard_of(WHITE_QUEEN) | pos.bitboard_of(BLACK_QUEEN),
                        pos.bitboard_of(WHITE_ROOK) | pos.bitboard_of(BLACK_ROOK),
                        pos.bitboard_of(WHITE_BISHOP) | pos.bitboard_of(BLACK_BISHOP),
                        pos.bitboard_of(WHITE_KNIGHT) | pos.bitboard_of(BLACK_KNIGHT),
                        pos.bitboard_of(WHITE_PAWN) | pos.bitboard_of(BLACK_PAWN),
                        syzygy::use_50_move_rule ? static_cast<unsigned>(pos.fifty()) : 0u,
                        syzygy::has_castling_rights(pos) ? 1u : 0u, ep_square(pos), pos.turn() == WHITE);
  }

  std::optional<syzygy::Wdl> decode_wdl(unsigned value) {
    if (value == TB_RESULT_FAILED || value > TB_WIN)
      return std::nullopt;
    if (value == TB_DRAW)
      return syzygy::Wdl::DRAW;
    if (value == TB_BLESSED_LOSS)
      return syzygy::use_50_move_rule ? syzygy::Wdl::DRAW : syzygy::Wdl::LOSS;
    if (value == TB_CURSED_WIN)
      return syzygy::use_50_move_rule ? syzygy::Wdl::DRAW : syzygy::Wdl::WIN;
    return value == TB_LOSS ? syzygy::Wdl::LOSS : syzygy::Wdl::WIN;
  }

  bool promotion_matches(Move move, unsigned promotion) {
    const MoveFlags flag = move.flags();
    if (promotion == TB_PROMOTES_NONE)
      return !(flag >= PR_KNIGHT && flag <= PR_QUEEN) && !(flag >= PC_KNIGHT && flag <= PC_QUEEN);

    const int piece = promotion == TB_PROMOTES_KNIGHT   ? KNIGHT
                      : promotion == TB_PROMOTES_BISHOP ? BISHOP
                      : promotion == TB_PROMOTES_ROOK   ? ROOK
                                                        : QUEEN;
    if (flag >= PR_KNIGHT && flag <= PR_QUEEN)
      return int(flag - PR_KNIGHT) == piece - KNIGHT;
    if (flag >= PC_KNIGHT && flag <= PC_QUEEN)
      return int(flag - PC_KNIGHT) == piece - KNIGHT;
    return false;
  }

  Move find_root_move(Position &pos, unsigned result) {
    const Square   from      = Square(TB_GET_FROM(result));
    const Square   to        = Square(TB_GET_TO(result));
    const unsigned promotion = TB_GET_PROMOTES(result);
    Move           moves[218];
    Move          *end = pos.turn() == WHITE ? pos.generate_legals<WHITE>(moves) : pos.generate_legals<BLACK>(moves);
    for (Move *it = moves; it != end; ++it)
      if (it->from() == from && it->to() == to && promotion_matches(*it, promotion))
        return *it;
    return Move();
  }

} // namespace

int  syzygy::probe_depth      = 1;
int  syzygy::probe_limit      = 7;
int  syzygy::cardinality      = 0;
bool syzygy::use_50_move_rule = true;

bool syzygy::set_path(const std::string &path_value) {
  const std::string next = path_value.empty() ? "<empty>" : path_value;
  if (!tb_init(next.c_str()))
    return false;
  g_path      = next;
  cardinality = std::min(probe_limit, static_cast<int>(TB_LARGEST));
  return true;
}

void syzygy::set_probe_depth(int depth) { probe_depth = std::clamp(depth, 1, 100); }

void syzygy::set_probe_limit(int limit) {
  probe_limit = std::clamp(limit, 0, 7);
  cardinality = std::min(probe_limit, static_cast<int>(TB_LARGEST));
}

void syzygy::set_50_move_rule(bool enabled) { use_50_move_rule = enabled; }

const std::string &syzygy::path() { return g_path; }

int syzygy::max_cardinality() { return static_cast<int>(TB_LARGEST); }

std::optional<syzygy::Wdl> syzygy::probe_wdl(const Position &pos) { return decode_wdl(raw_probe_wdl(pos)); }

std::optional<syzygy::RootResult> syzygy::probe_root(Position &pos) {
  if (cardinality == 0 || has_castling_rights(pos))
    return std::nullopt;

  const Bitboard white    = pos.all_pieces<WHITE>();
  const Bitboard black    = pos.all_pieces<BLACK>();
  const Bitboard occupied = white | black;
  if (pop_count(occupied) > cardinality)
    return std::nullopt;

  const unsigned result =
          tb_probe_root(white, black, pos.bitboard_of(WHITE_KING) | pos.bitboard_of(BLACK_KING),
                        pos.bitboard_of(WHITE_QUEEN) | pos.bitboard_of(BLACK_QUEEN),
                        pos.bitboard_of(WHITE_ROOK) | pos.bitboard_of(BLACK_ROOK),
                        pos.bitboard_of(WHITE_BISHOP) | pos.bitboard_of(BLACK_BISHOP),
                        pos.bitboard_of(WHITE_KNIGHT) | pos.bitboard_of(BLACK_KNIGHT),
                        pos.bitboard_of(WHITE_PAWN) | pos.bitboard_of(BLACK_PAWN), use_50_move_rule ? pos.fifty() : 0,
                        0, ep_square(pos), pos.turn() == WHITE, nullptr);
  if (result == TB_RESULT_FAILED)
    return std::nullopt;

  const Move move = find_root_move(pos, result);
  const auto wdl  = decode_wdl(TB_GET_WDL(result));
  if (move.to_from() == 0 || !wdl)
    return std::nullopt;
  return RootResult{move, *wdl};
}

void syzygy::shutdown() {
  tb_free();
  g_path      = "<empty>";
  cardinality = 0;
}
