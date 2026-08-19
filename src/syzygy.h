#pragma once

#include <optional>
#include <string>
#include "position.h"
#include "types.h"

namespace syzygy {

  enum class Wdl : int { LOSS = -1, DRAW = 0, WIN = 1 };

  struct RootResult {
    Move move{};
    Wdl  wdl = Wdl::DRAW;
  };

  extern int  probe_depth;
  extern int  probe_limit;
  extern int  cardinality;
  extern bool use_50_move_rule;

  [[gnu::pure, gnu::always_inline]] inline bool has_castling_rights(const Position &pos) {
    const Bitboard entry = pos.castle_entry();
    return !(entry & WHITE_OO_MASK) || !(entry & WHITE_OOO_MASK) || !(entry & BLACK_OO_MASK) ||
           !(entry & BLACK_OOO_MASK);
  }

  [[gnu::pure, gnu::always_inline]] inline bool can_probe(const Position &pos, int depth) {
    if (cardinality == 0 || depth < probe_depth || (use_50_move_rule && pos.fifty() != 0))
      return false;

    const Bitboard occupied = pos.all_pieces<WHITE>() | pos.all_pieces<BLACK>();
    return pop_count(occupied) <= cardinality && !has_castling_rights(pos);
  }

  bool set_path(const std::string &path);
  void set_probe_depth(int depth);
  void set_probe_limit(int limit);
  void set_50_move_rule(bool enabled);

  [[nodiscard]] const std::string        &path();
  [[nodiscard]] int                       max_cardinality();
  [[nodiscard]] std::optional<Wdl>        probe_wdl(const Position &pos);
  [[nodiscard]] std::optional<RootResult> probe_root(Position &pos);

  void shutdown();

} // namespace syzygy
