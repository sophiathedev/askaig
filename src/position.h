#pragma once

#include <ostream>
#include <string>
#include <utility>
#include "tables.h"
#include "types.h"

class PRNG {
  uint64_t s;

  uint64_t rand64() {
    s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
    return s * 2685821657736338717LL;
  }

public:
  PRNG(uint64_t seed) : s(seed) {}

  template<typename T>
  T rand() {
    return T(rand64());
  }

  template<typename T>
  T sparse_rand() {
    return T(rand64() & rand64() & rand64());
  }
};


namespace zobrist {
  extern uint64_t zobrist_table[NPIECES][NSQUARES];
  extern uint64_t zobrist_side; // black-to-move key
  extern void initialise_zobrist_keys();
} // namespace zobrist

struct UndoInfo {
  Bitboard entry;

  Piece captured;

  Square epsq;

  uint64_t hash;
  int      fifty;

  constexpr UndoInfo() : entry(0), captured(NO_PIECE), epsq(NO_SQUARE), hash(0), fifty(0) {}

  UndoInfo(const UndoInfo &prev) : entry(prev.entry), captured(NO_PIECE), epsq(NO_SQUARE), hash(0), fifty(prev.fifty) {}

  UndoInfo &operator=(const UndoInfo &) = default;
};

class Position {
private:
  Bitboard piece_bb[NPIECES];

  Piece board[NSQUARES];

  Color side_to_play;

  int game_ply;

  uint64_t hash;

public:
  // game and search plies share this stack
  static constexpr int MAX_HISTORY = 1024;

  UndoInfo history[MAX_HISTORY];

  Bitboard checkers;

  Bitboard pinned;


  Position() : piece_bb{0}, board{}, side_to_play(WHITE), game_ply(0), hash(0), checkers(0), pinned(0) {
    for (auto &i: board)
      i = NO_PIECE;
    history[0] = UndoInfo();
  }

  inline void put_piece(Piece pc, Square s) {
    board[s] = pc;
    piece_bb[pc] |= sq_bb(s);
    hash ^= zobrist::zobrist_table[pc][s];
  }

  inline void remove_piece(Square s) {
    const Piece pc = board[s];
    hash ^= zobrist::zobrist_table[pc][s];
    piece_bb[pc] &= ~sq_bb(s);
    board[s] = NO_PIECE;
  }

  inline void put_piece_nohash(Piece pc, Square s) {
    board[s] = pc;
    piece_bb[pc] |= sq_bb(s);
  }

  inline void remove_piece_nohash(Square s) {
    piece_bb[board[s]] &= ~sq_bb(s);
    board[s] = NO_PIECE;
  }

  inline void move_piece_quiet_nohash(Square from, Square to) {
    piece_bb[board[from]] ^= (sq_bb(from) | sq_bb(to));
    board[to]   = board[from];
    board[from] = NO_PIECE;
  }

  [[gnu::always_inline]] inline void move_piece(Square from, Square to) {
    const Piece moving = board[from];
    const Piece victim = board[to];
    hash ^= zobrist::zobrist_table[moving][from] ^ zobrist::zobrist_table[moving][to] ^
            zobrist::zobrist_table[victim][to];
    const Bitboard mask = sq_bb(from) | sq_bb(to);
    piece_bb[moving] ^= mask;
    piece_bb[victim] &= ~mask;
    board[to]   = moving;
    board[from] = NO_PIECE;
  }

  [[gnu::always_inline]] inline void move_piece_quiet(Square from, Square to) {
    const Piece moving = board[from];
    hash ^= zobrist::zobrist_table[moving][from] ^ zobrist::zobrist_table[moving][to];
    piece_bb[moving] ^= (sq_bb(from) | sq_bb(to));
    board[to]   = moving;
    board[from] = NO_PIECE;
  }

  friend std::ostream &operator<<(std::ostream &os, const Position &p);
  // malformed input may leave p partial
  static bool               set(const std::string &fen, Position &p);
  [[nodiscard]] std::string fen() const;

  Position(const Position &) = default;
  // assignment would corrupt UndoInfo history
  Position   &operator=(const Position &) = delete;
  inline bool operator==(const Position &other) const { return hash == other.hash; }

  [[nodiscard]] inline Bitboard bitboard_of(Piece pc) const { return piece_bb[pc]; }
  [[nodiscard]] inline Bitboard bitboard_of(Color c, PieceType pt) const { return piece_bb[make_piece(c, pt)]; }
  [[nodiscard]] inline Piece    at(Square sq) const { return board[sq]; }
  [[nodiscard]] inline Color    turn() const { return side_to_play; }
  [[nodiscard]] inline int      ply() const { return game_ply; }
  [[nodiscard]] inline uint64_t get_hash() const { return hash; }
  [[nodiscard]] inline int      fifty() const { return history[game_ply].fifty; }
  [[nodiscard]] inline Bitboard castle_entry() const { return history[game_ply].entry; }

  [[gnu::hot, nodiscard]] inline bool is_draw() const {
    const int f = history[game_ply].fifty;
    if (f >= 100) [[unlikely]]
      return true;
    if (!(piece_bb[WHITE_PAWN] | piece_bb[BLACK_PAWN])) [[unlikely]] {
      const Bitboard majors =
              piece_bb[WHITE_ROOK] | piece_bb[BLACK_ROOK] | piece_bb[WHITE_QUEEN] | piece_bb[BLACK_QUEEN];
      const Bitboard minors =
              piece_bb[WHITE_KNIGHT] | piece_bb[BLACK_KNIGHT] | piece_bb[WHITE_BISHOP] | piece_bb[BLACK_BISHOP];
      if (!majors && pop_count(minors) <= 1)
        return true;
    }
    const int end = game_ply - f;
    for (int i = game_ply - 4; i >= end && i >= 0; i -= 2)
      if (history[i].hash == hash)
        return true;
    return false;
  }

  template<Color C>
  [[nodiscard]] inline Bitboard diagonal_sliders() const;
  template<Color C>
  [[nodiscard]] inline Bitboard orthogonal_sliders() const;
  template<Color C>
  [[nodiscard]] inline Bitboard all_pieces() const;
  template<Color C>
  [[nodiscard]] inline Bitboard attackers_from(Square s, Bitboard occ) const;

  template<Color C>
  [[nodiscard]] inline bool in_check() const {
    return attackers_from<~C>(bsf(bitboard_of(C, KING)), all_pieces<WHITE>() | all_pieces<BLACK>());
  }

  template<Color C>
  void play(Move m);
  template<Color C>
  void undo(Move m);

  inline void play_null() {
    side_to_play = ~side_to_play;
    hash ^= zobrist::zobrist_side;
    ++game_ply;
    history[game_ply]       = UndoInfo(history[game_ply - 1]);
    history[game_ply].hash  = hash;
    history[game_ply].fifty = history[game_ply - 1].fifty + 1;
  }
  inline void undo_null() {
    side_to_play = ~side_to_play;
    hash ^= zobrist::zobrist_side;
    --game_ply;
  }

  // captures-only is invalid while in check
  template<Color Us, bool CAPTURES_ONLY = false>
  Move *generate_legals(Move *list);
};

template<Color C>
inline Bitboard Position::diagonal_sliders() const {
  return C == WHITE ? piece_bb[WHITE_BISHOP] | piece_bb[WHITE_QUEEN] : piece_bb[BLACK_BISHOP] | piece_bb[BLACK_QUEEN];
}

template<Color C>
inline Bitboard Position::orthogonal_sliders() const {
  return C == WHITE ? piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN] : piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN];
}

template<Color C>
inline Bitboard Position::all_pieces() const {
  return or_reduce6(&piece_bb[C == WHITE ? WHITE_PAWN : BLACK_PAWN]);
}

template<Color C>
inline Bitboard Position::attackers_from(Square s, Bitboard occ) const {
  if (C == WHITE) {
    const Bitboard att[4] = {pawn_attacks<BLACK>(s), attacks<KNIGHT>(s, occ), attacks<BISHOP>(s, occ),
                             attacks<ROOK>(s, occ)};
    const Bitboard msk[4] = {piece_bb[WHITE_PAWN], piece_bb[WHITE_KNIGHT],
                             piece_bb[WHITE_BISHOP] | piece_bb[WHITE_QUEEN],
                             piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN]};
    return and_or_reduce4(att, msk);
  } else {
    const Bitboard att[4] = {pawn_attacks<WHITE>(s), attacks<KNIGHT>(s, occ), attacks<BISHOP>(s, occ),
                             attacks<ROOK>(s, occ)};
    const Bitboard msk[4] = {piece_bb[BLACK_PAWN], piece_bb[BLACK_KNIGHT],
                             piece_bb[BLACK_BISHOP] | piece_bb[BLACK_QUEEN],
                             piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN]};
    return and_or_reduce4(att, msk);
  }
}


template<Color C>
[[gnu::hot]] void Position::play(const Move m) {
  side_to_play = ~side_to_play;
  hash ^= zobrist::zobrist_side;
  ++game_ply;

  MoveFlags type = m.flags();
  UndoInfo &st   = history[game_ply];
  st.entry       = history[game_ply - 1].entry | sq_bb(m.to()) | sq_bb(m.from());
  st.captured    = NO_PIECE;
  st.epsq        = NO_SQUARE;

  if (type == QUIET) [[likely]] {
    move_piece_quiet(m.from(), m.to());
  } else switch (type) {
    case DOUBLE_PUSH:
      move_piece_quiet(m.from(), m.to());

      history[game_ply].epsq = m.from() + relative_dir<C>(NORTH);
      break;
    case OO:
      if (C == WHITE) {
        move_piece_quiet(e1, g1);
        move_piece_quiet(h1, f1);
      } else {
        move_piece_quiet(e8, g8);
        move_piece_quiet(h8, f8);
      }
      break;
    case OOO:
      if (C == WHITE) {
        move_piece_quiet(e1, c1);
        move_piece_quiet(a1, d1);
      } else {
        move_piece_quiet(e8, c8);
        move_piece_quiet(a8, d8);
      }
      break;
    case EN_PASSANT:
      move_piece_quiet(m.from(), m.to());
      remove_piece(m.to() + relative_dir<C>(SOUTH));
      break;
    case PR_KNIGHT:
      remove_piece(m.from());
      put_piece(make_piece(C, KNIGHT), m.to());
      break;
    case PR_BISHOP:
      remove_piece(m.from());
      put_piece(make_piece(C, BISHOP), m.to());
      break;
    case PR_ROOK:
      remove_piece(m.from());
      put_piece(make_piece(C, ROOK), m.to());
      break;
    case PR_QUEEN:
      remove_piece(m.from());
      put_piece(make_piece(C, QUEEN), m.to());
      break;
    case PC_KNIGHT:
      remove_piece(m.from());
      history[game_ply].captured = board[m.to()];
      remove_piece(m.to());

      put_piece(make_piece(C, KNIGHT), m.to());
      break;
    case PC_BISHOP:
      remove_piece(m.from());
      history[game_ply].captured = board[m.to()];
      remove_piece(m.to());

      put_piece(make_piece(C, BISHOP), m.to());
      break;
    case PC_ROOK:
      remove_piece(m.from());
      history[game_ply].captured = board[m.to()];
      remove_piece(m.to());

      put_piece(make_piece(C, ROOK), m.to());
      break;
    case PC_QUEEN:
      remove_piece(m.from());
      history[game_ply].captured = board[m.to()];
      remove_piece(m.to());

      put_piece(make_piece(C, QUEEN), m.to());
      break;
    case CAPTURE:
      history[game_ply].captured = board[m.to()];
      move_piece(m.from(), m.to());

      break;
    default:
      __builtin_unreachable();
  }

  history[game_ply].hash  = hash;
  history[game_ply].fifty = (type != QUIET || type_of(board[m.to()]) == PAWN) ? 0 : history[game_ply - 1].fifty + 1;
}

template<Color C>
[[gnu::hot]] void Position::undo(const Move m) {
  MoveFlags type = m.flags();
  if (type <= DOUBLE_PUSH) [[likely]] {
    move_piece_quiet_nohash(m.to(), m.from());
  } else switch (type) {
    case OO:
      if (C == WHITE) {
        move_piece_quiet_nohash(g1, e1);
        move_piece_quiet_nohash(f1, h1);
      } else {
        move_piece_quiet_nohash(g8, e8);
        move_piece_quiet_nohash(f8, h8);
      }
      break;
    case OOO:
      if (C == WHITE) {
        move_piece_quiet_nohash(c1, e1);
        move_piece_quiet_nohash(d1, a1);
      } else {
        move_piece_quiet_nohash(c8, e8);
        move_piece_quiet_nohash(d8, a8);
      }
      break;
    case EN_PASSANT:
      move_piece_quiet_nohash(m.to(), m.from());
      put_piece_nohash(make_piece(~C, PAWN), m.to() + relative_dir<C>(SOUTH));
      break;
    case PR_KNIGHT:
    case PR_BISHOP:
    case PR_ROOK:
    case PR_QUEEN:
      remove_piece_nohash(m.to());
      put_piece_nohash(make_piece(C, PAWN), m.from());
      break;
    case PC_KNIGHT:
    case PC_BISHOP:
    case PC_ROOK:
    case PC_QUEEN:
      remove_piece_nohash(m.to());
      put_piece_nohash(make_piece(C, PAWN), m.from());
      put_piece_nohash(history[game_ply].captured, m.to());
      break;
    case CAPTURE:
      move_piece_quiet_nohash(m.to(), m.from());
      put_piece_nohash(history[game_ply].captured, m.to());
      break;
    default:
      __builtin_unreachable();
  }

  side_to_play = ~side_to_play;
  --game_ply;
  hash = history[game_ply].hash;
}


template<Color Us, bool CAPTURES_ONLY>
[[gnu::hot]] Move *Position::generate_legals(Move *list) {
  constexpr Color Them = ~Us;

  const Bitboard us_bb   = all_pieces<Us>();
  const Bitboard them_bb = all_pieces<Them>();
  const Bitboard all     = us_bb | them_bb;

  const Square our_king   = bsf(bitboard_of(Us, KING));
  const Square their_king = bsf(bitboard_of(Them, KING));

  const Bitboard our_diag_sliders   = diagonal_sliders<Us>();
  const Bitboard their_diag_sliders = diagonal_sliders<Them>();
  const Bitboard our_orth_sliders   = orthogonal_sliders<Us>();
  const Bitboard their_orth_sliders = orthogonal_sliders<Them>();

  Bitboard b1, b2, b3;

  Bitboard danger = 0;

  danger |= pawn_attacks<Them>(bitboard_of(Them, PAWN)) | attacks<KING>(their_king, all) |
            knight_attacks(bitboard_of(Them, KNIGHT));

  const Bitboard all_no_king = all ^ sq_bb(our_king);

  b1 = their_diag_sliders;
  while (b1)
    danger |= attacks<BISHOP>(pop_lsb(&b1), all_no_king);

  b1 = their_orth_sliders;
  while (b1)
    danger |= attacks<ROOK>(pop_lsb(&b1), all_no_king);

  b1 = attacks<KING>(our_king, all) & ~(us_bb | danger);
  if constexpr (!CAPTURES_ONLY)
    list = make<QUIET>(our_king, b1 & ~them_bb, list);
  list = make<CAPTURE>(our_king, b1 & them_bb, list);

  Bitboard capture_mask;

  Bitboard quiet_mask;

  Square s;

  Bitboard candidates = (attacks<ROOK>(our_king, them_bb) & their_orth_sliders) |
                        (attacks<BISHOP>(our_king, them_bb) & their_diag_sliders);

  pinned = 0;
  if (!(danger & sq_bb(our_king))) [[likely]] {
    checkers = 0;
    while (candidates) {
      s  = pop_lsb(&candidates);
      b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb;
      if (!(b1 & (b1 - 1)))
        pinned ^= b1;
    }
  } else {
    checkers = (attacks<KNIGHT>(our_king, all) & bitboard_of(Them, KNIGHT)) |
               (pawn_attacks<Us>(our_king) & bitboard_of(Them, PAWN));

    while (candidates) {
      s  = pop_lsb(&candidates);
      b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb;

      if (b1 == 0)
        checkers ^= sq_bb(s);
      else if (!(b1 & (b1 - 1)))
        pinned ^= b1;
    }
  }

  const Bitboard not_pinned = ~pinned;

  if (checkers) {
    if (checkers & (checkers - 1))
      return list;
    {

      Square checker_square = bsf(checkers);

      switch (board[checker_square]) {
        case make_piece(Them, PAWN):
          if (checkers == shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history[game_ply].epsq])) {
            b1 = pawn_attacks<Them>(history[game_ply].epsq) & bitboard_of(Us, PAWN) & not_pinned;
            while (b1)
              *list++ = Move(pop_lsb(&b1), history[game_ply].epsq, EN_PASSANT);
          }
        case make_piece(Them, KNIGHT):
          b1 = attackers_from<Us>(checker_square, all) & not_pinned;
          while (b1) {
            s = pop_lsb(&b1);
            if (board[s] == make_piece(Us, PAWN) && rank_of(checker_square) == relative_rank<Us>(RANK8)) {
              *list++ = Move(s, checker_square, PC_KNIGHT);
              *list++ = Move(s, checker_square, PC_BISHOP);
              *list++ = Move(s, checker_square, PC_ROOK);
              *list++ = Move(s, checker_square, PC_QUEEN);
            } else
              *list++ = Move(s, checker_square, CAPTURE);
          }

          return list;
        default:
          capture_mask = checkers;

          quiet_mask = SQUARES_BETWEEN_BB[our_king][checker_square];
          break;
      }
    }
  } else {
    {
      capture_mask = them_bb;

      quiet_mask = ~all;

      if (history[game_ply].epsq != NO_SQUARE) [[unlikely]] {
        b2 = pawn_attacks<Them>(history[game_ply].epsq) & bitboard_of(Us, PAWN);
        b1 = b2 & not_pinned;
        while (b1) {
          s = pop_lsb(&b1);


          // en passant may expose a rank attack on the king

          if ((sliding_attacks(our_king,
                               all ^ sq_bb(s) ^ shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history[game_ply].epsq]),
                               MASK_RANK[rank_of(our_king)]) &
               their_orth_sliders) == 0)
            *list++ = Move(s, history[game_ply].epsq, EN_PASSANT);
        }

        b1 = b2 & pinned & LINE[history[game_ply].epsq][our_king];
        if (b1) {
          *list++ = Move(bsf(b1), history[game_ply].epsq, EN_PASSANT);
        }
      }

      if constexpr (!CAPTURES_ONLY) {
        *list = Us == WHITE ? Move(e1, h1, OO) : Move(e8, h8, OO);
        list += !((history[game_ply].entry & oo_mask<Us>()) | ((all | danger) & oo_blockers_mask<Us>()));
        *list = Us == WHITE ? Move(e1, c1, OOO) : Move(e8, c8, OOO);
        list += !((history[game_ply].entry & ooo_mask<Us>()) |
                  ((all | (danger & ~ignore_ooo_danger<Us>())) & ooo_blockers_mask<Us>()));
      }

      b1 = pinned & ~(bitboard_of(Us, KNIGHT) | bitboard_of(Us, PAWN));
      while (b1) {
        s = pop_lsb(&b1);

        b2 = attacks(type_of(board[s]), s, all) & LINE[our_king][s];
        if constexpr (!CAPTURES_ONLY)
          list = make<QUIET>(s, b2 & quiet_mask, list);
        list = make<CAPTURE>(s, b2 & capture_mask, list);
      }

      b1 = ~not_pinned & bitboard_of(Us, PAWN);
      while (b1) {
        s = pop_lsb(&b1);

        if (rank_of(s) == relative_rank<Us>(RANK7)) {
          b2   = pawn_attacks<Us>(s) & capture_mask & LINE[our_king][s];
          list = make<PROMOTION_CAPTURES>(s, b2, list);
        } else {
          b2   = pawn_attacks<Us>(s) & them_bb & LINE[s][our_king];
          list = make<CAPTURE>(s, b2, list);

          if constexpr (!CAPTURES_ONLY) {
            b2 = shift<relative_dir<Us>(NORTH)>(sq_bb(s)) & ~all & LINE[our_king][s];
            b3   = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & ~all & LINE[our_king][s];
            list = make<QUIET>(s, b2, list);
            list = make<DOUBLE_PUSH>(s, b3, list);
          }
        }
      }

    }
  }

  b1 = bitboard_of(Us, KNIGHT) & not_pinned;
  while (b1) {
    s  = pop_lsb(&b1);
    b2 = attacks<KNIGHT>(s, all);
    if constexpr (!CAPTURES_ONLY)
      list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  b1 = our_diag_sliders & not_pinned;
  while (b1) {
    s  = pop_lsb(&b1);
    b2 = attacks<BISHOP>(s, all);
    if constexpr (!CAPTURES_ONLY)
      list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  b1 = our_orth_sliders & not_pinned;
  while (b1) {
    s  = pop_lsb(&b1);
    b2 = attacks<ROOK>(s, all);
    if constexpr (!CAPTURES_ONLY)
      list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  b1 = bitboard_of(Us, PAWN) & not_pinned & ~MASK_RANK[relative_rank<Us>(RANK7)];

  if constexpr (!CAPTURES_ONLY) {
    b2 = shift<relative_dir<Us>(NORTH)>(b1) & ~all;

    b3 = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & quiet_mask;

    b2 &= quiet_mask;

    while (b2) {
      s       = pop_lsb(&b2);
      *list++ = Move(s - relative_dir<Us>(NORTH), s, QUIET);
    }

    while (b3) {
      s       = pop_lsb(&b3);
      *list++ = Move(s - relative_dir<Us>(NORTH_NORTH), s, DOUBLE_PUSH);
    }
  }

  b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
  b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

  while (b2) {
    s       = pop_lsb(&b2);
    *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, CAPTURE);
  }

  while (b3) {
    s       = pop_lsb(&b3);
    *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, CAPTURE);
  }

  b1 = bitboard_of(Us, PAWN) & not_pinned & MASK_RANK[relative_rank<Us>(RANK7)];
  if (b1) [[unlikely]] {
    b2 = shift<relative_dir<Us>(NORTH)>(b1) & quiet_mask;
    while (b2) {
      s = pop_lsb(&b2);
      if constexpr (!CAPTURES_ONLY) {
        *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_KNIGHT);
        *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_BISHOP);
        *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_ROOK);
      }
      *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_QUEEN);
    }

    b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
    b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

    while (b2) {
      s = pop_lsb(&b2);
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_KNIGHT);
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_BISHOP);
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_ROOK);
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_QUEEN);
    }

    while (b3) {
      s = pop_lsb(&b3);
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_KNIGHT);
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_BISHOP);
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_ROOK);
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_QUEEN);
    }
  }

  return list;
}

template<Color Us>
class MoveList {
public:
  explicit MoveList(Position &p) : last(p.generate_legals<Us>(list)) {}

  [[nodiscard]] const Move *begin() const { return list; }
  [[nodiscard]] const Move *end() const { return last; }
  [[nodiscard]] size_t      size() const { return last - list; }

private:
  Move  list[218];
  Move *last;
};
