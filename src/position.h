#pragma once

#include <ostream>
#include <string>
#include <utility>
#include "tables.h"
#include "types.h"

// A psuedorandom number generator
// Source: Stockfish
class PRNG {
  uint64_t s;

  uint64_t rand64() {
    s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
    return s * 2685821657736338717LL;
  }

public:
  PRNG(uint64_t seed) : s(seed) {}

  // Generate psuedorandom number
  template<typename T>
  T rand() {
    return T(rand64());
  }

  // Generate psuedorandom number with only a few set bits
  template<typename T>
  T sparse_rand() {
    return T(rand64() & rand64() & rand64());
  }
};


namespace zobrist {
  extern uint64_t zobrist_table[NPIECES][NSQUARES];
  extern uint64_t zobrist_side; // XORed into the hash for black-to-move, so the key distinguishes
                                // side to move (required for null-move pruning and correct TT hits)
  extern void initialise_zobrist_keys();
} // namespace zobrist

// Stores position information which cannot be recovered on undo-ing a move
struct UndoInfo {
  // The bitboard of squares on which pieces have either moved from, or have been moved to. Used for castling
  // legality checks
  Bitboard entry;

  // The piece that was captured on the last move
  Piece captured;

  // The en passant square. This is the square which pawns can move to in order to en passant capture an enemy pawn that
  // has double pushed on the previous move
  Square epsq;

  // The Zobrist hash of the position reached AFTER this ply (set by play/play_null/set), so the
  // search can scan back for repetitions. And the halfmove clock: plies since the last irreversible
  // move (pawn move / capture / castle / promotion); it bounds the repetition scan and drives the
  // fifty-move rule. Both are recovered automatically on undo (which just rolls game_ply back).
  uint64_t hash;
  int      fifty;

  constexpr UndoInfo() : entry(0), captured(NO_PIECE), epsq(NO_SQUARE), hash(0), fifty(0) {}

  // This preserves the entry bitboard and halfmove clock across moves (play() overrides hash/fifty).
  UndoInfo(const UndoInfo &prev) : entry(prev.entry), captured(NO_PIECE), epsq(NO_SQUARE), hash(0), fifty(prev.fifty) {}
};

class Position {
private:
  // A bitboard of the locations of each piece
  Bitboard piece_bb[NPIECES];

  // A mailbox representation of the board. Stores the piece occupying each square on the board
  Piece board[NSQUARES];

  // The side whose turn it is to play next
  Color side_to_play;

  // The current game ply (depth), incremented after each move
  int game_ply;

  // The zobrist hash of the position, which can be incrementally updated and rolled back after each
  // make/unmake
  uint64_t hash;

public:
  // Upper bound on the undo stack: it must hold the whole *game* (every ply played via `play`) PLUS
  // the deepest search line stacked on top (the search makes its moves on the same Position, so it
  // keeps incrementing game_ply). Surge's original 256 overflowed in long games — a ~250-ply shuffle
  // plus a 50+-ply search wrote `history[300+]` out of bounds and corrupted memory. 1024 leaves room
  // for any realistic game (an arbiter adjudicates long before) plus the full search/quiescence depth;
  // the search also guards game_ply against this bound so it can never index past the array.
  static constexpr int MAX_HISTORY = 1024;

  // The history of non-recoverable information
  UndoInfo history[MAX_HISTORY];

  // The bitboard of enemy pieces that are currently attacking the king, updated whenever generate_moves()
  // is called
  Bitboard checkers;

  // The bitboard of pieces that are currently pinned to the king by enemy sliders, updated whenever
  // generate_moves() is called
  Bitboard pinned;


  Position() : piece_bb{0}, side_to_play(WHITE), game_ply(0), board{}, hash(0), pinned(0), checkers(0) {

    // Sets all squares on the board as empty
    for (auto &i: board)
      i = NO_PIECE;
    history[0] = UndoInfo();
  }

  // Places a piece on a particular square and updates the hash. Placing a piece on a square that is
  // already occupied is an error
  inline void put_piece(Piece pc, Square s) {
    board[s] = pc;
    piece_bb[pc] |= SQUARE_BB[s];
    hash ^= zobrist::zobrist_table[pc][s];
  }

  // Removes a piece from a particular square and updates the hash.
  inline void remove_piece(Square s) {
    const Piece pc = board[s];
    hash ^= zobrist::zobrist_table[pc][s];
    piece_bb[pc] &= ~SQUARE_BB[s];
    board[s] = NO_PIECE;
  }

  void move_piece(Square from, Square to);
  void move_piece_quiet(Square from, Square to);

  friend std::ostream &operator<<(std::ostream &os, const Position &p);
  // Parses `fen` into the freshly-constructed `p`. Returns false on a malformed board field
  // (an unrecognised piece letter, or a rank that over/underflows the 64 squares) and leaves
  // `p` in a WHATEVER-partial state was reached — the caller (the UCI `position` command,
  // where `fen` is untrusted external input) must check this and fall back to a known-good
  // position rather than continuing with a partially/incorrectly built one. Every other caller
  // in this codebase passes a hardcoded, already-known-valid FEN and is not required to check.
  static bool               set(const std::string &fen, Position &p);
  [[nodiscard]] std::string fen() const;

  Position   &operator=(const Position &) = delete;
  inline bool operator==(const Position &other) const { return hash == other.hash; }

  [[nodiscard]] inline Bitboard bitboard_of(Piece pc) const { return piece_bb[pc]; }
  [[nodiscard]] inline Bitboard bitboard_of(Color c, PieceType pt) const { return piece_bb[make_piece(c, pt)]; }
  [[nodiscard]] inline Piece    at(Square sq) const { return board[sq]; }
  [[nodiscard]] inline Color    turn() const { return side_to_play; }
  [[nodiscard]] inline int      ply() const { return game_ply; }
  [[nodiscard]] inline uint64_t get_hash() const { return hash; }
  [[nodiscard]] inline int      fifty() const { return history[game_ply].fifty; } // halfmove clock
  // Castling "entry" bitboard: squares whose king/rook have moved or been captured on. A side has
  // lost a castling right when its OO/OOO mask intersects this (used by eval's trapped-rook term).
  [[nodiscard]] inline Bitboard castle_entry() const { return history[game_ply].entry; }

  // True if the current position is a draw by repetition, the fifty-move rule, or dead
  // material. The repetition scan steps back two plies at a time (same side to move) and is
  // bounded by the halfmove clock: positions before the last irreversible move (pawn move /
  // capture / castle / promotion) had a different pawn structure or material and cannot recur.
  // A single earlier occurrence is enough — in a search line either side can force the
  // threefold, so it is treated as a draw immediately.
  [[gnu::hot, nodiscard]] inline bool is_draw() const {
    const int f = history[game_ply].fifty;
    if (f >= 100) [[unlikely]]
      return true; // fifty-move rule (100 halfmoves)
    // Dead material: bare kings plus at most one minor anywhere (KvK, KNvK, KBvK) cannot
    // mate by any play — without this the engine can hold an NNUE "+0.3" forever, or (with
    // Contempt) actively dodge a simplification into a position it could never lose OR win.
    // Gated on "no pawns" so the common case pays one OR and a taken branch. KNNvK stays
    // out: mates exist there (unforceable, but a position with legal mates is not dead).
    if (!(piece_bb[WHITE_PAWN] | piece_bb[BLACK_PAWN])) [[unlikely]] {
      const Bitboard majors =
              piece_bb[WHITE_ROOK] | piece_bb[BLACK_ROOK] | piece_bb[WHITE_QUEEN] | piece_bb[BLACK_QUEEN];
      const Bitboard minors =
              piece_bb[WHITE_KNIGHT] | piece_bb[BLACK_KNIGHT] | piece_bb[WHITE_BISHOP] | piece_bb[BLACK_BISHOP];
      if (!majors && pop_count(minors) <= 1)
        return true;
    }
    const int end = game_ply - f; // nothing before the last irreversible move can match
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

  // Plays/undoes a "null move" (pass the turn): flips the side to move, clears the en-passant
  // right, and toggles the side-to-move hash term. No piece moves. Used by null-move pruning.
  inline void play_null() {
    side_to_play = ~side_to_play;
    hash ^= zobrist::zobrist_side;
    ++game_ply;
    history[game_ply]       = UndoInfo(history[game_ply - 1]); // preserves castling rights, clears epsq
    history[game_ply].hash  = hash;
    history[game_ply].fifty = history[game_ply - 1].fifty + 1; // a pass changes nothing irreversibly
  }
  inline void undo_null() {
    side_to_play = ~side_to_play;
    hash ^= zobrist::zobrist_side;
    --game_ply;
  }

  template<Color Us>
  Move *generate_legals(Move *list);
};

// Returns the bitboard of all bishops and queens of a given color
template<Color C>
inline Bitboard Position::diagonal_sliders() const {
  return C == WHITE ? piece_bb[WHITE_BISHOP] | piece_bb[WHITE_QUEEN] : piece_bb[BLACK_BISHOP] | piece_bb[BLACK_QUEEN];
}

// Returns the bitboard of all rooks and queens of a given color
template<Color C>
inline Bitboard Position::orthogonal_sliders() const {
  return C == WHITE ? piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN] : piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN];
}

// Returns a bitboard containing all the pieces of a given color.
// A color's six piece bitboards are contiguous (WHITE: indices 0..5, BLACK: 8..13),
// so the OR-accumulation is vectorised via or_reduce6 (scalar OR chain when SIMD is off).
template<Color C>
inline Bitboard Position::all_pieces() const {
  return or_reduce6(&piece_bb[C == WHITE ? WHITE_PAWN : BLACK_PAWN]);
}

// Returns a bitboard containing all pieces of a given color attacking a particluar square
// The four attack sets (pawn/knight/bishop/rook) are computed scalarly (magic lookups
// cannot be vectorised cheaply); the per-piece AND and final OR-reduction are vectorised
// via and_or_reduce4 (scalar AND/OR expression when SIMD is off).
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


/*template<Color C>
Bitboard Position::pinned(Square s, Bitboard us, Bitboard occ) const {
  Bitboard pinned = 0;

  Bitboard pinners = get_xray_rook_attacks(s, occ, us) & orthogonal_sliders<~C>();
  while (pinners) pinned |= SQUARES_BETWEEN_BB[s][pop_lsb(&pinners)] & us;

  pinners = get_xray_bishop_attacks(s, occ, us) & diagonal_sliders<~C>();
  while (pinners) pinned |= SQUARES_BETWEEN_BB[s][pop_lsb(&pinners)] & us;

  return pinned;
}

template<Color C>
Bitboard Position::blockers_to(Square s, Bitboard occ) const {
  Bitboard blockers = 0;
  Bitboard candidates = get_rook_attacks(s, occ) & occ;
  Bitboard attackers = get_rook_attacks(s, occ ^ candidates) & orthogonal_sliders<~C>();

  candidates = get_bishop_attacks(s, occ) & occ;
  attackers |= get_bishop_attacks(s, occ ^ candidates) & diagonal_sliders<~C>();

  while (attackers) blockers |= SQUARES_BETWEEN_BB[s][pop_lsb(&attackers)];
  return blockers;
}*/

// Plays a move in the position
template<Color C>
[[gnu::hot]] void Position::play(const Move m) {
  side_to_play = ~side_to_play;
  hash ^= zobrist::zobrist_side;
  ++game_ply;
  history[game_ply] = UndoInfo(history[game_ply - 1]);

  MoveFlags type = m.flags();
  history[game_ply].entry |= SQUARE_BB[m.to()] | SQUARE_BB[m.from()];

  switch (type) {
    case QUIET:
      // The to square is guaranteed to be empty here
      move_piece_quiet(m.from(), m.to());
      break;
    case DOUBLE_PUSH:
      // The to square is guaranteed to be empty here
      move_piece_quiet(m.from(), m.to());

      // This is the square behind the pawn that was double-pushed
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
  }

  // Record this position's hash for repetition detection, and advance the halfmove clock — reset to
  // zero on an irreversible move (any non-quiet flag, or a quiet pawn push), else +1. `type != QUIET`
  // short-circuits, so board[m.to()] (the moved/promoted/capturing piece) is only read for quiets.
  history[game_ply].hash  = hash;
  history[game_ply].fifty = (type != QUIET || type_of(board[m.to()]) == PAWN) ? 0 : history[game_ply - 1].fifty + 1;
}

// Undos a move in the current position, rolling it back to the previous position
template<Color C>
[[gnu::hot]] void Position::undo(const Move m) {
  MoveFlags type = m.flags();
  switch (type) {
    case QUIET:
    case DOUBLE_PUSH:
      move_piece_quiet(m.to(), m.from());
      break;
    case OO:
      if (C == WHITE) {
        move_piece_quiet(g1, e1);
        move_piece_quiet(f1, h1);
      } else {
        move_piece_quiet(g8, e8);
        move_piece_quiet(f8, h8);
      }
      break;
    case OOO:
      if (C == WHITE) {
        move_piece_quiet(c1, e1);
        move_piece_quiet(d1, a1);
      } else {
        move_piece_quiet(c8, e8);
        move_piece_quiet(d8, a8);
      }
      break;
    case EN_PASSANT:
      move_piece_quiet(m.to(), m.from());
      put_piece(make_piece(~C, PAWN), m.to() + relative_dir<C>(SOUTH));
      break;
    case PR_KNIGHT:
    case PR_BISHOP:
    case PR_ROOK:
    case PR_QUEEN:
      remove_piece(m.to());
      put_piece(make_piece(C, PAWN), m.from());
      break;
    case PC_KNIGHT:
    case PC_BISHOP:
    case PC_ROOK:
    case PC_QUEEN:
      remove_piece(m.to());
      put_piece(make_piece(C, PAWN), m.from());
      put_piece(history[game_ply].captured, m.to());
      break;
    case CAPTURE:
      move_piece_quiet(m.to(), m.from());
      put_piece(history[game_ply].captured, m.to());
      break;
  }

  side_to_play = ~side_to_play;
  hash ^= zobrist::zobrist_side;
  --game_ply;
}


// Generates all legal moves in a position for the given side. Advances the move pointer and returns it.
template<Color Us>
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

  // General purpose bitboards for attacks, masks, etc.
  Bitboard b1, b2, b3;

  // Squares that our king cannot move to
  Bitboard danger = 0;

  // For each enemy piece, add all of its attacks to the danger bitboard
  danger |= pawn_attacks<Them>(bitboard_of(Them, PAWN)) | attacks<KING>(their_king, all);

  b1 = bitboard_of(Them, KNIGHT);
  while (b1)
    danger |= attacks<KNIGHT>(pop_lsb(&b1), all);

  b1 = their_diag_sliders;
  // all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
  // by enemy bishops and queens
  while (b1)
    danger |= attacks<BISHOP>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);

  b1 = their_orth_sliders;
  // all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
  // by enemy rooks and queens
  while (b1)
    danger |= attacks<ROOK>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);

  // The king can move to all of its surrounding squares, except ones that are attacked, and
  // ones that have our own pieces on them
  b1   = attacks<KING>(our_king, all) & ~(us_bb | danger);
  list = make<QUIET>(our_king, b1 & ~them_bb, list);
  list = make<CAPTURE>(our_king, b1 & them_bb, list);

  // The capture mask filters destination squares to those that contain an enemy piece that is checking the
  // king and must be captured
  Bitboard capture_mask;

  // The quiet mask filter destination squares to those where pieces must be moved to block an incoming attack
  // to the king
  Bitboard quiet_mask;

  // A general purpose square for storing destinations, etc.
  Square s;

  // Checkers of each piece type are identified by:
  // 1. Projecting attacks FROM the king square
  // 2. Intersecting this bitboard with the enemy bitboard of that piece type
  checkers = attacks<KNIGHT>(our_king, all) & bitboard_of(Them, KNIGHT) |
             pawn_attacks<Us>(our_king) & bitboard_of(Them, PAWN);

  // Here, we identify slider checkers and pinners simultaneously, and candidates for such pinners
  // and checkers are represented by the bitboard <candidates>
  Bitboard candidates = attacks<ROOK>(our_king, them_bb) & their_orth_sliders |
                        attacks<BISHOP>(our_king, them_bb) & their_diag_sliders;

  pinned = 0;
  while (candidates) {
    s  = pop_lsb(&candidates);
    b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb;

    // Do the squares in between the enemy slider and our king contain any of our pieces?
    // If not, add the slider to the checker bitboard
    if (b1 == 0)
      checkers ^= SQUARE_BB[s];
    // If there is only one of our pieces between them, add our piece to the pinned bitboard
    else if (std::has_single_bit(b1))
      pinned ^= b1;
  }

  // This makes it easier to mask pieces
  const Bitboard not_pinned = ~pinned;

  switch (sparse_pop_count(checkers)) {
    case 2:
      // If there is a double check, the only legal moves are king moves out of check
      return list;
    case 1: {
      // It's a single check!

      Square checker_square = bsf(checkers);

      switch (board[checker_square]) {
        case make_piece(Them, PAWN):
          // If the checker is a pawn, we must check for e.p. moves that can capture it
          // This evaluates to true if the checking piece is the one which just double pushed
          if (checkers == shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history[game_ply].epsq])) {
            // b1 contains our pawns that can capture the checker e.p.
            b1 = pawn_attacks<Them>(history[game_ply].epsq) & bitboard_of(Us, PAWN) & not_pinned;
            while (b1)
              *list++ = Move(pop_lsb(&b1), history[game_ply].epsq, EN_PASSANT);
          }
          // FALL THROUGH INTENTIONAL
        case make_piece(Them, KNIGHT):
          // If the checker is either a pawn or a knight, the only legal moves are to capture
          // the checker. Only non-pinned pieces can capture it
          b1 = attackers_from<Us>(checker_square, all) & not_pinned;
          while (b1) {
            s = pop_lsb(&b1);
            // A pawn capturing the checker on its promotion rank must promote. Upstream surge
            // emitted a plain CAPTURE here, which both prints an illegal UCI move (no promotion
            // suffix -> forfeits the game) and, when played internally, leaves an unpromoted pawn
            // on the last rank. Only a knight checker is capturable there (an enemy pawn never
            // stands on our 8th rank), and only a pawn promotes.
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
          // We must capture the checking piece
          capture_mask = checkers;

          //...or we can block it since it is guaranteed to be a slider
          quiet_mask = SQUARES_BETWEEN_BB[our_king][checker_square];
          break;
      }

      break;
    }

    default:
      // We can capture any enemy piece
      capture_mask = them_bb;

      //...and we can play a quiet move to any square which is not occupied
      quiet_mask = ~all;

      if (history[game_ply].epsq != NO_SQUARE) {
        // b1 contains our pawns that can perform an e.p. capture
        b2 = pawn_attacks<Them>(history[game_ply].epsq) & bitboard_of(Us, PAWN);
        b1 = b2 & not_pinned;
        while (b1) {
          s = pop_lsb(&b1);

          // This piece of evil bit-fiddling magic prevents the infamous 'pseudo-pinned' e.p. case,
          // where the pawn is not directly pinned, but on moving the pawn and capturing the enemy pawn
          // e.p., a rook or queen attack to the king is revealed

          /*
          .nbqkbnr
          ppp.pppp
          ........
          r..pP..K
          ........
          ........
          PPPP.PPP
          RNBQ.BNR

          Here, if white plays exd5 e.p., the black rook on a5 attacks the white king on h5
          */

          if ((sliding_attacks(our_king,
                               all ^ SQUARE_BB[s] ^ shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history[game_ply].epsq]),
                               MASK_RANK[rank_of(our_king)]) &
               their_orth_sliders) == 0)
            *list++ = Move(s, history[game_ply].epsq, EN_PASSANT);
        }

        // Pinned pawns can only capture e.p. if they are pinned diagonally and the e.p. square is in line with the king
        b1 = b2 & pinned & LINE[history[game_ply].epsq][our_king];
        if (b1) {
          *list++ = Move(bsf(b1), history[game_ply].epsq, EN_PASSANT);
        }
      }

      // Only add castling if:
      // 1. The king and the rook have both not moved
      // 2. No piece is attacking between the the rook and the king
      // 3. The king is not in check
      if (!((history[game_ply].entry & oo_mask<Us>()) | ((all | danger) & oo_blockers_mask<Us>())))
        *list++ = Us == WHITE ? Move(e1, h1, OO) : Move(e8, h8, OO);
      if (!((history[game_ply].entry & ooo_mask<Us>()) |
            ((all | (danger & ~ignore_ooo_danger<Us>())) & ooo_blockers_mask<Us>())))
        *list++ = Us == WHITE ? Move(e1, c1, OOO) : Move(e8, c8, OOO);

      // For each pinned rook, bishop or queen...
      b1 = ~(not_pinned | bitboard_of(Us, KNIGHT));
      while (b1) {
        s = pop_lsb(&b1);

        //...only include attacks that are aligned with our king, since pinned pieces
        // are constrained to move in this direction only
        b2   = attacks(type_of(board[s]), s, all) & LINE[our_king][s];
        list = make<QUIET>(s, b2 & quiet_mask, list);
        list = make<CAPTURE>(s, b2 & capture_mask, list);
      }

      // For each pinned pawn...
      b1 = ~not_pinned & bitboard_of(Us, PAWN);
      while (b1) {
        s = pop_lsb(&b1);

        if (rank_of(s) == relative_rank<Us>(RANK7)) {
          // Quiet promotions are impossible since the square in front of the pawn will
          // either be occupied by the king or the pinner, or doing so would leave our king
          // in check
          b2   = pawn_attacks<Us>(s) & capture_mask & LINE[our_king][s];
          list = make<PROMOTION_CAPTURES>(s, b2, list);
        } else {
          b2   = pawn_attacks<Us>(s) & them_bb & LINE[s][our_king];
          list = make<CAPTURE>(s, b2, list);

          // Single pawn pushes
          b2 = shift<relative_dir<Us>(NORTH)>(SQUARE_BB[s]) & ~all & LINE[our_king][s];
          // Double pawn pushes (only pawns on rank 3/6 are eligible)
          b3   = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & ~all & LINE[our_king][s];
          list = make<QUIET>(s, b2, list);
          list = make<DOUBLE_PUSH>(s, b3, list);
        }
      }

      // Pinned knights cannot move anywhere, so we're done with pinned pieces!

      break;
  }

  // Non-pinned knight moves
  b1 = bitboard_of(Us, KNIGHT) & not_pinned;
  while (b1) {
    s    = pop_lsb(&b1);
    b2   = attacks<KNIGHT>(s, all);
    list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  // Non-pinned bishops and queens
  b1 = our_diag_sliders & not_pinned;
  while (b1) {
    s    = pop_lsb(&b1);
    b2   = attacks<BISHOP>(s, all);
    list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  // Non-pinned rooks and queens
  b1 = our_orth_sliders & not_pinned;
  while (b1) {
    s    = pop_lsb(&b1);
    b2   = attacks<ROOK>(s, all);
    list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  // b1 contains non-pinned pawns which are not on the last rank
  b1 = bitboard_of(Us, PAWN) & not_pinned & ~MASK_RANK[relative_rank<Us>(RANK7)];

  // Single pawn pushes
  b2 = shift<relative_dir<Us>(NORTH)>(b1) & ~all;

  // Double pawn pushes (only pawns on rank 3/6 are eligible)
  b3 = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & quiet_mask;

  // We & this with the quiet mask only later, as a non-check-blocking single push does NOT mean that the
  // corresponding double push is not blocking check either.
  b2 &= quiet_mask;

  while (b2) {
    s       = pop_lsb(&b2);
    *list++ = Move(s - relative_dir<Us>(NORTH), s, QUIET);
  }

  while (b3) {
    s       = pop_lsb(&b3);
    *list++ = Move(s - relative_dir<Us>(NORTH_NORTH), s, DOUBLE_PUSH);
  }

  // Pawn captures
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

  // b1 now contains non-pinned pawns which ARE on the last rank (about to promote)
  b1 = bitboard_of(Us, PAWN) & not_pinned & MASK_RANK[relative_rank<Us>(RANK7)];
  if (b1) {
    // Quiet promotions
    b2 = shift<relative_dir<Us>(NORTH)>(b1) & quiet_mask;
    while (b2) {
      s = pop_lsb(&b2);
      // One move is added for each promotion piece
      *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_KNIGHT);
      *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_BISHOP);
      *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_ROOK);
      *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_QUEEN);
    }

    // Promotion captures
    b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
    b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

    while (b2) {
      s = pop_lsb(&b2);
      // One move is added for each promotion piece
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_KNIGHT);
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_BISHOP);
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_ROOK);
      *list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_QUEEN);
    }

    while (b3) {
      s = pop_lsb(&b3);
      // One move is added for each promotion piece
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_KNIGHT);
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_BISHOP);
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_ROOK);
      *list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_QUEEN);
    }
  }

  return list;
}

// A convenience class for interfacing with legal moves, rather than using the low-level
// generate_legals() function directly. It can be iterated over.
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
