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

  template<typename T>
  T rand() {
    return T(rand64());
  }

  // Generate pseudorandom number with only a few set bits
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

  // Plain memberwise copy (used by play_null(), copying an already-adjusted UndoInfo). Declared
  // explicitly alongside the custom copy constructor above to satisfy the rule of three.
  UndoInfo &operator=(const UndoInfo &) = default;
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


  Position() : piece_bb{0}, board{}, side_to_play(WHITE), game_ply(0), hash(0), checkers(0), pinned(0) {
    for (auto &i: board)
      i = NO_PIECE;
    history[0] = UndoInfo();
  }

  // Places a piece on a particular square and updates the hash. Placing a piece on a square that is
  // already occupied is an error
  inline void put_piece(Piece pc, Square s) {
    board[s] = pc;
    piece_bb[pc] |= sq_bb(s);
    hash ^= zobrist::zobrist_table[pc][s];
  }

  // Removes a piece from a particular square and updates the hash.
  inline void remove_piece(Square s) {
    const Piece pc = board[s];
    hash ^= zobrist::zobrist_table[pc][s];
    piece_bb[pc] &= ~sq_bb(s);
    board[s] = NO_PIECE;
  }

  // Hash-free piece movers, for undo() only: the pre-move hash is restored wholesale from
  // history[] (play()/set() recorded it there), so re-XORing zobrist keys piece-by-piece while
  // un-moving would be dead work — these skip the zobrist table entirely.
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

  // Both piece movers are defined here in the class body (not position.cpp) and force-inlined:
  // LTO left move_piece() as an out-of-line call inside play<C>'s CAPTURE case (verified in the
  // disassembly — with argument spills around the call), and captures are the second most common
  // move type on every search path.
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

  // Moves a piece to an empty square. Note that it is an error if the <to> square contains a piece
  [[gnu::always_inline]] inline void move_piece_quiet(Square from, Square to) {
    const Piece moving = board[from];
    hash ^= zobrist::zobrist_table[moving][from] ^ zobrist::zobrist_table[moving][to];
    piece_bb[moving] ^= (sq_bb(from) | sq_bb(to));
    board[to]   = moving;
    board[from] = NO_PIECE;
  }

  friend std::ostream &operator<<(std::ostream &os, const Position &p);
  // Parses `fen` into the freshly-constructed `p`. Returns false on a malformed board field
  // (an unrecognised piece letter, or a rank that over/underflows the 64 squares) and leaves
  // `p` in a WHATEVER-partial state was reached — the caller (the UCI `position` command,
  // where `fen` is untrusted external input) must check this and fall back to a known-good
  // position rather than continuing with a partially/incorrectly built one. Every other caller
  // in this codebase passes a hardcoded, already-known-valid FEN and is not required to check.
  static bool               set(const std::string &fen, Position &p);
  [[nodiscard]] std::string fen() const;

  // Assignment is deleted (accidental `posA = posB;` would go through UndoInfo's copy
  // constructor per history[] entry and reset captured/epsq/hash for all of them). The copy
  // constructor is left implicit (elementwise, same UndoInfo-ctor caveat) rather than also
  // deleted: uci.cpp's clone_position() byte-memcpy's a Position and returns it by value, which
  // needs a callable copy constructor to exist even though the return is always elided (NRVO) in
  // every build config this project uses — verified empirically the elision is NOT optional in
  // practice here (identical under -O0 and -O2+LTO; only `-fno-elide-constructors`, which no
  // build of this project passes, forces the real copy and its would-be corruption).
  Position(const Position &) = default;
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

  // Draw by repetition (a single earlier occurrence counts — either side could force the
  // threefold), the fifty-move rule, or dead material. The repetition scan is bounded by the
  // halfmove clock: nothing before the last irreversible move can recur.
  [[gnu::hot, nodiscard]] inline bool is_draw() const {
    const int f = history[game_ply].fifty;
    if (f >= 100) [[unlikely]]
      return true; // fifty-move rule (100 halfmoves)
    // Dead material: KvK, KNvK, KBvK cannot mate by any play. KNNvK stays out (legal mates
    // exist). Gated on "no pawns" so the common case pays one OR and a taken branch.
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

  // CAPTURES_ONLY skips every quiet emission except quiet QUEEN promotions (quiescence
  // moves), leaving the legality machinery identical. Not for use in check: quiet evasions
  // are skipped too — the in-check quiescence path generates fully instead.
  template<Color Us, bool CAPTURES_ONLY = false>
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


template<Color C>
[[gnu::hot]] void Position::play(const Move m) {
  side_to_play = ~side_to_play;
  hash ^= zobrist::zobrist_side;
  ++game_ply;

  // Written out field-by-field instead of `history[game_ply] = UndoInfo(prev)`: the
  // copy-constructor form also stored hash=0 and prev.fifty, both dead — they are
  // unconditionally overwritten at the end of this function.
  MoveFlags type = m.flags();
  UndoInfo &st   = history[game_ply];
  st.entry       = history[game_ply - 1].entry | sq_bb(m.to()) | sq_bb(m.from());
  st.captured    = NO_PIECE;
  st.epsq        = NO_SQUARE;

  // Quiet moves dominate every game tree; peeling them off as one predictable branch keeps
  // the common path free of the switch's indirect jump (a frequent branch-predictor miss).
  if (type == QUIET) [[likely]] {
    // The to square is guaranteed to be empty here
    move_piece_quiet(m.from(), m.to());
  } else switch (type) {
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
    default: // QUIET: excluded by the if/else above, never reaches here
      __builtin_unreachable();
  }

  // Record this position's hash for repetition detection, and advance the halfmove clock — reset to
  // zero on an irreversible move (any non-quiet flag, or a quiet pawn push), else +1. `type != QUIET`
  // short-circuits, so board[m.to()] (the moved/promoted/capturing piece) is only read for quiets.
  history[game_ply].hash  = hash;
  history[game_ply].fifty = (type != QUIET || type_of(board[m.to()]) == PAWN) ? 0 : history[game_ply - 1].fifty + 1;
}

template<Color C>
[[gnu::hot]] void Position::undo(const Move m) {
  MoveFlags type = m.flags();
  // Same common-path peel as play(): QUIET (0) and DOUBLE_PUSH (1) both just move the piece
  // back, and together they dominate — one predictable branch instead of the indirect jump.
  // All piece movers are the _nohash variants: the hash is restored in one load at the end.
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
    default: // QUIET/DOUBLE_PUSH: excluded by the if/else above, never reaches here
      __builtin_unreachable();
  }

  side_to_play = ~side_to_play;
  --game_ply;
  // The zobrist hash of the position being returned to, recorded when it was first reached
  // (play/play_null/set all store it). One load replaces the per-piece zobrist XOR chain AND
  // the side-to-move toggle.
  hash = history[game_ply].hash;
}


// Generates all legal moves in a position for the given side. Advances the move pointer and returns it.
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

  // General purpose bitboards for attacks, masks, etc.
  Bitboard b1, b2, b3;

  // Squares that our king cannot move to
  Bitboard danger = 0;

  // For each enemy piece, add all of its attacks to the danger bitboard. Pawns and knights are
  // added setwise (one shift/mask tree covers every piece at once) — no per-piece loop.
  danger |= pawn_attacks<Them>(bitboard_of(Them, PAWN)) | attacks<KING>(their_king, all) |
            knight_attacks(bitboard_of(Them, KNIGHT));

  // all ^ sq_bb(our_king) prevents the king from moving to squares which are 'x-rayed'
  // by enemy sliders (the king does not block the ray it is standing on)
  const Bitboard all_no_king = all ^ sq_bb(our_king);

  b1 = their_diag_sliders;
  while (b1)
    danger |= attacks<BISHOP>(pop_lsb(&b1), all_no_king);

  b1 = their_orth_sliders;
  while (b1)
    danger |= attacks<ROOK>(pop_lsb(&b1), all_no_king);

  // The king can move to all of its surrounding squares, except ones that are attacked, and
  // ones that have our own pieces on them
  b1 = attacks<KING>(our_king, all) & ~(us_bb | danger);
  if constexpr (!CAPTURES_ONLY)
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

  // Candidate slider checkers/pinners: enemy sliders seen from the king with only ENEMY pieces
  // as occupancy, so the ray continues through our own men (a candidate is then classified as a
  // checker — no blocker — or a pinner — exactly one of ours between). Needed on both paths below.
  Bitboard candidates = (attacks<ROOK>(our_king, them_bb) & their_orth_sliders) |
                        (attacks<BISHOP>(our_king, them_bb) & their_diag_sliders);

  pinned = 0;
  // `danger` was built from EVERY enemy piece with the king removed from the occupancy, so
  // "our king square is in danger" is EXACTLY "we are in check" — one AND against the bitboard
  // already in hand. The common not-in-check path then skips the knight/pawn checker probes
  // entirely, and its candidates loop drops the checker case (a candidate with no blocker would
  // mean a slider attacks the king, contradicting not-in-check).
  if (!(danger & sq_bb(our_king))) [[likely]] {
    checkers = 0;
    while (candidates) {
      s  = pop_lsb(&candidates);
      b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb; // non-empty here: no blocker would be a check
      // Exactly one of our pieces between slider and king pins it. `b1 & (b1 - 1)` clears the
      // lowest bit: nothing left means exactly one bit — a two-op GPR test instead of a popcount
      // (which detours through the vector unit on ARM).
      if (!(b1 & (b1 - 1)))
        pinned ^= b1;
    }
  } else {
    // In check. Checkers of each piece type are identified by:
    // 1. Projecting attacks FROM the king square
    // 2. Intersecting this bitboard with the enemy bitboard of that piece type
    checkers = (attacks<KNIGHT>(our_king, all) & bitboard_of(Them, KNIGHT)) |
               (pawn_attacks<Us>(our_king) & bitboard_of(Them, PAWN));

    while (candidates) {
      s  = pop_lsb(&candidates);
      b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb;

      // Do the squares in between the enemy slider and our king contain any of our pieces?
      // If not, add the slider to the checker bitboard
      if (b1 == 0)
        checkers ^= sq_bb(s);
      // If there is only one of our pieces between them, add our piece to the pinned bitboard
      // (same two-op exactly-one-bit test as above).
      else if (!(b1 & (b1 - 1)))
        pinned ^= b1;
    }
  }

  // This makes it easier to mask pieces
  const Bitboard not_pinned = ~pinned;

  // Dispatch on the number of checkers without a popcount: `checkers & (checkers - 1)` is
  // non-zero iff at least two pieces give check (saves the vector round-trip of a popcount
  // on this always-taken path; checkers can never hold more than two bits).
  if (checkers) {
    if (checkers & (checkers - 1))
      // If there is a double check, the only legal moves are king moves out of check
      return list;
    {
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
    }
  } else {
    // Not in check:
    {
      // We can capture any enemy piece
      capture_mask = them_bb;

      //...and we can play a quiet move to any square which is not occupied
      quiet_mask = ~all;

      // An en-passant square only exists right after a double push (~6% of nodes) — keep the
      // whole block off the straight-line path.
      if (history[game_ply].epsq != NO_SQUARE) [[unlikely]] {
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
                               all ^ sq_bb(s) ^ shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history[game_ply].epsq]),
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
      // Emitted branchlessly: the move is a compile-time constant, so it is stored
      // unconditionally and the pointer advances by the condition (0 or 1) — no data-dependent
      // branch. The speculative store is always in-bounds: only king moves and en-passant
      // (≤10 moves) precede it in the 218-slot list.
      if constexpr (!CAPTURES_ONLY) {
        *list = Us == WHITE ? Move(e1, h1, OO) : Move(e8, h8, OO);
        list += !((history[game_ply].entry & oo_mask<Us>()) | ((all | danger) & oo_blockers_mask<Us>()));
        *list = Us == WHITE ? Move(e1, c1, OOO) : Move(e8, c8, OOO);
        list += !((history[game_ply].entry & ooo_mask<Us>()) |
                  ((all | (danger & ~ignore_ooo_danger<Us>())) & ooo_blockers_mask<Us>()));
      }

      // For each pinned rook, bishop or queen... Pawns are excluded explicitly: they have their
      // own pinned loop below, and iterating them here only worked by accident (the runtime
      // attacks() falls through to PSEUDO_LEGAL_ATTACKS[PAWN], which is all-zeros) — each pinned
      // pawn cost a wasted loop iteration emitting nothing.
      b1 = pinned & ~(bitboard_of(Us, KNIGHT) | bitboard_of(Us, PAWN));
      while (b1) {
        s = pop_lsb(&b1);

        //...only include attacks that are aligned with our king, since pinned pieces
        // are constrained to move in this direction only
        b2 = attacks(type_of(board[s]), s, all) & LINE[our_king][s];
        if constexpr (!CAPTURES_ONLY)
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

          if constexpr (!CAPTURES_ONLY) {
            // Single pawn pushes
            b2 = shift<relative_dir<Us>(NORTH)>(sq_bb(s)) & ~all & LINE[our_king][s];
            // Double pawn pushes (only pawns on rank 3/6 are eligible)
            b3   = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & ~all & LINE[our_king][s];
            list = make<QUIET>(s, b2, list);
            list = make<DOUBLE_PUSH>(s, b3, list);
          }
        }
      }

      // Pinned knights cannot move anywhere, so we're done with pinned pieces!
    }
  }

  // Non-pinned knight moves
  b1 = bitboard_of(Us, KNIGHT) & not_pinned;
  while (b1) {
    s  = pop_lsb(&b1);
    b2 = attacks<KNIGHT>(s, all);
    if constexpr (!CAPTURES_ONLY)
      list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  // Non-pinned bishops and queens
  b1 = our_diag_sliders & not_pinned;
  while (b1) {
    s  = pop_lsb(&b1);
    b2 = attacks<BISHOP>(s, all);
    if constexpr (!CAPTURES_ONLY)
      list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  // Non-pinned rooks and queens
  b1 = our_orth_sliders & not_pinned;
  while (b1) {
    s  = pop_lsb(&b1);
    b2 = attacks<ROOK>(s, all);
    if constexpr (!CAPTURES_ONLY)
      list = make<QUIET>(s, b2 & quiet_mask, list);
    list = make<CAPTURE>(s, b2 & capture_mask, list);
  }

  // b1 contains non-pinned pawns which are not on the last rank
  b1 = bitboard_of(Us, PAWN) & not_pinned & ~MASK_RANK[relative_rank<Us>(RANK7)];

  if constexpr (!CAPTURES_ONLY) {
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
  // Pawns on the 7th are rare in any phase — keep all four promotion emit loops cold.
  if (b1) [[unlikely]] {
    // Quiet promotions. CAPTURES_ONLY keeps the queen promotion (a quiescence move, the same
    // one the QS picker used to keep from the full list) and drops the underpromotions.
    b2 = shift<relative_dir<Us>(NORTH)>(b1) & quiet_mask;
    while (b2) {
      s = pop_lsb(&b2);
      // One move is added for each promotion piece
      if constexpr (!CAPTURES_ONLY) {
        *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_KNIGHT);
        *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_BISHOP);
        *list++ = Move(s - relative_dir<Us>(NORTH), s, PR_ROOK);
      }
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
