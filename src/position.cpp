#include "position.h"
#include <sstream>
#include "tables.h"

// Zobrist keys for each piece and each square
// Used to incrementally update the hash key of a position
uint64_t zobrist::zobrist_table[NPIECES][NSQUARES];
uint64_t zobrist::zobrist_side;

// Initializes the zobrist table with random 64-bit numbers
void zobrist::initialise_zobrist_keys() {
  PRNG rng(70026072);
  for (auto &i: zobrist::zobrist_table)
    for (auto &key: i)
      key = rng.rand<uint64_t>();
  zobrist::zobrist_side = rng.rand<uint64_t>();
}

// Pretty-prints the position (including FEN and hash key)
std::ostream &operator<<(std::ostream &os, const Position &p) {
  const char *s = "   +---+---+---+---+---+---+---+---+\n";
  const char *t = "     A   B   C   D   E   F   G   H\n";
  os << t;
  for (int i = 56; i >= 0; i -= 8) {
    os << s << " " << i / 8 + 1 << " ";
    for (int j = 0; j < 8; j++)
      os << "| " << PIECE_STR[p.board[i + j]] << " ";
    os << "| " << i / 8 + 1 << "\n";
  }
  os << s;
  os << t << "\n";

  os << "FEN: " << p.fen() << "\n";
  os << "Hash: 0x" << std::hex << p.hash << std::dec << "\n";

  return os;
}

// Returns the FEN (Forsyth-Edwards Notation) representation of the position
std::string Position::fen() const {
  std::ostringstream fen;
  int                empty;

  for (int i = 56; i >= 0; i -= 8) {
    empty = 0;
    for (int j = 0; j < 8; j++) {
      Piece p = board[i + j];
      if (p == NO_PIECE)
        empty++;
      else {
        fen << (empty == 0 ? "" : std::to_string(empty)) << PIECE_STR[p];
        empty = 0;
      }
    }

    if (empty != 0)
      fen << empty;
    if (i > 0)
      fen << '/';
  }

  // Castling field: the available rights (a set bit in `entry` means that right was lost), or "-" when
  // none remain — a standard FEN castling field (the old code appended a stray "-" whenever ANY single
  // right was lost, producing e.g. "KQq- " for a partial-rights position).
  const Bitboard entry = history[game_ply].entry;
  std::string    castle;
  if (!(entry & WHITE_OO_MASK))
    castle += 'K';
  if (!(entry & WHITE_OOO_MASK))
    castle += 'Q';
  if (!(entry & BLACK_OO_MASK))
    castle += 'k';
  if (!(entry & BLACK_OOO_MASK))
    castle += 'q';
  if (castle.empty())
    castle = "-";

  fen << (side_to_play == WHITE ? " w " : " b ") << castle << ' '
      << (history[game_ply].epsq == NO_SQUARE ? "-" : SQSTR[history[game_ply].epsq]);

  return fen.str();
}

// Updates a position according to an FEN string
void Position::set(const std::string &fen, Position &p) {
  int square = a8;
  for (char ch: fen.substr(0, fen.find(' '))) {
    if (isdigit(ch))
      square += (ch - '0') * EAST;
    else if (ch == '/')
      square += 2 * SOUTH;
    else
      p.put_piece(Piece(PIECE_STR.find(ch)), Square(square++));
  }

  // Parse the side-to-move, castling-rights and en-passant fields as whitespace-separated
  // tokens. The two halfmove/fullmove counters (if present) are read past and ignored.
  std::istringstream ss(fen.substr(fen.find(' ')));
  std::string        side;
  std::string        castling;
  std::string        ep;
  ss >> side >> castling >> ep;

  p.side_to_play = side == "w" ? WHITE : BLACK;
  if (p.side_to_play == BLACK)
    p.hash ^= zobrist::zobrist_side; // keep the side-to-move hash convention for the root position

  p.history[p.game_ply].entry = ALL_CASTLING_MASK;
  for (char ch: castling) {
    switch (ch) {
      case 'K':
        p.history[p.game_ply].entry &= ~WHITE_OO_MASK;
        break;
      case 'Q':
        p.history[p.game_ply].entry &= ~WHITE_OOO_MASK;
        break;
      case 'k':
        p.history[p.game_ply].entry &= ~BLACK_OO_MASK;
        break;
      case 'q':
        p.history[p.game_ply].entry &= ~BLACK_OOO_MASK;
        break;
      default:
        break;
    }
  }

  if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && ep[1] >= '1' && ep[1] <= '8')
    p.history[p.game_ply].epsq = create_square(File(ep[0] - 'a'), Rank(ep[1] - '1'));

  // Halfmove clock (5th FEN field, when present): plies since the last irreversible move. It
  // drives is_draw()'s fifty-move rule and the eval's fifty-move damping, so analysing a mid-game
  // FEN must not silently reset it to 0 (the engine would overvalue an unconvertible advantage and
  // miss imminent fifty-move draws). A missing or garbled field leaves 0; the fullmove number (6th
  // field) stays ignored — game_ply only ever matters relatively. The repetition scan still cannot
  // see positions from BEFORE the FEN (they were never pushed onto the history), which is inherent
  // to starting from a FEN.
  int halfmove = 0;
  if (ss >> halfmove && halfmove > 0)
    p.history[p.game_ply].fifty = halfmove;
  else
    p.history[p.game_ply].fifty = 0;

  // Seed the root position's hash (now fully built) for repetition detection.
  p.history[p.game_ply].hash = p.hash;
}


// Moves a piece to a (possibly empty) square on the board and updates the hash
void Position::move_piece(Square from, Square to) {
  hash ^= zobrist::zobrist_table[board[from]][from] ^ zobrist::zobrist_table[board[from]][to] ^
          zobrist::zobrist_table[board[to]][to];
  Bitboard mask = SQUARE_BB[from] | SQUARE_BB[to];
  piece_bb[board[from]] ^= mask;
  piece_bb[board[to]] &= ~mask;
  board[to]   = board[from];
  board[from] = NO_PIECE;
}

// Moves a piece to an empty square. Note that it is an error if the <to> square contains a piece
void Position::move_piece_quiet(Square from, Square to) {
  hash ^= zobrist::zobrist_table[board[from]][from] ^ zobrist::zobrist_table[board[from]][to];
  piece_bb[board[from]] ^= (SQUARE_BB[from] | SQUARE_BB[to]);
  board[to]   = board[from];
  board[from] = NO_PIECE;
}
