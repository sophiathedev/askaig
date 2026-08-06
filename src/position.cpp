#include "position.h"
#include <sstream>
#include "tables.h"

uint64_t zobrist::zobrist_table[NPIECES][NSQUARES];
uint64_t zobrist::zobrist_side;

void zobrist::initialise_zobrist_keys() {
  PRNG rng(70026072);
  for (auto &i: zobrist::zobrist_table)
    for (auto &key: i)
      key = rng.rand<uint64_t>();
  zobrist::zobrist_side = rng.rand<uint64_t>();
}

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

bool Position::set(const std::string &fen, Position &p) {
  const size_t sp = fen.find(' ');
  if (sp == std::string::npos)
    return false;

  int square = a8;
  for (char ch: fen.substr(0, sp)) {
    if (isdigit(ch)) {
      if (ch == '0' || ch - '0' > 8)
        return false; // fen runs are 1..8
      square += (ch - '0') * EAST;
    } else if (ch == '/') {
      square += 2 * SOUTH;
    } else {
      const size_t idx = PIECE_STR.find(ch);
      if (idx == std::string::npos || idx == 6 || idx == 7 || idx == 14)
        return false;
      if (square < a1 || square > h8)
        return false; // the board field ran past 64 squares
      p.put_piece(Piece(idx), Square(square++));
    }
  }

  std::istringstream ss(fen.substr(sp));
  std::string        side;
  std::string        castling;
  std::string        ep;
  ss >> side >> castling >> ep;

  p.side_to_play = side == "w" ? WHITE : BLACK;
  if (p.side_to_play == BLACK)
    p.hash ^= zobrist::zobrist_side; // root side key

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

  int halfmove = 0;
  if (ss >> halfmove && halfmove > 0)
    p.history[p.game_ply].fifty = halfmove;
  else
    p.history[p.game_ply].fifty = 0;

  p.history[p.game_ply].hash = p.hash;
  return true;
}
