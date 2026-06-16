#include "kpk.h"

#include <bitset>
#include <vector>

#include "simd.h" // pop_lsb
#include "tables.h" // KING_ATTACKS, PAWN_ATTACKS

namespace kpk {

  namespace {

    // The pawn has 24 legal squares (files a-d × ranks 2-7); the full index also encodes both kings
    // (6 bits each) and the side to move (1 bit). 2 * 24 * 64 * 64 = 196608 positions, one bit each.
    constexpr unsigned MAX_INDEX = 2 * 24 * 64 * 64;

    std::bitset<MAX_INDEX> Bitbase;

    // index = wksq | bksq<<6 | stm<<12 | file(psq)<<13 | (rank(psq)-RANK2)<<15  (Stockfish layout).
    [[nodiscard]] unsigned index(Color stm, Square bksq, Square wksq, Square psq) noexcept {
      return unsigned(wksq) | (unsigned(bksq) << 6) | (unsigned(stm) << 12) | (unsigned(file_of(psq)) << 13) |
             (unsigned(int(rank_of(psq)) - int(RANK2)) << 15);
    }

    // A 2-bit lattice so positions resolve by OR-ing successors: WIN dominates for the pawn's side,
    // DRAW for the defender; INVALID (0) contributes nothing, UNKNOWN keeps a position pending.
    enum Res { INVALID = 0, UNKNOWN = 1, DRAW = 2, WIN = 4 };

    // King-distance (Chebyshev).
    [[nodiscard]] int kdist(Square a, Square b) noexcept {
      int df = int(file_of(a)) - int(file_of(b));
      int dr = int(rank_of(a)) - int(rank_of(b));
      df     = df < 0 ? -df : df;
      dr     = dr < 0 ? -dr : dr;
      return df > dr ? df : dr;
    }

    struct KPK {
      Color  stm;
      Square wk, bk, psq;
      Res    result;

      explicit KPK(unsigned idx) noexcept {
        wk  = Square(idx & 0x3F);
        bk  = Square((idx >> 6) & 0x3F);
        stm = Color((idx >> 12) & 0x01);
        psq = create_square(File((idx >> 13) & 0x3), Rank(int(RANK2) + int((idx >> 15) & 0x7)));

        const Square push = Square(int(psq) + NORTH); // the pawn's stop square

        if (kdist(wk, bk) <= 1 || wk == psq || bk == psq ||
            (stm == WHITE && (PAWN_ATTACKS[WHITE][psq] & SQUARE_BB[bk])))
          result = INVALID; // kings adjacent / overlap, or White-to-move with Black's king en prise
        else if (stm == WHITE && rank_of(psq) == RANK7 && wk != push && (kdist(bk, push) > 1 || kdist(wk, push) == 1))
          result = WIN; // promote safely
        else if (stm == BLACK && (!(KING_ATTACKS[bk] & ~(KING_ATTACKS[wk] | PAWN_ATTACKS[WHITE][psq])) || // stalemate
                                  (KING_ATTACKS[bk] & ~KING_ATTACKS[wk] & SQUARE_BB[psq]))) // king grabs the pawn
          result = DRAW;
        else
          result = UNKNOWN;
      }

      // Resolve from the already-classified successors. White-to-move wins if ANY child wins; Black-to-
      // move draws if ANY child draws. Illegal children index INVALID (0) and drop out of the OR.
      Res classify(const std::vector<KPK> &db) noexcept {
        const Res Good = stm == WHITE ? WIN : DRAW;
        const Res Bad  = stm == WHITE ? DRAW : WIN;
        unsigned  r    = INVALID;

        for (Bitboard b = KING_ATTACKS[stm == WHITE ? wk : bk]; b;) {
          const Square to = pop_lsb(&b);
          r |= stm == WHITE ? db[index(BLACK, bk, to, psq)].result : db[index(WHITE, to, wk, psq)].result;
        }
        if (stm == WHITE) {
          if (rank_of(psq) < RANK7) // single push
            r |= db[index(BLACK, bk, wk, Square(int(psq) + NORTH))].result;
          const Square push2 = Square(int(psq) + NORTH + NORTH);
          if (rank_of(psq) == RANK2 && Square(int(psq) + NORTH) != wk && Square(int(psq) + NORTH) != bk &&
              push2 != wk && push2 != bk) // double push
            r |= db[index(BLACK, bk, wk, push2)].result;
        }
        result = (r & Good) ? Good : (r & UNKNOWN) ? UNKNOWN : Bad;
        return result;
      }
    };

  } // namespace

  void init() {
    std::vector<KPK> db;
    db.reserve(MAX_INDEX);
    for (unsigned i = 0; i < MAX_INDEX; ++i)
      db.emplace_back(i);

    // Fixed-point iteration: keep resolving UNKNOWN positions until none change.
    for (bool changed = true; changed;) {
      changed = false;
      for (unsigned i = 0; i < MAX_INDEX; ++i)
        if (db[i].result == UNKNOWN && db[i].classify(db) != UNKNOWN)
          changed = true;
    }

    for (unsigned i = 0; i < MAX_INDEX; ++i)
      if (db[i].result == WIN)
        Bitbase.set(i);
  }

  bool probe(Square wksq, Square wpsq, Square bksq, Color stm) { return Bitbase[index(stm, bksq, wksq, wpsq)]; }

} // namespace kpk
