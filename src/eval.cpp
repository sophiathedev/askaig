#include "eval.h"
#include "position.h"
#include "types.h"

namespace eval {

  namespace {

    // --- Pawn structure ---------------------------------------------------------------------

    constexpr Bitboard file_bb(int f) { return UINT64_C(0x0101010101010101) << f; }
    constexpr Bitboard rank_bb(int r) { return UINT64_C(0xff) << (8 * r); }

    // Penalties (centipawns) and a passed-pawn bonus indexed by the pawn's own rank of advancement
    // (rank 1 = just left home ... rank 6 = one step from promotion).
    constexpr int DOUBLED_PENALTY  = 12;
    constexpr int ISOLATED_PENALTY = 15;
    // Passed-pawn bonus, indexed by rank of advancement, tapered: passers are worth far more in the
    // endgame (often decisive) than the middlegame (where pieces blockade/round them up).
    constexpr int PASSED_MG[8] = {0, 5, 10, 15, 25, 40, 60, 0};
    constexpr int PASSED_EG[8] = {0, 15, 25, 40, 65, 100, 150, 0};
    // Passed-pawn refinements: a blockaded passer (any piece on its stop square) keeps only 2/3 of
    // the bonus, and in the endgame the kings join the race — per square of Chebyshev distance from
    // the stop square, the ENEMY king being far is worth +5 eg and our own king being far costs -2
    // eg, weighted by how advanced the passer is (irrelevant for a pawn still at home, decisive on
    // the 7th). Starting values for SPRT tuning.
    constexpr int PASSED_KDIST_W[8] = {0, 0, 0, 1, 2, 3, 5, 0}; // by rank of advancement
    constexpr int CENTERED_BONUS    = 10; // pawn on a central square (d4/e4/d5/e5)
    constexpr int OUTPOST_BONUS     = 25; // knight on a hole, defended by a pawn, in advanced ranks

    // King safety: penalty per missing pawn-shield file and per open/semi-open file by the king,
    // scaled by the opponent's attacking material (KS_MAX_POWER = full middlegame) so it fades out
    // in the endgame, where the king should instead be active.
    constexpr int SHIELD_DEFICIT    = 18;
    constexpr int OPEN_FILE_PENALTY = 20;
    constexpr int KS_MAX_POWER      = 12;

    // Mobility: bonus per safe square a piece can move to (not onto own pieces or enemy-pawn-
    // controlled squares), indexed by PieceType and tapered — sliders are worth more per square in
    // the (open) endgame. Pinned: penalty per piece pinned to its own king.
    constexpr int MOB_MG[NPIECE_TYPES] = {0, 4, 3, 2, 1, 0}; // P N B R Q K
    constexpr int MOB_EG[NPIECE_TYPES] = {0, 4, 5, 4, 2, 0};
    constexpr int PIN_PENALTY          = 18;

    // Piece bonuses: the bishop pair (worth more in the open endgame), a rook on a fully open or
    // semi-open (no friendly pawn) file, and a rook on the relative 7th rank (decisive in endgames).
    constexpr int BISHOP_PAIR_MG = 25;
    constexpr int BISHOP_PAIR_EG = 45;
    constexpr int ROOK_OPEN      = 25; // file with no pawns at all
    constexpr int ROOK_SEMIOPEN  = 12; // file with no friendly pawns
    constexpr int ROOK_7TH_MG    = 15;
    constexpr int ROOK_7TH_EG    = 35;

    // A middlegame/endgame score pair and its interpolation by game phase (PHASE_MAX = middlegame).
    struct Score {
      int mg = 0;
      int eg = 0;
    };

    // Threats: a piece attacking a more (or equally) valuable enemy piece, and "hanging" pieces (a
    // piece we attack that the enemy does not defend). Tapered. Values are deliberately modest and
    // SHOULD be tuned by SPRT self-play (tools/sprt.sh) — these are reasonable starting points, not
    // proven optima. Indexed by the *threatened* (victim) piece type; bigger victims = bigger threat.
    constexpr Score THREAT_BY_PAWN                = {80, 55}; // a pawn attacks any enemy minor/rook/queen
    constexpr Score THREAT_BY_MINOR[NPIECE_TYPES] = {{0, 0},   {22, 35}, {24, 35},
                                                     {45, 50}, {48, 55}, {0, 0}}; // victim: P N B R Q K
    constexpr Score THREAT_BY_ROOK[NPIECE_TYPES]  = {{0, 0},  {20, 40}, {22, 40},
                                                     {0, 20}, {40, 40}, {0, 0}}; // rooks chiefly threaten R/Q
    constexpr Score HANGING                       = {28, 22}; // per undefended enemy piece we attack

    // King-zone attacks: each knight/bishop/rook/queen attack into the ring around the enemy king
    // (the king's square + its 8 neighbours) earns weight units; the total is then scaled by the
    // NUMBER of distinct attacking pieces — one piece alone cannot mate, so danger grows steeply as
    // attackers join (0% for <2 attackers). Score = units * scale% * KING_ATT_UNIT cp, mostly a
    // middlegame term (quartered in the endgame). Starting values — to be SPRT-tuned like threats.
    constexpr int KING_ATT_WEIGHT[NPIECE_TYPES] = {0, 2, 2, 3, 5, 0}; // P N B R Q K (per zone square)
    constexpr int KING_ATT_SCALE[8]             = {0, 0, 50, 75, 88, 94, 97, 99}; // % by attacker count
    constexpr int KING_ATT_UNIT                 = 7; // cp per weighted unit (after the % scale)

    // Tempo: a small bonus for simply being the side to move (the mover usually has the option to
    // improve their position). Also damps eval oscillation between plies. Starting values for SPRT.
    constexpr Score TEMPO = {18, 10};

    [[gnu::const, gnu::always_inline]] inline int taper(Score s, int phase) noexcept {
      return (s.mg * phase + s.eg * (psqt::PHASE_MAX - phase)) / psqt::PHASE_MAX;
    }

    // Central squares (d4/e4/d5/e5): a pawn here gets a small "centered pawn" bonus.
    constexpr Bitboard CENTER = (file_bb(3) | file_bb(4)) & (rank_bb(3) | rank_bb(4));

    // Precomputed (at compile time) pawn masks:
    //  - adjacent[f]: the files either side of file f (a pawn is isolated if no friendly pawn is here)
    //  - passed[color][sq]: the squares an enemy pawn must avoid for the pawn on sq to be passed
    //  - span[color][sq]: the pawn attack span — squares on adjacent files ahead of sq that the
    //    pawn (or its advanced self) could attack. A square not in either side's span is a "hole".
    struct PawnMasks {
      Bitboard adjacent[8]{};
      Bitboard passed[NCOLORS][NSQUARES]{};
      Bitboard span[NCOLORS][NSQUARES]{};
      Bitboard king_shield[NCOLORS][NSQUARES]{}; // pawn-shield squares (3 files, 2 ranks in front)

      constexpr PawnMasks() {
        for (int f = 0; f < 8; ++f)
          adjacent[f] = (f > 0 ? file_bb(f - 1) : 0) | (f < 7 ? file_bb(f + 1) : 0);
        for (int sq = 0; sq < 64; ++sq) {
          int      f     = sq & 7;
          int      r     = sq >> 3;
          Bitboard files = file_bb(f) | adjacent[f];
          Bitboard above = 0;
          Bitboard below = 0;
          for (int rr = r + 1; rr < 8; ++rr)
            above |= rank_bb(rr);
          for (int rr = r - 1; rr >= 0; --rr)
            below |= rank_bb(rr);
          passed[WHITE][sq]      = files & above; // white promotes upward
          passed[BLACK][sq]      = files & below; // black promotes downward
          span[WHITE][sq]        = adjacent[f] & above;
          span[BLACK][sq]        = adjacent[f] & below;
          Bitboard wFront        = (r + 1 < 8 ? rank_bb(r + 1) : 0) | (r + 2 < 8 ? rank_bb(r + 2) : 0);
          Bitboard bFront        = (r - 1 >= 0 ? rank_bb(r - 1) : 0) | (r - 2 >= 0 ? rank_bb(r - 2) : 0);
          king_shield[WHITE][sq] = files & wFront;
          king_shield[BLACK][sq] = files & bFront;
        }
      }
    };
    constexpr PawnMasks PM;

    // Chebyshev (king-move) distance between two squares.
    [[gnu::const, gnu::always_inline]] inline int cheb(Square a, Square b) noexcept {
      int df = static_cast<int>(file_of(a)) - static_cast<int>(file_of(b));
      int dr = static_cast<int>(rank_of(a)) - static_cast<int>(rank_of(b));
      if (df < 0)
        df = -df;
      if (dr < 0)
        dr = -dr;
      return df > dr ? df : dr;
    }

    // Evaluates pawn structure (doubled, isolated, passed) from White's perspective. The passed-pawn
    // bonus is tapered by `phase`; the other terms are phase-independent.
    [[gnu::pure, gnu::hot]] int pawn_structure(const Position &pos, int phase) noexcept {
      const Bitboard wp = pos.bitboard_of(WHITE, PAWN);
      const Bitboard bp = pos.bitboard_of(BLACK, PAWN);
      int            s  = 0;

      for (int f = 0; f < 8; ++f) {
        int wc = pop_count(wp & file_bb(f));
        int bc = pop_count(bp & file_bb(f));
        if (wc > 1)
          s -= DOUBLED_PENALTY * (wc - 1);
        if (bc > 1)
          s += DOUBLED_PENALTY * (bc - 1);
        if (wc > 0 && !(wp & PM.adjacent[f]))
          s -= ISOLATED_PENALTY * wc;
        if (bc > 0 && !(bp & PM.adjacent[f]))
          s += ISOLATED_PENALTY * bc;
      }

      // Passed pawns: the rank-scaled base bonus, reduced when the stop square is blocked, with an
      // endgame king-race adjustment (see PASSED_KDIST_W above).
      const Bitboard occ = pos.all_pieces<WHITE>() | pos.all_pieces<BLACK>();
      const Square   wk  = bsf(pos.bitboard_of(WHITE, KING));
      const Square   bk  = bsf(pos.bitboard_of(BLACK, KING));

      Bitboard w = wp;
      while (w) {
        Square sq = pop_lsb(&w);
        if (!(bp & PM.passed[WHITE][sq])) {
          const int    r    = rank_of(sq);
          const Square stop = sq + NORTH;
          Score        ps{PASSED_MG[r], PASSED_EG[r]};
          if (occ & SQUARE_BB[stop])
            ps.mg = ps.mg * 2 / 3, ps.eg = ps.eg * 2 / 3; // blockaded: it cannot advance
          ps.eg += (5 * cheb(bk, stop) - 2 * cheb(wk, stop)) * PASSED_KDIST_W[r];
          s += taper(ps, phase);
        }
      }
      Bitboard b = bp;
      while (b) {
        Square sq = pop_lsb(&b);
        if (!(wp & PM.passed[BLACK][sq])) {
          const int    r    = 7 - rank_of(sq);
          const Square stop = sq + SOUTH;
          Score        ps{PASSED_MG[r], PASSED_EG[r]};
          if (occ & SQUARE_BB[stop])
            ps.mg = ps.mg * 2 / 3, ps.eg = ps.eg * 2 / 3;
          ps.eg += (5 * cheb(wk, stop) - 2 * cheb(bk, stop)) * PASSED_KDIST_W[r];
          s -= taper(ps, phase);
        }
      }

      // Centered pawns.
      s += CENTERED_BONUS * pop_count(wp & CENTER);
      s -= CENTERED_BONUS * pop_count(bp & CENTER);

      // Holes / knight outposts. A square is a hole for one side if no pawn of that side can ever
      // attack it (not in its pawn attack span). A knight sitting on the enemy's hole — defended by
      // a friendly pawn, in advanced ranks — is a strong outpost.
      Bitboard wspan = 0;
      Bitboard bspan = 0;
      for (Bitboard t = wp; t;)
        wspan |= PM.span[WHITE][pop_lsb(&t)];
      for (Bitboard t = bp; t;)
        bspan |= PM.span[BLACK][pop_lsb(&t)];

      for (Bitboard wn = pos.bitboard_of(WHITE, KNIGHT); wn;) {
        Square sq = pop_lsb(&wn);
        if (rank_of(sq) >= RANK4 && rank_of(sq) <= RANK6 && !(bspan & SQUARE_BB[sq]) &&
            (wp & pawn_attacks<BLACK>(SQUARE_BB[sq])))
          s += OUTPOST_BONUS;
      }
      for (Bitboard bn = pos.bitboard_of(BLACK, KNIGHT); bn;) {
        Square sq = pop_lsb(&bn);
        if (rank_of(sq) <= RANK5 && rank_of(sq) >= RANK3 && !(wspan & SQUARE_BB[sq]) &&
            (bp & pawn_attacks<WHITE>(SQUARE_BB[sq])))
          s -= OUTPOST_BONUS;
      }
      return s;
    }

    // Attacking material of side `c`, used to scale how much the enemy king's exposure matters.
    [[gnu::pure]] int attack_power(const Position &pos, Color c) noexcept {
      return pop_count(pos.bitboard_of(c, KNIGHT)) + pop_count(pos.bitboard_of(c, BISHOP)) +
             2 * pop_count(pos.bitboard_of(c, ROOK)) + 4 * pop_count(pos.bitboard_of(c, QUEEN));
    }

    // Unscaled danger for `c`'s king: missing pawn-shield files and open/semi-open files beside it.
    [[gnu::pure]] int king_danger(Square ks, Bitboard pawns, Color c) noexcept {
      const int kf      = file_of(ks);
      const int shield  = pop_count(pawns & PM.king_shield[c][ks]);
      int       penalty = SHIELD_DEFICIT * (shield < 3 ? 3 - shield : 0);
      const int lo      = kf > 0 ? kf - 1 : 0;
      const int hi      = kf < 7 ? kf + 1 : 7;
      for (int f = lo; f <= hi; ++f)
        if (!(pawns & file_bb(f)))
          penalty += OPEN_FILE_PENALTY;
      return penalty;
    }

    // King safety from White's perspective: each king's danger scaled by the opponent's attacking
    // material (so it vanishes once the heavy pieces are traded off).
    [[gnu::pure, gnu::hot]] int king_safety(const Position &pos) noexcept {
      const Square   wk    = bsf(pos.bitboard_of(WHITE, KING));
      const Square   bk    = bsf(pos.bitboard_of(BLACK, KING));
      const Bitboard wp    = pos.bitboard_of(WHITE, PAWN);
      const Bitboard bp    = pos.bitboard_of(BLACK, PAWN);
      const int      bPow  = attack_power(pos, BLACK);
      const int      wPow  = attack_power(pos, WHITE);
      const int      wScal = bPow < KS_MAX_POWER ? bPow : KS_MAX_POWER;
      const int      bScal = wPow < KS_MAX_POWER ? wPow : KS_MAX_POWER;

      int s = 0;
      s -= king_danger(wk, wp, WHITE) * wScal / KS_MAX_POWER;
      s += king_danger(bk, bp, BLACK) * bScal / KS_MAX_POWER;
      return s;
    }

    // (Mobility is computed together with the threat attack-maps in one shared pass over the pieces —
    // see `mobility_threats` below — so each piece's attack set is looked up only once.)

    // The bitboard of `C`'s pieces pinned against their own king by an enemy slider.
    template<Color C>
    [[gnu::pure]] Bitboard pinned_of(const Position &pos) noexcept {
      const Square   ks  = bsf(pos.bitboard_of(C, KING));
      const Bitboard us  = pos.all_pieces<C>();
      const Bitboard occ = us | pos.all_pieces<~C>();
      Bitboard       pin = 0;

      Bitboard pinners = get_xray_rook_attacks(ks, occ, us) & pos.orthogonal_sliders<~C>();
      while (pinners)
        pin |= SQUARES_BETWEEN_BB[ks][pop_lsb(&pinners)] & us;
      pinners = get_xray_bishop_attacks(ks, occ, us) & pos.diagonal_sliders<~C>();
      while (pinners)
        pin |= SQUARES_BETWEEN_BB[ks][pop_lsb(&pinners)] & us;
      return pin;
    }

    [[gnu::pure, gnu::hot]] int pin_penalty(const Position &pos) noexcept {
      return PIN_PENALTY * (pop_count(pinned_of<BLACK>(pos)) - pop_count(pinned_of<WHITE>(pos)));
    }

    // Bishop pair + rook placement (open/semi-open files, 7th rank), White's perspective, tapered.
    [[gnu::pure, gnu::hot]] int piece_bonuses(const Position &pos, int phase) noexcept {
      const Bitboard wp = pos.bitboard_of(WHITE, PAWN);
      const Bitboard bp = pos.bitboard_of(BLACK, PAWN);
      Score          s;

      if (pop_count(pos.bitboard_of(WHITE, BISHOP)) >= 2)
        s.mg += BISHOP_PAIR_MG, s.eg += BISHOP_PAIR_EG;
      if (pop_count(pos.bitboard_of(BLACK, BISHOP)) >= 2)
        s.mg -= BISHOP_PAIR_MG, s.eg -= BISHOP_PAIR_EG;

      for (Bitboard r = pos.bitboard_of(WHITE, ROOK); r;) {
        Square   sq = pop_lsb(&r);
        Bitboard fl = file_bb(file_of(sq));
        if (!((wp | bp) & fl))
          s.mg += ROOK_OPEN, s.eg += ROOK_OPEN;
        else if (!(wp & fl))
          s.mg += ROOK_SEMIOPEN, s.eg += ROOK_SEMIOPEN;
        if (rank_of(sq) == RANK7)
          s.mg += ROOK_7TH_MG, s.eg += ROOK_7TH_EG;
      }
      for (Bitboard r = pos.bitboard_of(BLACK, ROOK); r;) {
        Square   sq = pop_lsb(&r);
        Bitboard fl = file_bb(file_of(sq));
        if (!((wp | bp) & fl))
          s.mg -= ROOK_OPEN, s.eg -= ROOK_OPEN;
        else if (!(bp & fl))
          s.mg -= ROOK_SEMIOPEN, s.eg -= ROOK_SEMIOPEN;
        if (rank_of(sq) == RANK2)
          s.mg -= ROOK_7TH_MG, s.eg -= ROOK_7TH_EG;
      }
      return taper(s, phase);
    }

    // The squares attacked by side C, split by attacker tier (pawns / minors / rooks) plus the full
    // union (`all`, used to decide whether an enemy piece is defended). Computed once per side.
    struct AttackMap {
      Bitboard pawn  = 0;
      Bitboard minor = 0; // knights + bishops
      Bitboard rook  = 0;
      Bitboard all   = 0; // every attacker, including queens and the king
    };

    // One pass over side C's pieces, shared by mobility, the threat attack-map AND the king-zone
    // attack count: every knight / bishop / rook / queen attack set (the magic-bitboard lookups —
    // the dominant eval cost) is looked up ONCE, then used to (1) count mobility over `targets`,
    // (2) OR into the tiered attack map, and (3) count attacks into `kingZone` (the ring around the
    // ENEMY king) for the king-attack term. `ownPawnAtt` is C's pawn attacks (the map's pawn tier).
    // Previously mobility() and attack_maps() each ran this loop independently, doubling the slider
    // lookups per eval.
    template<Color C>
    [[gnu::hot]] void side_mob_att(const Position &pos, Bitboard occ, Bitboard targets, Bitboard ownPawnAtt,
                                   Bitboard kingZone, Score &mob, AttackMap &m, int &kzUnits,
                                   int &kzAttackers) noexcept {
      m.pawn = ownPawnAtt;
      for (Bitboard b = pos.bitboard_of(C, KNIGHT); b;) {
        Bitboard a = attacks<KNIGHT>(pop_lsb(&b), occ);
        m.minor |= a;
        int c = pop_count(a & targets);
        mob.mg += MOB_MG[KNIGHT] * c, mob.eg += MOB_EG[KNIGHT] * c;
        if (Bitboard z = a & kingZone) {
          kzUnits += KING_ATT_WEIGHT[KNIGHT] * pop_count(z);
          ++kzAttackers;
        }
      }
      for (Bitboard b = pos.bitboard_of(C, BISHOP); b;) {
        Bitboard a = attacks<BISHOP>(pop_lsb(&b), occ);
        m.minor |= a;
        int c = pop_count(a & targets);
        mob.mg += MOB_MG[BISHOP] * c, mob.eg += MOB_EG[BISHOP] * c;
        if (Bitboard z = a & kingZone) {
          kzUnits += KING_ATT_WEIGHT[BISHOP] * pop_count(z);
          ++kzAttackers;
        }
      }
      for (Bitboard b = pos.bitboard_of(C, ROOK); b;) {
        Bitboard a = attacks<ROOK>(pop_lsb(&b), occ);
        m.rook |= a;
        int c = pop_count(a & targets);
        mob.mg += MOB_MG[ROOK] * c, mob.eg += MOB_EG[ROOK] * c;
        if (Bitboard z = a & kingZone) {
          kzUnits += KING_ATT_WEIGHT[ROOK] * pop_count(z);
          ++kzAttackers;
        }
      }
      Bitboard queens = 0;
      for (Bitboard b = pos.bitboard_of(C, QUEEN); b;) {
        Bitboard a = attacks<QUEEN>(pop_lsb(&b), occ);
        queens |= a;
        int c = pop_count(a & targets);
        mob.mg += MOB_MG[QUEEN] * c, mob.eg += MOB_EG[QUEEN] * c;
        if (Bitboard z = a & kingZone) {
          kzUnits += KING_ATT_WEIGHT[QUEEN] * pop_count(z);
          ++kzAttackers;
        }
      }
      m.all = m.pawn | m.minor | m.rook | queens | attacks<KING>(bsf(pos.bitboard_of(C, KING)), occ);
    }

    // Threat bonus for side C against ~C: pieces of ~C attacked by our pawns / minors / rooks, plus
    // any enemy piece we attack that ~C does not defend (`theyDefend` = ~C's full attack map).
    template<Color C>
    [[gnu::pure, gnu::hot]] Score side_threats(const Position &pos, const AttackMap &us, Bitboard theyDefend) noexcept {
      const Color    Them        = ~C;
      const Bitboard theirMinors = pos.bitboard_of(Them, KNIGHT) | pos.bitboard_of(Them, BISHOP);
      const Bitboard theirR      = pos.bitboard_of(Them, ROOK);
      const Bitboard theirQ      = pos.bitboard_of(Them, QUEEN);
      const Bitboard theirPieces = theirMinors | theirR | theirQ; // enemy non-pawn, non-king pieces

      Score s;
      // Pawn threats: any enemy piece attacked by one of our pawns (flat — the pawn is cheap).
      s.mg += THREAT_BY_PAWN.mg * pop_count(us.pawn & theirPieces);
      s.eg += THREAT_BY_PAWN.eg * pop_count(us.pawn & theirPieces);
      // Minor-piece threats, by the threatened piece's type.
      for (Bitboard t = us.minor & theirPieces; t;) {
        const int vt = type_of(pos.at(pop_lsb(&t)));
        s.mg += THREAT_BY_MINOR[vt].mg, s.eg += THREAT_BY_MINOR[vt].eg;
      }
      // Rook threats (chiefly enemy rooks/queens, but minors too).
      for (Bitboard t = us.rook & theirPieces; t;) {
        const int vt = type_of(pos.at(pop_lsb(&t)));
        s.mg += THREAT_BY_ROOK[vt].mg, s.eg += THREAT_BY_ROOK[vt].eg;
      }
      // Hanging: enemy pieces we attack that they do not defend.
      const int hung = pop_count(theirPieces & us.all & ~theyDefend);
      s.mg += HANGING.mg * hung, s.eg += HANGING.eg * hung;
      return s;
    }

    // Mobility + threats + king-zone attacks combined, White's perspective, tapered by game phase.
    // The three terms share a single pass over the pieces (see `side_mob_att`), so the per-piece
    // attack lookups happen once.
    [[gnu::hot]] int piece_activity(const Position &pos, int phase) noexcept {
      const Bitboard wpieces  = pos.all_pieces<WHITE>();
      const Bitboard bpieces  = pos.all_pieces<BLACK>();
      const Bitboard occ      = wpieces | bpieces;
      const Bitboard wPawnAtt = pawn_attacks<WHITE>(pos.bitboard_of(WHITE, PAWN));
      const Bitboard bPawnAtt = pawn_attacks<BLACK>(pos.bitboard_of(BLACK, PAWN));

      // King zones: the king's square plus its 8 neighbours (the king attack table is occupancy-
      // independent). White's pieces are scored against BLACK's zone and vice versa.
      const Square   wk     = bsf(pos.bitboard_of(WHITE, KING));
      const Square   bk     = bsf(pos.bitboard_of(BLACK, KING));
      const Bitboard wkZone = attacks<KING>(wk, occ) | SQUARE_BB[wk];
      const Bitboard bkZone = attacks<KING>(bk, occ) | SQUARE_BB[bk];

      Score     wmob, bmob;
      AttackMap wm, bm;
      int       wU = 0, wC = 0, bU = 0, bC = 0; // king-zone units / distinct attackers, per side
      side_mob_att<WHITE>(pos, occ, ~wpieces & ~bPawnAtt, wPawnAtt, bkZone, wmob, wm, wU, wC);
      side_mob_att<BLACK>(pos, occ, ~bpieces & ~wPawnAtt, bPawnAtt, wkZone, bmob, bm, bU, bC);

      const int   mob = taper({wmob.mg - bmob.mg, wmob.eg - bmob.eg}, phase);
      const Score sw  = side_threats<WHITE>(pos, wm, bm.all);
      const Score sb  = side_threats<BLACK>(pos, bm, wm.all);
      const int   thr = taper({sw.mg - sb.mg, sw.eg - sb.eg}, phase);

      // King-zone attack score: weighted units, gated/scaled by the attacker count (a lone attacker
      // scores nothing), in centipawns. Mostly middlegame — quartered in the endgame.
      const int katt_w = wU * KING_ATT_SCALE[wC < 7 ? wC : 7] / 100 * KING_ATT_UNIT;
      const int katt_b = bU * KING_ATT_SCALE[bC < 7 ? bC : 7] / 100 * KING_ATT_UNIT;
      const int katt   = taper({katt_w - katt_b, (katt_w - katt_b) / 4}, phase);

      return mob + thr + katt;
    }

  } // namespace

  [[gnu::pure]] bool is_passed_pawn(const Position &pos, Color c, Square sq) noexcept {
    return !(pos.bitboard_of(~c, PAWN) & PM.passed[c][sq]);
  }

  // A full static evaluation. NOTE: there used to be a thread_local Zobrist-keyed cache of this whole
  // result, but with the incremental material+PST and the (now single-pass) attack maps the eval is
  // cheap enough that the 2 MiB/thread cache cost more (CPU-cache pressure + per-call lookup/store)
  // than its ~5% hit rate saved — removing it measured +6-8% nps with identical node counts. So the
  // eval is recomputed every call.
  [[gnu::hot]] int evaluate(const Position &pos) noexcept {
    // Taper the (incremental) material + piece-square score between the middlegame and endgame
    // tables by the game phase (full board -> middlegame, few pieces -> endgame).
    int phase = pop_count(pos.bitboard_of(WHITE, KNIGHT) | pos.bitboard_of(BLACK, KNIGHT)) +
                pop_count(pos.bitboard_of(WHITE, BISHOP) | pos.bitboard_of(BLACK, BISHOP)) +
                2 * pop_count(pos.bitboard_of(WHITE, ROOK) | pos.bitboard_of(BLACK, ROOK)) +
                4 * pop_count(pos.bitboard_of(WHITE, QUEEN) | pos.bitboard_of(BLACK, QUEEN));
    if (phase > psqt::PHASE_MAX)
      phase = psqt::PHASE_MAX; // promotions can exceed the starting material

    int s = (pos.psqt_mg() * phase + pos.psqt_eg() * (psqt::PHASE_MAX - phase)) / psqt::PHASE_MAX;
    s += pawn_structure(pos, phase);
    s += king_safety(pos);
    s += piece_activity(pos, phase); // mobility + threats + king-zone attacks, one shared attack pass
    s += pin_penalty(pos);
    s += piece_bonuses(pos, phase);

    // White-perspective score flipped to the side to move, plus the tempo bonus for that side.
    return (pos.turn() == WHITE ? s : -s) + taper(TEMPO, phase);
  }

} // namespace eval
