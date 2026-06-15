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
    int DOUBLED_PENALTY  = 16; // tunable (see eval::params)
    int ISOLATED_PENALTY = 10; // tunable
    // Passed-pawn bonus, indexed by rank of advancement, tapered: passers are worth far more in the
    // endgame (often decisive) than the middlegame (where pieces blockade/round them up).
    int PASSED_MG[8] = {0, 2, -31, 0, 22, 59, 115, 0}; // tunable [1..6]
    int PASSED_EG[8] = {0, 6, 28, 25, 37, 1, -33, 0}; // tunable [1..6]
    // Passed-pawn refinements: a blockaded passer (any piece on its stop square) keeps only 2/3 of
    // the bonus, and in the endgame the kings join the race — per square of Chebyshev distance from
    // the stop square, the ENEMY king being far is worth +5 eg and our own king being far costs -2
    // eg, weighted by how advanced the passer is (irrelevant for a pawn still at home, decisive on
    // the 7th). Starting values for SPRT tuning.
    int PASSED_KDIST_W[8] = {0, 0, 0, 2, 4, 8, 7, 0}; // by rank of advancement; tunable [3..6]
    int CENTERED_BONUS    = -8; // pawn on a central square (d4/e4/d5/e5); tunable
    int OUTPOST_BONUS     = 12; // knight on a hole, defended by a pawn, in advanced ranks; tunable
    // Connected pawns (part of a phalanx OR defended by a friendly pawn), bonus by rank of
    // advancement — a connected chain is harder to break and far stronger as it nears promotion.
    int CONNECTED[8] = {0, -2, -1, 5, 17, 54, 110, 0}; // by relative rank; Texel-tuned, tunable [1..6]
    // Backward pawn: no friendly pawn level-or-behind on an adjacent file to support it, and its
    // advance square is controlled by an enemy pawn — a lasting weakness on a (semi-)open file.
    int BACKWARD_PENALTY = 16; // Texel-tuned; tunable

    // King safety: penalty per missing pawn-shield file and per open/semi-open file by the king,
    // scaled by the opponent's attacking material (KS_MAX_POWER = full middlegame) so it fades out
    // in the endgame, where the king should instead be active.
    int           SHIELD_DEFICIT    = 8; // tunable (Texel ~zeroed — safe-check now carries king safety)
    int           OPEN_FILE_PENALTY = 19; // tunable
    constexpr int KS_MAX_POWER      = 12;

    // Mobility: bonus per safe square a piece can move to (not onto own pieces or enemy-pawn-
    // controlled squares), indexed by PieceType and tapered — sliders are worth more per square in
    // the (open) endgame. Pinned: penalty per piece pinned to its own king.
    int MOB_MG[NPIECE_TYPES] = {0, 6, 8, 4, 5, 0}; // P N B R Q K; tunable [N..Q]
    int MOB_EG[NPIECE_TYPES] = {0, 2, 3, 4, 2, 0}; // tunable [N..Q]
    int PIN_PENALTY          = 23; // tunable

    // Piece bonuses: the bishop pair (worth more in the open endgame), a rook on a fully open or
    // semi-open (no friendly pawn) file, and a rook on the relative 7th rank (decisive in endgames).
    int BISHOP_PAIR_MG = 13; // tunable
    int BISHOP_PAIR_EG = 97; // tunable
    int ROOK_OPEN      = 15; // file with no pawns at all; tunable
    int ROOK_SEMIOPEN  = 11; // file with no friendly pawns; tunable
    int ROOK_7TH_MG    = -42; // tunable
    int ROOK_7TH_EG    = 25; // tunable

    // Imbalance (Stockfish-style): a quadratic form over the two sides' piece counts — the marginal
    // value of a piece depends on the rest of the material (knights like having pawns, a second rook is
    // worth less, queen vs minors, …). This is ORTHOGONAL to PeSTO's linear material/PST (which gives
    // every piece a fixed value regardless of the board), so it adds expressiveness PeSTO cannot. Phase-
    // independent. Lower-triangular tables indexed [0=P 1=N 2=B 3=R 4=Q]: IMB_OURS[p1][p2] weights our
    // count of p2 in p1's bonus, IMB_THEIRS[p1][p2] the enemy's. Starting values are SF's (divided by
    // IMB_SCALE at the end); Texel-tuned for this engine. SF's bishop-pair pseudo-piece is dropped — the
    // existing tapered BISHOP_PAIR term already carries it.
    int IMB_OURS[5][5] = {
            {54, 0, 0, 0, 0}, // P:  P
            {111, 1, 0, 0, 0}, // N:  P  N
            {36, 4, -2, 0, 0}, // B:  P  N  B
            {6, 60, 141, 152, 0}, // R:  P  N  B  R
            {34, 98, 145, 33, -6}, // Q:  P  N  B  R  Q
    };
    int IMB_THEIRS[5][5] = {
            {0, 0, 0, 0, 0}, // P
            {67, 0, 0, 0, 0}, // N:  P
            {61, 22, 0, 0, 0}, // B:  P  N
            {115, 32, -24, 0, 0}, // R:  P  N  B
            {245, -14, 187, 161, 0}, // Q:  P  N  B  R
    };
    constexpr int IMB_SCALE = 16; // final divisor (SF convention)

    // A middlegame/endgame score pair and its interpolation by game phase (PHASE_MAX = middlegame).
    struct Score {
      int mg = 0;
      int eg = 0;
    };

    // Threats: a piece attacking a more (or equally) valuable enemy piece, and "hanging" pieces (a
    // piece we attack that the enemy does not defend). Tapered. Values are deliberately modest and
    // SHOULD be tuned by SPRT self-play (tools/sprt.sh) — these are reasonable starting points, not
    // proven optima. Indexed by the *threatened* (victim) piece type; bigger victims = bigger threat.
    Score THREAT_BY_PAWN                = {36, 39}; // a pawn attacks any enemy minor/rook/queen; tunable
    Score THREAT_BY_MINOR[NPIECE_TYPES] = {{0, 0},   {27, 16},  {30, 26},
                                           {40, -3}, {34, -37}, {0, 0}}; // victim: P N B R Q K; tunable [N..Q]
    Score THREAT_BY_ROOK[NPIECE_TYPES]  = {{0, 0},      {13, 9}, {23, 11},
                                           {-338, 579}, {73, 6}, {0, 0}}; // rooks chiefly threaten R/Q; tunable [N..Q]
    Score HANGING                       = {15, 10}; // per undefended enemy piece we attack; tunable
    Score THREAT_BY_KING                = {0, 18}; // undefended enemy piece our king attacks (eg); tunable
    Score THREAT_BY_PUSH                = {12, 9}; // enemy piece a safe pawn push would attack; tunable
    // Rook behind a passed pawn (Tarrasch): a friendly rook on the passer's file behind it supports the
    // push (+); an enemy rook behind it restrains the passer (−). Mostly an endgame term. Used in
    // pawn_structure (piece-dependent, so outside the pawn cache).
    Score ROOK_BEHIND_PASSER = {3, 26}; // tunable

    // Restricted squares: squares we attack that the enemy also attacks but does NOT defend with a
    // pawn — contesting them cramps the enemy's pieces. Small per-square bonus (Stockfish-style).
    Score RESTRICTED = {5, -7}; // tunable (Texel ~neutralised — kept for re-tuning)
    // Space: safe squares in the centre files on our own half of the board (a middlegame term, scaled
    // by our piece count). Squares behind our pawns count double (sheltered territory).
    int SPACE_WEIGHT = 1; // tunable (Texel near-zero on this dataset — space term ~inert)
    // Trapped pieces (penalty MAGNITUDES, subtracted from the side that owns the trapped piece):
    // a corner bishop hemmed by an enemy pawn (a7/h7), and a rook boxed in by its own king on the
    // back rank after castling on that side is already lost.
    Score TRAPPED_BISHOP = {52, 60}; // tunable
    Score TRAPPED_ROOK   = {40, 20}; // tunable

    // King-zone attacks: each knight/bishop/rook/queen attack into the ring around the enemy king
    // (the king's square + its 8 neighbours) earns weight units; the total is then scaled by the
    // NUMBER of distinct attacking pieces — one piece alone cannot mate, so danger grows steeply as
    // attackers join (0% for <2 attackers). Score = units * scale% * KING_ATT_UNIT cp, mostly a
    // middlegame term (quartered in the endgame). Starting values — to be SPRT-tuned like threats.
    int           KING_ATT_WEIGHT[NPIECE_TYPES] = {0, 4, 2, 3, 13, 0}; // P N B R Q K (per zone square); tunable [N..Q]
    constexpr int KING_ATT_SCALE[8]             = {0, 0, 50, 75, 88, 94, 97, 99}; // % by attacker count
    int           KING_ATT_UNIT                 = 6; // cp per weighted unit (after the % scale); tunable

    // Safe checks: a square from which we could check the enemy king that the enemy does NOT defend
    // (so we are not simply recaptured) and we do not already occupy. One of the strongest king-safety
    // signals — counted per checking piece type, in cp, tapered like the king-zone term and NOT gated
    // by attacker count (a lone queen check can be deadly). Starting values for SPRT/Texel tuning.
    int SAFE_CHECK[NPIECE_TYPES] = {0, 21, 16, 39, 49, 0}; // P N B R Q K; tunable [N..Q]

    // Tempo: a small bonus for simply being the side to move (the mover usually has the option to
    // improve their position). Also damps eval oscillation between plies. Starting values for SPRT.
    Score TEMPO = {13, 4}; // tunable

    [[gnu::const, gnu::always_inline]] inline int taper(Score s, int phase) noexcept {
      return (s.mg * phase + s.eg * (psqt::PHASE_MAX - phase)) / psqt::PHASE_MAX;
    }

    // Central squares (d4/e4/d5/e5): a pawn here gets a small "centered pawn" bonus.
    constexpr Bitboard CENTER = (file_bb(3) | file_bb(4)) & (rank_bb(3) | rank_bb(4));

    // Space areas: files c..f on each side's own half (ranks 2-4 for White, 5-7 for Black).
    constexpr Bitboard SPACE_FILES  = file_bb(2) | file_bb(3) | file_bb(4) | file_bb(5);
    constexpr Bitboard SPACE_MASK_W = SPACE_FILES & (rank_bb(1) | rank_bb(2) | rank_bb(3));
    constexpr Bitboard SPACE_MASK_B = SPACE_FILES & (rank_bb(4) | rank_bb(5) | rank_bb(6));

    // The set of files strictly greater / less than file f (for the trapped-rook side test).
    [[gnu::const]] inline Bitboard files_above(int f) noexcept {
      Bitboard r = 0;
      for (int x = f + 1; x < 8; ++x)
        r |= file_bb(x);
      return r;
    }
    [[gnu::const]] inline Bitboard files_below(int f) noexcept {
      Bitboard r = 0;
      for (int x = 0; x < f; ++x)
        r |= file_bb(x);
      return r;
    }

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
      Bitboard back[NCOLORS][NSQUARES]{}; // adjacent files at the pawn's rank and behind it
                                          // (a friendly pawn here can support/advance — not backward)

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
          back[WHITE][sq]        = adjacent[f] & (below | rank_bb(r)); // ranks 0..r on adjacent files
          back[BLACK][sq]        = adjacent[f] & (above | rank_bb(r)); // ranks r..7 on adjacent files
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

    // Pawn cache: the pawn-only parts of pawn_structure (doubled/isolated/centered score, the
    // passed-pawn bitboards, the attack spans) depend ONLY on the two pawn bitboards, which change
    // on a small fraction of moves — so they are cached per thread, keyed by an exact match of
    // BOTH bitboards (like the perft TT: a hit is provably the same structure, never a hash
    // collision). The piece-dependent refinements (passer blockade / king race, knight outposts,
    // the phase taper) are recomputed per call on top. A zeroed entry is the correct answer for
    // the no-pawns position (everything 0), so the zero-initialised table needs no sentinel.
    struct PawnEntry {
      Bitboard wp, bp; // the key (exact match)
      Bitboard wpassed, bpassed; // passed pawns of each side
      Bitboard wspan, bspan; // pawn attack spans (their complement = holes, for outposts)
      int      base; // doubled + isolated + centered, White's perspective (phase-independent)
    };
    constexpr size_t       PAWN_CACHE_SIZE = 8192; // entries (power of two), ~0.5 MiB per thread
    thread_local PawnEntry t_pawn_cache[PAWN_CACHE_SIZE];

    [[gnu::const, gnu::always_inline]] inline size_t pawn_cache_index(Bitboard wp, Bitboard bp) noexcept {
      return ((wp ^ (bp * UINT64_C(0x9E3779B97F4A7C15))) * UINT64_C(0xC2B2AE3D27D4EB4F)) >> 51; // top 13 bits
    }

    // Evaluates pawn structure (doubled, isolated, passed) from White's perspective. The passed-pawn
    // bonus is tapered by `phase`; the other terms are phase-independent.
    [[gnu::hot]] int pawn_structure(const Position &pos, int phase) noexcept {
      const Bitboard wp = pos.bitboard_of(WHITE, PAWN);
      const Bitboard bp = pos.bitboard_of(BLACK, PAWN);

      PawnEntry &e = t_pawn_cache[pawn_cache_index(wp, bp)];
      if (e.wp != wp || e.bp != bp) { // miss: recompute the pawn-only parts into this slot
        e.wp      = wp;
        e.bp      = bp;
        e.wpassed = e.bpassed = 0;
        e.wspan = e.bspan = 0;
        e.base            = 0;

        for (int f = 0; f < 8; ++f) {
          int wc = pop_count(wp & file_bb(f));
          int bc = pop_count(bp & file_bb(f));
          if (wc > 1)
            e.base -= DOUBLED_PENALTY * (wc - 1);
          if (bc > 1)
            e.base += DOUBLED_PENALTY * (bc - 1);
          if (wc > 0 && !(wp & PM.adjacent[f]))
            e.base -= ISOLATED_PENALTY * wc;
          if (bc > 0 && !(bp & PM.adjacent[f]))
            e.base += ISOLATED_PENALTY * bc;
        }

        e.base += CENTERED_BONUS * pop_count(wp & CENTER);
        e.base -= CENTERED_BONUS * pop_count(bp & CENTER);

        // Connected (phalanx neighbour OR pawn-defended) and backward pawns. Both phase-independent.
        const Bitboard wAtt = pawn_attacks<WHITE>(wp);
        const Bitboard bAtt = pawn_attacks<BLACK>(bp);
        for (Bitboard t = wp & (shift<EAST>(wp) | shift<WEST>(wp) | wAtt); t;) {
          const Square sq = pop_lsb(&t);
          e.base += CONNECTED[rank_of(sq)];
        }
        for (Bitboard t = bp & (shift<EAST>(bp) | shift<WEST>(bp) | bAtt); t;) {
          const Square sq = pop_lsb(&t);
          e.base -= CONNECTED[7 - rank_of(sq)];
        }
        for (Bitboard t = wp; t;) {
          const Square sq = pop_lsb(&t);
          const int    f  = file_of(sq);
          if ((wp & PM.adjacent[f]) && !(wp & PM.back[WHITE][sq]) && (bAtt & SQUARE_BB[sq + NORTH]))
            e.base -= BACKWARD_PENALTY;
        }
        for (Bitboard t = bp; t;) {
          const Square sq = pop_lsb(&t);
          const int    f  = file_of(sq);
          if ((bp & PM.adjacent[f]) && !(bp & PM.back[BLACK][sq]) && (wAtt & SQUARE_BB[sq + SOUTH]))
            e.base += BACKWARD_PENALTY;
        }

        for (Bitboard t = wp; t;) {
          const Square sq = pop_lsb(&t);
          if (!(bp & PM.passed[WHITE][sq]))
            e.wpassed |= SQUARE_BB[sq];
          e.wspan |= PM.span[WHITE][sq];
        }
        for (Bitboard t = bp; t;) {
          const Square sq = pop_lsb(&t);
          if (!(wp & PM.passed[BLACK][sq]))
            e.bpassed |= SQUARE_BB[sq];
          e.bspan |= PM.span[BLACK][sq];
        }
      }

      int s = e.base;

      // Passed pawns: the rank-scaled base bonus, reduced when the stop square is blocked, with an
      // endgame king-race adjustment (see PASSED_KDIST_W above). Piece/king-dependent, so this
      // part stays outside the cache.
      const Bitboard occ = pos.all_pieces<WHITE>() | pos.all_pieces<BLACK>();
      const Square   wk  = bsf(pos.bitboard_of(WHITE, KING));
      const Square   bk  = bsf(pos.bitboard_of(BLACK, KING));

      const Bitboard wrooks = pos.bitboard_of(WHITE, ROOK);
      const Bitboard brooks = pos.bitboard_of(BLACK, ROOK);

      Bitboard w = e.wpassed;
      while (w) {
        Square       sq   = pop_lsb(&w);
        const int    r    = rank_of(sq);
        const Square stop = sq + NORTH;
        Score        ps{PASSED_MG[r], PASSED_EG[r]};
        if (occ & SQUARE_BB[stop])
          ps.mg = ps.mg * 2 / 3, ps.eg = ps.eg * 2 / 3; // blockaded: it cannot advance
        ps.eg += (5 * cheb(bk, stop) - 2 * cheb(wk, stop)) * PASSED_KDIST_W[r];
        s += taper(ps, phase);
        const Bitboard behind = file_bb(file_of(sq)) & (SQUARE_BB[sq] - 1); // file squares below the pawn
        if (wrooks & behind)
          s += taper(ROOK_BEHIND_PASSER, phase); // our rook supports the push
        if (brooks & behind)
          s -= taper(ROOK_BEHIND_PASSER, phase); // enemy rook restrains it from behind
      }
      Bitboard b = e.bpassed;
      while (b) {
        Square       sq   = pop_lsb(&b);
        const int    r    = 7 - rank_of(sq);
        const Square stop = sq + SOUTH;
        Score        ps{PASSED_MG[r], PASSED_EG[r]};
        if (occ & SQUARE_BB[stop])
          ps.mg = ps.mg * 2 / 3, ps.eg = ps.eg * 2 / 3;
        ps.eg += (5 * cheb(wk, stop) - 2 * cheb(bk, stop)) * PASSED_KDIST_W[r];
        s -= taper(ps, phase);
        const Bitboard behind = file_bb(file_of(sq)) & ~((SQUARE_BB[sq] - 1) | SQUARE_BB[sq]); // above the pawn
        if (brooks & behind)
          s -= taper(ROOK_BEHIND_PASSER, phase);
        if (wrooks & behind)
          s += taper(ROOK_BEHIND_PASSER, phase);
      }

      // Holes / knight outposts. A square is a hole for one side if no pawn of that side can ever
      // attack it (not in its pawn attack span). A knight sitting on the enemy's hole — defended by
      // a friendly pawn, in advanced ranks — is a strong outpost.
      const Bitboard wspan = e.wspan;
      const Bitboard bspan = e.bspan;

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

    // One side's imbalance bonus: sum over our piece types p1 of our_count[p1] * (sum over p2<=p1 of
    // IMB_OURS[p1][p2]*our_count[p2] + IMB_THEIRS[p1][p2]*their_count[p2]). Quadratic in the counts.
    [[gnu::pure]] int imbalance_side(const int our[5], const int they[5]) noexcept {
      int bonus = 0;
      for (int p1 = 0; p1 < 5; ++p1) {
        if (!our[p1])
          continue;
        int v = 0;
        for (int p2 = 0; p2 <= p1; ++p2)
          v += IMB_OURS[p1][p2] * our[p2] + IMB_THEIRS[p1][p2] * they[p2];
        bonus += our[p1] * v;
      }
      return bonus;
    }

    // Material imbalance, White's perspective, phase-independent (see IMB_OURS/IMB_THEIRS).
    [[gnu::pure]] int imbalance(const Position &pos) noexcept {
      int w[5], b[5];
      for (int i = 0; i < 5; ++i) { // 0=P 1=N 2=B 3=R 4=Q
        w[i] = pop_count(pos.bitboard_of(WHITE, PieceType(i)));
        b[i] = pop_count(pos.bitboard_of(BLACK, PieceType(i)));
      }
      return (imbalance_side(w, b) - imbalance_side(b, w)) / IMB_SCALE;
    }

    // The squares attacked by side C, split by attacker tier (pawns / minors / rooks) plus the full
    // union (`all`, used to decide whether an enemy piece is defended). Computed once per side.
    struct AttackMap {
      Bitboard pawn   = 0;
      Bitboard minor  = 0; // knights + bishops
      Bitboard rook   = 0;
      Bitboard knight = 0; // kept split (from `minor`) for safe-check detection
      Bitboard bishop = 0;
      Bitboard queen  = 0;
      Bitboard all    = 0; // every attacker, including queens and the king
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
        m.knight |= a;
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
        m.bishop |= a;
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
      for (Bitboard b = pos.bitboard_of(C, QUEEN); b;) {
        Bitboard a = attacks<QUEEN>(pop_lsb(&b), occ);
        m.queen |= a;
        int c = pop_count(a & targets);
        mob.mg += MOB_MG[QUEEN] * c, mob.eg += MOB_EG[QUEEN] * c;
        if (Bitboard z = a & kingZone) {
          kzUnits += KING_ATT_WEIGHT[QUEEN] * pop_count(z);
          ++kzAttackers;
        }
      }
      m.all = m.pawn | m.minor | m.rook | m.queen | attacks<KING>(bsf(pos.bitboard_of(C, KING)), occ);
    }

    // Threat bonus for side C against ~C: pieces of ~C attacked by our pawns / minors / rooks / king /
    // safe pawn pushes, plus any enemy piece we attack that ~C does not defend (`theyDefend` = ~C's full
    // attack map). `occ` = full occupancy, `theirPawnAtt` = ~C's pawn attacks (for safe-push detection).
    template<Color C>
    [[gnu::pure, gnu::hot]] Score side_threats(const Position &pos, const AttackMap &us, Bitboard theyDefend,
                                               Bitboard occ, Bitboard theirPawnAtt) noexcept {
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

      // King threats: undefended enemy pieces our king attacks (the king is a real attacker, esp. eg).
      const int kingThr = pop_count(theirPieces & attacks<KING>(bsf(pos.bitboard_of(C, KING)), occ) & ~theyDefend);
      s.mg += THREAT_BY_KING.mg * kingThr, s.eg += THREAT_BY_KING.eg * kingThr;

      // Pawn-push threats: a pawn that can safely push (to an empty square no enemy pawn attacks) and
      // would then attack an enemy piece. Single push, plus the double push from the home rank.
      constexpr Direction Up    = C == WHITE ? NORTH : SOUTH;
      const Bitboard      pawns = pos.bitboard_of(C, PAWN);
      Bitboard            push  = shift<Up>(pawns) & ~occ;
      push |= shift<Up>(push & (C == WHITE ? MASK_RANK[RANK3] : MASK_RANK[RANK6])) & ~occ;
      push &= ~theirPawnAtt; // the pushed pawn must not be en-prise to an enemy pawn
      const int pushThr = pop_count(pawn_attacks<C>(push) & theirPieces);
      s.mg += THREAT_BY_PUSH.mg * pushThr, s.eg += THREAT_BY_PUSH.eg * pushThr;
      return s;
    }

    // Safe-check pressure (cp) on the enemy king at `ek`: for each checking piece type, the squares
    // from which that piece would check the king (the king's own slider/knight rays) that we actually
    // attack (`m`), the enemy does not defend (`enemyDef` = their full attack map) and we do not
    // occupy (`ourPieces`). A capture-check of an undefended blocker counts; a king-defended square
    // does not. Slider rays use full occupancy, so a check square must be reachable past the blockers.
    [[gnu::pure]] int safe_checks(Square ek, Bitboard occ, const AttackMap &m, Bitboard enemyDef,
                                  Bitboard ourPieces) noexcept {
      const Bitboard bishopRay = attacks<BISHOP>(ek, occ);
      const Bitboard rookRay   = attacks<ROOK>(ek, occ);
      const Bitboard knightRay = attacks<KNIGHT>(ek, occ);
      const Bitboard safe      = ~enemyDef & ~ourPieces;
      int            s         = 0;
      s += SAFE_CHECK[KNIGHT] * pop_count(knightRay & m.knight & safe);
      s += SAFE_CHECK[BISHOP] * pop_count(bishopRay & m.bishop & safe);
      s += SAFE_CHECK[ROOK] * pop_count(rookRay & m.rook & safe);
      s += SAFE_CHECK[QUEEN] * pop_count((bishopRay | rookRay) & m.queen & safe);
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
      const Score sw  = side_threats<WHITE>(pos, wm, bm.all, occ, bm.pawn);
      const Score sb  = side_threats<BLACK>(pos, bm, wm.all, occ, wm.pawn);
      const int   thr = taper({sw.mg - sb.mg, sw.eg - sb.eg}, phase);

      // King pressure = king-zone attacks (weighted units, scaled by attacker count — a lone attacker
      // scores nothing) PLUS safe checks (cp, NOT attacker-count gated — a single deadly check counts).
      // Each side attacks the OTHER king. Mostly middlegame — quartered in the endgame.
      const int kw =
              wU * KING_ATT_SCALE[wC < 7 ? wC : 7] / 100 * KING_ATT_UNIT + safe_checks(bk, occ, wm, bm.all, wpieces);
      const int kb =
              bU * KING_ATT_SCALE[bC < 7 ? bC : 7] / 100 * KING_ATT_UNIT + safe_checks(wk, occ, bm, wm.all, bpieces);
      const int katt = taper({kw - kb, (kw - kb) / 4}, phase);

      // Restricted squares: squares we attack that the enemy attacks but does not protect with a pawn.
      const int wRestr = pop_count(wm.all & bm.all & ~bm.pawn);
      const int bRestr = pop_count(bm.all & wm.all & ~wm.pawn);
      const int restr  = taper({RESTRICTED.mg * (wRestr - bRestr), RESTRICTED.eg * (wRestr - bRestr)}, phase);

      // Space: safe central squares on our own half, weighted by piece count (a middlegame term).
      const Bitboard wp = pos.bitboard_of(WHITE, PAWN), bp = pos.bitboard_of(BLACK, PAWN);
      const Bitboard wSafe   = SPACE_MASK_W & ~wp & ~bPawnAtt;
      const Bitboard bSafe   = SPACE_MASK_B & ~bp & ~wPawnAtt;
      Bitboard       wBehind = wp >> 8;
      wBehind |= wBehind >> 8, wBehind |= wBehind >> 16, wBehind |= wBehind >> 32; // squares behind white pawns
      Bitboard bBehind = bp << 8;
      bBehind |= bBehind << 8, bBehind |= bBehind << 16, bBehind |= bBehind << 32;
      const int wSpace  = pop_count(wSafe) + pop_count(wSafe & wBehind);
      const int bSpace  = pop_count(bSafe) + pop_count(bSafe & bBehind);
      const int wPc     = pop_count(wpieces & ~wp & ~pos.bitboard_of(WHITE, KING)); // White minors+majors
      const int bPc     = pop_count(bpieces & ~bp & ~pos.bitboard_of(BLACK, KING));
      const int spaceMg = SPACE_WEIGHT * (wSpace * wPc - bSpace * bPc) / 16;
      const int space   = taper({spaceMg, 0}, phase); // middlegame only

      return mob + thr + katt + restr + space;
    }

    // Trapped pieces, White's perspective, tapered. (1) A bishop in the enemy corner (a7/h7 for White)
    // hemmed by an enemy pawn that it cannot escape. (2) A rook boxed in by its own king on the back
    // rank, on the side where castling is already lost (the king blocks the rook's only exit).
    [[gnu::pure]] int trapped_pieces(const Position &pos, int phase) noexcept {
      Score          s;
      const Bitboard wp = pos.bitboard_of(WHITE, PAWN), bp = pos.bitboard_of(BLACK, PAWN);
      const Bitboard wB = pos.bitboard_of(WHITE, BISHOP), bB = pos.bitboard_of(BLACK, BISHOP);

      if ((wB & SQUARE_BB[a7]) && (bp & SQUARE_BB[b6]))
        s.mg -= TRAPPED_BISHOP.mg, s.eg -= TRAPPED_BISHOP.eg;
      if ((wB & SQUARE_BB[h7]) && (bp & SQUARE_BB[g6]))
        s.mg -= TRAPPED_BISHOP.mg, s.eg -= TRAPPED_BISHOP.eg;
      if ((bB & SQUARE_BB[a2]) && (wp & SQUARE_BB[b3]))
        s.mg += TRAPPED_BISHOP.mg, s.eg += TRAPPED_BISHOP.eg;
      if ((bB & SQUARE_BB[h2]) && (wp & SQUARE_BB[g3]))
        s.mg += TRAPPED_BISHOP.mg, s.eg += TRAPPED_BISHOP.eg;

      const Bitboard wR = pos.bitboard_of(WHITE, ROOK), bR = pos.bitboard_of(BLACK, ROOK);
      const Bitboard entry = pos.castle_entry();
      const Square   wk    = bsf(pos.bitboard_of(WHITE, KING));
      const Square   bk    = bsf(pos.bitboard_of(BLACK, KING));

      if (rank_of(wk) == RANK1) {
        const int kf = file_of(wk);
        if (kf >= FFILE && (entry & WHITE_OO_MASK) && (wR & MASK_RANK[RANK1] & files_above(kf)))
          s.mg -= TRAPPED_ROOK.mg, s.eg -= TRAPPED_ROOK.eg; // kingside rook stuck behind the king
        else if (kf <= CFILE && (entry & WHITE_OOO_MASK) && (wR & MASK_RANK[RANK1] & files_below(kf)))
          s.mg -= TRAPPED_ROOK.mg, s.eg -= TRAPPED_ROOK.eg; // queenside
      }
      if (rank_of(bk) == RANK8) {
        const int kf = file_of(bk);
        if (kf >= FFILE && (entry & BLACK_OO_MASK) && (bR & MASK_RANK[RANK8] & files_above(kf)))
          s.mg += TRAPPED_ROOK.mg, s.eg += TRAPPED_ROOK.eg;
        else if (kf <= CFILE && (entry & BLACK_OOO_MASK) && (bR & MASK_RANK[RANK8] & files_below(kf)))
          s.mg += TRAPPED_ROOK.mg, s.eg += TRAPPED_ROOK.eg;
      }
      return taper(s, phase);
    }

    // --- Drawish-endgame scaling -------------------------------------------------------------
    // Many material configurations are dead or near-dead draws even with a nominal material edge; the
    // engine must not score them as wins (or it trades into them). scale_factor returns a
    // 0..SCALE_NORMAL multiplier for the White-perspective score `s` — it only narrows a winning eval
    // toward 0, never flips the sign. White-POV: s > 0 means White is the (nominally) stronger side.
    // Pawnless cases are tested first, so a pawnless opposite-bishop position hits the dead-draw
    // branch rather than the OCB branch (which is for OCB *with* pawns). Starting values, SPRT-tune.
    constexpr int      SCALE_NORMAL  = 64;
    constexpr int      SCALE_DRAWISH = 16; // only ~a minor up with no pawns (rook-vs-minor, R+B vs R…)
    constexpr int      SCALE_OCB     = 26; // opposite-coloured bishops (with pawns) — notoriously drawish
    constexpr Bitboard LIGHT_SQUARES = 0x55AA55AA55AA55AAULL;

    [[gnu::pure]] int scale_factor(const Position &pos, int s) noexcept {
      const Color strong = s > 0 ? WHITE : BLACK;
      const Color weak   = ~strong;
      const auto  cnt    = [&](Color c, PieceType pt) noexcept { return pop_count(pos.bitboard_of(c, pt)); };

      const int sP = cnt(strong, PAWN);
      const int sN = cnt(strong, KNIGHT), sB = cnt(strong, BISHOP), sR = cnt(strong, ROOK), sQ = cnt(strong, QUEEN);
      const int wN = cnt(weak, KNIGHT), wB = cnt(weak, BISHOP), wR = cnt(weak, ROOK), wQ = cnt(weak, QUEEN);
      const int sNPM = 3 * (sN + sB) + 5 * sR + 9 * sQ; // non-pawn material in pawn-ish units
      const int wNPM = 3 * (wN + wB) + 5 * wR + 9 * wQ;

      // Stronger side has NO pawns: forcing a win needs real material superiority it may lack.
      if (sP == 0) {
        if (sNPM <= 3) // at most one minor: cannot force mate (KN/KB vs K, minor vs minor)
          return 0;
        if (sN == 2 && sB == 0 && sR == 0 && sQ == 0 && wNPM == 0) // two knights vs lone king: no forced mate
          return 0;
        if (sNPM - wNPM <= 3) // only ~a minor up (rook vs minor, R+B vs R, equal heavy material…)
          return SCALE_DRAWISH;
      }

      // Opposite-coloured bishops with no other pieces (only B + pawns): notoriously drawish.
      if (sQ == 0 && wQ == 0 && sR == 0 && wR == 0 && sN == 0 && wN == 0 && sB == 1 && wB == 1) {
        const bool sLight = (pos.bitboard_of(strong, BISHOP) & LIGHT_SQUARES) != 0;
        const bool wLight = (pos.bitboard_of(weak, BISHOP) & LIGHT_SQUARES) != 0;
        if (sLight != wLight)
          return SCALE_OCB;
      }
      return SCALE_NORMAL;
    }

    // --- Mop-up (elementary-mate driving) ----------------------------------------------------
    // When one side is reduced to a BARE KING (no pawns, no pieces) and the other has mating
    // material, add a small White-perspective gradient that (1) drives the lone king toward the
    // edge/corner and (2) brings the winning king closer — so KQK / KRK / KBNK are converted within
    // the fifty-move rule instead of shuffled. The magnitudes are tiny next to the material they ride
    // on (Q ≈ 900), so they only shape the PATH to mate and never flip the eval; they are therefore
    // NOT Texel-tuned (a positional dataset has too few such positions — tuning would zero them like
    // the space term). SPRT / conversion tests judge them.
    constexpr int MOPUP_CORNER = 12; // per unit of centre-distance of the lone king (edge/corner drive)
    constexpr int MOPUP_KBNK   = 16; // per unit toward the bishop-coloured corner (KBNK needs THAT corner)
    constexpr int MOPUP_CLOSE  = 10; // per unit the winning king is closer to the lone king

    [[gnu::pure]] int mopup(const Position &pos) noexcept {
      const auto bare = [&](Color c) noexcept {
        return (pos.bitboard_of(c, PAWN) | pos.bitboard_of(c, KNIGHT) | pos.bitboard_of(c, BISHOP) |
                pos.bitboard_of(c, ROOK) | pos.bitboard_of(c, QUEEN)) == 0;
      };
      Color winner;
      if (bare(BLACK) && !bare(WHITE))
        winner = WHITE;
      else if (bare(WHITE) && !bare(BLACK))
        winner = BLACK;
      else
        return 0;

      const Bitboard wbb = pos.bitboard_of(winner, BISHOP);
      const int      wN = pop_count(pos.bitboard_of(winner, KNIGHT)), wB = pop_count(wbb);
      const int      wR = pop_count(pos.bitboard_of(winner, ROOK)), wQ = pop_count(pos.bitboard_of(winner, QUEEN));
      const bool     hasPawn = pos.bitboard_of(winner, PAWN) != 0;
      // Pure-piece mate only (a winner WITH pawns is handled by the normal passer/PST eval), and the
      // material must actually force mate: Q, R, bishop+knight, or two OPPOSITE-coloured bishops (a
      // lone minor, two knights, or two same-coloured bishops cannot force mate).
      const bool twoColourB = (wbb & LIGHT_SQUARES) != 0 && (wbb & ~LIGHT_SQUARES) != 0;
      const bool canMate    = wQ > 0 || wR > 0 || twoColourB || (wB >= 1 && wN >= 1);
      if (hasPawn || !canMate)
        return 0;

      const Square lk = bsf(pos.bitboard_of(~winner, KING)); // the lone king
      const Square wk = bsf(pos.bitboard_of(winner, KING));
      const int    lf = file_of(lk), lr = rank_of(lk);
      const int    fd     = lf < 4 ? 3 - lf : lf - 4; // file distance from the centre files (d/e)
      const int    rd     = lr < 4 ? 3 - lr : lr - 4;
      int          corner = MOPUP_CORNER * (fd + rd); // 0 at the centre, up to 72 in a corner

      // KBN vs K: the mate only works in a corner of the bishop's colour — steer to THAT corner.
      if (wQ == 0 && wR == 0 && wB == 1 && wN == 1) {
        const bool light = (pos.bitboard_of(winner, BISHOP) & LIGHT_SQUARES) != 0;
        const int  da    = cheb(lk, light ? h1 : a1); // light corners: h1/a8; dark corners: a1/h8
        const int  db    = cheb(lk, light ? a8 : h8);
        const int  d     = da < db ? da : db;
        corner           = MOPUP_KBNK * (7 - d);
      }

      const int bonus = corner + MOPUP_CLOSE * (7 - cheb(wk, lk));
      return winner == WHITE ? bonus : -bonus;
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
    s += imbalance(pos); // material imbalance (quadratic piece-interaction, phase-independent)
    s += trapped_pieces(pos, phase);
    s += mopup(pos); // elementary-mate driving in bare-king endgames (KQK / KRK / KBNK conversion)

    // Drawish-endgame scaling: narrow a nominal material edge toward 0 in dead / near-dead-drawn
    // material configurations (insufficient minors, rook-vs-minor, opposite-coloured bishops) so the
    // engine neither overvalues them nor trades into them. Applied before the side-to-move flip.
    s = s * scale_factor(pos, s) / SCALE_NORMAL;

    // White-perspective score flipped to the side to move, plus the tempo bonus for that side.
    const int v = (pos.turn() == WHITE ? s : -s) + taper(TEMPO, phase);

    // Fifty-move damping: scale the whole evaluation linearly toward 0 as the halfmove clock
    // approaches the fifty-move rule (down to 50% at 100 halfmoves, where is_draw() takes over and
    // scores it 0 exactly). An advantage the engine cannot convert within the rule is not a real
    // advantage: this makes shuffling lines decay, so the better side prefers PROGRESS (a pawn push
    // or capture resets the clock and restores the full eval) over repetition-adjacent drift, and
    // dead-drawn games converge into the match-runner's draw-adjudication window instead of
    // shuffling for hundreds of plies. The 200 divisor is the standard (Stockfish-style) starting
    // value, unproven here — SPRT decides.
    return v * (200 - pos.fifty()) / 200;
  }

  // --- Texel tuning registry (see eval.h) -----------------------------------------------------
  const std::vector<Param> &params() noexcept {
    static const std::vector<Param> P = [] {
      std::vector<Param> v;
      const auto         add = [&v](std::string name, int &x) { v.push_back({std::move(name), &x}); };

      add("DOUBLED_PENALTY", DOUBLED_PENALTY);
      add("ISOLATED_PENALTY", ISOLATED_PENALTY);
      for (int r = 1; r <= 6; ++r)
        add("PASSED_MG_" + std::to_string(r), PASSED_MG[r]);
      for (int r = 1; r <= 6; ++r)
        add("PASSED_EG_" + std::to_string(r), PASSED_EG[r]);
      for (int r = 3; r <= 6; ++r)
        add("PASSED_KDIST_W_" + std::to_string(r), PASSED_KDIST_W[r]);
      add("CENTERED_BONUS", CENTERED_BONUS);
      add("OUTPOST_BONUS", OUTPOST_BONUS);
      for (int r = 1; r <= 6; ++r)
        add("CONNECTED_" + std::to_string(r), CONNECTED[r]);
      add("BACKWARD_PENALTY", BACKWARD_PENALTY);
      add("SHIELD_DEFICIT", SHIELD_DEFICIT);
      add("OPEN_FILE_PENALTY", OPEN_FILE_PENALTY);
      static const char *PT = "PNBRQK";
      for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
        add(std::string("MOB_MG_") + PT[pt], MOB_MG[pt]);
        add(std::string("MOB_EG_") + PT[pt], MOB_EG[pt]);
      }
      add("PIN_PENALTY", PIN_PENALTY);
      add("BISHOP_PAIR_MG", BISHOP_PAIR_MG);
      add("BISHOP_PAIR_EG", BISHOP_PAIR_EG);
      add("ROOK_OPEN", ROOK_OPEN);
      add("ROOK_SEMIOPEN", ROOK_SEMIOPEN);
      add("ROOK_7TH_MG", ROOK_7TH_MG);
      add("ROOK_7TH_EG", ROOK_7TH_EG);
      {
        static const char *IPT = "PNBRQ";
        for (int p1 = 0; p1 < 5; ++p1)
          for (int p2 = 0; p2 <= p1; ++p2)
            add(std::string("IMB_OURS_") + IPT[p1] + IPT[p2], IMB_OURS[p1][p2]);
        for (int p1 = 0; p1 < 5; ++p1)
          for (int p2 = 0; p2 < p1; ++p2) // theirs diagonal is structurally 0 (kept fixed)
            add(std::string("IMB_THEIRS_") + IPT[p1] + IPT[p2], IMB_THEIRS[p1][p2]);
      }
      add("THREAT_BY_PAWN_MG", THREAT_BY_PAWN.mg);
      add("THREAT_BY_PAWN_EG", THREAT_BY_PAWN.eg);
      for (int pt = KNIGHT; pt <= QUEEN; ++pt) { // victims a minor/rook can threaten
        add(std::string("THREAT_BY_MINOR_MG_") + PT[pt], THREAT_BY_MINOR[pt].mg);
        add(std::string("THREAT_BY_MINOR_EG_") + PT[pt], THREAT_BY_MINOR[pt].eg);
        add(std::string("THREAT_BY_ROOK_MG_") + PT[pt], THREAT_BY_ROOK[pt].mg);
        add(std::string("THREAT_BY_ROOK_EG_") + PT[pt], THREAT_BY_ROOK[pt].eg);
      }
      add("HANGING_MG", HANGING.mg);
      add("HANGING_EG", HANGING.eg);
      add("THREAT_BY_KING_MG", THREAT_BY_KING.mg);
      add("THREAT_BY_KING_EG", THREAT_BY_KING.eg);
      add("THREAT_BY_PUSH_MG", THREAT_BY_PUSH.mg);
      add("THREAT_BY_PUSH_EG", THREAT_BY_PUSH.eg);
      add("ROOK_BEHIND_PASSER_MG", ROOK_BEHIND_PASSER.mg);
      add("ROOK_BEHIND_PASSER_EG", ROOK_BEHIND_PASSER.eg);
      add("RESTRICTED_MG", RESTRICTED.mg);
      add("RESTRICTED_EG", RESTRICTED.eg);
      add("SPACE_WEIGHT", SPACE_WEIGHT);
      add("TRAPPED_BISHOP_MG", TRAPPED_BISHOP.mg);
      add("TRAPPED_BISHOP_EG", TRAPPED_BISHOP.eg);
      add("TRAPPED_ROOK_MG", TRAPPED_ROOK.mg);
      add("TRAPPED_ROOK_EG", TRAPPED_ROOK.eg);
      for (int pt = KNIGHT; pt <= QUEEN; ++pt)
        add(std::string("KING_ATT_WEIGHT_") + PT[pt], KING_ATT_WEIGHT[pt]);
      add("KING_ATT_UNIT", KING_ATT_UNIT);
      for (int pt = KNIGHT; pt <= QUEEN; ++pt)
        add(std::string("SAFE_CHECK_") + PT[pt], SAFE_CHECK[pt]);
      add("TEMPO_MG", TEMPO.mg);
      add("TEMPO_EG", TEMPO.eg);
      return v;
    }();
    return P;
  }

} // namespace eval
