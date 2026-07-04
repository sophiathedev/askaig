#include "search.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include "history.h"
#include "movepick.h"
#include "nnue.h"
#include "see.h"
#include "smp.h"
#include "tt.h"

// Fail-soft negamax with, in rough order of gating: TT cutoffs, static-eval correction
// history, IIR, razoring, RFP, NMP, ProbCut, singular extensions (with multicut and
// double/negative extensions), LMP, futility pruning, history pruning, SEE pruning (captures
// and quiets), log-formula LMR, PVS, and a SEE/delta-pruned quiescence. Move ordering in
// movepick.h uses the TT move, MVV + capture history (SEE-verified lazily at yield time),
// killers, and butterfly + continuation (CMH/FMH) history.
//
// Parallelism is Lazy SMP (smp.h/smp.cpp): every helper thread runs its own full
// iterative-deepening search of the root on a private Position copy, sharing only the
// transposition table and the history tables with the main thread. Nothing is ever read back
// from a helper — their value is the TT/history warm-up. The main thread alone reports
// iterations, manages time and produces the bestmove; helpers stop when it raises
// g_helper_stop at the end of think(). Perft keeps its own worker scheme in uci.cpp.
namespace {

  using namespace search;
  using Clock = std::chrono::steady_clock;

  // Shared search state. History tables are shared by all threads with plain (benign-race)
  // int16 updates, exactly like the TT — both are statistics, not correctness-critical.
  Histories          g_hist;
  ThreadData         g_main;
  std::atomic<bool>  g_stop{false};
  // Raised by think() once ITS search is done, so the lazy-SMP helpers (each running an
  // open-ended iterative deepening of their own) wind down before think() returns. A separate
  // flag from g_stop on purpose: g_stop belongs to the caller (see clear_stop()'s contract) —
  // think() may never touch it, but it fully owns the helpers' lifetime.
  std::atomic<bool>  g_helper_stop{false};
  int64_t            g_hard_ms = 0;
  Clock::time_point  g_t0;
  int                g_contempt   = 0; // the "Contempt" UCI option, in centipawns
  Color              g_root_color = WHITE; // side to move at the root of the current think()

  // Node-based time management instrumentation (ThreadData::root_m1_nodes/root_m1_move): how
  // many of the just-completed iteration's nodes went into the root's first-searched move
  // (the TT/PV move from move ordering — the one we already believe is best). Reset at the
  // start of every root call so a widened aspiration re-try only keeps the LAST attempt's
  // count; think() reads the MAIN thread's copy back once the iteration returns (per-thread
  // fields so helper roots can't clobber it). It usually IS the eventual best move — think()
  // checks that and skips the adjustment on the rarer iterations where it wasn't.

  // LMR reduction table, ln(depth)*ln(moves) scaled (the "most principled" log formula).
  int  g_lmr[64][64];
  bool g_lmr_init = false;
  void init_lmr() {
    for (int d = 1; d < 64; ++d)
      for (int m = 1; m < 64; ++m)
        g_lmr[d][m] = int(0.8 + std::log(d) * std::log(m) / 2.3);
    g_lmr_init = true;
  }
  // Reads a global filled once (init_lmr, before any search starts) and never written again —
  // safe as `pure` (no side effects; the memory it reads is effectively frozen for the caller).
  [[gnu::pure, gnu::always_inline]] inline int lmr_base(int depth, int movecount) {
    return g_lmr[std::min(depth, 63)][std::min(movecount, 63)];
  }

  // --- small helpers ----------------------------------------------------------------------------
  // Tiny, extremely hot leaf functions (called at every node/move): `always_inline` matches the
  // existing idiom in simd.h/tables.h for functions this small and this frequently called.
  // `const`/`pure` follow GNU semantics precisely: `const` only for functions with zero memory
  // reads (pure arithmetic on by-value arguments); `pure` for functions that read memory
  // (Position, history tables) but have no side effects of their own.

  // tt_key (the castling/en-passant mix over the incremental Zobrist hash) lives in smp.h
  // with the other shared make/unmake helpers.

  // Correction-history keys. Each is a cheap position-derived hash into a CORR_SIZE table —
  // collisions just blend unrelated positions' corrections, no correctness impact.
  [[gnu::pure, gnu::always_inline]] inline size_t pawn_corr_index(const Position &p) {
    const uint64_t w = p.bitboard_of(WHITE_PAWN) * 0x9E3779B97F4A7C15ull;
    const uint64_t b = p.bitboard_of(BLACK_PAWN) * 0xC2B2AE3D27D4EB4Full;
    return (w ^ (b + 0x165667B19E3779F9ull + (w << 6) + (w >> 2))) & (Histories::CORR_SIZE - 1);
  }
  // Non-pawn material imbalance: piece-TYPE COUNTS (not placement), so it fires on trades and
  // promotions rather than on where the pieces happen to stand.
  [[gnu::pure, gnu::always_inline]] inline size_t material_corr_index(const Position &p) {
    uint64_t key = 0;
    for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
      key = key * 9 + uint64_t(pop_count(p.bitboard_of(make_piece(WHITE, PieceType(pt)))));
      key = key * 9 + uint64_t(pop_count(p.bitboard_of(make_piece(BLACK, PieceType(pt)))));
    }
    return (key * 0x9E3779B97F4A7C15ull) & (Histories::CORR_SIZE - 1);
  }
  [[gnu::pure, gnu::always_inline]] inline size_t minor_corr_index(const Position &p) {
    const Bitboard minors = p.bitboard_of(WHITE_KNIGHT) | p.bitboard_of(WHITE_BISHOP) |
                            p.bitboard_of(BLACK_KNIGHT) | p.bitboard_of(BLACK_BISHOP);
    return (minors * 0x9E3779B97F4A7C15ull) & (Histories::CORR_SIZE - 1);
  }
  [[gnu::pure, gnu::always_inline]] inline size_t major_corr_index(const Position &p) {
    const Bitboard majors = p.bitboard_of(WHITE_ROOK) | p.bitboard_of(WHITE_QUEEN) |
                            p.bitboard_of(BLACK_ROOK) | p.bitboard_of(BLACK_QUEEN);
    return (majors * 0xC2B2AE3D27D4EB4Full) & (Histories::CORR_SIZE - 1);
  }

  // Mate scores are stored ply-relative in the TT ("mate in N from HERE"). Pure arithmetic on
  // scalar arguments only, no memory touched at all — `const` (stronger than `pure`) applies.
  [[gnu::const, gnu::always_inline]] inline int to_tt(int v, int ply) {
    return v >= MATE_IN_MAX ? v + ply : v <= -MATE_IN_MAX ? v - ply : v;
  }
  [[gnu::const, gnu::always_inline]] inline int from_tt(int v, int ply) {
    return v >= MATE_IN_MAX ? v - ply : v <= -MATE_IN_MAX ? v + ply : v;
  }

  // Draw value, from the current node's side-to-move perspective (as every search function
  // must return). g_contempt is a fixed cost the ROOT side pays for any draw, however deep and
  // whoever's move it's detected on: negamax flips sign once per ply on the way back up, so
  // returning -contempt when it's the root color's own turn here and +contempt when it's the
  // opponent's turn here both collapse to "root is down `contempt` cp" once propagated to the
  // root. g_contempt == 0 (default) makes this identical to the old bare `return 0`.
  [[gnu::pure, gnu::always_inline]] inline int draw_score(const Position &pos) {
    return pos.turn() == g_root_color ? -g_contempt : g_contempt;
  }

  // Sum of all correction-history tables for this node: pawn skeleton, non-pawn material,
  // minor- and major-piece placement, and the continuation table keyed by the previous move
  // (piece, to) — skipped at the root and right after a null move, where there is none.
  // Divided by a larger constant than a single table would need (256 vs. the old 64) so five
  // tables agreeing doesn't blow the correction past what one confident table used to give.
  [[gnu::pure, gnu::hot]] int correction(const Position &pos, const Stack *ss) {
    const Color c    = pos.turn();
    int         corr = g_hist.corr_pawn[c][pawn_corr_index(pos)] + g_hist.corr_material[c][material_corr_index(pos)] +
                g_hist.corr_minor[c][minor_corr_index(pos)] + g_hist.corr_major[c][major_corr_index(pos)];
    if ((ss - 1)->move.to_from() != 0)
      corr += g_hist.corr_cont[pos.at((ss - 1)->move.to())][(ss - 1)->move.to()];
    return corr / 256;
  }

  // Static eval + the learned corrections, clamped out of the mate range. NOT pure: t.ev's
  // lazy accumulator walk-back writes into t's (real, cross-call-persistent) accumulator cache
  // as a side effect — not just a local computation — so `pure` would be incorrect here even
  // though the observable result is deterministic. Called at nearly every node; discarding the
  // result would always be a bug.
  //
  // Fifty-move damping: the raw net output is scaled toward zero as the halfmove clock climbs
  // (full at clock 0, about half at 99), so an advantage the search cannot convert decays
  // instead of the engine shuffling forever at "+0.3" — and, symmetrically, a defender sees
  // the incoming fifty-move draw as increasingly survivable. The min() guards hand-written
  // FENs with a clock past 100; in play the search returns a draw at 100 before evaluating.
  // The damped value is what lands in the TT eval field, carrying the storing node's clock —
  // same-hash nodes at a different clock get a slightly stale eval, an accepted imprecision.
  [[gnu::hot, nodiscard]] int evaluate(ThreadData &t, const Position &pos, const Stack *ss) {
    const int raw = t.ev.evaluate(pos) * (200 - std::min(pos.fifty(), 100)) / 200;
    return std::clamp(raw + correction(pos, ss), -MATE_IN_MAX + 1, MATE_IN_MAX - 1);
  }

  // True when this thread's search is moot: the caller's stop, or think() winding the lazy-SMP
  // helpers down. Every consumer must discard results once this holds.
  [[gnu::hot]] inline bool stopped() {
    return g_stop.load(std::memory_order_relaxed) || g_helper_stop.load(std::memory_order_relaxed);
  }

  // A stop/abort poll for the tree: stopped() plus the hard clock. NOT pure — it can flip
  // g_stop on timeout, a real side effect every other node in this file relies on. Helpers
  // poll the hard clock too, so they never outlive the time budget on their own.
  [[gnu::hot]] bool time_up(ThreadData &t) {
    if (stopped())
      return true;
    if (g_hard_ms > 0 && (t.nodes & 2047) == 0) {
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_t0).count();
      if (ms >= g_hard_ms) {
        g_stop.store(true, std::memory_order_relaxed);
        return true;
      }
    }
    return false;
  }

  [[gnu::const, gnu::always_inline]] inline int hist_bonus(int depth) { return std::min(160 * depth - 80, 2000); }

  [[gnu::pure, gnu::always_inline]] inline int quiet_hist(const Stack *ss, const Position &pos, Move m) {
    const Piece pc = pos.at(m.from());
    int         h  = g_hist.butterfly[pos.turn()][m.from()][m.to()];
    if ((ss - 1)->ch)
      h += (*(ss - 1)->ch)[pc][m.to()];
    if ((ss - 2)->ch)
      h += (*(ss - 2)->ch)[pc][m.to()];
    return h;
  }

  [[gnu::hot]] void update_quiet_hists(Stack *ss, const Position &pos, Move best, const Move *tried, int n_tried,
                                       int depth) {
    const int  bonus = hist_bonus(depth);
    const auto touch = [&](Move m, int b) {
      const Piece pc = pos.at(m.from());
      hist_update(g_hist.butterfly[pos.turn()][m.from()][m.to()], b);
      if ((ss - 1)->ch)
        hist_update((*(ss - 1)->ch)[pc][m.to()], b);
      if ((ss - 2)->ch)
        hist_update((*(ss - 2)->ch)[pc][m.to()], b);
    };
    touch(best, bonus);
    for (int i = 0; i < n_tried; ++i)
      touch(tried[i], -bonus);
    if (ss->killers[0].to_from() != best.to_from()) { // killer slots shift
      ss->killers[1] = ss->killers[0];
      ss->killers[0] = best;
    }
  }

  [[gnu::hot]] void update_capture_hists(const Position &pos, Move best, const Move *tried, int n_tried, int depth) {
    const int  bonus = hist_bonus(depth);
    const auto touch = [&](Move m, int b) {
      const PieceType captured = m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
      hist_update(g_hist.capture[pos.at(m.from())][m.to()][captured], b);
    };
    if (best.is_capture())
      touch(best, bonus);
    for (int i = 0; i < n_tried; ++i)
      touch(tried[i], -bonus);
  }

  // The two hottest functions in the engine: every node passes through one of these.
  template<bool PV>
  [[gnu::hot]] int qsearch(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int ply);
  template<bool PV>
  [[gnu::hot]] int negamax(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int depth, int ply,
                           bool cutnode);

  // The reported node/nps totals: the main thread plus every lazy-SMP helper, summed live
  // (see ThreadPool::total_nodes for the intentionally-unsynchronized read).
  uint64_t all_nodes() { return g_main.nodes + pool().total_nodes(); }

  // --- quiescence -------------------------------------------------------------------------------

  template<bool PV>
  [[gnu::hot]] int qsearch(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int ply) {
    if constexpr (PV)
      ss->pv_len = 0;
    t.seldepth = std::max(t.seldepth, ply);
    if (time_up(t))
      return 0;
    if (pos.is_draw())
      return draw_score(pos);
    if (ply >= MAX_PLY) [[unlikely]] // needs a 120-ply-deep line; essentially never happens
      return evaluate(t, pos, ss);

    const bool     in_check = stm_in_check(pos);
    const uint64_t key      = tt_key(pos);

    const tt::Probe tp   = tt::probe(key);
    const int       ttsc = tp.hit && tp.score != tt::VALUE_NONE_TT ? from_tt(tp.score, ply) : tt::VALUE_NONE_TT;
    if (!PV && tp.hit && ttsc != tt::VALUE_NONE_TT) {
      if (tp.bound == tt::EXACT || (tp.bound == tt::LOWER && ttsc >= beta) ||
          (tp.bound == tt::UPPER && ttsc <= alpha))
        return ttsc;
    }

    int best = -INF, raw_eval = tt::VALUE_NONE_TT;
    if (!in_check) {
      raw_eval = tp.hit && tp.eval != tt::VALUE_NONE_TT ? tp.eval : evaluate(t, pos, ss);
      best     = raw_eval;
      // The TT score is a tighter bound on this position than the static eval — use it.
      if (tp.hit && ttsc != tt::VALUE_NONE_TT) {
        if (tp.bound == tt::EXACT || (tp.bound == tt::LOWER && ttsc > best) ||
            (tp.bound == tt::UPPER && ttsc < best))
          best = ttsc;
      }
      if (best >= beta)
        return best;
      alpha = std::max(alpha, best);
    }
    const int futility_base = best + 120;

    const Move ttm = tp.move;
    MovePicker picker(pos, g_hist, ttm, nullptr, nullptr, nullptr, /*quiescence=*/true);
    if (in_check && picker.total() == 0)
      return -MATE + ply;

    Move best_move{};
    for (Move m; (m = picker.next()).to_from() != 0;) {
      if (!in_check) {
        // QS SEE pruning: don't even try losing captures. The lazy-SEE picker already proved
        // its capture bands (winning kept, losing demoted) — only a move it never verified
        // (the TT move, or a quiet queen push onto a defended promotion square) needs the
        // see_ge call here.
        const auto band = picker.yielded_see();
        if (band == MovePicker::SEE_LOSING || (band == MovePicker::SEE_UNKNOWN && !see_ge(pos, m, 0)))
          continue;
        // QS futility (delta) pruning: stand pat + margin + captured value can't reach alpha.
        const PieceType captured = m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
        if (futility_base + PIECE_VAL[captured] <= alpha && m.flags() != PR_QUEEN && m.flags() != PC_QUEEN)
          continue;
      }

      ++t.nodes;
      t.ev.push(pos, m);
      do_move(pos, m);
      tt::prefetch(tt_key(pos));
      const int v = -qsearch<PV>(t, pos, ss + 1, -beta, -alpha, ply + 1);
      undo_move(pos, m);
      t.ev.pop();
      // Checked on every move at every node; true at most once per thread for the whole
      // search (time out / wind-down) — about as skewed a branch as exists in this file.
      if (stopped()) [[unlikely]]
        return 0;

      if (v > best) {
        best = v;
        if (v > alpha) {
          best_move = m;
          alpha     = v;
          if constexpr (PV) {
            ss->pv[0] = m;
            if (v < beta) { // fail-high values come from zero-window searches: stale child pv
              std::copy((ss + 1)->pv, (ss + 1)->pv + (ss + 1)->pv_len, ss->pv + 1);
              ss->pv_len = (ss + 1)->pv_len + 1;
            } else
              ss->pv_len = 1;
          }
          if (v >= beta)
            break;
        }
      }
    }

    tt::store(tp.slot, key, best_move, to_tt(best, ply), raw_eval, /*depth=*/0,
              best >= beta ? tt::LOWER : tt::UPPER, PV);
    return best;
  }

  // --- main search ------------------------------------------------------------------------------

  template<bool PV>
  [[gnu::hot]] int negamax(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int depth, int ply,
                           bool cutnode) {
    if (depth <= 0)
      return qsearch<PV>(t, pos, ss, alpha, beta, ply);

    if constexpr (PV)
      ss->pv_len = 0;
    if (time_up(t))
      return 0;
    t.seldepth = std::max(t.seldepth, ply);

    const bool root = ply == 0;
    if (root) // fresh attempt (first call, or an aspiration re-try after a widened window)
      t.root_m1_nodes = 0, t.root_m1_move = Move();
    if (!root) {
      if (pos.is_draw())
        return draw_score(pos);
      if (ply >= MAX_PLY) [[unlikely]] // needs a 120-ply-deep line; essentially never happens
        return evaluate(t, pos, ss);
      // Mate distance pruning: only bites once a shorter mate is already known on another
      // line, so alpha/beta collapsing here is rare outside forced-mate endgames.
      alpha = std::max(alpha, -MATE + ply);
      beta  = std::min(beta, MATE - ply - 1);
      if (alpha >= beta) [[unlikely]]
        return alpha;
    }

    const bool excluded = ss->excluded.to_from() != 0;
    const bool in_check = stm_in_check(pos);
    ss->in_check        = in_check;
    (ss + 1)->excluded  = Move();
    ss->double_ext      = root ? 0 : (ss - 1)->double_ext;

    // --- TT probe (skipped entirely under a singular-exclusion search) ---
    // The Probe defaults (no slot, no hit, VALUE_NONE fields) cover the excluded case: every
    // consumer below already guards on hit/!excluded, and no store happens without a slot.
    const uint64_t key = tt_key(pos);
    tt::Probe      tp;
    if (!excluded) {
      tp = tt::probe(key);
      const int sc = tp.hit && tp.score != tt::VALUE_NONE_TT ? from_tt(tp.score, ply) : tt::VALUE_NONE_TT;
      if (!PV && tp.hit && tp.depth >= depth && sc != tt::VALUE_NONE_TT &&
          (tp.bound == tt::EXACT || (tp.bound == tt::LOWER && sc >= beta) || (tp.bound == tt::UPPER && sc <= alpha)))
        return sc;
    }
    const Move ttm  = tp.move;
    const int  ttsc = tp.hit && tp.score != tt::VALUE_NONE_TT ? from_tt(tp.score, ply) : tt::VALUE_NONE_TT;

    // --- static eval + improving ---
    int raw_eval = tt::VALUE_NONE_TT;
    if (in_check)
      ss->static_eval = tt::VALUE_NONE_TT;
    else {
      raw_eval        = tp.eval != tt::VALUE_NONE_TT ? tp.eval : evaluate(t, pos, ss);
      ss->static_eval = raw_eval;
      // Sharpen with the TT score when it bounds in the right direction.
      if (tp.hit && ttsc != tt::VALUE_NONE_TT &&
          (tp.bound == tt::EXACT || (tp.bound == tt::LOWER && ttsc > raw_eval) ||
           (tp.bound == tt::UPPER && ttsc < raw_eval)))
        ss->static_eval = ttsc;
    }
    const bool improving = !in_check && ply >= 2 && (ss - 2)->static_eval != tt::VALUE_NONE_TT &&
                           ss->static_eval > (ss - 2)->static_eval;

    // --- Internal Iterative Reduction: no TT move at a node that should have one ---
    if ((PV || cutnode) && depth >= 4 && ttm.to_from() == 0)
      --depth;

    // --- whole-node pruning (never at PV nodes, in check, or under exclusion) ---
    if (!PV && !in_check && !excluded) {
      // Razoring: the eval is hopelessly below alpha at shallow depth — verify with a
      // quiescence search and trust its fail-low.
      if (depth <= 4 && ss->static_eval + 300 * depth < alpha) {
        const int v = qsearch<false>(t, pos, ss, alpha - 1, alpha, ply);
        if (v < alpha && std::abs(v) < MATE_IN_MAX)
          return v;
      }

      // Reverse futility pruning: eval is so far above beta a shallow search won't drop below.
      if (depth <= 8 && std::abs(beta) < MATE_IN_MAX &&
          ss->static_eval - 80 * (depth - improving) >= beta)
        return ss->static_eval;

      // Null move pruning: hand the opponent a free move; a fail-high still above beta means
      // the position is too good. Guarded by non-pawn material (zugzwang) and no double null.
      const Color    us       = pos.turn();
      const Bitboard non_pawn = pos.bitboard_of(us, KNIGHT) | pos.bitboard_of(us, BISHOP) |
                                pos.bitboard_of(us, ROOK) | pos.bitboard_of(us, QUEEN);
      if (depth >= 3 && ss->static_eval >= beta && non_pawn && (ss - 1)->move.to_from() != 0 &&
          beta > -MATE_IN_MAX) {
        const int R = 3 + depth / 3 + std::min((ss->static_eval - beta) / 200, 3);
        ss->move    = Move();
        ss->ch      = nullptr;
        t.ev.push_null();
        pos.play_null();
        tt::prefetch(tt_key(pos));
        const int v = -negamax<false>(t, pos, ss + 1, -beta, -beta + 1, depth - R, ply + 1, !cutnode);
        pos.undo_null();
        t.ev.pop();
        if (stopped()) [[unlikely]] // see the qsearch move loop
          return 0;
        if (v >= beta)
          return v >= MATE_IN_MAX ? beta : v; // don't return unproven mates
      }

      // ProbCut: a good capture that beats beta by a margin at reduced depth almost certainly
      // produces a full-depth beta cutoff too. Skipped when the TT already says otherwise.
      const int pc_beta = beta + 180 - 60 * improving;
      if (depth >= 5 && std::abs(beta) < MATE_IN_MAX &&
          !(tp.hit && tp.depth >= depth - 3 && ttsc != tt::VALUE_NONE_TT && ttsc < pc_beta)) {
        MovePicker pcpick(pos, g_hist, ttm.is_capture() ? ttm : Move(), nullptr, nullptr, nullptr,
                          /*quiescence=*/true);
        for (Move m; (m = pcpick.next()).to_from() != 0;) {
          if (!m.is_capture() || !see_ge(pos, m, pc_beta - ss->static_eval))
            continue;
          ss->move = m;
          ss->ch   = &g_hist.cont[pos.at(m.from())][m.to()];
          ++t.nodes;
          t.ev.push(pos, m);
          do_move(pos, m);
          tt::prefetch(tt_key(pos));
          int v = -qsearch<false>(t, pos, ss + 1, -pc_beta, -pc_beta + 1, ply + 1);
          if (v >= pc_beta) // qsearch agrees: confirm with a reduced full search
            v = -negamax<false>(t, pos, ss + 1, -pc_beta, -pc_beta + 1, depth - 4, ply + 1, !cutnode);
          undo_move(pos, m);
          t.ev.pop();
          if (stopped()) [[unlikely]] // see the qsearch move loop
            return 0;
          if (v >= pc_beta) {
            tt::store(tp.slot, key, m, to_tt(v, ply), raw_eval, depth - 3, tt::LOWER, false);
            return v;
          }
        }
      }
    }

    // --- move loop ---
    MovePicker picker(pos, g_hist, ttm, ss->killers, (ss - 1)->ch, (ss - 2)->ch, /*quiescence=*/false);
    if (picker.total() == 0) {
      if (excluded) // all-moves-excluded can't happen (1 move excluded), but be safe
        return alpha;
      return in_check ? -MATE + ply : 0; // checkmate / stalemate
    }

    const int lmp_limit = (3 + depth * depth) / (2 - improving);

    Move best_move{};
    int  best       = -INF;
    int  move_count = 0, quiet_count = 0;
    Move quiets_tried[64];
    Move capts_tried[32];
    int  n_quiets = 0, n_capts = 0;
    auto bound = tt::UPPER;

    for (Move m; (m = picker.next()).to_from() != 0;) {
      if (m.to_from() == ss->excluded.to_from())
        continue;
      ++move_count;
      const bool quiet = is_quiet(m);

      // --- move-loop pruning: only once a real score is on the board (best > mated) ---
      if (!root && best > -MATE_IN_MAX) {
        if (quiet && !in_check) {
          // Late move pruning: beyond this move count, quiets almost never rescue the node.
          if (depth <= 8 && quiet_count >= lmp_limit)
            continue;
          // Futility pruning: eval + margin can't reach alpha -> skip quiets.
          if (depth <= 6 && std::abs(alpha) < MATE_IN_MAX &&
              ss->static_eval + 100 + 120 * depth <= alpha)
            continue;
          // History pruning: consistently bad quiets die early at shallow depth.
          if (depth <= 4 && quiet_hist(ss, pos, m) < -2048 * depth)
            continue;
        }
        // PVS SEE pruning, depth-scaled thresholds (captures and quiets). A winning-band
        // capture from the lazy-SEE picker already passed see_ge(m, 0), which implies any
        // negative threshold (see_ge is monotone in the threshold) — skip the call for those.
        if (depth <= 8 && picker.yielded_see() != MovePicker::SEE_WINNING &&
            !see_ge(pos, m, quiet ? -50 * depth : -90 * depth))
          continue;
      }

      // --- singular extension / multicut on the TT move ---
      int extension = 0;
      if (!root && !excluded && depth >= 8 && m.to_from() == ttm.to_from() && tp.depth >= depth - 3 &&
          (tp.bound & tt::LOWER) && std::abs(ttsc) < MATE_IN_MAX && ply < 2 * t.root_depth) {
        const int s_beta = ttsc - 2 * depth;
        ss->excluded     = m;
        const int v      = negamax<false>(t, pos, ss, s_beta - 1, s_beta, (depth - 1) / 2, ply, cutnode);
        ss->excluded     = Move();
        if (v < s_beta) {
          extension = 1; // the TT move is singular: nothing else comes close -> look deeper
          if (!PV && v < s_beta - 25 && ss->double_ext < 6) {
            extension = 2 + (quiet && v < s_beta - 100); // double (rarely triple) extension
            ++ss->double_ext;
          }
        } else if (v >= beta && std::abs(v) < MATE_IN_MAX)
          return v; // multicut: even without the TT move we beat beta
        else if (ttsc >= beta)
          extension = -2; // negative extension: TT move not singular and already >= beta
        else if (cutnode)
          extension = -1;
      } else if (in_check && ply < 2 * t.root_depth)
        extension = 1; // capped check extension

      const Piece moved = pos.at(m.from());
      ss->move          = m;
      ss->ch            = &g_hist.cont[moved][m.to()];

      const uint64_t nodes_before = t.nodes; // for the root's node-based time management below
      ++t.nodes;
      t.ev.push(pos, m);
      do_move(pos, m);
      tt::prefetch(tt_key(pos));

      const int new_depth = depth - 1 + extension;
      int       v         = -INF;

      // --- LMR + PVS ---
      if (depth >= 3 && move_count > 1 + 2 * PV && (quiet || move_count > 6)) {
        int r = lmr_base(depth, move_count);
        r += cutnode;
        r += !improving;
        r -= PV;
        if (quiet)
          r -= std::clamp(quiet_hist(ss, pos, m) / 8192, -2, 2);
        else
          r /= 2; // captures get a gentler reduction
        r = std::clamp(r, 0, new_depth - 1);

        v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
        if (v > alpha && r > 0) {
          // The reduced search beat alpha: pick a confirmation depth from how convincingly it
          // did so against the best move found so far, then re-search at that depth. Well past
          // best -> the move looks genuinely strong, so look one ply deeper than normal to
          // confirm it properly; only just past alpha -> treat the reduced score as noise from
          // the cheap search and confirm one ply shallower instead of the full new_depth.
          const int confirm_depth =
                  std::clamp(new_depth + int(v > best + 40) - int(v < best + 15), 1, new_depth + 1);
          v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, confirm_depth, ply + 1, !cutnode);
        }
      } else if (!PV || move_count > 1)
        v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, new_depth, ply + 1, PV ? true : !cutnode);

      if (PV && (move_count == 1 || (v > alpha && v < beta)))
        v = -negamax<true>(t, pos, ss + 1, -beta, -alpha, new_depth, ply + 1, false);

      undo_move(pos, m);
      t.ev.pop();
      if (root && move_count == 1) { // node-based time management reads this back in think()
        t.root_m1_nodes = t.nodes - nodes_before;
        t.root_m1_move  = m;
      }
      if (time_up(t)) [[unlikely]] // see the qsearch move loop
        return 0;

      if (quiet) {
        ++quiet_count;
        if (n_quiets < 64)
          quiets_tried[n_quiets++] = m;
      } else if (m.is_capture() && n_capts < 32)
        capts_tried[n_capts++] = m;

      if (v > best) {
        best = v;
        if (v > alpha) {
          best_move = m;
          alpha     = v;
          bound     = tt::EXACT;
          if constexpr (PV) {
            ss->pv[0] = m;
            if (v < beta) { // fail-high values come from zero-window searches: stale child pv
              std::copy((ss + 1)->pv, (ss + 1)->pv + (ss + 1)->pv_len, ss->pv + 1);
              ss->pv_len = (ss + 1)->pv_len + 1;
            } else
              ss->pv_len = 1;
          }
          if (v >= beta) {
            bound = tt::LOWER;
            // History updates: reward the cutoff move, punish the tried-and-failed ones.
            if (quiet)
              update_quiet_hists(ss, pos, m, quiets_tried, n_quiets - 1, depth);
            update_capture_hists(pos, m, capts_tried, n_capts - (m.is_capture() ? 1 : 0), depth);
            break;
          }
        }
      }

    }

    if (best == -INF) // every move was pruned away: fall back to a fail-low bound
      best = alpha;

    if (!excluded) {
      tt::store(tp.slot, key, best_move, to_tt(best, ply), raw_eval, depth, bound, PV);

      // Static-eval correction history: when the search result disagrees with the static eval
      // in a bound-consistent way, learn the offset — in EVERY table at once (pawn skeleton,
      // material, minor/major placement, and the previous-move continuation), each moving
      // toward the same target independently at its own key.
      if (!in_check && (best_move.to_from() == 0 || is_quiet(best_move)) &&
          !(bound == tt::LOWER && best <= ss->static_eval) &&
          !(bound == tt::UPPER && best >= ss->static_eval) && std::abs(best) < MATE_IN_MAX) {
        const int   b = std::clamp((best - ss->static_eval) * depth / 8, -256, 256);
        const Color c = pos.turn();
        hist_update(g_hist.corr_pawn[c][pawn_corr_index(pos)], b);
        hist_update(g_hist.corr_material[c][material_corr_index(pos)], b);
        hist_update(g_hist.corr_minor[c][minor_corr_index(pos)], b);
        hist_update(g_hist.corr_major[c][major_corr_index(pos)], b);
        if ((ss - 1)->move.to_from() != 0)
          hist_update(g_hist.corr_cont[pos.at((ss - 1)->move.to())][(ss - 1)->move.to()], b);
      }
    }
    return best;
  }

  // Aspiration windows around the previous iteration's score, widening on failure. Runs on
  // `t` — the main thread from think(), or a helper from smp_worker_iterate.
  int aspiration(ThreadData &t, Position &pos, int depth, int prev) {
    Stack *ss    = t.stack + 4;
    int    delta = 14;
    int    alpha = -INF, beta = INF;
    if (depth >= 4) {
      alpha = std::max(prev - delta, -INF);
      beta  = std::min(prev + delta, INF);
    }
    while (true) {
      const int v = negamax<true>(t, pos, ss, alpha, beta, depth, 0, false);
      if (stopped())
        return v;
      if (v <= alpha) {
        beta  = (alpha + beta) / 2;
        alpha = std::max(v - delta, -INF);
      } else if (v >= beta)
        beta = std::min(v + delta, INF);
      else
        return v;
      delta += delta / 2;
    }
  }

} // namespace

// One lazy-SMP helper's whole search (called from ThreadPool::worker in smp.cpp): a private
// iterative-deepening loop over the helper's own copy of the root, stopping when the main
// thread raises g_helper_stop (or the caller's g_stop / the hard clock fires — time_up polls
// all of them). Odd-indexed helpers start one ply deeper so the pool doesn't move in
// lockstep; everything a helper learns reaches the main thread through the shared TT and
// history tables, its Result is never read.
void search::smp_worker_iterate(ThreadData &t, Position &pos, int max_depth, int idx) {
  std::memset(t.stack, 0, sizeof(t.stack));
  t.ev.reset(pos);
  int prev = 0;
  for (int d = 1 + (idx & 1); d <= max_depth && !stopped(); ++d) {
    t.root_depth = d;
    const int v  = aspiration(t, pos, d, prev);
    if (stopped())
      break;
    prev = v;
  }
  // A helper that exhausts max_depth (fixed-depth searches) simply idles out: searching past
  // the requested depth could only produce results the main thread never looks at.
}

void search::request_stop() { g_stop.store(true, std::memory_order_relaxed); }

void search::clear_stop() { g_stop.store(false, std::memory_order_relaxed); }

void search::new_game() { g_hist.clear(); }

void search::set_threads(int n) { pool().set_size(std::max(0, n - 1)); }

void search::set_contempt(int cp) { g_contempt = cp; }

// The caller must have already called search::clear_stop() SYNCHRONOUSLY on whatever thread
// might race a subsequent request_stop() (see search.h) before invoking this — think() itself
// does not reset g_stop, precisely so a stop requested before this function starts running (on
// a freshly-spawned search thread, for instance) is never silently discarded.
search::Result search::think(Position &pos, int max_depth, const InfoFn &info, int64_t soft_ms, int64_t hard_ms) {
  if (!g_lmr_init)
    init_lmr();
  g_root_color = pos.turn(); // fixes the draw_score() reference side for this whole search
  g_main.nodes = 0;
  g_hard_ms    = hard_ms;
  g_t0         = Clock::now();
  std::memset(g_main.stack, 0, sizeof(g_main.stack));
  g_main.ev.reset(pos);
  pool().reset_counters();
  tt::new_search();

  max_depth = std::clamp(max_depth, 1, MAX_PLY - 1);

  // Kick the lazy-SMP helpers into their own iterative-deepening loops. Safe to reset the
  // wind-down flag here: wait_idle() at the end of the previous think() guaranteed every
  // helper is asleep again, so nothing can still be reading it.
  g_helper_stop.store(false, std::memory_order_relaxed);
  pool().start_search(pos, max_depth);

  Result res;
  int    prev      = 0;
  Move   last_best{}; // best move of the previous completed iteration (stability tracking)
  int    stability = 0; // consecutive iterations that confirmed the same best move
  for (int d = 1; d <= max_depth; ++d) {
    g_main.root_depth = d;
    g_main.seldepth   = 0;
    const int v = aspiration(g_main, pos, d, prev);
    if (g_stop.load(std::memory_order_relaxed) && d > 1)
      break; // discard the aborted iteration; the previous full one stands

    Stack   *ss         = g_main.stack + 4;
    const int prev_score = prev; // previous ITERATION's score, before it's overwritten below
    prev         = v;
    res.score    = v;
    res.nodes    = all_nodes();
    res.seldepth = g_main.seldepth;
    res.pv.assign(ss->pv, ss->pv + ss->pv_len);
    if (!res.pv.empty()) {
      res.best = res.pv[0];
      // Best-move stability: count consecutive completed iterations confirming one move.
      if (res.pv[0].to_from() == last_best.to_from())
        ++stability;
      else
        stability = 0;
      last_best = res.pv[0];
    }

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_t0).count();
    if (info)
      info(d, res, res.nodes, ms);
    if (g_stop.load(std::memory_order_relaxed))
      break;

    // Time management: scale the soft deadline from three signals of the just-completed
    // iteration, multiplied together (the hard deadline always caps the result).
    int64_t effective_soft = soft_ms;
    if (soft_ms > 0 && d >= 5 && res.nodes >= 1000 && !res.pv.empty()) {
      double scale = 1.0;
      // Node concentration: how much of this iteration's effort went into the move we
      // already believe is best (see ThreadData::root_m1_nodes — the MAIN thread's own
      // counters, measured against its own nodes, not the pool total). Heavily focused
      // (frac -> 1) => confident, cut toward half; spread thin (frac -> 0) => unclear, allow
      // up to 1.5x. Only meaningful when root_m1_move actually IS the reported best move —
      // the rarer case where a later move overtook it gets no concentration scaling.
      if (g_main.root_m1_move.to_from() == res.pv[0].to_from() && g_main.nodes > 0) {
        const double frac = double(g_main.root_m1_nodes) / double(g_main.nodes);
        scale *= std::clamp(1.5 - frac, 0.5, 1.5);
      }
      // Best-move stability: a move that keeps being re-confirmed needs less and less
      // verification (each confirmation shaves 5%, floor 0.85x after 8); a move that JUST
      // changed is suspect — spend up to 1.25x resolving it.
      scale *= 1.25 - 0.05 * double(std::min(stability, 8));
      // Falling eval: the score dropping since the last iteration means the position is
      // turning out worse than believed — stretch the budget (~1.25x at a 50cp drop, capped
      // 1.4x); a rising score shrinks it mildly, floored at 0.85x.
      scale *= std::clamp(1.0 + double(prev_score - v) * 0.005, 0.85, 1.4);
      // Keep the product inside sane bounds: never below 0.4x nor past 2x the base budget.
      scale          = std::clamp(scale, 0.4, 2.0);
      effective_soft = int64_t(double(soft_ms) * scale);
      if (hard_ms > 0)
        effective_soft = std::min(effective_soft, hard_ms);
    }
    if (soft_ms > 0 && ms >= effective_soft)
      break;
  }

  // Wind the helpers down and wait them out BEFORE returning: the caller prints bestmove the
  // moment this returns, and a UCI engine must not keep burning CPU past that. Also makes the
  // final node total below exact rather than a live racing read.
  g_helper_stop.store(true, std::memory_order_relaxed);
  pool().wait_idle();

  // Guarantee a legal bestmove even if depth 1 was aborted before completing.
  if (res.best.to_from() == 0) {
    Move buf[218];
    const size_t n = pos.turn() == WHITE ? size_t(pos.generate_legals<WHITE>(buf) - buf)
                                         : size_t(pos.generate_legals<BLACK>(buf) - buf);
    if (n > 0)
      res.best = buf[0];
  }
  res.nodes = all_nodes();
  return res;
}
