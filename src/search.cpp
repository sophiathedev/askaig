#include "search.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include "history.h"
#include "movepick.h"
#include "nnue.h"
#include "see.h"
#include "tt.h"

// Fail-soft negamax with, in rough order of gating: TT cutoffs, static-eval correction
// history, IIR, RFP, NMP, singular extensions (with multicut and double/negative extensions),
// LMP, futility pruning, history pruning, SEE pruning (captures and quiets), LMR (log
// formula), PVS, and a SEE/delta-pruned quiescence. Move ordering in movepick.h uses the TT
// move, MVV + capture history, killers, and butterfly + continuation (CMH/FMH) history.
namespace {

  using namespace search;
  using Clock = std::chrono::steady_clock;

  // --- shared state (single-threaded search) ---------------------------------------------------

  struct Stack {
    Move       pv[MAX_PLY + 1];
    int        pv_len;
    Move       killers[2];
    Move       move; // the move made AT this ply (null move: to_from() == 0 with ch == nullptr)
    Move       excluded; // the move excluded by a singular-verification search
    int        static_eval;
    ContTable *ch; // continuation-history slice of `move`
    int        double_ext; // double-extension budget spent on this path
    bool       in_check;
  };

  struct ThreadData {
    nnue::Evaluator            ev;
    std::unique_ptr<Histories> hist = std::make_unique<Histories>();
    Stack                      stack[MAX_PLY + 8];
    uint64_t                   nodes = 0;
    int                        seldepth = 0, root_depth = 1;
    int64_t                    hard_ms = 0;
    Clock::time_point          t0;
  };

  ThreadData        td;
  std::atomic<bool> g_stop{false};

  // LMR reduction table, ln(depth)*ln(moves) scaled (the "most principled" log formula).
  int  g_lmr[64][64];
  bool g_lmr_init = false;
  void init_lmr() {
    for (int d = 1; d < 64; ++d)
      for (int m = 1; m < 64; ++m)
        g_lmr[d][m] = int(0.8 + std::log(d) * std::log(m) / 2.3);
    g_lmr_init = true;
  }
  int lmr_base(int depth, int movecount) { return g_lmr[std::min(depth, 63)][std::min(movecount, 63)]; }

  // --- small helpers ----------------------------------------------------------------------------

  void do_move(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.play<WHITE>(m);
    else
      p.play<BLACK>(m);
  }
  void undo_move(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.undo<BLACK>(m);
    else
      p.undo<WHITE>(m);
  }
  bool stm_in_check(const Position &p) { return p.turn() == WHITE ? p.in_check<WHITE>() : p.in_check<BLACK>(); }

  bool is_quiet(Move m) {
    const MoveFlags f = m.flags();
    return f == QUIET || f == DOUBLE_PUSH || f == OO || f == OOO;
  }

  // The engine Zobrist hash omits castling rights and the en-passant square (they change the
  // legal moves, so two positions differing only there must not share a TT slot) — mix them in.
  uint64_t tt_key(const Position &p) {
    return p.get_hash() ^ ((p.castle_entry() & ALL_CASTLING_MASK) * 0x9E3779B97F4A7C15ull) ^
           (uint64_t(uint16_t(p.history[p.ply()].epsq)) * 0xC2B2AE3D27D4EB4Full);
  }

  // Pawn-structure key for the static-eval correction history.
  size_t corr_index(const Position &p) {
    const uint64_t w = p.bitboard_of(WHITE_PAWN) * 0x9E3779B97F4A7C15ull;
    const uint64_t b = p.bitboard_of(BLACK_PAWN) * 0xC2B2AE3D27D4EB4Full;
    return (w ^ (b + 0x165667B19E3779F9ull + (w << 6) + (w >> 2))) & (Histories::CORR_SIZE - 1);
  }

  // Mate scores are stored ply-relative in the TT ("mate in N from HERE").
  int to_tt(int v, int ply) { return v >= MATE_IN_MAX ? v + ply : v <= -MATE_IN_MAX ? v - ply : v; }
  int from_tt(int v, int ply) { return v >= MATE_IN_MAX ? v - ply : v <= -MATE_IN_MAX ? v + ply : v; }

  // Static eval + the learned per-pawn-structure correction, clamped out of the mate range.
  int evaluate(const Position &pos) {
    const int raw  = td.ev.evaluate(pos);
    const int corr = td.hist->corr[pos.turn()][corr_index(pos)] / 64;
    return std::clamp(raw + corr, -MATE_IN_MAX + 1, MATE_IN_MAX - 1);
  }

  bool time_up() {
    if (g_stop.load(std::memory_order_relaxed))
      return true;
    if (td.hard_ms > 0 && (td.nodes & 2047) == 0) {
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - td.t0).count();
      if (ms >= td.hard_ms) {
        g_stop.store(true, std::memory_order_relaxed);
        return true;
      }
    }
    return false;
  }

  int hist_bonus(int depth) { return std::min(160 * depth - 80, 2000); }

  // Butterfly + continuation-history score of a quiet move (ordering and history pruning).
  int quiet_hist(const Stack *ss, const Position &pos, Move m) {
    const Piece pc = pos.at(m.from());
    int         h  = td.hist->butterfly[pos.turn()][m.from()][m.to()];
    if ((ss - 1)->ch)
      h += (*(ss - 1)->ch)[pc][m.to()];
    if ((ss - 2)->ch)
      h += (*(ss - 2)->ch)[pc][m.to()];
    return h;
  }

  void update_quiet_hists(Stack *ss, const Position &pos, Move best, const Move *tried, int n_tried, int depth) {
    const int bonus = hist_bonus(depth);
    const auto touch = [&](Move m, int b) {
      const Piece pc = pos.at(m.from());
      hist_update(td.hist->butterfly[pos.turn()][m.from()][m.to()], b);
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

  void update_capture_hists(const Position &pos, Move best, const Move *tried, int n_tried, int depth) {
    const int  bonus = hist_bonus(depth);
    const auto touch = [&](Move m, int b) {
      const PieceType captured = m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
      hist_update(td.hist->capture[pos.at(m.from())][m.to()][captured], b);
    };
    if (best.is_capture())
      touch(best, bonus);
    for (int i = 0; i < n_tried; ++i)
      touch(tried[i], -bonus);
  }

  // --- quiescence -------------------------------------------------------------------------------

  template<bool PV>
  int qsearch(Position &pos, Stack *ss, int alpha, int beta, int ply) {
    if constexpr (PV)
      ss->pv_len = 0;
    td.seldepth = std::max(td.seldepth, ply);
    if (time_up())
      return 0;
    if (pos.is_draw())
      return 0;
    if (ply >= MAX_PLY)
      return evaluate(pos);

    const bool     in_check = stm_in_check(pos);
    const uint64_t key      = tt_key(pos);

    bool       tthit = false;
    tt::Entry *tte   = tt::probe(key, tthit);
    const int  ttsc  = tthit && tte->score != tt::VALUE_NONE_TT ? from_tt(tte->score, ply) : tt::VALUE_NONE_TT;
    if (!PV && tthit && ttsc != tt::VALUE_NONE_TT) {
      const auto b = tt::Bound(tte->genbound & 0x3);
      if (b == tt::EXACT || (b == tt::LOWER && ttsc >= beta) || (b == tt::UPPER && ttsc <= alpha))
        return ttsc;
    }

    int best = -INF, raw_eval = tt::VALUE_NONE_TT;
    if (!in_check) {
      raw_eval = tthit && tte->eval != tt::VALUE_NONE_TT ? tte->eval : evaluate(pos);
      best     = raw_eval;
      // The TT score is a tighter bound on this position than the static eval — use it.
      if (tthit && ttsc != tt::VALUE_NONE_TT) {
        const auto b = tt::Bound(tte->genbound & 0x3);
        if (b == tt::EXACT || (b == tt::LOWER && ttsc > best) || (b == tt::UPPER && ttsc < best))
          best = ttsc;
      }
      if (best >= beta)
        return best;
      alpha = std::max(alpha, best);
    }
    const int futility_base = best + 120;

    const Move ttm = tthit ? Move(tte->move) : Move();
    MovePicker picker(pos, *td.hist, ttm, nullptr, nullptr, nullptr, /*quiescence=*/true);
    if (in_check && picker.total() == 0)
      return -MATE + ply;

    Move best_move{};
    for (Move m; (m = picker.next()).to_from() != 0;) {
      if (!in_check) {
        // QS SEE pruning: don't even try losing captures.
        if (!see_ge(pos, m, 0))
          continue;
        // QS futility (delta) pruning: stand pat + margin + captured value can't reach alpha.
        const PieceType captured = m.flags() == EN_PASSANT ? PAWN : type_of(pos.at(m.to()));
        if (futility_base + PIECE_VAL[captured] <= alpha && m.flags() != PR_QUEEN && m.flags() != PC_QUEEN)
          continue;
      }

      ++td.nodes;
      td.ev.push(pos, m);
      do_move(pos, m);
      const int v = -qsearch<PV>(pos, ss + 1, -beta, -alpha, ply + 1);
      undo_move(pos, m);
      td.ev.pop();
      if (g_stop.load(std::memory_order_relaxed))
        return 0;

      if (v > best) {
        best = v;
        if (v > alpha) {
          best_move = m;
          alpha     = v;
          if constexpr (PV) {
            ss->pv[0] = m;
            std::memcpy(ss->pv + 1, (ss + 1)->pv, size_t((ss + 1)->pv_len) * sizeof(Move));
            ss->pv_len = (ss + 1)->pv_len + 1;
          }
          if (v >= beta)
            break;
        }
      }
    }

    tt::store(tte, key, best_move, to_tt(best, ply), raw_eval, /*depth=*/0,
              best >= beta ? tt::LOWER : tt::UPPER, PV);
    return best;
  }

  // --- main search ------------------------------------------------------------------------------

  template<bool PV>
  int negamax(Position &pos, Stack *ss, int alpha, int beta, int depth, int ply, bool cutnode) {
    if (depth <= 0)
      return qsearch<PV>(pos, ss, alpha, beta, ply);

    if constexpr (PV)
      ss->pv_len = 0;
    if (time_up())
      return 0;
    td.seldepth = std::max(td.seldepth, ply);

    const bool root = ply == 0;
    if (!root) {
      if (pos.is_draw())
        return 0;
      if (ply >= MAX_PLY)
        return evaluate(pos);
      // Mate distance pruning.
      alpha = std::max(alpha, -MATE + ply);
      beta  = std::min(beta, MATE - ply - 1);
      if (alpha >= beta)
        return alpha;
    }

    const bool excluded = ss->excluded.to_from() != 0;
    const bool in_check = stm_in_check(pos);
    ss->in_check        = in_check;
    (ss + 1)->excluded  = Move();
    ss->double_ext      = root ? 0 : (ss - 1)->double_ext;

    // --- TT probe (skipped entirely under a singular-exclusion search) ---
    const uint64_t key   = tt_key(pos);
    bool           tthit = false;
    tt::Entry     *tte   = nullptr;
    Move           ttm{};
    int            ttsc     = tt::VALUE_NONE_TT;
    int            ttdepth  = -tt::DEPTH_OFFSET;
    tt::Bound      ttbound  = tt::NONE;
    int            tteval   = tt::VALUE_NONE_TT;
    if (!excluded) {
      tte = tt::probe(key, tthit);
      if (tthit) {
        ttm     = Move(tte->move);
        ttsc    = tte->score != tt::VALUE_NONE_TT ? from_tt(tte->score, ply) : tt::VALUE_NONE_TT;
        ttdepth = int(tte->depth) - tt::DEPTH_OFFSET;
        ttbound = tt::Bound(tte->genbound & 0x3);
        tteval  = tte->eval;
      }
      if (!PV && tthit && ttdepth >= depth && ttsc != tt::VALUE_NONE_TT &&
          (ttbound == tt::EXACT || (ttbound == tt::LOWER && ttsc >= beta) || (ttbound == tt::UPPER && ttsc <= alpha)))
        return ttsc;
    }

    // --- static eval + improving ---
    int raw_eval = tt::VALUE_NONE_TT;
    if (in_check)
      ss->static_eval = tt::VALUE_NONE_TT;
    else {
      raw_eval        = tteval != tt::VALUE_NONE_TT ? tteval : evaluate(pos);
      ss->static_eval = raw_eval;
      // Sharpen with the TT score when it bounds in the right direction.
      if (tthit && ttsc != tt::VALUE_NONE_TT &&
          (ttbound == tt::EXACT || (ttbound == tt::LOWER && ttsc > raw_eval) ||
           (ttbound == tt::UPPER && ttsc < raw_eval)))
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
        const int v = qsearch<false>(pos, ss, alpha - 1, alpha, ply);
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
        td.ev.push_null();
        pos.play_null();
        const int v = -negamax<false>(pos, ss + 1, -beta, -beta + 1, depth - R, ply + 1, !cutnode);
        pos.undo_null();
        td.ev.pop();
        if (g_stop.load(std::memory_order_relaxed))
          return 0;
        if (v >= beta)
          return v >= MATE_IN_MAX ? beta : v; // don't return unproven mates
      }

      // ProbCut: a good capture that beats beta by a margin at reduced depth almost certainly
      // produces a full-depth beta cutoff too. Skipped when the TT already says otherwise.
      const int pc_beta = beta + 180 - 60 * improving;
      if (depth >= 5 && std::abs(beta) < MATE_IN_MAX &&
          !(tthit && ttdepth >= depth - 3 && ttsc != tt::VALUE_NONE_TT && ttsc < pc_beta)) {
        MovePicker pcpick(pos, *td.hist, ttm.is_capture() ? ttm : Move(), nullptr, nullptr, nullptr,
                          /*quiescence=*/true);
        for (Move m; (m = pcpick.next()).to_from() != 0;) {
          if (!m.is_capture() || !see_ge(pos, m, pc_beta - ss->static_eval))
            continue;
          ss->move = m;
          ss->ch   = &td.hist->cont[pos.at(m.from())][m.to()];
          ++td.nodes;
          td.ev.push(pos, m);
          do_move(pos, m);
          int v = -qsearch<false>(pos, ss + 1, -pc_beta, -pc_beta + 1, ply + 1);
          if (v >= pc_beta) // qsearch agrees: confirm with a reduced full search
            v = -negamax<false>(pos, ss + 1, -pc_beta, -pc_beta + 1, depth - 4, ply + 1, !cutnode);
          undo_move(pos, m);
          td.ev.pop();
          if (g_stop.load(std::memory_order_relaxed))
            return 0;
          if (v >= pc_beta) {
            tt::store(tte, key, m, to_tt(v, ply), raw_eval, depth - 3, tt::LOWER, false);
            return v;
          }
        }
      }
    }

    // --- move loop ---
    MovePicker picker(pos, *td.hist, ttm, ss->killers, (ss - 1)->ch, (ss - 2)->ch, /*quiescence=*/false);
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
        // PVS SEE pruning, depth-scaled thresholds (captures and quiets).
        if (depth <= 8 && !see_ge(pos, m, quiet ? -50 * depth : -90 * depth))
          continue;
      }

      // --- singular extension / multicut on the TT move ---
      int extension = 0;
      if (!root && !excluded && depth >= 8 && m.to_from() == ttm.to_from() && ttdepth >= depth - 3 &&
          (ttbound & tt::LOWER) && std::abs(ttsc) < MATE_IN_MAX && ply < 2 * td.root_depth) {
        const int s_beta = ttsc - 2 * depth;
        ss->excluded     = m;
        const int v      = negamax<false>(pos, ss, s_beta - 1, s_beta, (depth - 1) / 2, ply, cutnode);
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
      } else if (in_check && ply < 2 * td.root_depth)
        extension = 1; // capped check extension

      const Piece moved = pos.at(m.from());
      ss->move          = m;
      ss->ch            = &td.hist->cont[moved][m.to()];

      ++td.nodes;
      td.ev.push(pos, m);
      do_move(pos, m);

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

        v = -negamax<false>(pos, ss + 1, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
        if (v > alpha && r > 0) // reduced search beat alpha: verify at full depth
          v = -negamax<false>(pos, ss + 1, -alpha - 1, -alpha, new_depth, ply + 1, !cutnode);
      } else if (!PV || move_count > 1)
        v = -negamax<false>(pos, ss + 1, -alpha - 1, -alpha, new_depth, ply + 1, PV ? true : !cutnode);

      if (PV && (move_count == 1 || (v > alpha && v < beta)))
        v = -negamax<true>(pos, ss + 1, -beta, -alpha, new_depth, ply + 1, false);

      undo_move(pos, m);
      td.ev.pop();
      if (g_stop.load(std::memory_order_relaxed))
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
            std::memcpy(ss->pv + 1, (ss + 1)->pv, size_t((ss + 1)->pv_len) * sizeof(Move));
            ss->pv_len = (ss + 1)->pv_len + 1;
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
      tt::store(tte, key, best_move, to_tt(best, ply), raw_eval, depth, bound, PV);

      // Static-eval correction history: when the search result disagrees with the static eval
      // in a bound-consistent way, learn the offset for this pawn structure.
      if (!in_check && (best_move.to_from() == 0 || is_quiet(best_move)) &&
          !(bound == tt::LOWER && best <= ss->static_eval) &&
          !(bound == tt::UPPER && best >= ss->static_eval) && std::abs(best) < MATE_IN_MAX) {
        const int b = std::clamp((best - ss->static_eval) * depth / 8, -256, 256);
        hist_update(td.hist->corr[pos.turn()][corr_index(pos)], b);
      }
    }
    return best;
  }

  // Aspiration windows around the previous iteration's score, widening on failure.
  int aspiration(Position &pos, int depth, int prev) {
    Stack *ss    = td.stack + 4;
    int    delta = 14;
    int    alpha = -INF, beta = INF;
    if (depth >= 4) {
      alpha = std::max(prev - delta, -INF);
      beta  = std::min(prev + delta, INF);
    }
    while (true) {
      const int v = negamax<true>(pos, ss, alpha, beta, depth, 0, false);
      if (g_stop.load(std::memory_order_relaxed))
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

void search::request_stop() { g_stop.store(true, std::memory_order_relaxed); }

void search::new_game() { td.hist->clear(); }

search::Result search::think(Position &pos, int max_depth, const InfoFn &info, int64_t soft_ms, int64_t hard_ms) {
  if (!g_lmr_init)
    init_lmr();
  g_stop.store(false, std::memory_order_relaxed);
  td.nodes   = 0;
  td.hard_ms = hard_ms;
  td.t0      = Clock::now();
  std::memset(td.stack, 0, sizeof(td.stack));
  td.ev.reset(pos);
  tt::new_search();

  max_depth = std::clamp(max_depth, 1, MAX_PLY - 1);

  Result res;
  int    prev = 0;
  for (int d = 1; d <= max_depth; ++d) {
    td.root_depth = d;
    td.seldepth   = 0;
    const int v   = aspiration(pos, d, prev);
    if (g_stop.load(std::memory_order_relaxed) && d > 1)
      break; // discard the aborted iteration; the previous full one stands

    Stack *ss = td.stack + 4;
    prev      = v;
    res.score = v;
    res.nodes = td.nodes;
    res.seldepth = td.seldepth;
    res.pv.assign(ss->pv, ss->pv + ss->pv_len);
    if (!res.pv.empty())
      res.best = res.pv[0];

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - td.t0).count();
    if (info)
      info(d, res, td.nodes, ms);
    if (g_stop.load(std::memory_order_relaxed))
      break;
    if (soft_ms > 0 && ms >= soft_ms)
      break;
  }

  // Guarantee a legal bestmove even if depth 1 was aborted before completing.
  if (res.best.to_from() == 0) {
    Move buf[218];
    const size_t n = pos.turn() == WHITE ? size_t(pos.generate_legals<WHITE>(buf) - buf)
                                         : size_t(pos.generate_legals<BLACK>(buf) - buf);
    if (n > 0)
      res.best = buf[0];
  }
  res.nodes = td.nodes;
  return res;
}
