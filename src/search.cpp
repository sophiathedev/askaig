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

namespace {

  using namespace search;
  using Clock = std::chrono::steady_clock;

  Histories          g_hist;
  ThreadData         g_main;
  std::atomic<bool>  g_stop{false};
  std::atomic<bool>  g_helper_stop{false};
  int64_t            g_hard_ms = 0;
  uint64_t           g_node_limit = 0; // 0 = off
  Clock::time_point  g_t0;
  int                g_contempt   = 0;
  Color              g_root_color = WHITE; // draw reference

  int  g_lmr[64][64];
  bool g_lmr_init = false;
  void init_lmr() {
    for (int d = 1; d < 64; ++d)
      for (int m = 1; m < 64; ++m)
        g_lmr[d][m] = int(prm.LMR_BASE / 100.0 + std::log(d) * std::log(m) / (prm.LMR_DIV / 100.0));
    g_lmr_init = true;
  }
  [[gnu::pure, gnu::always_inline]] inline int lmr_base(int depth, int movecount) {
    return g_lmr[std::min(depth, 63)][std::min(movecount, 63)];
  }

  [[gnu::pure, gnu::always_inline]] inline size_t pawn_corr_index(const Position &p) {
    const uint64_t w = p.bitboard_of(WHITE_PAWN) * 0x9E3779B97F4A7C15ull;
    const uint64_t b = p.bitboard_of(BLACK_PAWN) * 0xC2B2AE3D27D4EB4Full;
    return (w ^ (b + 0x165667B19E3779F9ull + (w << 6) + (w >> 2))) & (Histories::CORR_SIZE - 1);
  }
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

  [[gnu::const, gnu::always_inline]] inline int to_tt(int v, int ply) {
    return v >= MATE_IN_MAX ? v + ply : v <= -MATE_IN_MAX ? v - ply : v;
  }
  [[gnu::const, gnu::always_inline]] inline int from_tt(int v, int ply) {
    return v >= MATE_IN_MAX ? v - ply : v <= -MATE_IN_MAX ? v + ply : v;
  }

  [[gnu::pure, gnu::always_inline]] inline int draw_score(const Position &pos) {
    return pos.turn() == g_root_color ? -g_contempt : g_contempt;
  }

  struct EvalResult {
    int raw;
    int score;
  };

  // correction disagreement feeds pruning uncertainty
  [[gnu::hot, nodiscard]] EvalResult evaluate(ThreadData &t, const Position &pos, Stack *ss,
                                              int raw = tt::VALUE_NONE_TT) {
    const Color  c     = pos.turn();
    const size_t i_paw = pawn_corr_index(pos), i_mat = material_corr_index(pos);
    const size_t i_min = minor_corr_index(pos), i_maj = major_corr_index(pos);
    __builtin_prefetch(&g_hist.corr_pawn[c][i_paw]);
    __builtin_prefetch(&g_hist.corr_material[c][i_mat]);
    __builtin_prefetch(&g_hist.corr_minor[c][i_min]);
    __builtin_prefetch(&g_hist.corr_major[c][i_maj]);

    if (raw == tt::VALUE_NONE_TT) {
      const int npm = 3 * pop_count(pos.bitboard_of(WHITE_KNIGHT) | pos.bitboard_of(BLACK_KNIGHT) |
                                    pos.bitboard_of(WHITE_BISHOP) | pos.bitboard_of(BLACK_BISHOP)) +
                      5 * pop_count(pos.bitboard_of(WHITE_ROOK) | pos.bitboard_of(BLACK_ROOK)) +
                      9 * pop_count(pos.bitboard_of(WHITE_QUEEN) | pos.bitboard_of(BLACK_QUEEN));
      raw           = t.ev.evaluate(pos) * (prm.MAT_BASE + prm.MAT_MULT * npm) / 1024;
      raw           = raw * (200 - std::min(pos.fifty(), 100)) / 200;
      raw           = std::clamp(raw, -MATE_IN_MAX + 1, MATE_IN_MAX - 1);
    }

    const int c1 = g_hist.corr_pawn[c][i_paw], c2 = g_hist.corr_material[c][i_mat];
    const int c3 = g_hist.corr_minor[c][i_min], c4 = g_hist.corr_major[c][i_maj];
    int       corr = c1 + c2 + c3 + c4;
    int       mass = std::abs(c1) + std::abs(c2) + std::abs(c3) + std::abs(c4);
    if ((ss - 1)->move.to_from() != 0) {
      const int c5 = g_hist.corr_cont[pos.at((ss - 1)->move.to())][(ss - 1)->move.to()];
      corr += c5;
      mass += std::abs(c5);
    }
    ss->eval_unc = mass - std::abs(corr);
    return {raw, std::clamp(raw + corr / 256, -MATE_IN_MAX + 1, MATE_IN_MAX - 1)};
  }

  [[gnu::hot]] inline bool stopped() {
    return g_stop.load(std::memory_order_relaxed) || g_helper_stop.load(std::memory_order_relaxed);
  }

  [[gnu::hot]] bool time_up(ThreadData &t) {
    if (stopped())
      return true;
    if ((t.nodes & 2047) == 0) {
      if (g_node_limit && g_main.nodes + pool().total_nodes() >= g_node_limit) {
        g_stop.store(true, std::memory_order_relaxed);
        return true;
      }
      if (g_hard_ms > 0) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_t0).count();
        if (ms >= g_hard_ms) {
          g_stop.store(true, std::memory_order_relaxed);
          return true;
        }
      }
    }
    return false;
  }

  [[gnu::pure, gnu::always_inline]] inline int hist_bonus(int depth) {
    return std::min(prm.HB_MULT * depth - prm.HB_SUB, prm.HB_MAX);
  }

  [[gnu::pure, gnu::always_inline]] inline int quiet_hist(const Stack *ss, const Position &pos, Move m) {
    const Piece pc = pos.at(m.from());
    int         h  = g_hist.butterfly[pos.turn()][m.from()][m.to()];
    if ((ss - 1)->ch)
      h += (*(ss - 1)->ch)[pc][m.to()];
    if ((ss - 2)->ch)
      h += (*(ss - 2)->ch)[pc][m.to()];
    return h;
  }

  [[gnu::hot]] inline void update_tt_quiet_hist(Stack *ss, const Position &pos, Move m, int depth) {
    const Piece pc = pos.at(m.from());
    if (depth < 8 || m.to_from() == 0 || !is_quiet(m) || pc == NO_PIECE || color_of(pc) != pos.turn() ||
        pos.at(m.to()) != NO_PIECE)
      return;

    const int bonus = hist_bonus(depth) / 8;
    hist_update(g_hist.butterfly[pos.turn()][m.from()][m.to()], bonus);
    if ((ss - 1)->ch)
      hist_update((*(ss - 1)->ch)[pc][m.to()], bonus);
    if ((ss - 2)->ch)
      hist_update((*(ss - 2)->ch)[pc][m.to()], bonus);
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
    if (ss->killers[0].to_from() != best.to_from()) {
      ss->killers[1] = ss->killers[0];
      ss->killers[0] = best;
    }
    if ((ss - 1)->move.to_from() != 0)
      counter_store(g_hist.counter[pos.at((ss - 1)->move.to())][(ss - 1)->move.to()], best);
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

  template<bool PV>
  [[gnu::hot]] int qsearch(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int ply);
  template<bool PV>
  [[gnu::hot]] int negamax(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int depth, int ply,
                           bool cutnode);

  uint64_t all_nodes() { return g_main.nodes + pool().total_nodes(); }

  template<bool PV>
  [[gnu::hot]] int qsearch(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int ply) {
    if constexpr (PV)
      ss->pv_len = 0;
    t.seldepth = std::max(t.seldepth, ply);
    if (time_up(t))
      return 0;
    if (pos.is_draw())
      return draw_score(pos);
    if (ply >= MAX_PLY) [[unlikely]]
      return evaluate(t, pos, ss).score;

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
      const EvalResult ev = evaluate(t, pos, ss, tp.eval);
      raw_eval            = ev.raw;
      ss->static_eval     = ev.score;
      best                = ev.score;
      if (tp.hit && ttsc != tt::VALUE_NONE_TT) {
        if (tp.bound == tt::EXACT || (tp.bound == tt::LOWER && ttsc > best) ||
            (tp.bound == tt::UPPER && ttsc < best))
          best = ttsc;
      }
      if (best >= beta)
        return best;
      alpha = std::max(alpha, best);
    } else
      ss->static_eval = tt::VALUE_NONE_TT;
    const int futility_base = ss->static_eval + prm.QS_FUT;

    const Move ttm = tp.move;
    MovePicker picker(pos, g_hist, ttm, nullptr, Move(), nullptr, nullptr, /*quiescence=*/true);
    if (in_check && picker.total() == 0)
      return -MATE + ply;

    Move best_move{};
    for (Move m; (m = picker.next()).to_from() != 0;) {
      if (!in_check) {
        const auto band = picker.yielded_see();
        if (band == MovePicker::SEE_LOSING || (band == MovePicker::SEE_UNKNOWN && !see_ge(pos, m, 0)))
          continue;
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
      if (stopped()) [[unlikely]]
        return 0;

      if (v > best) {
        best = v;
        if (v > alpha) {
          best_move = m;
          alpha     = v;
          if constexpr (PV) {
            ss->pv[0] = m;
            if (v < beta) { // pv is stale on fail-high
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
    if (root)
      t.root_m1_nodes = 0, t.root_m1_move = Move();
    if (!root) {
      if (pos.is_draw())
        return draw_score(pos);
      if (ply >= MAX_PLY) [[unlikely]]
        return evaluate(t, pos, ss).score;
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

    const uint64_t key = tt_key(pos);
    tt::Probe      tp;
    if (!excluded) {
      tp = tt::probe(key);
      const int sc = tp.hit && tp.score != tt::VALUE_NONE_TT ? from_tt(tp.score, ply) : tt::VALUE_NONE_TT;
      if (!PV && tp.hit && tp.depth >= depth && sc != tt::VALUE_NONE_TT &&
          (tp.bound == tt::EXACT || (tp.bound == tt::LOWER && sc >= beta) || (tp.bound == tt::UPPER && sc <= alpha))) {
        if (tp.bound == tt::LOWER && sc >= beta)
          update_tt_quiet_hist(ss, pos, tp.move, depth);
        return sc;
      }
    }
    const Move ttm  = tp.move;
    const int  ttsc = tp.hit && tp.score != tt::VALUE_NONE_TT ? from_tt(tp.score, ply) : tt::VALUE_NONE_TT;

    int raw_eval = tt::VALUE_NONE_TT, prune_eval = tt::VALUE_NONE_TT;
    ss->eval_unc = 0;
    if (in_check)
      ss->static_eval = tt::VALUE_NONE_TT;
    else {
      const EvalResult ev = evaluate(t, pos, ss, tp.eval);
      raw_eval            = ev.raw;
      ss->static_eval     = ev.score;
      prune_eval          = ev.score;
      if (tp.hit && ttsc != tt::VALUE_NONE_TT &&
          (tp.bound == tt::EXACT || (tp.bound == tt::LOWER && ttsc > prune_eval) ||
           (tp.bound == tt::UPPER && ttsc < prune_eval)))
        prune_eval = ttsc;
    }
    bool improving = !in_check && ply >= 2 && (ss - 2)->static_eval != tt::VALUE_NONE_TT &&
                     ss->static_eval > (ss - 2)->static_eval;

    if ((PV || cutnode) && depth >= prm.IIR_DEPTH && ttm.to_from() == 0)
      --depth;

    if (!PV && !in_check && !excluded) {
      if (depth <= prm.RAZOR_DEPTH &&
          prune_eval + prm.RAZOR_MULT * depth + prm.RAZOR_UNC * ss->eval_unc / 1024 < alpha) {
        const int v = qsearch<false>(t, pos, ss, alpha - 1, alpha, ply);
        if (v < alpha && std::abs(v) < MATE_IN_MAX)
          return v;
      }

      if (depth <= prm.RFP_DEPTH && std::abs(beta) < MATE_IN_MAX &&
          prune_eval - prm.RFP_MULT * (depth - improving) - prm.RFP_UNC * ss->eval_unc / 1024 >= beta)
        return prune_eval;

      const Color    us       = pos.turn();
      const Bitboard non_pawn = pos.bitboard_of(us, KNIGHT) | pos.bitboard_of(us, BISHOP) |
                                pos.bitboard_of(us, ROOK) | pos.bitboard_of(us, QUEEN);
      if (depth >= prm.NMP_DEPTH && ss->static_eval >= beta && non_pawn && (ss - 1)->move.to_from() != 0 &&
          beta > -MATE_IN_MAX) {
        const int R = prm.NMP_BASE + depth / prm.NMP_DDIV + std::min((ss->static_eval - beta) / prm.NMP_EDIV, prm.NMP_ECAP);
        ss->move    = Move();
        ss->ch      = nullptr;
        t.ev.push_null();
        pos.play_null();
        tt::prefetch(tt_key(pos));
        const int v = -negamax<false>(t, pos, ss + 1, -beta, -beta + 1, depth - R, ply + 1, !cutnode);
        pos.undo_null();
        t.ev.pop();
        if (stopped()) [[unlikely]]
          return 0;
        if (v >= beta) {
          if (depth >= prm.NMP_VDEPTH && std::abs(v) < MATE_IN_MAX) {
            const int w = negamax<false>(t, pos, ss, beta - 1, beta, depth - R, ply, cutnode);
            if (stopped()) [[unlikely]]
              return 0;
            if (w >= beta)
              return v;
          } else
            return v >= MATE_IN_MAX ? beta : v; // don't return unproven mates
        }
      }
    }

    if (!in_check)
      improving |= ss->static_eval >= beta;
    if (!PV && !in_check && !excluded) {
      const int pc_beta  = beta + prm.PC_MARGIN - prm.PC_IMP * improving;
      const int pc_depth = std::max(depth - (improving ? 5 : 3), 0);
      if (depth >= prm.PC_DEPTH && std::abs(beta) < MATE_IN_MAX &&
          !(tp.hit && tp.depth >= depth - 3 && ttsc != tt::VALUE_NONE_TT && ttsc < pc_beta)) {
        MovePicker pcpick(pos, g_hist, ttm.is_capture() ? ttm : Move(), nullptr, Move(), nullptr, nullptr,
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
          if (v >= pc_beta && pc_depth > 0)
            v = -negamax<false>(t, pos, ss + 1, -pc_beta, -pc_beta + 1, pc_depth, ply + 1, !cutnode);
          undo_move(pos, m);
          t.ev.pop();
          if (stopped()) [[unlikely]]
            return 0;
          if (v >= pc_beta) {
            tt::store(tp.slot, key, m, to_tt(v, ply), raw_eval, pc_depth + 1, tt::LOWER, false);
            return v;
          }
        }
      }
    }

    const Move counter = (ss - 1)->move.to_from() != 0
                                 ? counter_load(g_hist.counter[pos.at((ss - 1)->move.to())][(ss - 1)->move.to()])
                                 : Move();
    MovePicker picker(pos, g_hist, ttm, ss->killers, counter, (ss - 1)->ch, (ss - 2)->ch, /*quiescence=*/false);
    if (picker.total() == 0) {
      if (excluded)
        return alpha;
      return in_check ? -MATE + ply : 0;
    }

    const int lmp_limit = (prm.LMP_BASE + depth * depth) / (2 - improving);

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

      if (!root && best > -MATE_IN_MAX) {
        if (quiet && !in_check) {
          if (depth <= prm.LMP_DEPTH && quiet_count >= lmp_limit)
            continue;
          if (depth <= prm.FUT_DEPTH && std::abs(alpha) < MATE_IN_MAX &&
              ss->static_eval + prm.FUT_BASE + prm.FUT_MULT * depth + prm.FUT_UNC * ss->eval_unc / 1024 <= alpha)
            continue;
          if (depth <= prm.HP_DEPTH && quiet_hist(ss, pos, m) < -prm.HP_MULT * depth)
            continue;
        }
        if (depth <= prm.SEEP_DEPTH && picker.yielded_see() != MovePicker::SEE_WINNING &&
            !see_ge(pos, m, quiet ? -prm.SEEP_QUIET * depth : -prm.SEEP_CAPT * depth))
          continue;
      }

      int extension = 0;
      if (!root && !excluded && depth >= prm.SE_DEPTH && m.to_from() == ttm.to_from() &&
          tp.depth >= depth - prm.SE_TTSUB && (tp.bound & tt::LOWER) && std::abs(ttsc) < MATE_IN_MAX &&
          ply < 2 * t.root_depth) {
        const int s_beta = ttsc - prm.SE_BMULT * depth;
        ss->excluded     = m;
        const int v      = negamax<false>(t, pos, ss, s_beta - 1, s_beta, (depth - 1) / 2, ply, cutnode);
        ss->excluded     = Move();
        if (v < s_beta) {
          extension = 1; // singular tt move
          if (!PV && v < s_beta - prm.SE_DBL && ss->double_ext < prm.SE_DBLMAX) {
            extension = 2 + (quiet && v < s_beta - prm.SE_TRI); // double (rarely triple) extension
            ++ss->double_ext;
          }
        } else if (v >= beta && std::abs(v) < MATE_IN_MAX)
          return v;
        else if (ttsc >= beta)
          extension = -2; // negative extension
        else if (cutnode)
          extension = -1;
      } else if (in_check && ply < 2 * t.root_depth)
        extension = 1;

      const bool  lmr         = depth >= 3 && move_count > 1 + 2 * PV && (quiet || move_count > prm.LMR_TACT_MC);
      const int   lmr_history = lmr && quiet ? quiet_hist(ss, pos, m) : 0;
      const Piece moved       = pos.at(m.from());
      ss->move                = m;
      ss->ch                  = &g_hist.cont[moved][m.to()];

      const uint64_t nodes_before = t.nodes; // root effort
      ++t.nodes;
      t.ev.push(pos, m);
      do_move(pos, m);
      tt::prefetch(tt_key(pos));

      const int new_depth = depth - 1 + extension;
      int       v         = -INF;

      if (lmr) {
        int r = lmr_base(depth, move_count);
        r += cutnode;
        r += !improving;
        r -= PV;
        r -= tp.pv; // reduce less on former pv nodes
        if (quiet)
          r -= std::clamp(lmr_history / 8192, -2, 2);
        else
          r /= 2;
        r = std::clamp(r, 0, new_depth - 1);

        v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
        if (v > alpha && r > 0) {
          const int confirm_depth = std::clamp(new_depth + int(v > best + prm.LMR_CONF_HI) -
                                                       int(v < best + prm.LMR_CONF_LO),
                                               1, new_depth + 1);
          v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, confirm_depth, ply + 1, !cutnode);
        }
      } else if (!PV || move_count > 1)
        v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, new_depth, ply + 1, PV ? true : !cutnode);

      if (PV && (move_count == 1 || (v > alpha && v < beta)))
        v = -negamax<true>(t, pos, ss + 1, -beta, -alpha, new_depth, ply + 1, false);

      undo_move(pos, m);
      t.ev.pop();
      if (root && move_count == 1) { // root effort sample
        t.root_m1_nodes = t.nodes - nodes_before;
        t.root_m1_move  = m;
      }
      if (time_up(t)) [[unlikely]]
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
            if (v < beta) { // pv is stale on fail-high
              std::copy((ss + 1)->pv, (ss + 1)->pv + (ss + 1)->pv_len, ss->pv + 1);
              ss->pv_len = (ss + 1)->pv_len + 1;
            } else
              ss->pv_len = 1;
          }
          if (v >= beta) {
            bound = tt::LOWER;
            if (quiet)
              update_quiet_hists(ss, pos, m, quiets_tried, n_quiets - 1, depth);
            update_capture_hists(pos, m, capts_tried, n_capts - (m.is_capture() ? 1 : 0), depth);
            break;
          }
        }
      }

    }

    if (best == -INF) // all moves pruned
      best = alpha;

    if (!excluded) {
      tt::store(tp.slot, key, best_move, to_tt(best, ply), raw_eval, depth, bound, PV);

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

  int aspiration(ThreadData &t, Position &pos, int depth, int prev) {
    Stack *ss    = t.stack + 4;
    int    delta = prm.ASP_DELTA;
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
}

search::Params search::prm;

const std::vector<search::ParamInfo> &search::tunables() {
  static const std::vector<ParamInfo> t = {
          {"LMR_BASE", &prm.LMR_BASE, 80, 30, 150},    {"LMR_DIV", &prm.LMR_DIV, 230, 140, 360},
          {"LMR_TACT_MC", &prm.LMR_TACT_MC, 6, 2, 12}, {"LMR_CONF_HI", &prm.LMR_CONF_HI, 40, 10, 90},
          {"LMR_CONF_LO", &prm.LMR_CONF_LO, 15, 2, 45},
          {"MAT_BASE", &prm.MAT_BASE, 736, 550, 950},  {"MAT_MULT", &prm.MAT_MULT, 5, 1, 10},
          {"HB_MULT", &prm.HB_MULT, 160, 60, 320},     {"HB_SUB", &prm.HB_SUB, 80, 0, 250},
          {"HB_MAX", &prm.HB_MAX, 2000, 800, 4000},
          {"QS_FUT", &prm.QS_FUT, 120, 40, 260},
          {"IIR_DEPTH", &prm.IIR_DEPTH, 4, 2, 8},
          {"RAZOR_DEPTH", &prm.RAZOR_DEPTH, 4, 2, 7},  {"RAZOR_MULT", &prm.RAZOR_MULT, 300, 120, 560},
          {"RFP_DEPTH", &prm.RFP_DEPTH, 8, 4, 12},     {"RFP_MULT", &prm.RFP_MULT, 80, 30, 160},
          {"NMP_DEPTH", &prm.NMP_DEPTH, 3, 2, 6},      {"NMP_BASE", &prm.NMP_BASE, 3, 2, 6},
          {"NMP_DDIV", &prm.NMP_DDIV, 3, 2, 8},        {"NMP_EDIV", &prm.NMP_EDIV, 200, 80, 400},
          {"NMP_ECAP", &prm.NMP_ECAP, 3, 1, 7},        {"NMP_VDEPTH", &prm.NMP_VDEPTH, 12, 7, 18},
          {"PC_MARGIN", &prm.PC_MARGIN, 180, 70, 350}, {"PC_IMP", &prm.PC_IMP, 60, 0, 140},
          {"PC_DEPTH", &prm.PC_DEPTH, 5, 3, 8},
          {"LMP_BASE", &prm.LMP_BASE, 3, 1, 8},        {"LMP_DEPTH", &prm.LMP_DEPTH, 8, 4, 12},
          {"FUT_DEPTH", &prm.FUT_DEPTH, 6, 3, 10},     {"FUT_BASE", &prm.FUT_BASE, 100, 20, 250},
          {"FUT_MULT", &prm.FUT_MULT, 120, 50, 240},
          {"HP_DEPTH", &prm.HP_DEPTH, 4, 2, 8},        {"HP_MULT", &prm.HP_MULT, 2048, 700, 4500},
          {"SEEP_DEPTH", &prm.SEEP_DEPTH, 8, 4, 12},   {"SEEP_QUIET", &prm.SEEP_QUIET, 50, 15, 110},
          {"SEEP_CAPT", &prm.SEEP_CAPT, 90, 30, 180},
          {"SE_DEPTH", &prm.SE_DEPTH, 8, 5, 12},       {"SE_TTSUB", &prm.SE_TTSUB, 3, 1, 6},
          {"SE_BMULT", &prm.SE_BMULT, 2, 1, 5},        {"SE_DBL", &prm.SE_DBL, 25, 8, 70},
          {"SE_TRI", &prm.SE_TRI, 100, 40, 220},       {"SE_DBLMAX", &prm.SE_DBLMAX, 6, 2, 12},
          {"ASP_DELTA", &prm.ASP_DELTA, 14, 6, 35},
          {"RAZOR_UNC", &prm.RAZOR_UNC, 8, 0, 64},     {"RFP_UNC", &prm.RFP_UNC, 8, 0, 64},
          {"FUT_UNC", &prm.FUT_UNC, 8, 0, 64},
  };
  return t;
}

void search::params_dirty() { g_lmr_init = false; }

void search::request_stop() { g_stop.store(true, std::memory_order_relaxed); }

void search::clear_stop() { g_stop.store(false, std::memory_order_relaxed); }

void search::new_game() { g_hist.clear(); }

void search::set_threads(int n) { pool().set_size(std::max(0, n - 1)); }

void search::set_contempt(int cp) { g_contempt = cp; }

void search::set_node_limit(uint64_t n) { g_node_limit = n; }

search::Result search::think(Position &pos, int max_depth, const InfoFn &info, int64_t soft_ms, int64_t hard_ms) {
  if (!g_lmr_init)
    init_lmr();
  g_root_color = pos.turn();
  g_main.nodes = 0;
  g_hard_ms    = hard_ms;
  g_t0         = Clock::now();
  std::memset(g_main.stack, 0, sizeof(g_main.stack));
  g_main.ev.reset(pos);
  pool().reset_counters();
  tt::new_search();

  max_depth = std::clamp(max_depth, 1, MAX_PLY - 1);

  g_helper_stop.store(false, std::memory_order_relaxed);
  pool().start_search(pos, max_depth);

  Result res;
  int    prev      = 0;
  Move   last_best{};
  int    stability = 0;
  for (int d = 1; d <= max_depth; ++d) {
    g_main.root_depth = d;
    g_main.seldepth   = 0;
    const int v = aspiration(g_main, pos, d, prev);
    if (g_stop.load(std::memory_order_relaxed) && d > 1)
      break;

    Stack   *ss         = g_main.stack + 4;
    const int prev_score = prev;
    prev         = v;
    res.score    = v;
    res.nodes    = all_nodes();
    res.seldepth = g_main.seldepth;
    res.pv.assign(ss->pv, ss->pv + ss->pv_len);
    if (!res.pv.empty()) {
      res.best = res.pv[0];
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

    int64_t effective_soft = soft_ms;
    if (soft_ms > 0 && d >= 5 && res.nodes >= 1000 && !res.pv.empty()) {
      double scale = 1.0;
      if (g_main.root_m1_move.to_from() == res.pv[0].to_from() && g_main.nodes > 0) {
        const double frac = double(g_main.root_m1_nodes) / double(g_main.nodes);
        scale *= std::clamp(1.5 - frac, 0.5, 1.5);
      }
      scale *= 1.25 - 0.05 * double(std::min(stability, 8));
      scale *= std::clamp(1.0 + double(prev_score - v) * 0.005, 0.85, 1.4);
      scale          = std::clamp(scale, 0.4, 2.0);
      effective_soft = int64_t(double(soft_ms) * scale);
      if (hard_ms > 0)
        effective_soft = std::min(effective_soft, hard_ms);
    }
    if (soft_ms > 0 && ms >= effective_soft)
      break;
  }

  g_helper_stop.store(true, std::memory_order_relaxed);
  pool().wait_idle();

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
