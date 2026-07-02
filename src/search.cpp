#include "search.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "history.h"
#include "movepick.h"
#include "nnue.h"
#include "see.h"
#include "tt.h"

// Fail-soft negamax with, in rough order of gating: TT cutoffs, static-eval correction
// history, IIR, razoring, RFP, NMP, ProbCut, singular extensions (with multicut and
// double/negative extensions), LMP, futility pruning, history pruning, SEE pruning (captures
// and quiets), log-formula LMR, PVS, and a SEE/delta-pruned quiescence. Move ordering in
// movepick.h uses the TT move, MVV + capture history, killers, and butterfly + continuation
// (CMH/FMH) history.
//
// Parallelism is YBWC (Young Brothers Wait Concept), NOT Lazy SMP: at a node with
// depth >= the "Split" UCI option, the first ("eldest") move is always searched sequentially;
// only if it fails to cut are the remaining siblings distributed over the helper threads of a
// global pool. Helpers claim sibling moves from a SplitPoint, sharing alpha/best atomically;
// a beta cutoff raises the split's abort flag, which every claimant polls (and which chains
// through nested splits). Perft keeps its own independent worker scheme in uci.cpp.
namespace {

  using namespace search;
  using Clock = std::chrono::steady_clock;

  struct SplitPoint;

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
    nnue::Evaluator ev;
    Stack           stack[MAX_PLY + 8];
    uint64_t        nodes    = 0;
    int             seldepth = 0, root_depth = 1;
    SplitPoint     *cur_split = nullptr; // innermost split this thread is claiming from
  };

  // Shared search state. History tables are shared by all threads with plain (benign-race)
  // int16 updates, exactly like the TT — both are statistics, not correctness-critical.
  Histories          g_hist;
  ThreadData         g_main;
  std::atomic<bool>  g_stop{false};
  int64_t            g_hard_ms = 0;
  Clock::time_point  g_t0;
  int                g_split_depth = 10; // the "Split" UCI option

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

  Position clone_position(const Position &src) { // Position is plain data; memcpy is a faithful copy
    Position dst;
    std::memcpy(static_cast<void *>(&dst), static_cast<const void *>(&src), sizeof(Position));
    return dst;
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
  int evaluate(ThreadData &t, const Position &pos) {
    const int raw  = t.ev.evaluate(pos);
    const int corr = g_hist.corr[pos.turn()][corr_index(pos)] / 64;
    return std::clamp(raw + corr, -MATE_IN_MAX + 1, MATE_IN_MAX - 1);
  }

  struct SplitPoint {
    SplitPoint *parent; // the split this thread was itself claiming from (abort chain)
    // Snapshots taken at split creation. The master keeps making/unmaking moves on its LIVE
    // position and stack while claiming — workers must attach to a frozen copy, never to the
    // master's live objects (cloning those mid-make was a nasty corruption bug).
    Position             snapshot;
    static constexpr int CTX = 8; // stack entries [ctx_lo .. 4+ply] the children may read below the node
    Stack                ctx[CTX];
    int                  ctx_lo;
    int                  ply, depth, base_count, lmp_limit, root_depth;
    bool                 pv_node, cutnode, improving;
    int                  beta;
    std::atomic<int>     alpha, best;
    std::atomic<size_t>  next{0};
    std::atomic<int>     quiets{0}; // shared quiet count (LMP in the parallel phase)
    std::atomic<int>     active{0};
    std::atomic<bool>    cutoff{false};
    std::vector<Move>    moves;
    std::mutex           mtx; // guards best/alpha/best_move/pv merges
    Move                 best_move{};
    Move                 pv[MAX_PLY + 1];
    int                  pv_len = 0;
  };

  // True when this thread's work is moot: global stop, or a beta cutoff anywhere in the chain
  // of splits it is working under. A search aborted this way returns garbage values — every
  // consumer (claim loops, think) must discard results once aborted() holds.
  bool aborted(ThreadData &t) {
    if (g_stop.load(std::memory_order_relaxed))
      return true;
    for (SplitPoint *s = t.cur_split; s; s = s->parent)
      if (s->cutoff.load(std::memory_order_relaxed))
        return true;
    return false;
  }

  // A stop/abort poll for the tree: aborted() plus the hard clock.
  bool time_up(ThreadData &t) {
    if (aborted(t))
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

  int hist_bonus(int depth) { return std::min(160 * depth - 80, 2000); }

  int quiet_hist(const Stack *ss, const Position &pos, Move m) {
    const Piece pc = pos.at(m.from());
    int         h  = g_hist.butterfly[pos.turn()][m.from()][m.to()];
    if ((ss - 1)->ch)
      h += (*(ss - 1)->ch)[pc][m.to()];
    if ((ss - 2)->ch)
      h += (*(ss - 2)->ch)[pc][m.to()];
    return h;
  }

  void update_quiet_hists(Stack *ss, const Position &pos, Move best, const Move *tried, int n_tried, int depth) {
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

  void update_capture_hists(const Position &pos, Move best, const Move *tried, int n_tried, int depth) {
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
  int qsearch(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int ply);
  template<bool PV>
  int negamax(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int depth, int ply, bool cutnode);

  // --- YBWC split machinery ---------------------------------------------------------------------


  // One parallel sibling: the shared move-loop tail (pruning, LMR, PVS) for moves searched
  // after the eldest brother. Mirrors the sequential loop's post-first-move logic; singular
  // extension never applies here (it is TT-move-only, and the TT move is searched first).
  // Returns INT_MIN when the move was pruned, else the search value.
  int split_move(ThreadData &t, SplitPoint &sp, Position &pos, Stack *ss, Move m, int move_count) {
    const int  depth     = sp.depth;
    const int  ply       = sp.ply;
    const bool quiet     = is_quiet(m);
    const bool in_check  = ss->in_check;
    const int  alpha_now = sp.alpha.load(std::memory_order_relaxed);
    const int  beta      = sp.beta;

    // Move-loop pruning (a real best already exists: the eldest brother completed).
    if (quiet && !in_check) {
      if (depth <= 8 && sp.quiets.load(std::memory_order_relaxed) >= sp.lmp_limit)
        return INT32_MIN;
      if (depth <= 6 && std::abs(alpha_now) < MATE_IN_MAX && ss->static_eval + 100 + 120 * depth <= alpha_now)
        return INT32_MIN;
      if (depth <= 4 && quiet_hist(ss, pos, m) < -2048 * depth)
        return INT32_MIN;
    }
    if (depth <= 8 && !see_ge(pos, m, quiet ? -50 * depth : -90 * depth))
      return INT32_MIN;

    int extension = 0;
    if (in_check && ply < 2 * t.root_depth)
      extension = 1; // capped check extension

    const Piece moved = pos.at(m.from());
    ss->move          = m;
    ss->ch            = &g_hist.cont[moved][m.to()];

    ++t.nodes;
    if (quiet)
      sp.quiets.fetch_add(1, std::memory_order_relaxed);
    t.ev.push(pos, m);
    do_move(pos, m);

    const int new_depth = depth - 1 + extension;
    int       v;

    int r = 0;
    if (depth >= 3 && move_count > 1 + 2 * sp.pv_node && (quiet || move_count > 6)) {
      r = lmr_base(depth, move_count);
      r += sp.cutnode;
      r += !sp.improving;
      r -= sp.pv_node;
      if (quiet)
        r -= std::clamp(quiet_hist(ss, pos, m) / 8192, -2, 2);
      else
        r /= 2;
      r = std::clamp(r, 0, new_depth - 1);
    }

    v = -negamax<false>(t, pos, ss + 1, -alpha_now - 1, -alpha_now, new_depth - r, ply + 1, true);
    if (v > alpha_now && r > 0)
      v = -negamax<false>(t, pos, ss + 1, -alpha_now - 1, -alpha_now, new_depth, ply + 1, !sp.cutnode);
    if (sp.pv_node && v > alpha_now && v < beta)
      v = -negamax<true>(t, pos, ss + 1, -beta, -alpha_now, new_depth, ply + 1, false);

    undo_move(pos, m);
    t.ev.pop();
    return v;
  }

  // Claims and searches sibling moves from `sp` until exhausted/cut. Used by the master (on
  // its own position) and by pool helpers (on a clone). Returns via sp.{best,alpha,cutoff,...}.
  void split_claim_loop(ThreadData &t, SplitPoint &sp, Position &pos, Stack *ss) {
    while (!aborted(t)) {
      const size_t idx = sp.next.fetch_add(1, std::memory_order_relaxed);
      if (idx >= sp.moves.size())
        break;
      const Move m = sp.moves[idx];
      if (sp.alpha.load(std::memory_order_relaxed) >= sp.beta)
        break;

      const int v = split_move(t, sp, pos, ss, m, sp.base_count + int(idx) + 1);
      if (v == INT32_MIN)
        continue; // pruned
      if (aborted(t))
        break; // an aborted search returns garbage — discard it (outer-split cutoffs included)

      std::lock_guard<std::mutex> lk(sp.mtx);
      if (v > sp.best.load(std::memory_order_relaxed)) {
        sp.best.store(v, std::memory_order_relaxed);
        if (v > sp.alpha.load(std::memory_order_relaxed)) {
          sp.best_move = m;
          sp.alpha.store(v, std::memory_order_relaxed);
          if (sp.pv_node) {
            // The child pv is only valid when the value came from the PV re-search (exact);
            // a fail-high came from a zero-window search, which never writes child pvs.
            sp.pv[0] = m;
            if (v < sp.beta) {
              std::copy((ss + 1)->pv, (ss + 1)->pv + (ss + 1)->pv_len, sp.pv + 1);
              sp.pv_len = (ss + 1)->pv_len + 1;
            } else
              sp.pv_len = 1;
          }
          if (v >= sp.beta)
            sp.cutoff.store(true, std::memory_order_relaxed);
        }
      }
    }
  }

  // Helper-thread pool. Helpers sleep until a split with unclaimed moves is registered, attach
  // to it (own Position clone + accumulator refresh + a copy of the master's stack prefix for
  // the history/eval context), and claim siblings until the split is drained or cut.
  class SplitPool {
  public:
    void set_size(int helpers) {
      shutdown();
      exit_flag = false;
      tds.clear();
      for (int i = 0; i < helpers; ++i)
        tds.emplace_back(std::make_unique<ThreadData>());
      for (int i = 0; i < helpers; ++i)
        threads.emplace_back(&SplitPool::worker, this, i);
    }
    void shutdown() {
      {
        std::lock_guard<std::mutex> lk(mtx);
        exit_flag = true;
      }
      cv.notify_all();
      for (auto &th: threads)
        th.join();
      threads.clear();
    }
    ~SplitPool() { shutdown(); }

    bool has_helpers() const { return !tds.empty(); }

    void register_split(SplitPoint *sp) {
      {
        std::lock_guard<std::mutex> lk(mtx);
        active.push_back(sp);
      }
      cv.notify_all();
    }
    void unregister_split(SplitPoint *sp) {
      std::lock_guard<std::mutex> lk(mtx);
      std::erase(active, sp);
    }

    uint64_t total_nodes() const {
      uint64_t n = 0;
      for (const auto &t: tds)
        n += t->nodes;
      return n;
    }
    void reset_counters(int root_depth) {
      for (auto &t: tds) {
        t->nodes      = 0;
        t->root_depth = root_depth;
      }
    }
    void set_root_depth(int d) {
      for (auto &t: tds)
        t->root_depth = d;
    }

  private:
    SplitPoint *grab() { // caller holds mtx
      for (SplitPoint *sp: active)
        if (!sp->cutoff.load(std::memory_order_relaxed) &&
            sp->next.load(std::memory_order_relaxed) < sp->moves.size())
          return sp;
      return nullptr;
    }

    void worker(int idx) {
      ThreadData &t = *tds[size_t(idx)];
      while (true) {
        SplitPoint *sp = nullptr;
        {
          std::unique_lock<std::mutex> lk(mtx);
          // active++ must happen INSIDE the lock: the master unregisters (same lock) and then
          // waits for active == 0 — incrementing after release would race its teardown.
          cv.wait(lk, [&] {
            sp = nullptr;
            if (exit_flag)
              return true;
            if ((sp = grab()) != nullptr)
              sp->active.fetch_add(1, std::memory_order_relaxed);
            return sp != nullptr;
          });
          if (exit_flag)
            return;
        }
        // Attach: clone the frozen snapshot, rebuild the accumulator there, and restore the
        // stack context around the node so (ss-1)/(ss-2) conthist and improving work.
        Position pos = clone_position(sp->snapshot);
        std::memset(t.stack, 0, sizeof(Stack) * size_t(4 + sp->ply + 1));
        for (int i = sp->ctx_lo; i <= 4 + sp->ply; ++i)
          t.stack[i] = sp->ctx[i - sp->ctx_lo];
        Stack *ss    = t.stack + 4 + sp->ply;
        t.ev.reset(pos);
        t.cur_split  = sp;
        t.root_depth = sp->root_depth;
        split_claim_loop(t, *sp, pos, ss);
        t.cur_split = nullptr; // fully detach BEFORE releasing the split (sp->parent may die with it)
        sp->active.fetch_sub(1, std::memory_order_relaxed);
      }
    }

    std::vector<std::unique_ptr<ThreadData>> tds;
    std::vector<std::thread>                 threads;
    std::vector<SplitPoint *>                active;
    std::mutex                               mtx;
    std::condition_variable                  cv;
    bool                                     exit_flag = false;
  };

  SplitPool g_pool;

  uint64_t all_nodes() { return g_main.nodes + g_pool.total_nodes(); }

  // --- quiescence -------------------------------------------------------------------------------

  template<bool PV>
  int qsearch(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int ply) {
    if constexpr (PV)
      ss->pv_len = 0;
    t.seldepth = std::max(t.seldepth, ply);
    if (time_up(t))
      return 0;
    if (pos.is_draw())
      return 0;
    if (ply >= MAX_PLY)
      return evaluate(t, pos);

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
      raw_eval = tthit && tte->eval != tt::VALUE_NONE_TT ? tte->eval : evaluate(t, pos);
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
    MovePicker picker(pos, g_hist, ttm, nullptr, nullptr, nullptr, /*quiescence=*/true);
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

      ++t.nodes;
      t.ev.push(pos, m);
      do_move(pos, m);
      const int v = -qsearch<PV>(t, pos, ss + 1, -beta, -alpha, ply + 1);
      undo_move(pos, m);
      t.ev.pop();
      if (g_stop.load(std::memory_order_relaxed))
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

    tt::store(tte, key, best_move, to_tt(best, ply), raw_eval, /*depth=*/0,
              best >= beta ? tt::LOWER : tt::UPPER, PV);
    return best;
  }

  // --- main search ------------------------------------------------------------------------------

  template<bool PV>
  int negamax(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int depth, int ply, bool cutnode) {
    if (depth <= 0)
      return qsearch<PV>(t, pos, ss, alpha, beta, ply);

    if constexpr (PV)
      ss->pv_len = 0;
    if (time_up(t))
      return 0;
    t.seldepth = std::max(t.seldepth, ply);

    const bool root = ply == 0;
    if (!root) {
      if (pos.is_draw())
        return 0;
      if (ply >= MAX_PLY)
        return evaluate(t, pos);
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
    int            ttsc    = tt::VALUE_NONE_TT;
    int            ttdepth = -tt::DEPTH_OFFSET;
    tt::Bound      ttbound = tt::NONE;
    int            tteval  = tt::VALUE_NONE_TT;
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
      raw_eval        = tteval != tt::VALUE_NONE_TT ? tteval : evaluate(t, pos);
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
        const int v = -negamax<false>(t, pos, ss + 1, -beta, -beta + 1, depth - R, ply + 1, !cutnode);
        pos.undo_null();
        t.ev.pop();
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
          int v = -qsearch<false>(t, pos, ss + 1, -pc_beta, -pc_beta + 1, ply + 1);
          if (v >= pc_beta) // qsearch agrees: confirm with a reduced full search
            v = -negamax<false>(t, pos, ss + 1, -pc_beta, -pc_beta + 1, depth - 4, ply + 1, !cutnode);
          undo_move(pos, m);
          t.ev.pop();
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
        // PVS SEE pruning, depth-scaled thresholds (captures and quiets).
        if (depth <= 8 && !see_ge(pos, m, quiet ? -50 * depth : -90 * depth))
          continue;
      }

      // --- singular extension / multicut on the TT move ---
      int extension = 0;
      if (!root && !excluded && depth >= 8 && m.to_from() == ttm.to_from() && ttdepth >= depth - 3 &&
          (ttbound & tt::LOWER) && std::abs(ttsc) < MATE_IN_MAX && ply < 2 * t.root_depth) {
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

      ++t.nodes;
      t.ev.push(pos, m);
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

        v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
        if (v > alpha && r > 0) // reduced search beat alpha: verify at full depth
          v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, new_depth, ply + 1, !cutnode);
      } else if (!PV || move_count > 1)
        v = -negamax<false>(t, pos, ss + 1, -alpha - 1, -alpha, new_depth, ply + 1, PV ? true : !cutnode);

      if (PV && (move_count == 1 || (v > alpha && v < beta)))
        v = -negamax<true>(t, pos, ss + 1, -beta, -alpha, new_depth, ply + 1, false);

      undo_move(pos, m);
      t.ev.pop();
      if (time_up(t))
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

      // --- YBWC split: the eldest brother is done and didn't cut -> farm out the rest ---
      if (best < beta && depth >= g_split_depth && g_pool.has_helpers() && move_count >= 1 &&
          picker.remaining() >= 2) {
        // Heap, NOT a local: SplitPoint embeds a Position snapshot (~37 KB — history[1024]),
        // and negamax recurses 30+ frames deep on a 512 KB std::thread stack. A stack-local
        // here blew the search thread's stack (SIGBUS) the moment depth reached the split gate.
        const auto  spp = std::make_unique<SplitPoint>();
        SplitPoint &sp  = *spp;
        for (Move mm; (mm = picker.next()).to_from() != 0;)
          if (mm.to_from() != ss->excluded.to_from())
            sp.moves.push_back(mm);

        sp.parent = t.cur_split;
        // Position::operator= is deleted (it would reset UndoInfo state); byte-copy like clone_position.
        std::memcpy(static_cast<void *>(&sp.snapshot), static_cast<const void *>(&pos), sizeof(Position));
        sp.ctx_lo = std::max(0, 4 + ply - (SplitPoint::CTX - 1));
        for (int i = sp.ctx_lo; i <= 4 + ply; ++i)
          sp.ctx[i - sp.ctx_lo] = t.stack[i];
        sp.ply          = ply;
        sp.depth        = depth;
        sp.base_count   = move_count;
        sp.lmp_limit    = lmp_limit;
        sp.root_depth   = t.root_depth;
        sp.pv_node      = PV;
        sp.cutnode      = cutnode;
        sp.improving    = improving;
        sp.beta         = beta;
        sp.alpha.store(alpha);
        sp.best.store(best);
        sp.quiets.store(quiet_count);

        g_pool.register_split(&sp);
        SplitPoint *prev_split = t.cur_split;
        t.cur_split            = &sp;
        split_claim_loop(t, sp, pos, ss); // the master helps drain its own split
        t.cur_split = prev_split;
        if (aborted(t)) // aborted from above: tell the helpers their work is moot too
          sp.cutoff.store(true, std::memory_order_relaxed);
        g_pool.unregister_split(&sp);
        while (sp.active.load(std::memory_order_relaxed) > 0)
          std::this_thread::yield(); // wait for helpers to finish/abort their claimed moves

        if (g_stop.load(std::memory_order_relaxed))
          return 0;

        // Merge the split result.
        const int sp_best = sp.best.load(std::memory_order_relaxed);
        if (sp_best > best) {
          best = sp_best;
          if (sp.best_move.to_from() != 0) {
            best_move = sp.best_move;
            alpha     = std::max(alpha, best);
            bound     = best >= beta ? tt::LOWER : tt::EXACT;
            if constexpr (PV) {
              if (sp.pv_len > 0) {
                std::copy(sp.pv, sp.pv + sp.pv_len, ss->pv);
                ss->pv_len = sp.pv_len;
              }
            }
            if (best >= beta) {
              // Parallel cutoff: reward the cutoff move (tried-move penalties are skipped —
              // the tried set is scattered across threads).
              if (is_quiet(best_move))
                update_quiet_hists(ss, pos, best_move, quiets_tried, n_quiets, depth);
              update_capture_hists(pos, best_move, nullptr, 0, depth);
            }
          }
        }
        break; // the split consumed the rest of the move list
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
        hist_update(g_hist.corr[pos.turn()][corr_index(pos)], b);
      }
    }
    return best;
  }

  // Aspiration windows around the previous iteration's score, widening on failure.
  int aspiration(Position &pos, int depth, int prev) {
    Stack *ss    = g_main.stack + 4;
    int    delta = 14;
    int    alpha = -INF, beta = INF;
    if (depth >= 4) {
      alpha = std::max(prev - delta, -INF);
      beta  = std::min(prev + delta, INF);
    }
    while (true) {
      const int v = negamax<true>(g_main, pos, ss, alpha, beta, depth, 0, false);
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

void search::new_game() { g_hist.clear(); }

void search::set_threads(int n) { g_pool.set_size(std::max(0, n - 1)); }

void search::set_split_depth(int d) { g_split_depth = std::max(1, d); }

search::Result search::think(Position &pos, int max_depth, const InfoFn &info, int64_t soft_ms, int64_t hard_ms) {
  if (!g_lmr_init)
    init_lmr();
  g_stop.store(false, std::memory_order_relaxed);
  g_main.nodes = 0;
  g_hard_ms    = hard_ms;
  g_t0         = Clock::now();
  std::memset(g_main.stack, 0, sizeof(g_main.stack));
  g_main.ev.reset(pos);
  g_main.cur_split = nullptr;
  g_pool.reset_counters(1);
  tt::new_search();

  max_depth = std::clamp(max_depth, 1, MAX_PLY - 1);

  Result res;
  int    prev = 0;
  for (int d = 1; d <= max_depth; ++d) {
    g_main.root_depth = d;
    g_main.seldepth   = 0;
    g_pool.set_root_depth(d);
    const int v = aspiration(pos, d, prev);
    if (g_stop.load(std::memory_order_relaxed) && d > 1)
      break; // discard the aborted iteration; the previous full one stands

    Stack *ss    = g_main.stack + 4;
    prev         = v;
    res.score    = v;
    res.nodes    = all_nodes();
    res.seldepth = g_main.seldepth;
    res.pv.assign(ss->pv, ss->pv + ss->pv_len);
    if (!res.pv.empty())
      res.best = res.pv[0];

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_t0).count();
    if (info)
      info(d, res, res.nodes, ms);
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
  res.nodes = all_nodes();
  return res;
}
