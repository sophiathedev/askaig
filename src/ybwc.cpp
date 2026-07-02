#include "ybwc.h"
#include <algorithm>
#include <cstring>
#include "movepick.h" // (via history.h users) — see_ge/PIECE_VAL come from see.h
#include "see.h"

// The YBWC split machinery: SplitPoint claim loops, the helper pool, and the abort chain.
// The per-move search logic here (split_move) mirrors the sequential move loop's
// post-first-move tail in search.cpp — pruning, LMR, PVS — reaching the actual negamax
// through the ybwc_search hook. Singular extension never applies to parallel siblings (it is
// TT-move-only, and the TT move is always the sequentially-searched eldest brother).
namespace {

  using namespace search;

  Position clone_position(const Position &src) { // Position is plain data; memcpy is a faithful copy
    Position dst;
    std::memcpy(static_cast<void *>(&dst), static_cast<const void *>(&src), sizeof(Position));
    return dst;
  }

  // One parallel sibling. Returns INT32_MIN when the move was pruned, else the search value.
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
      if (depth <= 4 && quiet_history(ss, pos, m) < -2048 * depth)
        return INT32_MIN;
    }
    if (depth <= 8 && !see_ge(pos, m, quiet ? -50 * depth : -90 * depth))
      return INT32_MIN;

    int extension = 0;
    if (in_check && ply < 2 * t.root_depth)
      extension = 1; // capped check extension

    const Piece moved = pos.at(m.from());
    ss->move          = m;
    ss->ch            = &shared_hist().cont[moved][m.to()];

    ++t.nodes;
    if (quiet)
      sp.quiets.fetch_add(1, std::memory_order_relaxed);
    t.ev.push(pos, m);
    do_move(pos, m);

    const int new_depth = depth - 1 + extension;
    int       v;

    int r = 0;
    if (depth >= 3 && move_count > 1 + 2 * sp.pv_node && (quiet || move_count > 6)) {
      r = lmr_reduction(depth, move_count);
      r += sp.cutnode;
      r += !sp.improving;
      r -= sp.pv_node;
      if (quiet)
        r -= std::clamp(quiet_history(ss, pos, m) / 8192, -2, 2);
      else
        r /= 2;
      r = std::clamp(r, 0, new_depth - 1);
    }

    v = -ybwc_search(t, pos, ss + 1, -alpha_now - 1, -alpha_now, new_depth - r, ply + 1, true, false);
    if (v > alpha_now && r > 0)
      v = -ybwc_search(t, pos, ss + 1, -alpha_now - 1, -alpha_now, new_depth, ply + 1, !sp.cutnode, false);
    if (sp.pv_node && v > alpha_now && v < beta)
      v = -ybwc_search(t, pos, ss + 1, -beta, -alpha_now, new_depth, ply + 1, false, true);

    undo_move(pos, m);
    t.ev.pop();
    return v;
  }

  // Claims and searches sibling moves from `sp` until exhausted/cut. Used by the master (on
  // its own position) and by pool helpers (on a clone). Results land in sp.{best,alpha,...}.
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

  SplitPool g_pool;

} // namespace

search::SplitPool &search::pool() { return g_pool; }

bool search::aborted(ThreadData &t) {
  if (stop_requested())
    return true;
  for (SplitPoint *s = t.cur_split; s; s = s->parent)
    if (s->cutoff.load(std::memory_order_relaxed))
      return true;
  return false;
}

void search::run_split(ThreadData &t, SplitPoint &sp, Position &pos, Stack *ss) {
  g_pool.register_split(&sp);
  SplitPoint *prev = t.cur_split;
  t.cur_split      = &sp;
  split_claim_loop(t, sp, pos, ss); // the master helps drain its own split
  t.cur_split = prev;
  if (aborted(t)) // aborted from above: tell the helpers their work is moot too
    sp.cutoff.store(true, std::memory_order_relaxed);
  g_pool.unregister_split(&sp);
  while (sp.active.load(std::memory_order_relaxed) > 0)
    std::this_thread::yield(); // wait for helpers to finish/abort their claimed moves
}

// --- SplitPool ---------------------------------------------------------------------------------

void search::SplitPool::set_size(int helpers) {
  shutdown();
  exit_flag = false;
  tds.clear();
  for (int i = 0; i < helpers; ++i)
    tds.emplace_back(std::make_unique<ThreadData>());
  for (int i = 0; i < helpers; ++i)
    threads.emplace_back(&SplitPool::worker, this, i);
}

void search::SplitPool::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mtx);
    exit_flag = true;
  }
  cv.notify_all();
  for (auto &th: threads)
    th.join();
  threads.clear();
}

void search::SplitPool::register_split(SplitPoint *sp) {
  {
    std::lock_guard<std::mutex> lk(mtx);
    active.push_back(sp);
  }
  cv.notify_all();
}

void search::SplitPool::unregister_split(SplitPoint *sp) {
  std::lock_guard<std::mutex> lk(mtx);
  std::erase(active, sp);
}

uint64_t search::SplitPool::total_nodes() const {
  uint64_t n = 0;
  for (const auto &t: tds)
    n += t->nodes;
  return n;
}

void search::SplitPool::reset_counters(int root_depth) {
  for (auto &t: tds) {
    t->nodes      = 0;
    t->root_depth = root_depth;
  }
}

void search::SplitPool::set_root_depth(int d) {
  for (auto &t: tds)
    t->root_depth = d;
}

search::SplitPoint *search::SplitPool::grab() { // caller holds mtx
  for (SplitPoint *sp: active)
    if (!sp->cutoff.load(std::memory_order_relaxed) && sp->next.load(std::memory_order_relaxed) < sp->moves.size())
      return sp;
  return nullptr;
}

void search::SplitPool::worker(int idx) {
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
