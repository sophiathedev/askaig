#include "smp.h"
#include <algorithm>
#include <cstring>

// Lazy SMP pool lifecycle — see smp.h for the model. All the actual searching lives in
// search.cpp (smp_worker_iterate); this file only owns threads, wake-ups and counters.
namespace {

  search::ThreadPool g_pool;

} // namespace

search::ThreadPool &search::pool() { return g_pool; }

void search::ThreadPool::set_size(int helpers) {
  shutdown();
  exit_flag = false;
  tds.clear();
  for (int i = 0; i < helpers; ++i)
    tds.emplace_back(std::make_unique<ThreadData>());
  for (int i = 0; i < helpers; ++i)
    threads.emplace_back(&ThreadPool::worker, this, i);
}

void search::ThreadPool::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mtx);
    exit_flag = true;
  }
  cv.notify_all();
  for (auto &th: threads)
    th.join();
  threads.clear();
}

ASKAIG_TSAN_IGNORE
uint64_t search::ThreadPool::total_nodes() const {
  uint64_t n = 0;
  for (const auto &t: tds)
    n += t->nodes;
  return n;
}

void search::ThreadPool::reset_counters() {
  for (auto &t: tds) {
    t->nodes    = 0;
    t->seldepth = 0;
  }
}

void search::ThreadPool::start_search(const Position &pos, int md) {
  if (tds.empty())
    return;
  {
    std::lock_guard<std::mutex> lk(mtx);
    // Position::operator= is deleted (it would reset UndoInfo state); byte-copy, same as the
    // helpers do when they clone it back out.
    std::memcpy(static_cast<void *>(&root), static_cast<const void *>(&pos), sizeof(Position));
    max_depth = md;
    ++gen;
    // Count every helper as searching UP FRONT, not when it actually wakes: think() may hit
    // wait_idle() before a slow helper has even woken for this generation, and must still
    // wait for it — a late helper decrements exactly once when it finishes, so the count
    // can never terminate early.
    searching = int(tds.size());
  }
  cv.notify_all();
}

void search::ThreadPool::wait_idle() {
  if (tds.empty())
    return;
  std::unique_lock<std::mutex> lk(mtx);
  cv_idle.wait(lk, [&] { return searching == 0; });
}

void search::ThreadPool::worker(int idx) {
  ThreadData &t    = *tds[size_t(idx)];
  uint64_t    seen = 0;
  while (true) {
    Position pos;
    int      md;
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [&] { return exit_flag || gen != seen; });
      if (exit_flag)
        return;
      seen = gen;
      std::memcpy(static_cast<void *>(&pos), static_cast<const void *>(&root), sizeof(Position));
      md = max_depth;
    }
    smp_worker_iterate(t, pos, md, idx);
    {
      std::lock_guard<std::mutex> lk(mtx);
      --searching;
    }
    cv_idle.notify_all();
  }
}
