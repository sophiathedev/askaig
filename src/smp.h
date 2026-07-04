#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "history.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "types.h"

// Lazy SMP. Every helper thread runs its OWN full iterative-deepening search on a private
// copy of the root position; the only shared state is the transposition table and the
// history tables (both intentionally lockless — see history.h). Helpers with an odd index
// start one ply deeper so the pool doesn't move in lockstep; beyond that, TT racing is what
// desynchronises the threads. Helper results are never read directly: their entire value is
// the TT/history warm-up they leave behind for the main thread, which alone owns iteration
// reporting, time management and the final bestmove.
//
// The pool is generation-based: start_search() copies the root, bumps the generation and
// marks every helper as searching; each helper wakes, searches until the stop flag rises
// (search.cpp's smp_worker_iterate), then decrements the searching count. wait_idle() blocks
// until that count reaches zero — think() raises the helper-stop flag first, so this is a
// bounded wait, and afterwards the helpers' node counters are final and safe to sum into the
// reported total. Perft keeps its own independent worker scheme in uci.cpp.
namespace search {

  // Per-ply search state (one array per thread).
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
    // Node-based time management instrumentation at this thread's root (see think()): how
    // many nodes the just-completed iteration spent on the first-searched root move, and
    // which move that was. Per-thread so helper roots can't clobber the main thread's.
    uint64_t root_m1_nodes = 0;
    Move     root_m1_move{};
  };

  // The helper-thread pool (`Threads` - 1 helpers; the main search thread is not in it).
  class ThreadPool {
  public:
    void set_size(int helpers);
    void shutdown();
    ~ThreadPool() { shutdown(); }

    [[nodiscard]] bool has_helpers() const { return !tds.empty(); }

    // Sums each helper's plain (non-atomic) ThreadData::nodes counter, live, while those
    // threads may still be incrementing it — an intentionally unsynchronized statistics read,
    // same reasoning as the Histories tables in history.h (nps/node totals don't need an
    // exact count mid-search, and the count IS exact once wait_idle() has returned).
    ASKAIG_TSAN_IGNORE
    [[nodiscard]] uint64_t total_nodes() const;
    void                   reset_counters();

    // Kicks every helper into a fresh search of `root` (see the file comment). The caller
    // must have reset the stop flags and counters first.
    void start_search(const Position &root, int max_depth);
    // Blocks until every helper is idle. The caller must already have raised the helper
    // stop flag (think() does) — this only waits, it never signals anything itself.
    void wait_idle();

  private:
    void worker(int idx);

    std::vector<std::unique_ptr<ThreadData>> tds;
    std::vector<std::thread>                 threads;
    std::mutex                               mtx;
    std::condition_variable                  cv; // helpers sleep here between searches
    std::condition_variable                  cv_idle; // think() waits here for searching == 0
    Position                                 root; // frozen copy helpers clone from
    int                                      max_depth = 1;
    uint64_t                                 gen       = 0; // search generation
    int                                      searching = 0; // helpers not yet done with `gen`
    bool                                     exit_flag = false;
  };

  ThreadPool &pool();

  // One helper's whole search, implemented in search.cpp (it needs the negamax/aspiration
  // internals): a private iterative-deepening loop that runs until the stop flag rises.
  void smp_worker_iterate(ThreadData &t, Position &pos, int max_depth, int idx);

  // --- tiny shared helpers ----------------------------------------------------------------------
  // Called at every make/unmake in the whole tree: force-inlined dispatch wrappers. Both mutate
  // `p`, so neither is `pure`/`const`.
  [[gnu::always_inline, gnu::hot]] inline void do_move(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.play<WHITE>(m);
    else
      p.play<BLACK>(m);
  }
  [[gnu::always_inline, gnu::hot]] inline void undo_move(Position &p, Move m) {
    if (p.turn() == WHITE)
      p.undo<BLACK>(m);
    else
      p.undo<WHITE>(m);
  }
  // Move is a value type (a plain uint16_t underneath) — zero memory access, so `const`
  // (the strongest attribute) applies, not just `pure`.
  [[gnu::const, gnu::always_inline]] inline bool is_quiet(Move m) {
    const MoveFlags f = m.flags();
    return f == QUIET || f == DOUBLE_PUSH || f == OO || f == OOO;
  }
  [[gnu::pure, gnu::always_inline]] inline bool stm_in_check(const Position &p) {
    return p.turn() == WHITE ? p.in_check<WHITE>() : p.in_check<BLACK>();
  }
  // The engine Zobrist hash omits castling rights and the en-passant square (they change the
  // legal moves, so two positions differing only there must not share a TT slot) — mix them in.
  // Shared by the tree search (probe/store) and the post-make child-key prefetches.
  [[gnu::pure, gnu::always_inline]] inline uint64_t tt_key(const Position &p) {
    return p.get_hash() ^ ((p.castle_entry() & ALL_CASTLING_MASK) * 0x9E3779B97F4A7C15ull) ^
           (uint64_t(uint16_t(p.history[p.ply()].epsq)) * 0xC2B2AE3D27D4EB4Full);
  }

} // namespace search
