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

// Lazy SMP. Every helper runs its own full iterative-deepening search on a private copy of
// the root; the only shared state is the TT and the history tables (both lockless, see
// history.h/tt.h). Odd-indexed helpers start one ply deeper to desync the pool. Helper
// results are never read — their value is the TT/history warm-up. The main thread alone
// reports iterations, manages time and produces bestmove.
//
// Pool protocol: start_search() copies the root, bumps the generation and counts every helper
// as searching up front (a slow-to-wake helper must still be waited on); helpers decrement
// when done; wait_idle() blocks until zero. think() raises the helper stop flag before
// wait_idle(), so the wait is bounded and node counters are final afterwards. Perft keeps its
// own worker scheme in uci.cpp.
namespace search {

  // Per-ply search state (one array per thread).
  struct Stack {
    Move       pv[MAX_PLY + 1];
    int        pv_len;
    Move       killers[2];
    Move       move; // move made AT this ply (null move: to_from() == 0 with ch == nullptr)
    Move       excluded; // move excluded by a singular-verification search
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
    // Node-TM instrumentation at this thread's root: nodes the last iteration spent on the
    // first root move, and which move that was. Per-thread so helpers can't clobber the main's.
    uint64_t root_m1_nodes = 0;
    Move     root_m1_move{};
  };

  // The helper pool (`Threads` - 1 helpers; the main search thread is not in it).
  class ThreadPool {
  public:
    void set_size(int helpers);
    void shutdown();
    ~ThreadPool() { shutdown(); }

    [[nodiscard]] bool has_helpers() const { return !tds.empty(); }

    // Live sum of plain per-thread node counters — intentionally unsynchronized (history.h);
    // exact once wait_idle() has returned.
    ASKAIG_TSAN_IGNORE
    [[nodiscard]] uint64_t total_nodes() const;
    void                   reset_counters();

    void start_search(const Position &root, int max_depth);
    // Blocks until every helper is idle. Caller must have raised the stop flag first.
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

  // One helper's whole search — implemented in search.cpp (needs negamax/aspiration).
  void smp_worker_iterate(ThreadData &t, Position &pos, int max_depth, int idx);

  // --- tiny shared helpers ----------------------------------------------------------------------

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
  [[gnu::const, gnu::always_inline]] inline bool is_quiet(Move m) {
    const MoveFlags f = m.flags();
    return f == QUIET || f == DOUBLE_PUSH || f == OO || f == OOO;
  }
  [[gnu::pure, gnu::always_inline]] inline bool stm_in_check(const Position &p) {
    return p.turn() == WHITE ? p.in_check<WHITE>() : p.in_check<BLACK>();
  }
  // The engine Zobrist hash omits castling rights and the en-passant square — two positions
  // differing only there must not share a TT slot, so mix them in.
  [[gnu::pure, gnu::always_inline]] inline uint64_t tt_key(const Position &p) {
    return p.get_hash() ^ ((p.castle_entry() & ALL_CASTLING_MASK) * 0x9E3779B97F4A7C15ull) ^
           (uint64_t(uint16_t(p.history[p.ply()].epsq)) * 0xC2B2AE3D27D4EB4Full);
  }

} // namespace search
