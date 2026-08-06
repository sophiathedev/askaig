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

namespace search {

  struct Stack {
    Move       pv[MAX_PLY + 1];
    int        pv_len;
    Move       killers[2];
    Move       move; // move at this ply; null when ch is null
    Move       excluded;
    int        static_eval;
    int        eval_unc;
    ContTable *ch;
    int        double_ext;
    bool       in_check;
  };

  struct ThreadData {
    nnue::Evaluator ev;
    Stack           stack[MAX_PLY + 8];
    uint64_t        nodes    = 0;
    int             seldepth = 0, root_depth = 1;
    uint64_t root_m1_nodes = 0;
    Move     root_m1_move{};
  };

  class ThreadPool {
  public:
    void set_size(int helpers);
    void shutdown();
    ~ThreadPool() { shutdown(); }

    [[nodiscard]] bool has_helpers() const { return !tds.empty(); }

    ASKAIG_TSAN_IGNORE
    [[nodiscard]] uint64_t total_nodes() const;
    void                   reset_counters();

    void start_search(const Position &root, int max_depth);
    void wait_idle();

  private:
    void worker(int idx);

    std::vector<std::unique_ptr<ThreadData>> tds;
    std::vector<std::thread>                 threads;
    std::mutex                               mtx;
    std::condition_variable                  cv;
    std::condition_variable                  cv_idle;
    Position                                 root;
    int                                      max_depth = 1;
    uint64_t                                 gen       = 0;
    int                                      searching = 0;
    bool                                     exit_flag = false;
  };

  ThreadPool &pool();

  void smp_worker_iterate(ThreadData &t, Position &pos, int max_depth, int idx);


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
  // base hash omits castling and en passant
  [[gnu::pure, gnu::always_inline]] inline uint64_t tt_key(const Position &p) {
    return p.get_hash() ^ ((p.castle_entry() & ALL_CASTLING_MASK) * 0x9E3779B97F4A7C15ull) ^
           (uint64_t(uint16_t(p.history[p.ply()].epsq)) * 0xC2B2AE3D27D4EB4Full);
  }

} // namespace search
