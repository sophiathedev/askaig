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

// YBWC (Young Brothers Wait Concept) parallel search — the split machinery, factored out of
// search.cpp. At a node with depth >= the "Split" UCI option the first ("eldest") move is
// always searched sequentially; only if it fails to cut are the remaining siblings drained
// from a SplitPoint by a global helper pool plus the master itself. Claimants share
// alpha/best atomically and merge best_move/pv under the split mutex; a beta cutoff raises an
// abort flag that chains through nested splits. Perft keeps its own worker scheme in uci.cpp.
//
// The split machinery calls back into the tree search through the ybwc_search /
// shared_hist / stop_requested / lmr_reduction / quiet_history hooks, all implemented in
// search.cpp (the negamax templates live in its anonymous namespace).
namespace search {

  struct SplitPoint;

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
    SplitPoint     *cur_split = nullptr; // innermost split this thread is claiming from
  };

  struct SplitPoint {
    SplitPoint *parent; // the split this thread was itself claiming from (abort chain)
    // Snapshots taken at split creation. The master keeps making/unmaking moves on its LIVE
    // position and stack while claiming — workers must attach to a frozen copy, never to the
    // master's live objects (cloning those mid-make was a nasty corruption bug).
    // NOTE for callers: allocate SplitPoints on the HEAP — the Position snapshot alone is
    // ~37 KB (history[1024]) and negamax recurses deeply on a 512 KB std::thread stack.
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

  // Helper-thread pool. Helpers sleep until a split with unclaimed moves is registered, attach
  // to it (own Position clone + accumulator refresh + a copy of the master's stack context),
  // and claim siblings until the split is drained or cut.
  class SplitPool {
  public:
    void set_size(int helpers);
    void shutdown();
    ~SplitPool() { shutdown(); }

    [[nodiscard]] bool has_helpers() const { return !tds.empty(); }

    void register_split(SplitPoint *sp);
    void unregister_split(SplitPoint *sp);

    // Sums each pool thread's plain (non-atomic) ThreadData::nodes counter, live, while those
    // threads may still be incrementing it — an intentionally unsynchronized statistics read,
    // same reasoning as the Histories tables in history.h (nps/NodeTM don't need an exact
    // count, and every parallel search engine sums live per-thread node counters this way).
    ASKAIG_TSAN_IGNORE
    [[nodiscard]] uint64_t total_nodes() const;
    void                   reset_counters(int root_depth);
    void                   set_root_depth(int d);

  private:
    SplitPoint *grab(); // caller holds mtx
    void        worker(int idx);

    std::vector<std::unique_ptr<ThreadData>> tds;
    std::vector<std::thread>                 threads;
    std::vector<SplitPoint *>                active;
    std::mutex                               mtx;
    std::condition_variable                  cv;
    bool                                     exit_flag = false;
  };

  SplitPool &pool();

  // True when this thread's work is moot: global stop, or a beta cutoff anywhere in the chain
  // of splits it is working under. A search aborted this way returns garbage values — every
  // consumer (claim loops, think) must discard results once aborted() holds.
  bool aborted(ThreadData &t);

  // The master's side of a split: register, help drain it, propagate an outer abort to the
  // helpers, unregister, and wait for the helpers to let go. The caller merges sp afterwards
  // (discarding it if stop_requested()).
  void run_split(ThreadData &t, SplitPoint &sp, Position &pos, Stack *ss);

  // --- hooks implemented in search.cpp (the tree search proper) --------------------------------
  int        ybwc_search(ThreadData &t, Position &pos, Stack *ss, int alpha, int beta, int depth, int ply,
                         bool cutnode, bool pv);
  Histories &shared_hist();
  bool       stop_requested();
  int        lmr_reduction(int depth, int movecount);
  int        quiet_history(const Stack *ss, const Position &pos, Move m);

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
  // Shared by the tree search (probe/store) and both make sites' child-key prefetches.
  [[gnu::pure, gnu::always_inline]] inline uint64_t tt_key(const Position &p) {
    return p.get_hash() ^ ((p.castle_entry() & ALL_CASTLING_MASK) * 0x9E3779B97F4A7C15ull) ^
           (uint64_t(uint16_t(p.history[p.ply()].epsq)) * 0xC2B2AE3D27D4EB4Full);
  }

} // namespace search
