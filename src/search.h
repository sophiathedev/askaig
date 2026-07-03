#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "position.h"
#include "types.h"

// The search: fail-soft negamax with alpha-beta, iterative deepening and (see search.cpp for
// the full list) PVS, aspiration windows, TT cutoffs, quiescence, and the standard modern
// pruning/extension/history stack. Single-threaded.
namespace search {

  constexpr int MAX_PLY = 120;
  constexpr int MATE    = 32000;
  constexpr int MATE_IN_MAX = MATE - MAX_PLY; // |score| >= this  <=>  a forced mate line
  constexpr int INF         = MATE + 1;

  struct Result {
    Move              best{};
    int               score = 0;
    std::vector<Move> pv;
    uint64_t          nodes    = 0;
    int               seldepth = 0;
  };

  // Called after every completed iteration with (depth, result-so-far, nodes, elapsed ms).
  using InfoFn = std::function<void(int, const Result &, uint64_t, long long)>;

  // Searches `pos` up to `max_depth`. Time limits in ms (<= 0 = none): the soft limit stops
  // between iterations, the hard limit aborts mid-tree. Runs until "stop" when both are 0 and
  // max_depth = MAX_PLY.
  Result think(Position &pos, int max_depth, const InfoFn &info, int64_t soft_ms, int64_t hard_ms);

  void request_stop(); // asynchronous (from the UCI thread)
  // Arms think() for a fresh search. MUST be called synchronously on whatever thread might
  // race a subsequent request_stop() — for a backgrounded search that's the UCI thread, right
  // before spawning the search thread, NOT inside think() itself (which used to reset this on
  // the NEW thread; a "go" immediately followed by "stop" could then have its stop request
  // wiped by that reset if it lost the race, silently running the full search anyway).
  void clear_stop();
  void new_game(); // clears history tables (TT cleared separately via tt::clear)

  // YBWC parallelism: `n` total threads (n-1 pool helpers; 1 = single-threaded, bit-identical
  // to no pool), splitting only at nodes with depth >= the split depth (the "Split" option).
  void set_threads(int n);
  void set_split_depth(int d);

  // Centipawns the root side is willing to give up to avoid a draw (the "Contempt" option;
  // negative seeks draws instead). 0 (default) reproduces the old unconditional draw==0 exactly.
  void set_contempt(int cp);

} // namespace search
