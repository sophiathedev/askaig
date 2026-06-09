#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "types.h"

class Position;

namespace search {

  // Mate scores live just below INF; a forced mate is reported as (MATE - ply), so a mate that
  // is closer to the root (smaller ply) scores higher and is therefore always preferred.
  constexpr int INF  = 32000;
  constexpr int MATE = 31000;

  // The search's hard ply ceiling, and the depth used for "go infinite": iterative deepening runs
  // up to here, which in practice no real position ever reaches before a "stop" arrives.
  constexpr int MAX_DEPTH = 128;

  struct Result {
    Move              best; // the chosen move (a null Move{} if there are no legal moves)
    int               score; // score in centipawns from the side-to-move's perspective (or a mate score)
    uint64_t          nodes; // nodes searched
    std::vector<Move> pv; // principal variation (best line), starting with `best`
  };

  // Called after each completed main-thread depth: (depth, partial result, cumulative nodes,
  // elapsed milliseconds). Used to emit UCI "info" lines.
  using InfoCallback = std::function<void(int depth, const Result &result, uint64_t nodes, long long ms)>;

  // Iterative-deepening negamax/alpha-beta search to `max_depth` using `threads` threads (Lazy
  // SMP, sharing the transposition table). Returns the best move; invokes `on_iteration` after
  // each main-thread depth. `pos` is not modified (each thread searches a copy). Runs until
  // `max_depth` or until `request_stop()` is called, in which case it returns the best move from
  // the last fully-completed iteration. Intended to be run on a background thread.
  Result think(const Position &pos, int max_depth, int threads, const InfoCallback &on_iteration);

  // Asks an in-progress `think()` to stop as soon as possible (the UCI "stop" command). Safe to
  // call from another thread; cleared automatically at the start of the next `think()`.
  void request_stop();

} // namespace search
