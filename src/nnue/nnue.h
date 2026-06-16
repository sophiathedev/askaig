#pragma once

// Public NNUE interface. The implementation lives in nnue.cpp; the network shape / feature indexing /
// accumulator are in network.h / features.h / accumulator.h.
class Position;

namespace nnue {

  // Loads the embedded default network. Call once at startup (main.cpp, after the move-gen databases).
  // v1: installs a deterministic PLACEHOLDER net so the inference path is exercisable before a real net
  // is trained — replace by loading the embedded trained net once it exists.
  void init();

  // Loads a network from a file (the UCI `EvalFile` option). Returns false on failure (the current net
  // is kept).
  bool load(const char *path);

  // True once a network is loaded.
  bool active();

  // Static evaluation in centipawns from the side-to-move's perspective (refresh-from-scratch in v1/M1;
  // incremental accumulator arrives in M2).
  int evaluate(const Position &pos);

  // Self-test: mirror-symmetry (eval(pos) == eval(color-mirrored pos)) over a fixed suite. Used by the
  // `nnuetest` CLI subcommand to verify the feature/perspective/mirror code (M1) and SIMD==portable (M3).
  // Returns the number of failures (0 = pass).
  int self_test();

} // namespace nnue
