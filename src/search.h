#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "position.h"
#include "types.h"

// The search: fail-soft negamax with alpha-beta, iterative deepening and (see search.cpp for
// the full list) PVS, aspiration windows, TT cutoffs, quiescence, and the standard modern
// pruning/extension/history stack. Parallelised with lazy SMP (smp.h/smp.cpp).
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

  // Lazy SMP: `n` total threads (n-1 pool helpers; 1 = single-threaded, bit-identical to no
  // pool). Helpers run their own iterative deepening and share only the TT/history tables.
  void set_threads(int n);

  // Centipawns the root side is willing to give up to avoid a draw (the "Contempt" option;
  // negative seeks draws instead). 0 (default) reproduces the old unconditional draw==0 exactly.
  void set_contempt(int cp);

  // Soft node cap for the next searches (0 = off). Polled on the same 2048-node throttle as
  // the hard clock; not reset automatically — go_cmd/bench/datagen each set what they need.
  void set_node_limit(uint64_t n);

  // --- SPSA-tunable search parameters -------------------------------------------------------
  // Every margin / depth gate / divisor the search uses, as runtime ints so a tuning run can
  // drive them per game through hidden UCI spin options (exposed only by `askaig --debug`,
  // see uci.cpp; tools/spsa.py discovers and tunes them). The defaults reproduce the committed
  // constants exactly — a default-valued build keeps the bench signature bit-for-bit.
  struct Params {
    // LMR (LMR_BASE/LMR_DIV are the table formula constants x100)
    int LMR_BASE = 80, LMR_DIV = 230, LMR_TACT_MC = 6, LMR_CONF_HI = 40, LMR_CONF_LO = 15;
    // static-eval material scaling: eval * (MAT_BASE + MAT_MULT*npm) / 1024
    int MAT_BASE = 736, MAT_MULT = 5;
    // history bonus: min(HB_MULT*depth - HB_SUB, HB_MAX)
    int HB_MULT = 160, HB_SUB = 80, HB_MAX = 2000;
    // quiescence futility margin
    int QS_FUT = 120;
    // internal iterative reduction depth gate
    int IIR_DEPTH = 4;
    // razoring: depth <= RAZOR_DEPTH, margin RAZOR_MULT*depth
    int RAZOR_DEPTH = 4, RAZOR_MULT = 300;
    // reverse futility: depth <= RFP_DEPTH, margin RFP_MULT*(depth-improving)
    int RFP_DEPTH = 8, RFP_MULT = 80;
    // null move: gate, R = NMP_BASE + depth/NMP_DDIV + min((eval-beta)/NMP_EDIV, NMP_ECAP),
    // fail-high verification from NMP_VDEPTH
    int NMP_DEPTH = 3, NMP_BASE = 3, NMP_DDIV = 3, NMP_EDIV = 200, NMP_ECAP = 3, NMP_VDEPTH = 12;
    // ProbCut: beta + PC_MARGIN - PC_IMP*improving, from PC_DEPTH
    int PC_MARGIN = 180, PC_IMP = 60, PC_DEPTH = 5;
    // late move pruning: (LMP_BASE + depth^2)/(2-improving), depth <= LMP_DEPTH
    int LMP_BASE = 3, LMP_DEPTH = 8;
    // futility pruning: depth <= FUT_DEPTH, margin FUT_BASE + FUT_MULT*depth
    int FUT_DEPTH = 6, FUT_BASE = 100, FUT_MULT = 120;
    // history pruning: depth <= HP_DEPTH, threshold -HP_MULT*depth
    int HP_DEPTH = 4, HP_MULT = 2048;
    // SEE pruning: depth <= SEEP_DEPTH, thresholds -SEEP_QUIET/-SEEP_CAPT * depth
    int SEEP_DEPTH = 8, SEEP_QUIET = 50, SEEP_CAPT = 90;
    // singular extensions: gate, tt-depth slack, s_beta = ttsc - SE_BMULT*depth, double/triple
    // extension margins, double-extension budget per line
    int SE_DEPTH = 8, SE_TTSUB = 3, SE_BMULT = 2, SE_DBL = 25, SE_TRI = 100, SE_DBLMAX = 6;
    // aspiration window half-width
    int ASP_DELTA = 14;
  };
  extern Params prm;

  // One row per tunable for the UCI layer: pointer into `prm` plus default and bounds.
  struct ParamInfo {
    const char *name;
    int        *p;
    int         def, lo, hi;
  };
  // The full registration table (name -> field), in a stable order.
  const std::vector<ParamInfo> &tunables();
  // Invalidate derived tables (the LMR reduction grid) after writing a parameter.
  void params_dirty();

} // namespace search
