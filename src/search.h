#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "position.h"
#include "types.h"

namespace search {

  constexpr int MAX_PLY       = 120;
  constexpr int MATE          = 32000;
  constexpr int MATE_IN_MAX   = MATE - MAX_PLY; // |score| >= this  <=>  a forced mate line
  constexpr int TABLEBASE_WIN = 30000;
  constexpr int INF           = MATE + 1;

  struct Result {
    Move              best{};
    int               score = 0;
    std::vector<Move> pv;
    uint64_t          nodes    = 0;
    uint64_t          tbhits   = 0;
    int               seldepth = 0;
  };

  using InfoFn = std::function<void(int, const Result &, uint64_t, long long)>;

  Result think(Position &pos, int max_depth, const InfoFn &info, int64_t soft_ms, int64_t hard_ms);

  void request_stop(); // asynchronous (from the UCI thread)
  // clear before launching the search thread
  void clear_stop();
  void new_game();

  void set_threads(int n);

  void set_contempt(int cp);

  void set_node_limit(uint64_t n);

  struct Params {
    int LMR_BASE = 93, LMR_DIV = 207, LMR_TACT_MC = 3, LMR_CONF_HI = 32, LMR_CONF_LO = 18;
    int MAT_BASE = 798, MAT_MULT = 4;
    int HB_MULT = 181, HB_SUB = 111, HB_MAX = 2797;
    int QS_FUT      = 103;
    int IIR_DEPTH   = 8;
    int RAZOR_DEPTH = 4, RAZOR_MULT = 280;
    int RFP_DEPTH = 12, RFP_MULT = 84;
    int NMP_DEPTH = 4, NMP_BASE = 4, NMP_DDIV = 5, NMP_EDIV = 103, NMP_ECAP = 5, NMP_VDEPTH = 13;
    int PC_MARGIN = 181, PC_IMP = 24, PC_DEPTH = 3;
    int LMP_BASE = 1, LMP_DEPTH = 11;
    int FUT_DEPTH = 4, FUT_BASE = 107, FUT_MULT = 111;
    int HP_DEPTH = 7, HP_MULT = 1989;
    int SEEP_DEPTH = 8, SEEP_QUIET = 57, SEEP_CAPT = 56;
    int SE_DEPTH = 6, SE_TTSUB = 3, SE_BMULT = 1, SE_DBL = 38, SE_TRI = 93, SE_DBLMAX = 5;
    int ASP_DELTA = 18;
    int RAZOR_UNC = 37, RFP_UNC = 3, FUT_UNC = 12;
  };
  extern Params prm;

  struct ParamInfo {
    const char *name;
    int        *p;
    int         def, lo, hi;
  };
  const std::vector<ParamInfo> &tunables();
  void                          params_dirty();
  bool                          selftest_correction();
  bool                          selftest_tt_eval();
  bool                          selftest_tt_rule50();

} // namespace search
