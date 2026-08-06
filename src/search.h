#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "position.h"
#include "types.h"

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
    int LMR_BASE = 80, LMR_DIV = 230, LMR_TACT_MC = 6, LMR_CONF_HI = 40, LMR_CONF_LO = 15;
    int MAT_BASE = 736, MAT_MULT = 5;
    int HB_MULT = 160, HB_SUB = 80, HB_MAX = 2000;
    int QS_FUT = 120;
    int IIR_DEPTH = 4;
    int RAZOR_DEPTH = 4, RAZOR_MULT = 300;
    int RFP_DEPTH = 8, RFP_MULT = 80;
    int NMP_DEPTH = 3, NMP_BASE = 3, NMP_DDIV = 3, NMP_EDIV = 200, NMP_ECAP = 3, NMP_VDEPTH = 12;
    int PC_MARGIN = 180, PC_IMP = 60, PC_DEPTH = 5;
    int LMP_BASE = 3, LMP_DEPTH = 8;
    int FUT_DEPTH = 6, FUT_BASE = 100, FUT_MULT = 120;
    int HP_DEPTH = 4, HP_MULT = 2048;
    int SEEP_DEPTH = 8, SEEP_QUIET = 50, SEEP_CAPT = 90;
    int SE_DEPTH = 8, SE_TTSUB = 3, SE_BMULT = 2, SE_DBL = 25, SE_TRI = 100, SE_DBLMAX = 6;
    int ASP_DELTA = 14;
    int RAZOR_UNC = 8, RFP_UNC = 8, FUT_UNC = 8;
  };
  extern Params prm;

  struct ParamInfo {
    const char *name;
    int        *p;
    int         def, lo, hi;
  };
  const std::vector<ParamInfo> &tunables();
  void params_dirty();

} // namespace search
