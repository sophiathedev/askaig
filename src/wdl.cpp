#include "wdl.h"

#include <algorithm>
#include <cmath>

#include "search.h" // search::MATE (the mate-score band)

namespace wdl {

  // The model: the expected score E(cp) uses the SAME logistic the Texel tuner fits the eval against
  // (tune.cpp: 1/(1+10^(-cp/scale))), so it is consistent with the engine's own eval calibration. The
  // draw rate D(cp) is modelled as a bell peaking at cp=0 (most draws in balanced positions) and
  // decaying as the score grows. Given E and D, W and L follow from E = W + D/2 and W + D + L = 1:
  //     W = E - D/2,   L = 1 - E - D/2.
  // Constants are reasonable starting points, NOT fit to askaig's own game results — they can be
  // calibrated later from the SPRT PGN (score-bucket → empirical W/D/L) if exact WDL accuracy matters.
  WDL from_cp(int cp) {
    constexpr double WIN_SCALE  = 350.0; // cp; bigger = slower rise of the win/expected-score curve
    constexpr double DRAW_MAX   = 0.58; // peak draw fraction at cp = 0
    constexpr double DRAW_WIDTH = 260.0; // cp; how fast the draw rate decays away from equality

    const double e = 1.0 / (1.0 + std::pow(10.0, -static_cast<double>(cp) / WIN_SCALE));
    const double x = static_cast<double>(cp) / DRAW_WIDTH;
    const double d = DRAW_MAX * std::exp(-x * x);
    double       w = e - d / 2.0;
    double       l = 1.0 - e - d / 2.0;
    w              = std::clamp(w, 0.0, 1.0);
    l              = std::clamp(l, 0.0, 1.0);

    int wi = static_cast<int>(std::lround(w * 1000.0));
    int li = static_cast<int>(std::lround(l * 1000.0));
    wi     = std::clamp(wi, 0, 1000);
    li     = std::clamp(li, 0, 1000 - wi);
    return {wi, 1000 - wi - li, li}; // draw absorbs the rounding residual so the three always sum to 1000
  }

  std::string format(int score) {
    constexpr int MAX_MATE_PLY = 256;
    if (score > search::MATE - MAX_MATE_PLY)
      return "wdl 1000 0 0";
    if (score < -(search::MATE - MAX_MATE_PLY))
      return "wdl 0 0 1000";
    const WDL w = from_cp(score);
    return "wdl " + std::to_string(w.win) + " " + std::to_string(w.draw) + " " + std::to_string(w.loss);
  }

} // namespace wdl
