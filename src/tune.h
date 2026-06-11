#pragma once

#include <string>

namespace tune {

  // Texel tuning driver (`askaig tune <book> [threads] [lambda]`): loads "<FEN> <result>" lines
  // (result in {1.0, 0.5, 0.0} from White's point of view, e.g. tools/extract.py output), fits the
  // sigmoid scale K, then runs coordinate descent over every eval::params() constant, minimising
  // mse(sigmoid(K * eval), results) + an L2 penalty pulling toward the compiled-in defaults
  // (`lambda`, default 1e-3; collinear features otherwise drift into absurd mutually-cancelling
  // pairs), with hard per-parameter bounds of default +/- max(2|default|, 30). Prints progress
  // (positions/s, per-iteration train/validation error, every accepted change) and ends with a
  // paste-ready dump of the tuned values (also written to <book>.tuned).
  void run(const std::string &book_path, int threads, double lambda = 1e-3);

} // namespace tune
