#pragma once

#include <string>

namespace tune {

  // Texel tuning driver (`askaig tune <book> [threads]`): loads "<FEN> <result>" lines (result in
  // {1.0, 0.5, 0.0} from White's point of view, e.g. tools/extract.py output), fits the sigmoid
  // scale K, then runs coordinate descent over every eval::params() constant, minimising the mean
  // squared error between sigmoid(K * eval) and the game results. Prints progress (positions/s,
  // per-iteration train/validation error, every accepted parameter change) and ends with a
  // paste-ready dump of the tuned values (also written to <book>.tuned).
  void run(const std::string &book_path, int threads);

} // namespace tune
