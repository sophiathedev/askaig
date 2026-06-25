#pragma once

#include <string>

namespace tune {

  // Gradient-descent eval tuner (`askaig tune <book> [threads] [lambda]`), after Andrew Grant's
  // *Evaluation & Tuning in Chess Engines* (Ethereal). Loads "<FEN> <result>" lines (result in
  // {1.0, 0.5, 0.0} from White's point of view, e.g. tools/extract.py output), then:
  //   1. extracts a per-position linear coefficient vector by central finite differences against the
  //      real eval (coeff[p][i] = (eval(w0+Δe_i) − eval(w0−Δe_i)) / 2Δ) plus a base value eval(w0),
  //      so the White-POV eval at any weights w is reconstructed as base[p] + Σ coeff[p][i]·(w−w0)
  //      without re-running evaluate() each epoch;
  //   2. fits the logistic scale K (base e);
  //   3. runs Adam over that linear model to minimise the logistic MSE, jointly over all eval::params()
  //      (the joint gradient avoids the collinearity artifacts of the old coordinate descent), with a
  //      90/10 train/validation split for early stopping and periodic re-linearization of the bilinear
  //      king-attack weights.
  // `lambda` is the AdamW weight-decay toward the compiled-in defaults (per-epoch decay fraction; 0 =
  // off). It regularizes the collinear blocks (the imbalance quadratic and king-safety terms), which an
  // unregularized fit drives into large mutually-cancelling values that lower MSE but lose Elo. Adam
  // hyper-parameters are env knobs: EPOCHS (5000), RELIN (50), PATIENCE (60), LR (1.0). Prints progress
  // and a paste-ready dump of the tuned values (also written to <book>.tuned).
  void run(const std::string &book_path, int threads, double lambda = 1e-3);

} // namespace tune
