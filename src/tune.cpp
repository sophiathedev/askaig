#include "tune.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "eval.h"
#include "position.h"
#include "types.h"

// Gradient-descent ("Ethereal"-style, after Andrew Grant's *Evaluation & Tuning in Chess Engines*)
// replacement for the old coordinate-descent Texel tuner.
//
// The key idea: the static eval is LINEAR in each tunable weight (the only exception is the king-zone
// term, bilinear in KING_ATT_WEIGHT × KING_ATT_UNIT). So for every training position we extract, ONCE,
// a coefficient vector by central finite differences against the real evaluate():
//
//     coeff[p][i] = ( eval(w0 + Δ·e_i)  −  eval(w0 − Δ·e_i) ) / (2Δ)       (White's point of view)
//
// and a base value base[p] = eval(w0). The White-POV eval at any weight vector w is then reconstructed
// WITHOUT re-running evaluate():  E_p(w) ≈ base[p] + Σ_i coeff[p][i]·(w_i − w0_i).  For the ~118 purely
// linear weights this slope is exact and constant for all w; the 5 bilinear king weights are
// re-linearized periodically (their coeff + base re-extracted at the current point). Adam then minimises
// the logistic MSE over this cheap linear model — every epoch is just sparse-ish dot products, so the
// whole tune (all ~123 weights jointly) takes seconds, and the joint gradient sidesteps the collinearity
// traps that made coordinate descent emit artifacts (the rook-7th / threat-by-rook noise).
//
// Extraction reproduces the engine's behaviour exactly (it calls evaluate()), so it correctly folds in
// the game-phase taper AND the drawish scale factor ξ (scale_factor/SCALE_NORMAL). The float model only
// smooths the eval's internal integer truncations; the verify() gate measures that residual.

namespace tune {

  namespace {

    // A training position, packed (a full Position is ~33 KB). The board is rebuilt into a per-thread
    // scratch Position before evaluation — evaluate() only reads the piece bitboards, the mailbox, the
    // side to move and the incremental PSQT accumulators, all of which put/remove_piece maintain.
    struct Packed {
      uint8_t board[NSQUARES]; // Piece per square (NO_PIECE = empty)
      uint8_t stm;             // side to move
      float   result;          // game result from White's point of view: 1 / 0.5 / 0
    };

    // Rebuild `scratch` to hold `pk` (square-by-square; remove_piece keeps the PSQT accumulators
    // consistent). No evaluation — callers may then evaluate the same board at several weight vectors.
    void build(Position &scratch, const Packed &pk) noexcept {
      for (int s = 0; s < static_cast<int>(NSQUARES); ++s) {
        if (scratch.at(Square(s)) != NO_PIECE)
          scratch.remove_piece(Square(s));
        if (pk.board[s] != NO_PIECE)
          scratch.put_piece(Piece(pk.board[s]), Square(s));
      }
      scratch.set_turn(Color(pk.stm));
    }

    // White-POV static eval of an already-built scratch (the scratch never plays a move, so its
    // halfmove clock stays 0 and the fifty-move damping inside evaluate() is the identity).
    [[gnu::always_inline]] inline int white_eval(Position &scratch, uint8_t stm) noexcept {
      const int v = eval::evaluate(scratch);
      return stm == WHITE ? v : -v;
    }

    // Evaluate ALL positions into out[] at the CURRENT global eval weights, split over `threads`
    // workers. Each worker spawns fresh (so its thread-local pawn cache starts empty and is filled with
    // the current weight — a just-changed weight is never answered from a stale cache entry).
    void eval_all(const std::vector<Packed> &data, std::vector<float> &out, int threads) {
      const size_t             n = data.size();
      std::vector<std::thread> ws;
      ws.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; ++t)
        ws.emplace_back([&, t]() {
          Position     scratch; // empty board (all NO_PIECE)
          const size_t a = n * static_cast<size_t>(t) / static_cast<size_t>(threads);
          const size_t b = n * (static_cast<size_t>(t) + 1) / static_cast<size_t>(threads);
          for (size_t i = a; i < b; ++i) {
            build(scratch, data[i]);
            out[i] = static_cast<float>(white_eval(scratch, data[i].stm));
          }
        });
      for (auto &w: ws)
        w.join();
    }

    // Logistic link, base e (Andrew Grant §2.3: dropping the base-10 / 400 constants simplifies the
    // gradient — they are absorbed into K).  σ(E) = 1 / (1 + e^(−K·E)),  dσ/dE = K·σ·(1−σ).
    [[gnu::const]] inline double sigmoid(double K, double E) noexcept { return 1.0 / (1.0 + std::exp(-K * E)); }

    // Mean squared error of σ(K·eval) vs results over [lo, hi), given precomputed White-POV evals.
    double mse(const std::vector<float> &evals, const std::vector<Packed> &data, size_t lo, size_t hi, double K) {
      double s = 0.0;
      for (size_t i = lo; i < hi; ++i) {
        const double e = static_cast<double>(data[i].result) - sigmoid(K, evals[i]);
        s += e * e;
      }
      return s / static_cast<double>(hi - lo);
    }

    // Loads "<FEN> <result>" lines. Only the board and side fields of the FEN matter to the (raw) eval;
    // the last whitespace-separated token is the result (1.0 / 0.5 / 0.0, White's point of view).
    std::vector<Packed> load(const std::string &path) {
      std::vector<Packed> data;
      std::ifstream       in(path);
      if (!in) {
        std::cout << "tune: cannot open '" << path << "'\n";
        return data;
      }
      const auto  t0 = std::chrono::steady_clock::now();
      std::string line;
      size_t      bad = 0;
      while (std::getline(in, line)) {
        if (line.empty())
          continue;
        Packed pk{};
        std::memset(pk.board, NO_PIECE, sizeof(pk.board));

        size_t i  = 0;
        int    sq = a8;
        for (; i < line.size() && line[i] != ' '; ++i) {
          const char ch = line[i];
          if (isdigit(static_cast<unsigned char>(ch)))
            sq += (ch - '0') * EAST;
          else if (ch == '/')
            sq += 2 * SOUTH;
          else {
            const size_t pc = PIECE_STR.find(ch);
            if (pc == std::string::npos || sq < a1 || sq > h8) {
              bad = bad + 1;
              goto next_line;
            }
            pk.board[sq++] = static_cast<uint8_t>(pc);
          }
        }
        pk.stm = (i + 1 < line.size() && line[i + 1] == 'b') ? BLACK : WHITE;
        {
          const size_t sp = line.find_last_of(' ');
          const double r  = std::strtod(line.c_str() + sp + 1, nullptr);
          if (r != 0.0 && r != 0.5 && r != 1.0) {
            ++bad;
            continue;
          }
          pk.result = static_cast<float>(r);
        }
        data.push_back(pk);
        if (data.size() % 100000 == 0)
          std::cout << "  loaded " << data.size() << " positions...\n" << std::flush;
      next_line:;
      }
      const auto ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
      std::cout << "loaded " << data.size() << " positions in " << ms << "ms";
      if (bad)
        std::cout << " (" << bad << " malformed lines skipped)";
      std::cout << "\n" << std::flush;
      return data;
    }

    // An integer environment knob (e.g. EPOCHS=4000 ./askaig tune ...), with a default.
    double envd(const char *name, double def) {
      const char *v = std::getenv(name);
      return v ? std::strtod(v, nullptr) : def;
    }

    // Finite-difference step (centipawns) for coefficient extraction. Large enough to average over the
    // eval's internal integer truncations, small enough to stay in each weight's locally-linear region.
    constexpr int FD_STEP = 8;

  } // namespace

  void run(const std::string &book_path, int threads, double lambda) {
    if (threads < 1)
      threads = 1;

    const auto &params = eval::params();
    const int   NP     = static_cast<int>(params.size());
    std::cout << "Gradient tuning: " << NP << " parameters, " << threads << " threads\n";

    // Starting weights = the compiled-in defaults (the regularization anchor + linearization point).
    std::vector<int> w0(NP);
    for (int i = 0; i < NP; ++i)
      w0[i] = *params[i].value;

    // The bilinear king-attack weights (KING_ATT_WEIGHT_* and KING_ATT_UNIT): their coefficient depends
    // on the OTHER king weights, so it is re-extracted periodically rather than fixed at w0.
    std::vector<int> kingIdx;
    for (int i = 0; i < NP; ++i)
      if (params[i].name.rfind("KING_ATT_", 0) == 0)
        kingIdx.push_back(i);

    std::vector<Packed> data = load(book_path);
    if (data.size() < 1000) {
      std::cout << "tune: not enough positions, aborting\n";
      return;
    }
    std::mt19937 rng(20260611);
    std::shuffle(data.begin(), data.end(), rng);
    const size_t N      = data.size();
    const size_t ntrain = N * 9 / 10; // 90/10 train/validation split (val = overfitting alarm)

    // === Coefficient extraction (central finite differences against the real eval) =================
    // Memory: N*NP floats. ~0.5 GB at 1M positions × 123 weights — quality beats quantity for the
    // dataset (Grant §2.2), so a few hundred-k clean positions is the intended scale.
    std::cout << "extracting coefficients (FD step ±" << FD_STEP << ", " << (2 * NP + 1)
              << " eval passes)...\n"
              << std::flush;
    std::vector<float> base(N), plus(N), minus(N);
    std::vector<float> coeff(N * static_cast<size_t>(NP));
    const auto         tExtract = std::chrono::steady_clock::now();

    eval_all(data, base, threads); // base[p] = eval(w0)
    for (int i = 0; i < NP; ++i) {
      int       *v = params[i].value;
      const int  o = w0[i];
      *v           = o + FD_STEP;
      eval_all(data, plus, threads);
      *v = o - FD_STEP;
      eval_all(data, minus, threads);
      *v                = o; // restore
      const double inv  = 1.0 / (2.0 * FD_STEP);
      for (size_t p = 0; p < N; ++p)
        coeff[p * NP + i] = static_cast<float>((plus[p] - minus[p]) * inv);
      if ((i + 1) % 20 == 0 || i + 1 == NP)
        std::cout << "  " << (i + 1) << "/" << NP << " weights\r" << std::flush;
    }
    {
      const auto s = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                           tExtract)
                             .count();
      std::printf("\nextraction done in %lldms\n", static_cast<long long>(s));
      std::cout << std::flush;
    }

    // === Verify gate: the linear model must reproduce the real eval under a JOINT random perturbation
    // (superposition holds for the linear weights; any large residual would flag a bug — only the king
    // bilinear cross-term is expected to contribute a small error).
    {
      eval::set_tuning(true); // bypass the pawn cache: we re-evaluate one board at many weight vectors
      std::mt19937 grng(777);
      Position     scratch;
      double       maxerr = 0, sumerr = 0;
      const int    trials = 400, gstep = 4;
      std::vector<int> dlt(NP);
      for (int tr = 0; tr < trials; ++tr) {
        const size_t p = grng() % N;
        build(scratch, data[p]);
        for (int i = 0; i < NP; ++i) {
          dlt[i]            = static_cast<int>(grng() % (2 * gstep + 1)) - gstep;
          *params[i].value  = w0[i] + dlt[i];
        }
        double model = base[p];
        for (int i = 0; i < NP; ++i)
          model += static_cast<double>(coeff[p * NP + i]) * dlt[i];
        const double real = white_eval(scratch, data[p].stm);
        const double err  = std::abs(real - model);
        maxerr            = err > maxerr ? err : maxerr;
        sumerr += err;
      }
      for (int i = 0; i < NP; ++i)
        *params[i].value = w0[i]; // restore
      eval::set_tuning(false);
      std::printf("verify: linear-model vs real eval over %d joint perturbations (±%d cp/weight): "
                  "mean %.2f cp, max %.2f cp\n",
                  trials, gstep, sumerr / trials, maxerr);
      std::cout << std::flush;
    }

    // === Fit the logistic scale K on the training set (ternary search on the convex-ish error). ====
    double klo = 0.0005, khi = 0.02;
    for (int it = 0; it < 32; ++it) {
      const double k1 = klo + (khi - klo) / 3, k2 = khi - (khi - klo) / 3;
      if (mse(base, data, 0, ntrain, k1) < mse(base, data, 0, ntrain, k2))
        khi = k2;
      else
        klo = k1;
    }
    const double K = (klo + khi) / 2;
    std::printf("K = %.6f   initial error: train %.6f  val %.6f  (%zu train / %zu val)\n", K,
                mse(base, data, 0, ntrain, K), mse(base, data, ntrain, N, K), ntrain, N - ntrain);
    std::cout << std::flush;

    // === Adam over the linear model ================================================================
    const int    EPOCHS   = static_cast<int>(envd("EPOCHS", 5000));
    const int    RELIN    = static_cast<int>(envd("RELIN", 50));  // re-linearize king weights every N epochs
    const int    PATIENCE = static_cast<int>(envd("PATIENCE", 60)); // early-stop after N epochs w/o val gain
    const double lr       = envd("LR", 1.0);
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;

    std::vector<double> theta(NP), m(NP, 0.0), vmom(NP, 0.0);
    std::vector<int>    wlin = w0; // current linearization point (where base[] and king coeffs are valid)
    for (int i = 0; i < NP; ++i)
      theta[i] = w0[i];

    auto model_eval = [&](size_t p) noexcept {
      double e = base[p];
      const float *row = &coeff[p * NP];
      for (int i = 0; i < NP; ++i)
        e += static_cast<double>(row[i]) * (theta[i] - wlin[i]);
      return e;
    };

    std::vector<double> best = theta;
    double              bestVal = 1e18;
    int                 sinceBest = 0;
    const auto          tAdam = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
      // Re-linearize the bilinear king weights (and refresh base[]) at the current point: set the global
      // weights to round(theta), recompute base = eval(theta), re-extract each king coefficient there.
      if (epoch > 1 && (epoch - 1) % RELIN == 0) {
        std::vector<int> wl(NP);
        for (int i = 0; i < NP; ++i) {
          wl[i]            = static_cast<int>(std::lround(theta[i]));
          *params[i].value = wl[i];
        }
        eval_all(data, base, threads);
        for (int idx: kingIdx) {
          int       *v = params[idx].value;
          const int  o = wl[idx];
          *v           = o + FD_STEP;
          eval_all(data, plus, threads);
          *v = o - FD_STEP;
          eval_all(data, minus, threads);
          *v               = o;
          const double inv = 1.0 / (2.0 * FD_STEP);
          for (size_t p = 0; p < N; ++p)
            coeff[p * NP + idx] = static_cast<float>((plus[p] - minus[p]) * inv);
        }
        wlin = wl;
      }

      // Full-batch gradient of the logistic MSE over the training set, split over threads.
      std::vector<std::vector<double>> tgrad(threads, std::vector<double>(NP, 0.0));
      std::vector<std::thread>         ws;
      ws.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; ++t)
        ws.emplace_back([&, t]() {
          const size_t a = ntrain * static_cast<size_t>(t) / static_cast<size_t>(threads);
          const size_t b = ntrain * (static_cast<size_t>(t) + 1) / static_cast<size_t>(threads);
          auto        &g = tgrad[static_cast<size_t>(t)];
          for (size_t p = a; p < b; ++p) {
            const double E   = model_eval(p);
            const double sg  = sigmoid(K, E);
            const double err = static_cast<double>(data[p].result) - sg;
            // d(mse_p)/dθ_i = −2·err·(dσ/dE)·(dE/dθ_i) ;  dσ/dE = K·sg·(1−sg) ;  dE/dθ_i = coeff[p][i]
            const double d   = -2.0 * err * K * sg * (1.0 - sg);
            const float *row = &coeff[p * NP];
            for (int i = 0; i < NP; ++i)
              g[static_cast<size_t>(i)] += d * row[i];
          }
        });
      for (auto &w: ws)
        w.join();

      // Adam update (gradient averaged over the training set) + decoupled weight decay toward the
      // defaults (AdamW). The decay is what keeps the COLLINEAR blocks sane: the imbalance quadratic
      // and the king-safety terms have many near-degenerate directions, so an unregularized fit drifts
      // them into huge mutually-cancelling values that lower MSE but lose Elo. Decay pulls weak-signal
      // (collinear) weights back toward their SF-derived defaults while letting strong-signal weights
      // (passers, threats, mobility) move freely. lambda = per-epoch decay fraction (0 = off).
      const double bc1 = 1.0 - std::pow(b1, epoch), bc2 = 1.0 - std::pow(b2, epoch);
      for (int i = 0; i < NP; ++i) {
        double gi = 0.0;
        for (int t = 0; t < threads; ++t)
          gi += tgrad[static_cast<size_t>(t)][static_cast<size_t>(i)];
        gi /= static_cast<double>(ntrain);
        m[i]              = b1 * m[i] + (1 - b1) * gi;
        vmom[i]           = b2 * vmom[i] + (1 - b2) * gi * gi;
        const double mhat = m[i] / bc1, vhat = vmom[i] / bc2;
        theta[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        if (lambda > 0.0)
          theta[i] -= lambda * (theta[i] - w0[i]);
      }

      // Validation error on the linear model (cheap) — the early-stopping signal.
      double val = 0.0;
      for (size_t p = ntrain; p < N; ++p) {
        const double e = static_cast<double>(data[p].result) - sigmoid(K, model_eval(p));
        val += e * e;
      }
      val /= static_cast<double>(N - ntrain);
      if (val + 1e-9 < bestVal) {
        bestVal   = val;
        best      = theta;
        sinceBest = 0;
      } else if (++sinceBest >= PATIENCE) {
        std::printf("early stop at epoch %d (no val gain for %d epochs)\n", epoch, PATIENCE);
        break;
      }
      if (epoch <= 5 || epoch % 50 == 0) {
        double tr = 0.0;
        for (size_t p = 0; p < ntrain; ++p) {
          const double e = static_cast<double>(data[p].result) - sigmoid(K, model_eval(p));
          tr += e * e;
        }
        tr /= static_cast<double>(ntrain);
        const auto el = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                                         tAdam)
                                .count();
        std::printf("epoch %4d: train %.6f  val %.6f  (best %.6f)  [%llds]\n", epoch, tr, val, bestVal,
                    static_cast<long long>(el));
        std::cout << std::flush;
      }
    }

    // === Final: round best weights, report the REAL (not linear-model) error, dump paste-ready. =====
    theta = best;
    std::vector<int> tuned(NP);
    for (int i = 0; i < NP; ++i) {
      tuned[i]         = static_cast<int>(std::lround(theta[i]));
      *params[i].value = tuned[i];
    }
    eval_all(data, base, threads); // real eval at the tuned weights
    std::printf("final (real eval): train %.6f  val %.6f\n", mse(base, data, 0, ntrain, K),
                mse(base, data, ntrain, N, K));

    std::cout << "\n=== tuned values (paste into src/eval.cpp) ===\n";
    std::ofstream out(book_path + ".tuned");
    for (int i = 0; i < NP; ++i) {
      std::printf("%-24s = %d%s\n", params[i].name.c_str(), tuned[i], tuned[i] != w0[i] ? "" : "");
      out << params[i].name << " = " << tuned[i] << "\n";
    }
    std::printf("(saved to %s.tuned)\n", book_path.c_str());
    std::cout << std::flush;

    for (int i = 0; i < NP; ++i)
      *params[i].value = w0[i]; // leave the process's globals as they were
  }

} // namespace tune
