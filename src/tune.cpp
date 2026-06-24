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

namespace tune {

  namespace {

    // A training position, packed (a full Position is ~33 KB because of its undo stack; storing
    // hundreds of thousands of those is impossible). The board is rebuilt into a per-thread
    // scratch Position before each evaluation — evaluate_raw() only reads the piece bitboards,
    // the mailbox, the side to move and the incremental PSQT accumulators, all of which
    // put_piece/remove_piece/set_turn maintain.
    struct Packed {
      uint8_t board[NSQUARES]; // Piece per square (NO_PIECE = empty)
      uint8_t stm; // side to move
      float   result; // game result from White's point of view: 1 / 0.5 / 0
    };

    // Rebuilds `scratch` to hold `pk` and returns White's static eval of it. The scratch is wiped
    // square-by-square (remove_piece keeps the PSQT accumulators consistent) — cheaper and safer
    // than re-parsing a FEN or value-initialising a fresh 33 KB Position per call.
    int white_eval(Position &scratch, const Packed &pk) noexcept {
      for (int s = 0; s < static_cast<int>(NSQUARES); ++s) {
        if (scratch.at(Square(s)) != NO_PIECE)
          scratch.remove_piece(Square(s));
        if (pk.board[s] != NO_PIECE)
          scratch.put_piece(Piece(pk.board[s]), Square(s));
      }
      scratch.set_turn(Color(pk.stm));
      // Side-to-move POV. The scratch never plays a move, so its halfmove clock stays 0 and the
      // fifty-move damping inside evaluate() is the identity — this IS the raw eval.
      const int v = eval::evaluate(scratch);
      return pk.stm == WHITE ? v : -v;
    }

    [[gnu::const]] inline double sigmoid(double k, int cp) noexcept {
      return 1.0 / (1.0 + std::pow(10.0, -k * cp / 400.0));
    }

    // Mean squared error of sigmoid(K*eval) vs the results over data[lo, hi), split over
    // `threads` workers (each with its own scratch Position; the eval's pawn cache is
    // thread_local, so the workers never share mutable state).
    double mse(const std::vector<Packed> &data, size_t lo, size_t hi, double k, int threads) {
      const size_t             n = hi - lo;
      std::vector<double>      sums(static_cast<size_t>(threads), 0.0);
      std::vector<std::thread> ws;
      ws.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; ++t)
        ws.emplace_back([&, t]() {
          Position     scratch; // starts empty (default-constructed board is all NO_PIECE)
          const size_t a = lo + n * static_cast<size_t>(t) / static_cast<size_t>(threads);
          const size_t b = lo + n * (static_cast<size_t>(t) + 1) / static_cast<size_t>(threads);
          double       s = 0.0;
          for (size_t i = a; i < b; ++i) {
            const double e = data[i].result - sigmoid(k, white_eval(scratch, data[i]));
            s += e * e;
          }
          sums[static_cast<size_t>(t)] = s;
        });
      for (auto &w: ws)
        w.join();
      double s = 0.0;
      for (double x: sums)
        s += x;
      return s / static_cast<double>(n);
    }

    // Loads "<FEN> <result>" lines. Only the board and side fields of the FEN matter to the
    // (raw) eval; the last whitespace-separated token is the result.
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

        // Board field (same walk as Position::set).
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

        // Result: the last token.
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

  } // namespace

  void run(const std::string &book_path, int threads, double lambda) {
    if (threads < 1)
      threads = 1;
    const auto &params = eval::params();
    std::cout << "Texel tuning: " << params.size() << " parameters, " << threads << " threads\n";

    // Anti-runaway guards, both relative to the COMPILED-IN defaults (run the tuner from the
    // hand-tuned baseline, not from a previous tuning round):
    //  - hard bounds: each parameter may move at most max(2*|default|, 30) from its default —
    //    collinear features (e.g. the passer bonus vs the king-race term, rook-threatens-rook vs
    //    everything) otherwise trade off against each other into absurd pairs like -466/+564 that
    //    happen to cancel on the training set;
    //  - L2 regularization: the objective is mse + lambda * mean(((v - default)/scale)^2), with
    //    scale = max(|default|, 10): big drifts must EARN their distance with real error
    //    reduction. lambda = 0 disables it; ~1e-3 blocks runaways while letting modest moves pass.
    std::vector<int> def(params.size()), plo(params.size()), phi(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
      def[i]         = *params[i].value;
      const int span = std::max(2 * std::abs(def[i]), 30);
      plo[i]         = def[i] - span;
      phi[i]         = def[i] + span;
    }
    const auto reg = [&]() {
      double r = 0.0;
      for (size_t i = 0; i < params.size(); ++i) {
        const double sc = std::max(std::abs(static_cast<double>(def[i])), 10.0);
        const double d  = (static_cast<double>(*params[i].value) - def[i]) / sc;
        r += d * d;
      }
      return lambda * r / static_cast<double>(params.size());
    };
    std::printf("regularization: lambda = %g, bounds = default +/- max(2|default|, 30)\n", lambda);

    std::vector<Packed> data = load(book_path);
    if (data.size() < 1000) {
      std::cout << "tune: not enough positions, aborting\n";
      return;
    }

    // Deterministic shuffle, then a 90/10 train/validation split: the validation error is the
    // overfitting alarm — it must fall along with the training error, or the "improvement" is
    // just memorising the data set.
    std::mt19937 rng(20260611);
    std::shuffle(data.begin(), data.end(), rng);
    const size_t ntrain = data.size() * 9 / 10;
    const size_t nval   = data.size() - ntrain;

    // Eval throughput (it bounds everything): time a few full training passes.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i)
      (void) mse(data, 0, ntrain, 1.0, threads);
    const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    std::printf("speed: %.2fM positions/s (one error pass = %lldms)\n", 3.0 * static_cast<double>(ntrain) / us,
                static_cast<long long>(us / 3000));
    std::cout << std::flush;

    // Fit the sigmoid scale K on the training set (ternary search on the convex-ish error curve).
    double klo = 0.3, khi = 3.0;
    for (int it = 0; it < 24; ++it) {
      const double k1 = klo + (khi - klo) / 3, k2 = khi - (khi - klo) / 3;
      if (mse(data, 0, ntrain, k1, threads) < mse(data, 0, ntrain, k2, threads))
        khi = k2;
      else
        klo = k1;
    }
    const double K     = (klo + khi) / 2;
    double       train = mse(data, 0, ntrain, K, threads);
    double       val   = mse(data, ntrain, data.size(), K, threads);
    std::printf("K = %.4f   initial error: train %.6f  val %.6f  (%zu train / %zu val)\n", K, train, val, ntrain, nval);
    std::cout << std::flush;

    // Coordinate descent with a shrinking step, minimising the REGULARIZED objective (see above);
    // the train/val errors printed stay raw mse so they remain comparable across lambdas. Each
    // accepted change is printed; an iteration that changes nothing at step 1 ends the run.
    double     obj    = train + reg();
    const auto tstart = std::chrono::steady_clock::now();
    for (int step: {8, 4, 2, 1}) {
      for (int iter = 1;; ++iter) {
        int changed = 0;
        for (size_t pi = 0; pi < params.size(); ++pi) {
          int      &v    = *params[pi].value;
          const int orig = v;
          for (const int trial: {orig + step, orig - step}) {
            if (trial < plo[pi] || trial > phi[pi])
              continue; // outside the hard bounds
            v                = trial;
            const double err = mse(data, 0, ntrain, K, threads);
            const double o   = err + reg();
            if (o + 1e-9 < obj) {
              obj   = o;
              train = err;
              ++changed;
              std::printf("  %-22s %4d -> %4d   train %.6f  obj %.6f\n", params[pi].name.c_str(), orig, trial, err, o);
              break;
            }
            v = orig; // not better: restore and (maybe) try the other direction
          }
        }
        val = mse(data, ntrain, data.size(), K, threads);
        const auto el =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - tstart).count();
        std::printf("step %d iter %d: %d change(s)   train %.6f  val %.6f   [%llds elapsed]\n", step, iter, changed,
                    train, val, static_cast<long long>(el));
        std::cout << std::flush;
        if (changed == 0)
          break;
      }
    }

    // Final dump, paste-ready, plus a copy on disk.
    std::cout << "\n=== tuned values (paste into src/eval.cpp) ===\n";
    std::ofstream out(book_path + ".tuned");
    for (const auto &p: params) {
      std::printf("%-22s = %d\n", p.name.c_str(), *p.value);
      out << p.name << " = " << *p.value << "\n";
    }
    std::printf("final: train %.6f  val %.6f   (saved to %s.tuned)\n", train, val, book_path.c_str());
    std::cout << std::flush;
  }

} // namespace tune
