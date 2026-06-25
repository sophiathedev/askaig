#include <cstdlib>
#include <random>
#include <string_view>
#include <thread>
#include "kpk.h"
#include "position.h"
#include "tables.h"
#include "tt.h"
#include "tune.h"
#include "uci.h"

int main(int argc, char *argv[]) {
  // Initialise the attack/magic databases and Zobrist keys before any move generation.
  initialise_all_databases();
  zobrist::initialise_zobrist_keys();
  kpk::init(); // KPK win/draw bitbase (needs the attack tables above)

  // `askaig bench [depth]`: run the fixed benchmark and exit (the OpenBench-style CLI convention).
  // bench allocates its own fixed-size TT, so the 2 GiB default is skipped on this path.
  if (argc > 1 && std::string_view(argv[1]) == "bench") {
    uci::bench(argc > 2 ? std::atoi(argv[2]) : 0);
    return 0;
  }

  // `askaig tune <book> [threads] [lambda]`: gradient-tune the eval constants against a labeled
  // position book (tools/extract.py output) and exit. No TT needed — only the static eval runs.
  // lambda is the AdamW weight-decay toward the defaults (regularizes the collinear imbalance /
  // king-safety blocks; default 1e-3, 0 = off). Adam knobs via env: EPOCHS, RELIN, PATIENCE, LR.
  if (argc > 2 && std::string_view(argv[1]) == "tune") {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    tune::run(argv[2], argc > 3 ? std::atoi(argv[3]) : (hw > 2 ? hw - 1 : 1), argc > 4 ? std::atof(argv[4]) : 1e-3);
    return 0;
  }

  // `--debug` anywhere on the command line starts in debug mode (exposes the search-tuning options).
  // A tuning harness that can't send the `debug on` command but can pass args (e.g. fastchess
  // `args="--debug"`) uses this so the options are settable from the first `uci`.
  for (int i = 1; i < argc; ++i)
    if (std::string_view(argv[i]) == "--debug")
      uci::set_debug(true);

  // Allocate the transposition table at its default size (2 GiB).
  tt::resize(tt::DEFAULT_HASH_MB);

  // Start the engine straight into the UCI command loop.
  uci::loop();

  return 0;
}
