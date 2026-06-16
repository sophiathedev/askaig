#include <cstdlib>
#include <random>
#include <string_view>
#include <thread>
#include "kpk.h"
#include "nnue/nnue.h"
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
  nnue::init(); // load the (placeholder) NNUE network

  // `askaig nnuetest`: run the NNUE self-test (mirror symmetry; SIMD==portable later) and exit.
  if (argc > 1 && std::string_view(argv[1]) == "nnuetest")
    return nnue::self_test() == 0 ? 0 : 1;

  // `askaig nnueverify`: assert the incremental accumulator == from-scratch over a make/unmake walk.
  if (argc > 1 && std::string_view(argv[1]) == "nnueverify")
    return nnue::verify_incremental() == 0 ? 0 : 1;

  // `askaig datagen <out> [games] [depth] [seed]`: self-play NNUE training data. Needs a TT for the
  // search. With no explicit seed a RANDOM one is drawn, so repeated runs produce DISTINCT data (the
  // chosen seed is logged, so a good run is reproducible by passing it back); an explicit seed is
  // honoured for reproducibility / parallel-shard control.
  if (argc > 1 && std::string_view(argv[1]) == "datagen") {
    tt::resize(64);
    uint64_t seed;
    if (argc > 5) {
      seed = std::strtoull(argv[5], nullptr, 10);
    } else {
      std::random_device rd;
      seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    }
    nnue::datagen(argc > 2 ? argv[2] : "data.txt", argc > 3 ? std::atoi(argv[3]) : 100,
                  argc > 4 ? std::atoi(argv[4]) : 8, seed);
    return 0;
  }

  // `askaig bench [depth]`: run the fixed benchmark and exit (the OpenBench-style CLI convention).
  // bench allocates its own fixed-size TT, so the 2 GiB default is skipped on this path.
  if (argc > 1 && std::string_view(argv[1]) == "bench") {
    uci::bench(argc > 2 ? std::atoi(argv[2]) : 0);
    return 0;
  }

  // `askaig tune <book> [threads]`: Texel-tune the eval constants against a labeled position
  // book (tools/extract.py output) and exit. No TT needed — only the static eval runs.
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
