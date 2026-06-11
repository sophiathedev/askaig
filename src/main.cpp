#include <cstdlib>
#include <string_view>
#include "position.h"
#include "tables.h"
#include "tt.h"
#include "uci.h"

int main(int argc, char *argv[]) {
  // Initialise the attack/magic databases and Zobrist keys before any move generation.
  initialise_all_databases();
  zobrist::initialise_zobrist_keys();

  // `askaig bench [depth]`: run the fixed benchmark and exit (the OpenBench-style CLI convention).
  // bench allocates its own fixed-size TT, so the 2 GiB default is skipped on this path.
  if (argc > 1 && std::string_view(argv[1]) == "bench") {
    uci::bench(argc > 2 ? std::atoi(argv[2]) : 0);
    return 0;
  }

  // Allocate the transposition table at its default size (2 GiB).
  tt::resize(tt::DEFAULT_HASH_MB);

  // Start the engine straight into the UCI command loop.
  uci::loop();

  return 0;
}
