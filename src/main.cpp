#include "nnue.h"
#include "position.h"
#include "tables.h"
#include "tt.h"
#include "uci.h"

int main() {
  // Initialise the attack/magic databases and Zobrist keys before any move generation.
  initialise_all_databases();
  zobrist::initialise_zobrist_keys();
  nnue::load_embedded(); // the build-time default net; EvalFile can override it later
  tt::resize(tt::DEFAULT_HASH_MB);

  // Start the engine straight into the UCI command loop.
  uci::loop();

  return 0;
}
