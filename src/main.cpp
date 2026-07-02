#include "position.h"
#include "tables.h"
#include "uci.h"

int main() {
  // Initialise the attack/magic databases and Zobrist keys before any move generation.
  initialise_all_databases();
  zobrist::initialise_zobrist_keys();

  // Start the engine straight into the UCI command loop.
  uci::loop();

  return 0;
}
