#include <cstring>
#include "nnue.h"
#include "position.h"
#include "tables.h"
#include "tt.h"
#include "uci.h"

int main(int argc, char **argv) {
  // Initialise the attack/magic databases and Zobrist keys before any move generation.
  initialise_all_databases();
  zobrist::initialise_zobrist_keys();
  nnue::load_embedded(); // the build-time default net; EvalFile can override it later
  tt::resize(tt::DEFAULT_HASH_MB);

  // --debug: expose the hidden search-tuning spin options (tools/spsa.py drives them).
  uci::loop(argc > 1 && std::strcmp(argv[1], "--debug") == 0);

  return 0;
}
