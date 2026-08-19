#include <cstring>
#include "nnue.h"
#include "position.h"
#include "syzygy.h"
#include "tables.h"
#include "tt.h"
#include "uci.h"

int main(int argc, char **argv) {
  initialise_all_databases();
  zobrist::initialise_zobrist_keys();
  nnue::load_embedded();
  tt::resize(tt::DEFAULT_HASH_MB);

  uci::loop(argc > 1 && std::strcmp(argv[1], "--debug") == 0);
  syzygy::shutdown();

  return 0;
}
