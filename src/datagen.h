#pragma once

#include <cstdint>
#include <string>

// In-engine self-play data generation (the hidden "datagen" stdin command): node-limited
// games from random openings, quiet positions appended to `out` as bulletformat ChessBoard
// records (the exact 32-byte layout tools/nnue/convert_fen.py documents and decodes).
// Single-threaded by design — run one process per core (tools/datagen-bf.sh).
namespace datagen {

  void run(uint64_t count, const std::string &out, uint64_t nodes, uint64_t seed);

} // namespace datagen
