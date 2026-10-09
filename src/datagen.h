#pragma once

#include <cstdint>
#include <string>

namespace datagen {

  enum class SearchMode { Nodes, Depth };

  void run(uint64_t count, const std::string &out, SearchMode mode, uint64_t limit, uint64_t seed);

  inline void run(uint64_t count, const std::string &out, uint64_t nodes, uint64_t seed) {
    run(count, out, SearchMode::Nodes, nodes, seed);
  }

} // namespace datagen
