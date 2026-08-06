#pragma once

#include <cstdint>
#include <string>

namespace datagen {

  void run(uint64_t count, const std::string &out, uint64_t nodes, uint64_t seed);

} // namespace datagen
