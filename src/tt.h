#pragma once

#include <cstddef>
#include <cstdint>
#include "history.h" // ASKAIG_TSAN_BUILD detection
#include "types.h"

#ifdef ASKAIG_TSAN_BUILD
  #define ASKAIG_TT_NOSAN [[gnu::noinline, gnu::no_sanitize("thread")]]
#else
  #define ASKAIG_TT_NOSAN
#endif

namespace tt {

  constexpr size_t DEFAULT_HASH_MB = 128;

  enum Bound : uint8_t { NONE = 0, UPPER = 1, LOWER = 2, EXACT = 3 };

  struct Entry {
    uint16_t key16; // upper 16 bits of the mixed key
    uint16_t move; // best/refutation move (raw Move bits; 0 = none)
    int16_t  score; // search score, mate scores ply-adjusted by the caller
    int16_t  eval; // static eval (VALUE_NONE_TT when absent)
    uint8_t  depth; // search depth + DEPTH_OFFSET (so QS fits)
    uint8_t  genbound; // generation (high 5 bits) | pv (bit 2) | bound (low 2 bits)
  };
  static_assert(sizeof(Entry) == 10);

  constexpr int16_t VALUE_NONE_TT = 32002;
  constexpr int     DEPTH_OFFSET  = 8;

  void resize(size_t mb); // rounds down to a power-of-two cluster count
  void clear();
  void new_search(); // bumps the generation (once per "go")

  struct Probe {
    Entry *slot  = nullptr;
    bool   hit   = false;
    Move   move{};
    int    score = VALUE_NONE_TT;
    int    eval  = VALUE_NONE_TT;
    int    depth = -DEPTH_OFFSET;
    Bound  bound = NONE;
    bool   pv    = false;
  };

  [[gnu::hot, nodiscard]] Probe probe(uint64_t key);
  [[gnu::hot]] void             store(Entry *e, uint64_t key, Move m, int score, int eval, int depth, Bound b,
                                      bool pv);

  [[gnu::hot]] void prefetch(uint64_t key);

  size_t size_mb();
  int    hashfull(); // per-mille of recently-written entries, sampled

} // namespace tt
