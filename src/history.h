#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "types.h"


#if defined(__has_feature)
  #if __has_feature(thread_sanitizer)
    #define ASKAIG_TSAN_BUILD 1
  #endif
#endif
#if !defined(ASKAIG_TSAN_BUILD) && defined(__SANITIZE_THREAD__)
  #define ASKAIG_TSAN_BUILD 1
#endif

// lockless search stats; suppress only their known races
#ifdef ASKAIG_TSAN_BUILD
  #define ASKAIG_TSAN_IGNORE [[gnu::no_sanitize("thread")]]
  #define ASKAIG_HIST_UPDATE_ATTRS [[gnu::hot, gnu::noinline, gnu::no_sanitize("thread")]]
#else
  #define ASKAIG_TSAN_IGNORE
  #define ASKAIG_HIST_UPDATE_ATTRS [[gnu::always_inline, gnu::hot]]
#endif

namespace search {

  constexpr int HIST_MAX = 16384;

  ASKAIG_HIST_UPDATE_ATTRS
  inline void hist_update(int16_t &h, int bonus) {
    h = static_cast<int16_t>(h + bonus - int(h) * std::abs(bonus) / HIST_MAX);
  }

  ASKAIG_HIST_UPDATE_ATTRS
  inline void counter_store(Move &slot, Move m) { slot = m; }

  ASKAIG_HIST_UPDATE_ATTRS
  inline Move counter_load(const Move &slot) { return slot; }

  using ContTable = int16_t[NPIECES][NSQUARES];

  struct Histories {
    int16_t butterfly[NCOLORS][NSQUARES][NSQUARES]; // quiet moves, [stm][from][to]
    int16_t capture[NPIECES][NSQUARES][NPIECE_TYPES]; // [moving piece][to][captured type]
    ContTable cont[NPIECES][NSQUARES]; // [prev piece][prev to] -> current move (~3.7 MB)
    Move counter[NPIECES][NSQUARES]; // [prev piece][prev to] -> the quiet move that refuted it

    static constexpr size_t CORR_SIZE = 16384;
    int16_t                 corr_pawn[NCOLORS][CORR_SIZE]; // pawn placement
    int16_t                 corr_material[NCOLORS][CORR_SIZE]; // non-pawn piece counts
    int16_t                 corr_minor[NCOLORS][CORR_SIZE]; // knight+bishop placement
    int16_t                 corr_major[NCOLORS][CORR_SIZE]; // rook+queen placement
    int16_t                 corr_cont[NPIECES][NSQUARES]; // keyed by the previous move

    void clear() { std::memset(static_cast<void *>(this), 0, sizeof(*this)); }
  };

} // namespace search
