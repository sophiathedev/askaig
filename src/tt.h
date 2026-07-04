#pragma once

#include <cstddef>
#include <cstdint>
#include "history.h" // ASKAIG_TSAN_BUILD detection
#include "types.h"

// Transposition table: 3-entry, 32-byte clusters with an 8-generation age in genbound. The
// caller passes a fully-mixed 64-bit key — the engine Zobrist hash does NOT cover castling
// rights or en passant, so search::tt_key() (smp.h) mixes them in first.
//
// The table is lockless by design: under lazy SMP every thread probes and stores the same
// memory unsynchronized. A torn/stale entry is bounded — key16 filters most of it, a garbage
// move gets matched against the node's own legal moves, scores/evals are advisory. Under TSan
// the accessors compile uninstrumented AND noinline (no_sanitize only covers accesses still
// inside the function after inlining, and LTO would fold them into the search), and probe()
// returns a decoded COPY so no Entry is ever read outside tt.cpp.
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

  // Decoded snapshot of the probed entry (or just the replacement slot on a miss). `slot` is
  // only ever passed back to store().
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

  // Starts pulling `key`'s cluster toward L1. Called with the CHILD's key right after every
  // make, so the probe at the top of the recursive call finds the line already in flight.
  [[gnu::hot]] void prefetch(uint64_t key);

  size_t size_mb();
  int    hashfull(); // per-mille of recently-written entries, sampled

} // namespace tt
