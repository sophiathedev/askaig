#pragma once

#include <cstddef>
#include <cstdint>
#include "history.h" // ASKAIG_TSAN_BUILD detection (see the intentional-race rationale there)
#include "types.h"

// Transposition table: fixed-size clusters of 3 entries, 32 bytes per cluster, with an
// 8-generation age in the low bits of genbound. The caller passes a fully-mixed 64-bit key —
// note the engine's Zobrist hash does NOT cover castling rights or the en-passant square, so
// search::tt_key() mixes them in before probing (see smp.h's tt_key).
//
// The table is LOCKLESS BY DESIGN: under lazy SMP every thread probes and stores the same
// memory with no synchronization, the same statistics-not-correctness trade as the Histories
// tables (history.h). A torn/stale entry is bounded: the key16 check filters almost all of it,
// a garbage move is matched against the node's own legal moves before use, and scores/evals
// are advisory inputs the search already treats as fallible. Under TSan the accessors below
// are compiled uninstrumented AND noinline (no_sanitize only covers accesses still inside the
// function after inlining — see ASKAIG_HIST_UPDATE_ATTRS in history.h — and IPO/LTO would
// otherwise fold them into the search); every Entry access must stay behind them, which is
// why probe() returns a decoded COPY rather than an Entry to read through.
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
    uint8_t  depth; // search depth (offset by DEPTH_OFFSET so QS fits)
    uint8_t  genbound; // generation (high 5 bits) | pv (bit 2) | bound (low 2 bits)
  };
  static_assert(sizeof(Entry) == 10);

  constexpr int16_t VALUE_NONE_TT = 32002; // "no value" marker for score/eval fields
  constexpr int     DEPTH_OFFSET  = 8; // stored depth = real depth + offset (QS uses <= 0)

  void resize(size_t mb); // allocates round-down-to-power-of-two clusters; keeps contents on same size
  void clear();
  void new_search(); // bumps the generation (called once per "go")

  // What probe() hands the search: a decoded SNAPSHOT of the matching entry (or just the
  // replacement slot on a miss), taken in one place inside tt.cpp so the intentional races
  // never leak into instrumented code. `slot` is only ever passed back to store().
  struct Probe {
    Entry *slot  = nullptr; // the hit entry, or the replacement victim on a miss
    bool   hit   = false;
    Move   move{}; // 0 when absent
    int    score = VALUE_NONE_TT;
    int    eval  = VALUE_NONE_TT;
    int    depth = -DEPTH_OFFSET; // already decoded (stored byte minus DEPTH_OFFSET)
    Bound  bound = NONE;
    bool   pv    = false;
  };

  [[gnu::hot, nodiscard]] Probe probe(uint64_t key);
  [[gnu::hot]] void             store(Entry *e, uint64_t key, Move m, int score, int eval, int depth, Bound b,
                                      bool pv);

  // Starts pulling `key`'s cluster toward L1 without blocking. Called right after every make
  // in the search with the CHILD's key, so the probe at the top of the recursive call finds
  // the line already in flight instead of eating the full memory latency there.
  [[gnu::hot]] void prefetch(uint64_t key);

  size_t size_mb();
  int    hashfull(); // per-mille of recently-written entries, sampled

} // namespace tt
