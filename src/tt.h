#pragma once

#include <cstddef>
#include <cstdint>
#include "types.h"

// Transposition table: fixed-size clusters of 3 entries, 32 bytes per cluster, with an
// 8-generation age in the low bits of genbound. The caller passes a fully-mixed 64-bit key —
// note the engine's Zobrist hash does NOT cover castling rights or the en-passant square, so
// search::tt_key() mixes them in before probing (see search.cpp).
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

  // Returns the matching entry (and sets found), or the replacement candidate slot to store into.
  [[gnu::hot, nodiscard]] Entry *probe(uint64_t key, bool &found);
  [[gnu::hot]] void              store(Entry *e, uint64_t key, Move m, int score, int eval, int depth, Bound b,
                                       bool pv);

  // Starts pulling `key`'s cluster toward L1 without blocking. Called right after every make
  // in the search with the CHILD's key, so the probe at the top of the recursive call finds
  // the line already in flight instead of eating the full memory latency there.
  [[gnu::hot]] void prefetch(uint64_t key);

  size_t size_mb();
  int    hashfull(); // per-mille of recently-written entries, sampled

} // namespace tt
