#pragma once

#include <cstddef>
#include <cstdint>
#include "types.h"

// Transposition table: caches search results keyed by the position's Zobrist hash so that
// transpositions (and earlier iterative-deepening iterations) are not re-searched.
namespace tt {

  // Default hash size in megabytes (2 GiB).
  constexpr size_t DEFAULT_HASH_MB = 2048;

  // Bound type of a stored score relative to the search window.
  enum Bound : uint8_t {
    NONE, // empty slot
    EXACT, // exact score (PV node)
    LOWER, // fail-high: true score >= stored score
    UPPER, // fail-low:  true score <= stored score
  };

  struct Entry {
    uint64_t key; // full Zobrist key, for collision verification
    uint16_t move; // best move (raw 16-bit Move encoding)
    int16_t  score; // score (mate scores are stored relative to this node, not the root)
    int16_t  depth; // search depth this entry was produced at
    uint8_t  bound; // Bound
  };

  // (Re)allocates the table to ~`mb` megabytes (entry count rounded down to a power of two; halves
  // on allocation failure). mb == 0 frees the table.
  void resize(size_t mb);

  // Zeroes every slot. Call on "ucinewgame".
  void clear();

  // Returns the slot for `key`. A hit is `slot->key == key && slot->bound != NONE`.
  [[nodiscard]] Entry *probe(uint64_t key) noexcept;

  // Stores an entry (depth-preferred replacement). `score` must already be node-relative.
  void store(uint64_t key, Move move, int score, int depth, Bound bound) noexcept;

  // Current allocated size in megabytes.
  size_t size_mb();

} // namespace tt
