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

  // Sentinel for "no static eval was computed at this node".
  constexpr int EVAL_NONE = INT16_MIN;

  // One 16-byte entry. The table is grouped into BUCKET_SIZE-entry buckets (4 x 16 B = one 64-byte
  // cache line, so a whole bucket arrives with the single memory access the probe already paid):
  // probe matches the full key against every entry of the bucket, store picks the least valuable
  // entry of the bucket to evict (see store()). `genbound` packs the table generation (6 high
  // bits, bumped by new_search() once per "go") with the Bound (2 low bits): replacement can then
  // prefer evicting entries from old searches, instead of the old single-slot always-replace —
  // which let depth-0-adjacent junk evict deep PV analysis on any index collision.
  struct Entry {
    uint64_t key; // full Zobrist key, for collision verification
    uint16_t move; // best move (raw 16-bit Move encoding)
    int16_t  score; // score (mate scores are stored relative to this node, not the root)
    int16_t  eval; // static eval of the node (EVAL_NONE if never computed); reused to skip evaluate()
    uint8_t  depth; // search depth this entry was produced at (0..MAX_DEPTH fits)
    uint8_t  genbound; // generation << 2 | Bound

    [[nodiscard]] Bound bound() const noexcept { return static_cast<Bound>(genbound & 0b11); }
  };
  static_assert(sizeof(Entry) == 16, "Entry must stay 16 bytes (4 per cache line)");

  constexpr int BUCKET_SIZE = 4;

  // (Re)allocates the table to ~`mb` megabytes (entry count rounded down to a power of two; halves
  // on allocation failure). mb == 0 frees the table.
  void resize(size_t mb);

  // Zeroes every slot. Call on "ucinewgame".
  void clear();

  // Bumps the table generation. Call once at the start of every search ("go"); entries stored or
  // touched after this are marked young, and replacement prefers evicting older generations.
  void new_search() noexcept;

  // Returns the matching entry for `key` (full-key match within its bucket, refreshed to the
  // current generation) or nullptr on a miss. NOTE (lockless table): the pointed-to entry can be
  // overwritten by concurrent/descendant stores — copy the fields needed later, don't re-read.
  [[nodiscard]] Entry *probe(uint64_t key) noexcept;

  // Hints the CPU to start pulling `key`'s 64-byte bucket into cache. Call right after make-move (the
  // child key is known) so the cache-line fetch overlaps the recursion setup before the child probes —
  // hiding the random multi-hundred-MB access latency that is the table's dominant cost.
  void prefetch(uint64_t key) noexcept;

  // Stores an entry. Same-key: depth-preferred (a deeper analysis of the position is kept, only
  // refreshed). Otherwise the bucket's least valuable entry — empty first, then lowest
  // depth - 8*age — is evicted. `score` must already be node-relative; `eval` is the node's static
  // eval or EVAL_NONE.
  void store(uint64_t key, Move move, int score, int depth, Bound bound, int eval) noexcept;

  // Current allocated size in megabytes.
  size_t size_mb();

} // namespace tt
