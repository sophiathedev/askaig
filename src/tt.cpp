#include "tt.h"
#include <cstring>
#include <new>

namespace tt {

  namespace {

    Entry *g_table = nullptr;
    size_t g_count = 0; // number of slots (a power of two)
    size_t g_mask  = 0; // g_count - 1

    // Returned by probe() when no table is allocated, so callers never dereference null.
    Entry g_empty{};

  } // namespace

  void resize(size_t mb) {
    delete[] g_table;
    g_table = nullptr;
    g_count = 0;
    g_mask  = 0;
    if (mb == 0)
      return;

    size_t want = (mb * 1024 * 1024) / sizeof(Entry);
    size_t n    = 1;
    while ((n << 1) <= want)
      n <<= 1;

    // Allocate, halving the request on failure so an over-large Hash never aborts the engine.
    while (n >= 1024) {
      try {
        g_table = new Entry[n];
        break;
      } catch (const std::bad_alloc &) {
        n >>= 1;
      }
    }
    if (g_table == nullptr)
      return;

    g_count = n;
    g_mask  = n - 1;
    clear();
  }

  void clear() {
    if (g_table != nullptr)
      std::memset(g_table, 0, g_count * sizeof(Entry));
  }

  [[gnu::hot]] Entry *probe(uint64_t key) noexcept {
    if (g_table == nullptr)
      return &g_empty;
    return &g_table[key & g_mask];
  }

  [[gnu::hot]] void store(uint64_t key, Move move, int score, int depth, Bound bound) noexcept {
    if (g_table == nullptr)
      return;
    Entry &e = g_table[key & g_mask];

    // Depth-preferred replacement: keep a deeper analysis of the *same* position; otherwise (empty
    // slot, collision, or shallower entry) overwrite with the fresh result.
    if (e.bound != NONE && e.key == key && e.depth > depth)
      return;

    e.key   = key;
    e.move  = static_cast<uint16_t>(move.to_from());
    e.score = static_cast<int16_t>(score);
    e.depth = static_cast<int16_t>(depth);
    e.bound = bound;
  }

  size_t size_mb() { return g_count * sizeof(Entry) / (1024 * 1024); }

} // namespace tt
