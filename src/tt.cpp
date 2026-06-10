#include "tt.h"
#include <cstring>
#include <new>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

namespace tt {

  namespace {

    Entry *g_table = nullptr;
    size_t g_count = 0; // number of slots (a power of two)
    size_t g_mask  = 0; // g_count - 1

    // Returned by probe() when no table is allocated, so callers never dereference null.
    Entry g_empty{};

    // The table is allocated with mmap/munmap rather than new[]/delete[]: macOS's malloc keeps a
    // freed multi-GiB block in its per-process large-entry cache for reuse, so it never returns to
    // the OS — after `setoption Hash 4096` the freed 2 GiB default table stayed in the process
    // footprint and Activity Monitor showed ~6 GiB for a 4 GiB hash (measured; shrinking back to 64
    // still showed ~2.1 GiB). munmap unconditionally removes the region from the footprint. Bonus:
    // anonymous mmap pages are zero-filled and faulted lazily, so a fresh table costs no physical
    // memory until slots are actually written (a zero slot reads as bound == NONE). Windows keeps
    // the plain allocator (no malloc large-cache issue to work around there).
    Entry *table_alloc(size_t bytes) noexcept {
#if defined(_WIN32)
      return static_cast<Entry *>(operator new(bytes, std::nothrow));
#else
      void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
      return p == MAP_FAILED ? nullptr : static_cast<Entry *>(p);
#endif
    }

    void table_free(Entry *p, size_t bytes) noexcept {
      if (p == nullptr)
        return;
#if defined(_WIN32)
      (void) bytes;
      operator delete(p);
#else
      munmap(p, bytes);
#endif
    }

  } // namespace

  void resize(size_t mb) {
    table_free(g_table, g_count * sizeof(Entry));
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
      g_table = table_alloc(n * sizeof(Entry));
      if (g_table != nullptr)
        break;
      n >>= 1;
    }
    if (g_table == nullptr)
      return;

    g_count = n;
    g_mask  = n - 1;
#if defined(_WIN32)
    clear(); // mmap pages arrive zeroed; operator new's do not
#endif
  }

  void clear() {
    if (g_table == nullptr)
      return;
#if defined(_WIN32)
    std::memset(g_table, 0, g_count * sizeof(Entry));
#else
    // Atomically replace the mapping with fresh zero pages (MAP_FIXED over the same range): O(1)
    // instead of memset-ing GiBs, and it returns the table's physical pages to the OS immediately.
    void *p = mmap(g_table, g_count * sizeof(Entry), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    if (p == MAP_FAILED) // cannot happen for an in-place replace, but never leave a dangling pointer
      std::memset(g_table, 0, g_count * sizeof(Entry));
#endif
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
