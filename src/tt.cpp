#include "tt.h"
#include <cstring>
#include <new>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

namespace tt {

  namespace {

    Entry  *g_table       = nullptr;
    size_t  g_count       = 0; // number of entries (a power of two, multiple of BUCKET_SIZE)
    size_t  g_bucket_mask = 0; // (g_count / BUCKET_SIZE) - 1
    uint8_t g_gen         = 0; // current generation (6 bits, wraps)

    [[gnu::const]] inline uint8_t pack(uint8_t gen, Bound b) noexcept { return static_cast<uint8_t>(gen << 2 | b); }
    // Age of an entry in generations, robust to the 6-bit wrap.
    [[gnu::pure]] inline int rel_age(const Entry &e) noexcept { return (64 + g_gen - (e.genbound >> 2)) & 63; }

    [[gnu::pure, gnu::hot]] inline Entry *bucket_of(uint64_t key) noexcept {
      return &g_table[(key & g_bucket_mask) * BUCKET_SIZE];
    }

    // The table is allocated with mmap/munmap rather than new[]/delete[]: macOS's malloc keeps a
    // freed multi-GiB block in its per-process large-entry cache for reuse, so it never returns to
    // the OS — after `setoption Hash 4096` the freed 2 GiB default table stayed in the process
    // footprint and Activity Monitor showed ~6 GiB for a 4 GiB hash (measured; shrinking back to 64
    // still showed ~2.1 GiB). munmap unconditionally removes the region from the footprint. Bonus:
    // anonymous mmap pages are zero-filled and faulted lazily, so a fresh table costs no physical
    // memory until slots are actually written (a zero slot reads as bound == NONE). Windows keeps
    // the plain allocator (no malloc large-cache issue to work around there). mmap also returns
    // page-aligned memory, so the 64-byte buckets never straddle a cache line.
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
    g_table       = nullptr;
    g_count       = 0;
    g_bucket_mask = 0;
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

    g_count       = n; // a power of two >= 1024, hence a multiple of BUCKET_SIZE
    g_bucket_mask = n / BUCKET_SIZE - 1;
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

  void new_search() noexcept { g_gen = (g_gen + 1) & 63; }

  [[gnu::hot]] void prefetch(uint64_t key) noexcept {
    if (g_table != nullptr)
      __builtin_prefetch(bucket_of(key), 0, 1); // rw=0 (read), locality=1 (low temporal locality)
  }

  [[gnu::hot]] Entry *probe(uint64_t key) noexcept {
    if (g_table == nullptr)
      return nullptr;
    Entry *b = bucket_of(key);
    for (int i = 0; i < BUCKET_SIZE; ++i)
      if (b[i].bound() != NONE && b[i].key == key) {
        b[i].genbound = pack(g_gen, b[i].bound()); // hit -> mark young so it survives replacement
        return &b[i];
      }
    return nullptr;
  }

  [[gnu::hot]] void store(uint64_t key, Move move, int score, int depth, Bound bound, int eval) noexcept {
    if (g_table == nullptr)
      return;
    Entry *b = bucket_of(key);

    // Same position already stored: depth-preferred — keep the deeper analysis (refreshed so aging
    // doesn't evict it), otherwise overwrite it in place below.
    Entry *e = nullptr;
    for (int i = 0; i < BUCKET_SIZE; ++i)
      if (b[i].bound() != NONE && b[i].key == key) {
        if (b[i].depth > depth) {
          b[i].genbound = pack(g_gen, b[i].bound());
          return;
        }
        e = &b[i];
        break;
      }

    // Otherwise evict the bucket's least valuable entry: an empty slot if any, else the lowest
    // depth - 8*age (a deep entry is worth keeping, but each generation it survives unused costs
    // it 8 plies of "depth" — stale analysis from old searches ages out instead of squatting).
    if (e == nullptr) {
      e         = &b[0];
      int worst = e->bound() == NONE ? INT32_MIN : e->depth - 8 * rel_age(*e);
      for (int i = 1; i < BUCKET_SIZE && worst != INT32_MIN; ++i) {
        const int v = b[i].bound() == NONE ? INT32_MIN : b[i].depth - 8 * rel_age(b[i]);
        if (v < worst) {
          e     = &b[i];
          worst = v;
        }
      }
    }

    e->key      = key;
    e->move     = static_cast<uint16_t>(move.to_from());
    e->score    = static_cast<int16_t>(score);
    e->eval     = static_cast<int16_t>(eval);
    e->depth    = static_cast<uint8_t>(depth);
    e->genbound = pack(g_gen, bound);
  }

  size_t size_mb() { return g_count * sizeof(Entry) / (1024 * 1024); }

} // namespace tt
