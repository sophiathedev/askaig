#include "tt.h"
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

  struct Cluster {
    tt::Entry e[3];
    uint16_t  pad; // 3*10 + 2 = 32 bytes
  };
  static_assert(sizeof(Cluster) == 32);

  Cluster *g_table    = nullptr;
  size_t   g_clusters = 0; // power of two
  size_t   g_mb       = 0;
  uint8_t  g_gen      = 0; // stepped by 8 per search; low 3 bits of genbound hold pv|bound

  inline uint8_t age_of(uint8_t genbound) { return uint8_t((g_gen - (genbound & 0xF8)) & 0xF8); }

} // namespace

void tt::resize(size_t mb) {
  if (mb == g_mb && g_table)
    return;
  std::free(g_table);
  size_t clusters = mb * 1024 * 1024 / sizeof(Cluster);
  size_t pow2     = 1;
  while (pow2 * 2 <= clusters)
    pow2 *= 2;
  g_clusters = pow2;
  g_mb       = mb;
  g_table    = static_cast<Cluster *>(std::aligned_alloc(64, g_clusters * sizeof(Cluster)));
  clear();
}

void tt::clear() {
  if (g_table)
    std::memset(g_table, 0, g_clusters * sizeof(Cluster));
  g_gen = 0;
}

void tt::new_search() { g_gen += 8; }

tt::Entry *tt::probe(uint64_t key, bool &found) {
  Cluster       &c   = g_table[key & (g_clusters - 1)];
  const uint16_t k16 = uint16_t(key >> 48);

  for (Entry &e: c.e)
    if (e.key16 == k16 && (e.genbound & 0x3)) { // bound != NONE -> a real entry
      e.genbound = uint8_t(g_gen | (e.genbound & 0x7)); // refresh the age on a hit
      found      = true;
      return &e;
    }

  // Replacement: the shallowest entry after an age penalty (old entries die first).
  Entry *victim = &c.e[0];
  for (Entry &e: c.e)
    if (int(e.depth) - age_of(e.genbound) < int(victim->depth) - age_of(victim->genbound))
      victim = &e;
  found = false;
  return victim;
}

void tt::store(Entry *e, uint64_t key, Move m, int score, int eval, int depth, Bound b, bool pv) {
  const uint16_t k16 = uint16_t(key >> 48);
  // Keep the old move when the new search found none; don't overwrite a same-key entry that is
  // deeper unless the new bound is EXACT (standard preserve-depth policy).
  if (m.to_from() != 0 || e->key16 != k16)
    e->move = uint16_t(m.to_from());
  if (b != EXACT && e->key16 == k16 && depth + DEPTH_OFFSET < int(e->depth) - 2)
    return;
  e->key16    = k16;
  e->score    = int16_t(score);
  e->eval     = int16_t(eval);
  e->depth    = uint8_t(depth + DEPTH_OFFSET);
  e->genbound = uint8_t(g_gen | (uint8_t(pv) << 2) | b);
}

size_t tt::size_mb() { return g_mb; }

int tt::hashfull() {
  if (!g_table)
    return 0;
  int n = 0;
  for (size_t i = 0; i < 1000; ++i)
    for (const Entry &e: g_table[i].e)
      n += (e.genbound & 0x3) && (e.genbound & 0xF8) == g_gen;
  return n / 3;
}
