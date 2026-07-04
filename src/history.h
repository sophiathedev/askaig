#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "types.h"

// Move-ordering / pruning statistics ("history"), shared by movepick.h and search.cpp.
// All tables use the standard gravity update: h += bonus - h*|bonus|/HIST_MAX, which decays
// toward zero and saturates at +-HIST_MAX without explicit clamping.

// A few counters/tables in the search (this file's Histories, and ThreadData::nodes in
// smp.h) are shared across threads with NO synchronization at all, on purpose: they are
// statistics — a torn/stale/lost update costs at most slightly worse move ordering or a
// slightly-off node count, never a wrong result — and the cost of an atomic or a mutex on the
// hottest code path in the engine (every cutoff, every node) isn't worth paying for that.
// Formally still a data race (the C++ memory model has no "benign" exception), so
// ThreadSanitizer correctly flags it. ASKAIG_TSAN_IGNORE/ASKAIG_TSAN_BUILD mark exactly these
// intentionally-accepted races so CI's TSan job stays a meaningful gate against any OTHER,
// unintended one — in the YBWC era that gate caught a real merge-path race that then got a
// proper acquire/release fix instead of a suppression.
#if defined(__has_feature)
  #if __has_feature(thread_sanitizer)
    #define ASKAIG_TSAN_BUILD 1
  #endif
#endif
#if !defined(ASKAIG_TSAN_BUILD) && defined(__SANITIZE_THREAD__)
  #define ASKAIG_TSAN_BUILD 1
#endif

#ifdef ASKAIG_TSAN_BUILD
  // For functions that are already real, out-of-line, never-inlined functions: just suppress.
  #define ASKAIG_TSAN_IGNORE [[gnu::no_sanitize("thread")]]
  // `no_sanitize` only suppresses instrumentation of the memory accesses INSIDE the function
  // it's attached to — it has nothing to attach to once always_inline has folded hist_update's
  // body into every caller before the sanitizer pass runs (confirmed by testing: without this,
  // TSan blames update_quiet_hists/negamax instead of a hist_update frame). So under TSan
  // specifically, force it to stay a real, separate, un-inlined function.
  #define ASKAIG_HIST_UPDATE_ATTRS [[gnu::hot, gnu::noinline, gnu::no_sanitize("thread")]]
#else
  #define ASKAIG_TSAN_IGNORE
  #define ASKAIG_HIST_UPDATE_ATTRS [[gnu::always_inline, gnu::hot]]
#endif

namespace search {

  constexpr int HIST_MAX = 16384;

  // Writes through `h` (a real side effect), so it can't be `pure`/`const` — but it's tiny and
  // called several times per history update at every cutoff node, so force-inline it (except
  // under TSan — see ASKAIG_HIST_UPDATE_ATTRS above).
  ASKAIG_HIST_UPDATE_ATTRS
  inline void hist_update(int16_t &h, int bonus) {
    h = static_cast<int16_t>(h + bonus - int(h) * std::abs(bonus) / HIST_MAX);
  }

  // One continuation-history slice: indexed by the CURRENT move's [piece][to], selected by the
  // PREVIOUS move's (piece, to) — countermove history at 1 ply, follow-up history at 2 plies.
  using ContTable = int16_t[NPIECES][NSQUARES];

  struct Histories {
    // Butterfly history: [stm][from][to] of quiet moves.
    int16_t butterfly[NCOLORS][NSQUARES][NSQUARES];
    // Capture history: [moving piece][to][captured piece type].
    int16_t capture[NPIECES][NSQUARES][NPIECE_TYPES];
    // Continuation history: [prev piece][prev to] -> ContTable over the current move. (~3.7 MB)
    ContTable cont[NPIECES][NSQUARES];

    // Static-eval correction history: several independent (stm, structural-key) tables, each
    // learning a per-structure offset between the static eval and what the search actually
    // found there. NNUE's raw output is still just one number per node — a small/young net has
    // systematic biases correlated with pawn structure, material balance, and piece placement
    // that a cheap, node-free learned correction removes without touching the network itself.
    // Applied (summed, then rescaled) and updated together in search.cpp.
    static constexpr size_t CORR_SIZE = 16384;
    int16_t                 corr_pawn[NCOLORS][CORR_SIZE]; // pawn skeleton (placement)
    int16_t                 corr_material[NCOLORS][CORR_SIZE]; // non-pawn piece counts (imbalance)
    int16_t                 corr_minor[NCOLORS][CORR_SIZE]; // knight+bishop placement, both sides
    int16_t                 corr_major[NCOLORS][CORR_SIZE]; // rook+queen placement, both sides
    int16_t                 corr_cont[NPIECES][NSQUARES]; // keyed by the previous move (piece, to)

    void clear() { std::memset(this, 0, sizeof(*this)); }
  };

} // namespace search
