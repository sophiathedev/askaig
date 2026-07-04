#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "position.h"
#include "types.h"

// NNUE evaluation, modern bullet-style architecture:
//
//   (768 x 8 king buckets -> 512)x2 -> 1 of 8 material output buckets, SCReLU activation.
//
// Feature set (per perspective): (king bucket, piece, square) with kings INCLUDED (HalfKA
// style) and HORIZONTAL MIRRORING — when the perspective's own king sits on files e-h the
// whole board is flipped (sq ^ 7) for that perspective, so the 8 coarse king buckets only
// need to cover files a-d. Feature index:
//
//   oriented(sq) = persp == WHITE ? sq : sq ^ 56, then ^ 7 when mirrored
//   idx = 768*bucket(own king) + 64*(6*(piece color != persp) + piece type) + oriented(sq)
//
// The accumulator updates incrementally per perspective; a king move crossing a (bucket,
// mirror) boundary rebuilds that half lazily, as a diff against the RefreshTable cache. The
// output layer picks 1 of 8 heads by material count: bucket = (popcount(occ) - 2) / 4.
//
// Quantization (matching tools/nnue/export.py): FT weights/biases int16 at scale QA, output
// weights int16 at scale QB, output biases int32 at QA*QB. SCReLU is computed as v*(v*w) with
// v = clamp(acc, 0, QA): v*w must fit int16, which the trainer guarantees by clipping float
// weights to +-1.98 (255*127 < 32768).
namespace nnue {

  constexpr int KING_BUCKETS = 8;
  constexpr int FEATURES     = KING_BUCKETS * 768; // 6144 per perspective
  constexpr int HL           = 512;
  constexpr int OUT_BUCKETS  = 8; // material output buckets
  constexpr int QA           = 255;
  constexpr int QB           = 64;
  constexpr int SCALE        = 400;

  // Coarse king-bucket map, indexed by the perspective-oriented king square AFTER mirroring
  // (files e-h are never hit; entries mirrored anyway for safety). Back ranks fine-grained,
  // everything past rank 4 merged — where king placement stops mattering much.
  constexpr int KING_BUCKET[64] = {
          0, 1, 2, 3, 3, 2, 1, 0, //
          4, 4, 5, 5, 5, 5, 4, 4, //
          6, 6, 6, 6, 6, 6, 6, 6, //
          6, 6, 6, 6, 6, 6, 6, 6, //
          7, 7, 7, 7, 7, 7, 7, 7, //
          7, 7, 7, 7, 7, 7, 7, 7, //
          7, 7, 7, 7, 7, 7, 7, 7, //
          7, 7, 7, 7, 7, 7, 7, 7,
  };

  // Deepest make/unmake stack the Evaluator supports (the search's ply ceiling).
  constexpr int MAX_PLY = 256;

  // --- Net file (.nnue) ------------------------------------------------------------------------
  // Little-endian, 32-byte header followed by the raw quantized weights:
  //   int16 ft_w[FEATURES][HL] (feature-major), int16 ft_b[HL],
  //   int16 out_w[OUT_BUCKETS][2*HL], int32 out_b[OUT_BUCKETS]
  struct NetHeader {
    char     magic[4]; // "AKNN"
    uint32_t version; // 2 (v1 was the plain-768 net)
    uint32_t features; // 6144
    uint32_t hl; // HL
    uint32_t buckets; // OUT_BUCKETS
    uint16_t qa, qb, scale; // 255, 64, 400
    uint8_t  activation; // 1 = SCReLU
    uint8_t  pad[5]; // zero
  };
  static_assert(sizeof(NetHeader) == 32);

  // Loads a net, replacing the current one. On failure the current net is kept and `err`
  // (when non-null) explains why. load_embedded loads the build-time default net.
  bool load_file(const std::string &path, std::string *err = nullptr);
  bool load_buffer(const unsigned char *data, size_t size, std::string *err = nullptr);
  bool load_embedded(std::string *err = nullptr);
  bool loaded();

  // --- Incremental evaluation ------------------------------------------------------------------

  // The feature changes one move makes: at most 2 additions and 2 removals (castling).
  struct DirtyPiece {
    int8_t n_add, n_sub;
    Piece  add_pc[2], sub_pc[2];
    Square add_sq[2], sub_sq[2];
  };

  // A perspective's king context: which bucket its features live in and whether the board is
  // horizontally mirrored. A king move that changes this invalidates the whole half.
  struct KingCtx {
    int8_t bucket;
    bool   mirror;
    bool   operator==(const KingCtx &o) const { return bucket == o.bucket && mirror == o.mirror; }
  };

  struct Accumulator {
    alignas(64) int16_t v[2][HL]; // [Color][HL]: the two perspective halves
    KingCtx    ctx[2]; // king context each half was built with
    DirtyPiece dp; // how to build this from the parent (when !computed)
    bool       computed[2];
  };

  // Accumulator-refresh cache ("finny tables"): one cached half per (perspective, mirror,
  // king bucket), so a refresh diffs against the cached placement instead of rebuilding from
  // scratch. ~37 KB per Evaluator, never shared.
  struct RefreshEntry {
    alignas(64) int16_t v[HL]; // bias + rows of every piece in bb, in this entry's context
    Bitboard bb[NPIECES]; // the piece placement v was built from (enum-gap slots stay 0)
  };
  struct RefreshTable {
    RefreshEntry e[NCOLORS][2][KING_BUCKETS]; // [perspective][mirror][king bucket]
    bool         inited = false; // false until first use: entries refill from the loaded net
  };

  // One per search thread. Usage around make/unmake:
  //   ev.reset(rootpos);                 // once per new root
  //   ev.push(pos, m); pos.play<C>(m);   // push BEFORE play (reads the captured piece)
  //   ... int cp = ev.evaluate(pos); ... // lazy: cost is paid only if a node is evaluated
  //   pos.undo<C>(m); ev.pop();
  class Evaluator {
  public:
    Evaluator();
    void reset(const Position &pos);
    void push(const Position &before, Move m);
    void push_null(); // pairs with Position::play_null (no feature changes)
    void pop();
    // Centipawns, side-to-move perspective (lazy: the walk-back runs here, not at push).
    [[nodiscard]] int evaluate(const Position &pos);

  private:
    void ensure_half(Color persp, const Position &pos);
    void refresh_half_cached(Color persp, const Position &pos); // refresh via the cache

    std::unique_ptr<Accumulator[]> stack; // MAX_PLY+8 entries — heap, not per-Position
    std::unique_ptr<RefreshTable>  finny; // refresh cache, invalidated by reset()
    int                            top;
  };

  // One-shot eval via a full LOCAL from-scratch refresh (UCI `eval`/`d`, parity tests).
  [[gnu::pure, nodiscard]] int evaluate_refresh(const Position &pos);

} // namespace nnue
