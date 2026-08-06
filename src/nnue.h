#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "position.h"
#include "types.h"

namespace nnue {

  constexpr int KING_BUCKETS = 8;
  constexpr int FEATURES     = KING_BUCKETS * 768; // 6144 per perspective
  constexpr int HL           = 512;
  constexpr int OUT_BUCKETS  = 8; // material output buckets
  constexpr int QA           = 255;
  constexpr int QB           = 64;
  constexpr int SCALE        = 400;

  constexpr int KING_BUCKET[64] = {
          0, 1, 2, 3, 3, 2, 1, 0,
          4, 4, 5, 5, 5, 5, 4, 4,
          6, 6, 6, 6, 6, 6, 6, 6,
          6, 6, 6, 6, 6, 6, 6, 6,
          7, 7, 7, 7, 7, 7, 7, 7,
          7, 7, 7, 7, 7, 7, 7, 7,
          7, 7, 7, 7, 7, 7, 7, 7,
          7, 7, 7, 7, 7, 7, 7, 7,
  };

  constexpr int MAX_PLY = 256;

  // v2 on-disk header
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

  bool load_file(const std::string &path, std::string *err = nullptr);
  bool load_buffer(const unsigned char *data, size_t size, std::string *err = nullptr);
  bool load_embedded(std::string *err = nullptr);
  bool loaded();


  struct DirtyPiece {
    int8_t n_add, n_sub;
    Piece  add_pc[2], sub_pc[2];
    Square add_sq[2], sub_sq[2];
  };

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

  struct RefreshEntry {
    alignas(64) int16_t v[HL]; // bias + rows of every piece in bb, in this entry's context
    Bitboard bb[NPIECES]; // cached placement
  };
  struct RefreshTable {
    RefreshEntry e[NCOLORS][2][KING_BUCKETS]; // [perspective][mirror][king bucket]
    bool         inited = false; // false until first use: entries refill from the loaded net
  };

  class Evaluator {
  public:
    Evaluator();
    void reset(const Position &pos);
    // push before making the move
    void push(const Position &before, Move m);
    void push_null(); // pairs with Position::play_null (no feature changes)
    void pop();
    [[nodiscard]] int evaluate(const Position &pos);

  private:
    void ensure_half(Color persp, const Position &pos);
    void refresh_half_cached(Color persp, const Position &pos); // refresh via the cache

    std::unique_ptr<Accumulator[]> stack; // MAX_PLY+8 entries — heap, not per-Position
    std::unique_ptr<RefreshTable>  finny; // refresh cache, invalidated by reset()
    int                            top;
  };

  [[gnu::pure, nodiscard]] int evaluate_refresh(const Position &pos);

} // namespace nnue
