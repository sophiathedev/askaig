#pragma once

#include <cstdint>

// NNUE network shape + quantisation constants (v1).
//
// Feature set: HalfKA + horizontal-mirror, SINGLE king bucket. The feature index already multiplies by
// the bucket (king_bucket() returns 0 for now), so growing to real king buckets later is a net-only
// change (bump NUM_BUCKETS + retrain) — the engine code does not change.
//
// These constants are the ENGINE side of the net-format contract with the trainer (bullet). They MUST
// match the trainer's architecture + `quantise` config exactly, or inference is garbage. See
// net_format.md.
namespace nnue {

  constexpr int NUM_BUCKETS  = 1; // king buckets (v1: one)
  constexpr int PIECE_PLANES = 12; // (us/them) x 6 piece types, perspective-relative
  constexpr int INPUT_DIM    = NUM_BUCKETS * PIECE_PLANES * 64; // 768 input features per perspective
  constexpr int L1           = 1024; // hidden width per perspective (must equal the trained net's)

  // SF/bullet-style quantisation. Activation is SCReLU: a = clamp(acc, 0, QA)^2.
  constexpr int QA    = 255; // feature-transformer weight + activation scale
  constexpr int QB    = 64; // output-weight scale
  constexpr int SCALE = 400; // network output -> centipawn scale

  // The network, laid out exactly as loaded from disk (and as the trainer exports it):
  //  - ft_weights is FEATURE-MAJOR: the L1-vector for input feature `f` is ft_weights[f*L1 .. f*L1+L1),
  //    so an active feature adds one contiguous L1 column to the accumulator.
  //  - out_weights is perspective-concatenated: [us L1][them L1].
  struct Network {
    int16_t ft_weights[size_t(INPUT_DIM) * L1];
    int16_t ft_bias[L1];
    int16_t out_weights[2 * L1];
    int32_t out_bias;
  };

} // namespace nnue
