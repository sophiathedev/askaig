"""The NNUE model, mirroring src/nnue.cpp exactly (float; quantization happens in export.py).

Modern bullet-style architecture: (768 x 8 king buckets -> HL)x2 -> 1 of 8 material output
buckets, SCReLU. Features are (own-king bucket, piece, square) with kings included and
horizontal mirroring when the perspective's king is on files e-h (see src/nnue.h for the
exact index formula — data.py implements the same one).

The feature transformer is an nn.Embedding used as a sparse Linear: a batch row is the <= 32
active feature indices (padded with FEATURES, whose row is pinned to zero) and the accumulator
is the sum of the selected rows plus a bias. The 8 output heads are a single Linear(2*HL, 8)
with the head picked per sample by the material bucket.

Weight clipping to +-CLIP after every optimizer step is what makes the quantized engine eval
exact: |w_q| <= 127 keeps QA*w inside int16 for the engine's mullo+madd SCReLU (see export.py).
"""

import torch
import torch.nn as nn

KING_BUCKETS = 8
FEATURES = KING_BUCKETS * 768  # 6144
HL = 512
OUT_BUCKETS = 8
SCALE = 400
CLIP = 1.98


class Net(nn.Module):
    def __init__(self, hl=HL):
        super().__init__()
        self.hl = hl
        self.ft = nn.Embedding(FEATURES + 1, hl, padding_idx=FEATURES)
        self.ft_bias = nn.Parameter(torch.zeros(hl))
        self.out = nn.Linear(2 * hl, OUT_BUCKETS)
        with torch.no_grad():
            self.ft.weight.normal_(0.0, 0.05)
            self.ft.weight[FEATURES].zero_()  # the padding row stays zero

    def forward(self, stm_idx, opp_idx, obkt):
        """stm_idx/opp_idx: (B, 32) int64 feature indices padded with FEATURES;
        obkt: (B,) int64 material output bucket. Returns (B, 1) in cp/SCALE units."""
        acc_stm = self.ft(stm_idx).sum(dim=1) + self.ft_bias
        acc_opp = self.ft(opp_idx).sum(dim=1) + self.ft_bias
        h = torch.cat([acc_stm, acc_opp], dim=1).clamp(0, 1).pow(2)  # SCReLU
        return self.out(h).gather(1, obkt.view(-1, 1))

    @torch.no_grad()
    def clip(self):
        """Post-step clamp; load-bearing for int16 quantization (see module docstring)."""
        self.ft.weight.clamp_(-CLIP, CLIP)
        self.ft_bias.clamp_(-CLIP, CLIP)
        self.out.weight.clamp_(-CLIP, CLIP)
        # MPS's embedding backward does NOT respect padding_idx (the pad row received real
        # gradients in testing) — re-pin it to zero every step or padded slots leak a phantom
        # per-piece-count bias into training that export then silently drops.
        self.ft.weight[FEATURES].zero_()
