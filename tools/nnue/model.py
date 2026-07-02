"""The NNUE model, mirroring src/nnue.cpp exactly (float; quantization happens in export.py).

(768 -> HL)x2 -> 1 perspective net, SCReLU activation. The feature transformer is an
nn.Embedding used as a sparse Linear: a batch row is the ~32 active feature indices (padded
with index 768, whose row is pinned to zero), and the accumulator is the sum of the selected
rows plus a bias — identical math to the engine's accumulator.

Weight clipping to +-CLIP after every optimizer step is what makes the quantized engine eval
exact: |w_q| <= 127 keeps QA*w inside int16 for the engine's mullo+madd SCReLU (see export.py).
"""

import torch
import torch.nn as nn

FEATURES = 768
HL = 256
SCALE = 400
CLIP = 1.98


class Net(nn.Module):
    def __init__(self, hl=HL):
        super().__init__()
        self.hl = hl
        self.ft = nn.Embedding(FEATURES + 1, hl, padding_idx=FEATURES)
        self.ft_bias = nn.Parameter(torch.zeros(hl))
        self.out = nn.Linear(2 * hl, 1)
        with torch.no_grad():
            self.ft.weight.normal_(0.0, 0.05)
            self.ft.weight[FEATURES].zero_()  # the padding row stays zero (gets no gradient)

    def forward(self, stm_idx, opp_idx):
        """stm_idx/opp_idx: (B, 32) int64 feature indices, padded with FEATURES.
        Returns (B, 1) in cp/SCALE units (sigmoid of it is the win probability)."""
        acc_stm = self.ft(stm_idx).sum(dim=1) + self.ft_bias
        acc_opp = self.ft(opp_idx).sum(dim=1) + self.ft_bias
        h = torch.cat([acc_stm, acc_opp], dim=1).clamp(0, 1).pow(2)  # SCReLU
        return self.out(h)

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
