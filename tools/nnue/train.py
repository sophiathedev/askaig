#!/usr/bin/env python3
"""Train the askaig NNUE on a (pre-shuffled!) bulletformat .bf file.

  python3 train.py data.bf out.pt [--epochs 30] [--batch 16384] [--lr 1e-3]
                   [--lam 0.0] [--device mps]

Loss: MSE(sigmoid(pred), target), pred in cp/SCALE units,
      target = lam * wdl_result + (1 - lam) * sigmoid(score_cp / SCALE).
Use --lam 0 for teacher-score-only data (e.g. the lichess eval book, whose "results" are
derived from the score); 0.2-0.4 when the data carries real game results.

MPS notes: float32 only (MPS has no float64), single-process loader, decode is vectorized
numpy on CPU. Weights are clamped to +-CLIP after every step (required by quantization).
"""

import argparse
import math
import time

import torch

from data import Batches
from model import CLIP, SCALE, Net  # noqa: F401  (CLIP re-exported for export.py sanity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out", help="checkpoint output (.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.0, help="WDL-result weight in the target blend")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    from data import open_bf
    total = len(open_bf(args.data))
    n_val = max(args.batch, int(total * args.val_frac)) // args.batch * args.batch
    train = Batches(args.data, args.batch, device, start=0, stop=total - n_val)
    val = Batches(args.data, args.batch, device, start=total - n_val, stop=total)
    print(f"{args.data}: {total} records -> train {total - n_val}, val {n_val}, "
          f"{len(train)} batches/epoch, device {device}, lam {args.lam}")

    def target_of(score, result):
        return args.lam * result + (1.0 - args.lam) * torch.sigmoid(score / SCALE)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for stm, opp, score, result, obkt in train:
            pred = model(stm, opp, obkt).squeeze(1)
            loss = torch.mean((torch.sigmoid(pred) - target_of(score, result)) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.clip()  # load-bearing for int16 quantization
            running += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss, vn = 0.0, 0
            for stm, opp, score, result, obkt in val:
                pred = model(stm, opp, obkt).squeeze(1)
                vloss += torch.mean((torch.sigmoid(pred) - target_of(score, result)) ** 2).item()
                vn += 1
        vloss /= max(vn, 1)
        pos_s = nb * args.batch / (time.time() - t0)
        print(f"epoch {epoch:3d}: train {running / max(nb, 1):.6f}  val {vloss:.6f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {pos_s / 1e6:.2f}M pos/s")

        if vloss < best_val:
            best_val = vloss
            torch.save({"model": model.state_dict(), "hl": model.hl, "val_loss": vloss,
                        "epoch": epoch, "lam": args.lam}, args.out)
    print(f"best val {best_val:.6f} -> {args.out}")


if __name__ == "__main__":
    main()
