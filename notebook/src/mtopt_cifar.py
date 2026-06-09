"""Gate-1 real-task check: MultiTimescaleOptimizer (clean Layer-A) on the
WiderConvNet/CIFAR. Confirms it trains + descends (the §10 step-1 gate beyond
the 1-D probes). Eval uses the averaged readout (deploy_ -> w=init+G*s_s)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F

from cifar_vanilla_adamw import WiderConvNet, evaluate
from cifar_in_memory import get_loaders_in_memory
from mtopt import MultiTimescaleOptimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.0)          # injection scale
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--kappa", type=float, default=0.03)
    ap.add_argument("--alpha_v", type=float, default=0.001)
    ap.add_argument("--whiten", action="store_true",
                    help="gated-Adam injection g/sqrt(v) (per-coord scaling)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="./cifar_data")
    ap.add_argument("--tag", default="mt")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    tl, vl = get_loaders_in_memory(args.bsz, device, data_dir=args.data_dir)
    model = WiderConvNet().to(device)
    opt = MultiTimescaleOptimizer(model.parameters(), lr=args.lr,
                                  alpha=args.alpha, kappa=args.kappa,
                                  alpha_v=args.alpha_v, whiten=args.whiten)
    print(f"[{args.tag}] mtopt lr={args.lr} alpha={args.alpha} "
          f"kappa={args.kappa} alpha_v={args.alpha_v} whiten={args.whiten}  "
          f"bsz={args.bsz}", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); run = seen = 0; nan = False
        for x, y in tl:
            x = x.to(device); y = y.to(device)
            for p in model.parameters():
                p.grad = None
            loss = F.cross_entropy(model(x).float(), y)
            loss.backward(); opt.step()
            lv = loss.item()
            if lv != lv:
                nan = True; break
            run += lv * x.size(0); seen += x.size(0)
        if nan:
            print(f"[{args.tag}] ep{ep+1} NaN -> injection too hot; lower --lr",
                  flush=True); return
        opt.deploy_(); acc, vloss = evaluate(model, vl, device); opt.restore_()
        print(f"[{args.tag}] ep{ep+1}/{args.epochs}  tr_loss={run/max(seen,1):.4f}"
              f"  val_acc={acc*100:.2f}%  val_loss={vloss:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
