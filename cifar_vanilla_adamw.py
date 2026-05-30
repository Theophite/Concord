"""Vanilla torch.optim.AdamW baseline on the SAME WiderConvNet as
cifar_concord_packed_b.py — fp32, plain nn layers, real AdamW (96 bits/param
of optimizer state). Fair reference for the concord packed-B numbers
(concord = 32 bits/param packed state, bf16 compute).

Matched to the concord 150-ep run: same architecture/forward, same
cifar_in_memory loader, same cosine schedule (lr_min_frac), bsz=16.
Only the optimizer + precision differ (that's the point).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_in_memory import get_loaders_in_memory


class WiderConvNet(nn.Module):
    """Identical architecture/forward to the packed-B WiderConvNet."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.float()
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--lr_min_frac", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--data_dir", type=str,
                    default=os.environ.get("CIFAR_DATA_DIR", "./cifar_data"))
    ap.add_argument("--tag", type=str, default="adamw")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    tl, vl = get_loaders_in_memory(args.batch_size, device, data_dir=args.data_dir)
    total_steps = args.epochs * len(tl)

    model = WiderConvNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    n = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] vanilla AdamW fp32  {n/1e6:.2f}M params  "
          f"bsz={args.batch_size}  lr={args.lr}  wd={args.weight_decay}",
          flush=True)

    best = 0.0; best_ep = -1; final = 0.0; step = 0; t0 = time.time()
    for epoch in range(args.epochs):
        model.train(); ep_t = time.time(); run_loss = seen = 0
        for x, y in tl:
            cur = (args.lr * args.lr_min_frac + 0.5 * args.lr
                   * (1 - args.lr_min_frac)
                   * (1 + math.cos(math.pi * step / max(total_steps, 1))))
            for g in opt.param_groups:
                g['lr'] = cur
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).float(), y)
            loss.backward(); opt.step()
            run_loss += loss.item() * x.size(0); seen += x.size(0); step += 1
        acc, vloss = evaluate(model, vl, device)
        final = acc
        if acc > best:
            best, best_ep = acc, epoch + 1
        if (epoch + 1) % args.log_every == 0 or epoch == 0 \
                or epoch == args.epochs - 1:
            print(f"[{args.tag}] ep {epoch+1:>3}/{args.epochs}  lr={cur:.4f}  "
                  f"tr_loss={run_loss/max(seen,1):.4f}  val_acc={acc*100:.2f}%  "
                  f"val_loss={vloss:.4f}  best={best*100:.2f}% (ep {best_ep})  "
                  f"({time.time()-ep_t:.1f}s)", flush=True)
    print(f"\n[{args.tag}] DONE {(time.time()-t0)/60:.1f} min")
    print(f"[{args.tag}] BEST val_acc = {best*100:.2f}% (epoch {best_ep})")
    print(f"[{args.tag}] FINAL val_acc = {final*100:.2f}% (epoch {args.epochs})")


if __name__ == "__main__":
    main()
