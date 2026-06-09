"""CIFAR-10 training with packed-B layers.

Same architecture as cifar_concord_adamw.py's WiderConvNet (~3.2M params,
4 conv + 3 fc + BN), but using ConcordConv2dPackedB / ConcordLinearPackedB
(single int32 word per param). For direct comparison to the existing int8
delta path, runs 80 epochs by default at bsz=128 with the AdamW three_accum
config that gets the int8 path's headline number.

The packed-B layers do their state update inside the autograd Function's
backward, so the training loop doesn't need a torch.optim wrapper — just
forward + backward. BN params are still nn.Parameters (fp32) updated by
torch.optim.SGD because they're tiny.

Run:
    python cifar_concord_packed_b.py --epochs 80 --batch_size 128
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
from concord_linear_fused import (ConcordLinearFusedInt8 as ConcordLinearPackedB,
                                    ConcordConv2dFusedInt8 as ConcordConv2dPackedB)


class WiderConvNet(nn.Module):
    """~3.2M-param CIFAR conv net with BatchNorm, using packed-B layers
    for conv + fc weights. BN params stay as standard nn.Parameter
    (fp32, small)."""

    def __init__(self, device='cuda'):
        super().__init__()
        self.conv1 = ConcordConv2dPackedB(3, 64, 3, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.conv2 = ConcordConv2dPackedB(64, 128, 3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(128).to(device)
        self.conv3 = ConcordConv2dPackedB(128, 256, 3, padding=1, device=device)
        self.bn3 = nn.BatchNorm2d(256).to(device)
        self.conv4 = ConcordConv2dPackedB(256, 256, 3, padding=1, device=device)
        self.bn4 = nn.BatchNorm2d(256).to(device)
        self.fc1 = ConcordLinearPackedB(256 * 4 * 4, 512, device=device)
        self.bn_fc1 = nn.BatchNorm1d(512).to(device)
        self.fc2 = ConcordLinearPackedB(512, 256, device=device)
        self.bn_fc2 = nn.BatchNorm1d(256).to(device)
        self.fc3 = ConcordLinearPackedB(256, 10, device=device)

    def forward(self, x):
        x = x.to(torch.bfloat16)
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.1,
                     help="Chase-scale lr. Conv lr = this; Linear lr = "
                          "lr * v_lr_scale.")
    ap.add_argument("--v_lr_scale", type=float, default=0.2,
                     help="AdamW-Linear lr scaling. Effective Linear lr = "
                          "lr * v_lr_scale.")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--wd_sv", type=float, default=1e-5)
    ap.add_argument("--wd_sf", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha_v_fast", type=float, default=0.001)
    ap.add_argument("--drift_cancel_C", type=float, default=None,
                     help="None = auto (compute_drift_cancel_C from rates).")
    ap.add_argument("--step_cap", type=float, default=10.0)
    ap.add_argument("--lr_min_frac", type=float, default=0.01)
    ap.add_argument("--bn_lr", type=float, default=0.01,
                     help="lr for BN params (handled by torch.optim.SGD).")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--data_dir", type=str,
                     default=os.environ.get(
                         "CIFAR_DATA_DIR", "./cifar_data"))
    ap.add_argument("--tag", type=str, default="int8-path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(f"[{args.tag}] CUDA not available, SKIP.", flush=True)
        return
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    tl, vl = get_loaders_in_memory(args.batch_size, device,
                                     data_dir=args.data_dir)
    total_steps = args.epochs * len(tl)

    model = WiderConvNet(device=device)

    # Configure all int8-path layers for AdamW three_accum on Linears
    # only (matching the existing int8 path's CIFAR recipe).
    concord_layers = [m for m in model.modules()
                      if isinstance(m, (ConcordLinearPackedB,
                                         ConcordConv2dPackedB))]
    for m in concord_layers:
        m.enable_v_slow_i8()  # 3rd accumulator (int8 × 128)
        # AdamW only on Linears (matches int8-path headline recipe)
        if not isinstance(m, ConcordConv2dPackedB):
            m.set_optimizer_kind('adamw',
                                  weight_decay=args.weight_decay,
                                  eps=1.0)
            m.optimizer_v_kind = 'three_accum'
            m.step_cap = args.step_cap
        m.alpha = args.alpha
        m.alpha_v_fast = args.alpha_v_fast
        if args.drift_cancel_C is None:
            from prototype_packed_b import compute_drift_cancel_C
            m.drift_cancel_C = compute_drift_cancel_C(m.alpha,
                                                       m.alpha_v_fast)
        else:
            m.drift_cancel_C = args.drift_cancel_C
        m.wd_sv = args.wd_sv
        m.wd_sf = args.wd_sf

    # BN params + biases go through torch.optim.SGD (tiny, fp32 native).
    bn_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and 'bn' in n.lower()]
    bias_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and 'bn' not in n.lower()]
    aux_opt = torch.optim.SGD(
        [{'params': bn_params, 'lr': args.bn_lr},
         {'params': bias_params, 'lr': args.lr * args.v_lr_scale}],
        momentum=0.0)

    n_conv = sum(1 for m in concord_layers
                  if isinstance(m, ConcordConv2dPackedB))
    n_lin = sum(1 for m in concord_layers
                 if isinstance(m, ConcordLinearPackedB)
                 and not isinstance(m, ConcordConv2dPackedB))
    # Param count: int8 path uses separate s_slow buffer.
    n_concord_params = sum(m.s_slow.numel() for m in concord_layers)
    print(f"[{args.tag}] WiderConvNet ({n_concord_params/1e6:.2f}M concord params)  "
          f"{n_conv} Conv2d + {n_lin} Linear  bsz={args.batch_size}",
          flush=True)
    print(f"[{args.tag}] AdamW(three_accum, packed) on ALL {n_conv} Conv2d + "
          f"{n_lin} Linear(s)  "
          f"lr={args.lr}  v_lr_scale={args.v_lr_scale}  "
          f"wd={args.weight_decay}  alpha={args.alpha}",
          flush=True)

    best_acc = 0.0
    best_epoch = -1
    final_acc = 0.0
    step = 0
    t_run = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_t0 = time.time()
        running_loss, seen = 0.0, 0
        for x, y in tl:
            cur_lr = (args.lr * args.lr_min_frac
                      + 0.5 * args.lr * (1.0 - args.lr_min_frac)
                      * (1.0 + math.cos(math.pi * step / max(total_steps, 1))))
            # Update concord layer LRs (manually, since they don't use
            # torch.optim).
            for m in concord_layers:
                if isinstance(m, ConcordConv2dPackedB):
                    m.lr = cur_lr
                else:
                    m.lr = cur_lr * args.v_lr_scale
            # Update aux opt LRs (BN + biases).
            for pg in aux_opt.param_groups:
                if pg['params'] is bn_params:
                    pg['lr'] = args.bn_lr * (cur_lr / args.lr)
                else:
                    pg['lr'] = cur_lr * args.v_lr_scale
            aux_opt.zero_grad(set_to_none=True)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).float()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            aux_opt.step()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            step += 1
        val_acc, val_loss = evaluate(model, vl, device)
        final_acc = val_acc
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
        ep_dt = time.time() - ep_t0
        if (epoch + 1) % args.log_every == 0 or epoch == 0 \
                or epoch == args.epochs - 1:
            tr_loss = running_loss / max(seen, 1)
            print(f"[{args.tag}] ep {epoch+1:>3}/{args.epochs}  "
                  f"lr={cur_lr:.4f}  tr_loss={tr_loss:.4f}  "
                  f"val_acc={val_acc*100:.2f}%  val_loss={val_loss:.4f}  "
                  f"best={best_acc*100:.2f}% (ep {best_epoch})  "
                  f"({ep_dt:.1f}s)", flush=True)
    tot = time.time() - t_run
    print()
    print(f"[{args.tag}] DONE  total {tot/60:.1f} min  "
          f"avg {tot/args.epochs:.1f}s/ep")
    print(f"[{args.tag}] BEST  val_acc = {best_acc*100:.2f}% (epoch {best_epoch})")
    print(f"[{args.tag}] FINAL val_acc = {final_acc*100:.2f}% (epoch {args.epochs})")

    # Diagnostic: print int8-path state magnitudes per layer.
    print(f"[{args.tag}] Final state stats:")
    for name, m in model.named_modules():
        if not isinstance(m, (ConcordLinearPackedB, ConcordConv2dPackedB)):
            continue
        s_slow = m.s_slow.float()
        s_fast = m.s_fast.float()
        v_slow_full = (m.v_slow_i8.float() * m.v_slow_factor
                       if getattr(m, 'v_slow_i8', None) is not None
                       else torch.zeros_like(s_slow))
        print(f"[{args.tag}]   {name:>10}: "
              f"|s_slow|={s_slow.abs().mean().item():7.1f}  "
              f"|s_fast|={s_fast.abs().mean().item():5.1f}  "
              f"|v_slow_full|={v_slow_full.abs().mean().item():7.1f}  "
              f"row_exp.max={m.row_exp.max().item()}  "
              f"col_exp.max={m.col_exp.max().item()}")


if __name__ == "__main__":
    main()
