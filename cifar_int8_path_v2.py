"""CIFAR-10 baseline with the int8 delta path, now with two corrections:

  1. drift_cancel_C is auto-computed from compute_drift_cancel_C(alpha,
     alpha_v_fast) instead of hardcoded 0.1. At our config that's ~0.009
     — about 11× smaller than the previous baseline used, which fixes a
     longstanding overweighting of d_sv in the noise estimator.

  2. Garbage-fraction trust region replaces the hard step_cap clamp on
     the AdamW Linear layers: v_proxy += δ²·v̂ (Adafactor rank-1), so
     the implied step is smoothly bounded by ~1/δ at low SNR and decays
     to 0 at high SNR. δ² defaults to 1/step_cap² so the bound magnitude
     matches the old clamp.

slow_scale=1 (canonical formula). For direct comparison to the
89.52% int8 baseline.

Run:
    python cifar_int8_path_v2.py --epochs 80 --batch_size 32
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
from prototype_packed_b import compute_drift_cancel_C


class WiderConvNet(nn.Module):
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
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--v_lr_scale", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--wd_sv", type=float, default=1e-5)
    ap.add_argument("--wd_sf", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha_v_fast", type=float, default=0.001)
    ap.add_argument("--drift_cancel_C", type=float, default=None,
                     help="None (default) = auto via compute_drift_cancel_C.")
    ap.add_argument("--step_cap", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=1e-12,
                     help="AdamW denominator floor. The OLD runs used 1.0, "
                          "which dominated v_proxy (~1e-9) and the gf floor "
                          "(~1e-10) by ~1e9x — making the preconditioner "
                          "numerically inert (pure SGD-in-weight-space). "
                          "1e-12 lets drift-cancel noise² be the primary "
                          "preconditioner and gf_floor be the trust region.")
    ap.add_argument("--gf_trust_delta_sq", type=float, default=None,
                     help="None (default) = 1/step_cap² so the gf-trust "
                          "soft bound matches the old hard clamp. Set 0 "
                          "to disable gf-trust (falls back to hard clamp).")
    ap.add_argument("--adafactor_beta2", type=float, default=0.999)
    ap.add_argument("--lr_min_frac", type=float, default=0.01)
    ap.add_argument("--bn_lr", type=float, default=0.01)
    ap.add_argument("--slow_scale", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--data_dir", type=str,
                     default=os.environ.get(
                         "CIFAR_DATA_DIR", "./cifar_data"))
    ap.add_argument("--tag", type=str, default="int8-v2")
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

    # Resolve auto-defaults so we can print them for the run header.
    if args.drift_cancel_C is None:
        drift_C = compute_drift_cancel_C(args.alpha, args.alpha_v_fast)
    else:
        drift_C = args.drift_cancel_C
    if args.gf_trust_delta_sq is None:
        gf_delta_sq = 1.0 / (args.step_cap ** 2)
    else:
        gf_delta_sq = args.gf_trust_delta_sq

    concord_layers = [m for m in model.modules()
                      if isinstance(m, (ConcordLinearPackedB,
                                         ConcordConv2dPackedB))]
    for m in concord_layers:
        m.enable_v_slow_i8()
        if not isinstance(m, ConcordConv2dPackedB):
            m.set_optimizer_kind('adamw',
                                  weight_decay=args.weight_decay,
                                  eps=args.eps)
            m.optimizer_v_kind = 'three_accum'
            m.step_cap = args.step_cap
            # gf-trust only meaningful on the AdamW path (the SGD conv
            # kernel doesn't have a step_cap clamp to begin with).
            if gf_delta_sq > 0:
                m.enable_gf_trust(delta_sq=gf_delta_sq,
                                    beta2=args.adafactor_beta2)
        m.alpha = args.alpha
        m.alpha_v_fast = args.alpha_v_fast
        m.drift_cancel_C = drift_C
        m.wd_sv = args.wd_sv
        m.wd_sf = args.wd_sf
        m.slow_scale = int(args.slow_scale)

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
    n_concord_params = sum(m.s_slow.numel() for m in concord_layers)
    print(f"[{args.tag}] WiderConvNet ({n_concord_params/1e6:.2f}M concord params)  "
          f"{n_conv} Conv2d + {n_lin} Linear  bsz={args.batch_size}",
          flush=True)
    print(f"[{args.tag}] AdamW(three_accum) on {n_lin} Linear(s); SGD on "
          f"{n_conv} Conv2d  "
          f"lr={args.lr}  v_lr_scale={args.v_lr_scale}  "
          f"wd={args.weight_decay}  alpha={args.alpha}  "
          f"drift_C={drift_C:.5f}  gf_d2={gf_delta_sq:.4f}  "
          f"eps={args.eps:.1e}  slow_scale={args.slow_scale}",
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
            for m in concord_layers:
                if isinstance(m, ConcordConv2dPackedB):
                    m.lr = cur_lr
                else:
                    m.lr = cur_lr * args.v_lr_scale
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

    print(f"[{args.tag}] Final state stats:")
    for name, m in model.named_modules():
        if not isinstance(m, (ConcordLinearPackedB, ConcordConv2dPackedB)):
            continue
        s_slow = m.s_slow.float()
        s_fast = m.s_fast.float()
        v_slow_full = (m.v_slow_i8.float() * m.v_slow_factor
                       if getattr(m, 'v_slow_i8', None) is not None
                       else torch.zeros_like(s_slow))
        gf_info = ""
        if getattr(m, 'v_row', None) is not None:
            gf_info = (f"  v_row.mean={m.v_row.mean().item():.2e}  "
                       f"sum_v_inv={m._sum_v_inv.item():.2e}")
        print(f"[{args.tag}]   {name:>10}: "
              f"|s_slow|={s_slow.abs().mean().item():7.1f}  "
              f"|s_fast|={s_fast.abs().mean().item():5.1f}  "
              f"|v_slow_full|={v_slow_full.abs().mean().item():7.1f}"
              f"{gf_info}")


if __name__ == "__main__":
    main()
