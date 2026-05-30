"""Concord AdamW reference: bigger CIFAR-10 + BatchNorm + 3-accumulator
int8 AdamW with Bayesian-anchored weight decay.

This is the "minimal correct config" script — running it with no flags
reproduces the champion result documented in CONCORD_README.md:

    bsz=32 → 90.91 best / 90.66 final  (~16 min on RTX 4090)
    bsz=128 → ~89.2 best / 89.2 final  (~6 min on RTX 4090)

Storage cost: 40 bits/param of optimizer state (vs torch.optim.AdamW's
96 bits/param). At bsz≥128 the wallclock matches vanilla AdamW.
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

import onetrainer_concord_patch
onetrainer_concord_patch.install()

from concord_optimizer import create_concord_optimizer
from concord_linear_fused import ConcordConv2dFused, ConcordLinearFused
from train_cifar import evaluate
from cifar_in_memory import get_loaders_in_memory


class WiderConvNet(nn.Module):
    """~3.2M-param CIFAR conv net with BatchNorm. Same architecture used
    for the CONCORD_README results table."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   64,  3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,  128, 3, padding=1)
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
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)


# The OneTrainer-compatible config objects the wrapping optimizer reads.
# These mirror what config.optimizer.concord_* would look like in a real
# OneTrainer run.
class _OptCfg:
    optimizer = None
    concord_aux_lr = 0.1
    concord_alpha = 0.1
    concord_beta1 = 0.0   # Vestigial; non-zero values overdamp the chase.
    concord_rebalance_every = 8
    concord_refit_every = 0
    concord_refit_target = 16384
    concord_tickdown = "off"
    concord_qtridiag = True
    concord_qt_refresh = 3000
    concord_qtridiag_pairs = "fc1->fc2"
    concord_lr_flat_after = 0
    concord_lr_flat_frac = 1.0
    concord_bma_obs_every = 0
    concord_polyak_bias = False
    concord_polyak_observe_every = 8
    concord_polyak_leak = 0.05
    concord_polyak_commit = 0.1
    concord_polyak_probe_every = 200
    concord_polyak_level = 1
    concord_polyak_warmup = 2
    concord_polyak_temperature = 0.0
    concord_target_modules = ".*"
    concord_aux_optimizer = "sgd"
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.0
    eps = 1e-8


class _TrainCfg:
    def __init__(self, lr):
        self.learning_rate = lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    # Champion config defaults — change only if you know what you're doing.
    ap.add_argument("--lr", type=float, default=0.1,
                     help="Chase-scale lr (the rate the s_fast tick uses). "
                          "Convs run at this rate; Linears run at lr*v_lr_scale.")
    ap.add_argument("--v_lr_scale", type=float, default=0.2,
                     help="AdamW-Linear lr scaling. Effective Linear lr = "
                          "lr * v_lr_scale.")
    ap.add_argument("--weight_decay", type=float, default=0.01,
                     help="Standard decoupled AdamW weight decay.")
    ap.add_argument("--wd_sv", type=float, default=1e-5,
                     help="Bayesian-anchored wd on (s_slow - v_slow_full).")
    ap.add_argument("--wd_sf", type=float, default=1e-5,
                     help="Bayesian-anchored wd on (s_fast - v_slow_full).")
    ap.add_argument("--lr_min_frac", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--data_dir", type=str,
                     default=os.environ.get(
                         "CIFAR_DATA_DIR", "./cifar_data"),
                     help="CIFAR-10 dataset dir. Defaults to "
                          "$CIFAR_DATA_DIR or ./cifar_data.")
    ap.add_argument("--tag", type=str, default="concord-adamw")
    ap.add_argument("--mass_preserve_chase", action="store_true",
                     help="Enable mass-preserving s_slow chase. Makes "
                          "s_fast act as a small-magnitude delta "
                          "(s_fast - s_slow stays O(1/alpha)). Required "
                          "for eventual int8 s_fast storage.")
    args = ap.parse_args()
    # Wire mass-preserve-chase BEFORE building the model.
    import concord_triton_fused
    concord_triton_fused.set_mass_preserve_chase(args.mass_preserve_chase)
    if args.mass_preserve_chase:
        print(f"[{args.tag}] mass-preserve chase ON", flush=True)

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

    cfg = _TrainCfg(lr=args.lr)
    ocfg = _OptCfg()
    ocfg.concord_aux_lr = args.lr
    ocfg.weight_decay = args.weight_decay

    model = WiderConvNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    onetrainer_concord_patch.cache_model(model)

    parameter_dicts = [{"name": "all", "params": list(model.parameters()),
                         "lr": args.lr, "initial_lr": args.lr}]
    opt = create_concord_optimizer(parameter_dicts, cfg, ocfg)
    concord = model._concord_layers

    # Three-accumulator + int8 v_slow on every wrapped Linear and Conv2d.
    # set_optimizer_kind('adamw') plants the canonical three-accumulator
    # defaults (v_scale=1, drift_cancel_C=C* via compute_drift_cancel_C,
    # etc.); we only override wd / wd_sv / wd_sf here from CLI. The
    # optimizer_v_kind line is explicit so the example self-documents the
    # variance source — 'three_accum' is the default; 'v_rank1' is the
    # lower-memory alternative (don't combine it with enable_v_slow_i8).
    n_adamw = 0
    for m in concord:
        if isinstance(m, ConcordLinearFused):
            m.enable_v_slow_i8()
            if not isinstance(m, ConcordConv2dFused):
                m.set_optimizer_kind("adamw",
                                      weight_decay=args.weight_decay,
                                      eps=1.0)
                m.optimizer_v_kind = "three_accum"
                m.wd_sv = float(args.wd_sv)
                m.wd_sf = float(args.wd_sf)
                n_adamw += 1
    n_conv = sum(1 for m in concord if isinstance(m, ConcordConv2dFused))
    n_lin = sum(1 for m in concord
                 if isinstance(m, ConcordLinearFused)
                 and not isinstance(m, ConcordConv2dFused))
    print(f"[{args.tag}] WiderConvNet ({n_params/1e6:.2f}M params)  "
          f"{n_conv} Conv2d + {n_lin} Linear  bsz={args.batch_size}",
          flush=True)
    print(f"[{args.tag}] AdamW(three_accum) on {n_adamw} Linear(s)  "
          f"lr={args.lr}  v_lr_scale={args.v_lr_scale}  "
          f"wd={args.weight_decay}  wd_sv={args.wd_sv}  wd_sf={args.wd_sf}",
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
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            # AdamW Linears need v_lr_scale applied AFTER zero_grad
            # (which propagates pg["lr"] to every concord layer).
            if args.v_lr_scale != 1.0:
                for m in concord:
                    if (isinstance(m, ConcordLinearFused)
                            and not isinstance(m, ConcordConv2dFused)):
                        m.lr = cur_lr * args.v_lr_scale
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
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

    # Diagnostic: print s_slow / s_fast magnitudes + their gap, to
    # validate the mass-preserving-chase claim that s_fast becomes a
    # small-magnitude delta (s_fast - s_slow ~ O(1/alpha) ~ 10).
    print(f"[{args.tag}] s_slow / s_fast / gap stats:")
    for name, m in model.named_modules():
        if not isinstance(m, (ConcordLinearFused, ConcordConv2dFused)):
            continue
        ss = m.s_slow.to(torch.int32)
        sf = m.s_fast.to(torch.int32)
        gap = (sf - ss).abs().float().mean().item()
        print(f"[{args.tag}]   {name}: |s_slow|={ss.abs().float().mean().item():.1f}  "
              f"|s_fast|={sf.abs().float().mean().item():.1f}  "
              f"|gap|={gap:.1f}")


if __name__ == "__main__":
    main()
