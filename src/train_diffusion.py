"""Toy DDPM diffusion: tiny UNet on CIFAR-10, Concord vs AdamW (eps-prediction).

Mirrors the enwik8 comparability harness: shared init (same --seed builds the
same nn.* model; Concord loads those exact weights via load_weights, AdamW keeps
them) + a dedicated --data_seed generator so both optimizers see IDENTICAL
(image, timestep, noise) draws from step 0. Reports eps-MSE: train (running) and
val (a FIXED eval set, re-seeded each eval -> clean curve).

Modes:
  adamw   : plain nn.Conv2d/Linear UNet + torch.optim.AdamW (lr1e-3 wd0.1 .9/.95)
  concord : conv+linear -> ConcordConv2d/LinearPackedB (load_weights = shared
            init), best packed recipe (rank-1 v-hat: v_scale0 gf_trust1 eps1e-10
            precond.5, optional --coh_gate); aux AdamW for GroupNorm + biases.

Run:
  python src/train_diffusion.py --mode adamw   --max_iters 5000 --tag d_adamw
  python src/train_diffusion.py --mode concord --max_iters 5000 \
         --concord_lr 5e-4 --coh_gate --tag d_conc
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype_packed_b import (ConcordLinearPackedB, ConcordConv2dPackedB,
                                 compute_drift_cancel_C, set_fixed_coh)
from cifar_in_memory import get_loaders_in_memory


# ----------------------- DDPM schedule -----------------------
def make_schedule(T, device, beta0=1e-4, betaT=0.02):
    betas = torch.linspace(beta0, betaT, T, device=device)
    acp = torch.cumprod(1.0 - betas, dim=0)
    return {"T": T, "sqrt_acp": acp.sqrt(), "sqrt_1macp": (1.0 - acp).sqrt()}


def q_sample(x0, t, noise, sch):
    sa = sch["sqrt_acp"][t].view(-1, 1, 1, 1)
    sm = sch["sqrt_1macp"][t].view(-1, 1, 1, 1)
    return sa * x0 + sm * noise


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)   # [B, dim]


# ----------------------- tiny UNet -----------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim_t, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(dim_t, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """~2.5M-param constant-channel UNet (32->16->8->16->32) with concat skips."""

    def __init__(self, ch=128, groups=8):
        super().__init__()
        self.ch = ch
        dim_t = ch * 4
        self.time_mlp = nn.Sequential(nn.Linear(ch, dim_t), nn.SiLU(),
                                      nn.Linear(dim_t, dim_t))
        self.in_conv = nn.Conv2d(3, ch, 3, padding=1)
        self.e1 = ResBlock(ch, ch, dim_t, groups)       # 32x32
        self.e2 = ResBlock(ch, ch, dim_t, groups)       # 16x16
        self.mid = ResBlock(ch, ch, dim_t, groups)      # 8x8
        self.d2 = ResBlock(2 * ch, ch, dim_t, groups)   # 16x16 (+ skip e2)
        self.d1 = ResBlock(2 * ch, ch, dim_t, groups)   # 32x32 (+ skip e1)
        self.out_norm = nn.GroupNorm(min(groups, ch), ch)
        self.out_conv = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.ch))
        x = self.in_conv(x)
        s1 = self.e1(x, temb)                                   # 32, ch
        s2 = self.e2(F.avg_pool2d(s1, 2), temb)                 # 16, ch
        x = self.mid(F.avg_pool2d(s2, 2), temb)                 # 8, ch
        x = F.interpolate(x.float(), scale_factor=2, mode='nearest').to(x.dtype)
        x = self.d2(torch.cat([x, s2], dim=1), temb)            # 16, ch
        x = F.interpolate(x.float(), scale_factor=2, mode='nearest').to(x.dtype)
        x = self.d1(torch.cat([x, s1], dim=1), temb)            # 32, ch
        return self.out_conv(F.silu(self.out_norm(x)))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------- Concord wrapping -----------------------
def wrap_with_concord(model, device, lr, alpha, beta1, weight_decay, eps,
                      step_cap, precond_p, v_scale, gf_trust_delta_sq,
                      alpha_v_fast=0.001):
    """Replace every nn.Conv2d / nn.Linear with the packed-B Concord layer,
    loading the from-scratch init via load_weights (shared init across modes)."""
    layers = []
    npacked = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                c = ConcordConv2dPackedB(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride[0], padding=child.padding[0],
                    bias=child.bias is not None, device=device,
                    alpha=alpha, beta1=beta1, lr=lr)
                w2d = child.weight.data.reshape(child.out_channels, -1).float()
            elif isinstance(child, nn.Linear):
                c = ConcordLinearPackedB(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, device=device,
                    alpha=alpha, beta1=beta1, lr=lr)
                w2d = child.weight.data.float()
            else:
                continue
            c.set_optimizer_kind('adamw', weight_decay=weight_decay, eps=eps,
                                 step_cap=step_cap)
            c.precond_p = precond_p
            c.v_scale = v_scale
            c.gf_trust_delta_sq = gf_trust_delta_sq
            c.alpha_v_fast = alpha_v_fast
            c.drift_cancel_C = compute_drift_cancel_C(c.alpha, c.alpha_v_fast)
            c.track_rebalance = True
            with torch.no_grad():
                c.load_weights(w2d)
                if child.bias is not None:
                    c.bias.data.copy_(child.bias.data.to(c.bias.dtype))
            setattr(parent, name, c)
            layers.append(c)
            npacked += w2d.numel()
    return layers, npacked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["concord", "adamw"], default="adamw")
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=1234)
    # Concord (best packed recipe defaults)
    ap.add_argument("--concord_lr", type=float, default=5e-4)
    ap.add_argument("--concord_wd", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-10)
    ap.add_argument("--precond_p", type=float, default=0.5)
    ap.add_argument("--v_scale", type=float, default=0.0)
    ap.add_argument("--gf_trust_delta_sq", type=float, default=1.0)
    ap.add_argument("--step_cap", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--coh_gate", action="store_true")
    ap.add_argument("--rebalance_every", type=int, default=1)
    ap.add_argument("--aux_lr", type=float, default=1e-3)
    # AdamW baseline
    ap.add_argument("--adamw_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    # schedule
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--lr_min_frac", type=float, default=0.0)
    ap.add_argument("--data_dir", type=str, default="./cifar_data")
    ap.add_argument("--tag", type=str, default="diff")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(f"[{args.tag}] CUDA not available, SKIP.", flush=True)
        return
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data: CIFAR-10 in [-1, 1], sampled by a dedicated generator.
    tl, vl = get_loaders_in_memory(256, device, data_dir=args.data_dir)
    train_imgs = tl.x.float().div_(127.5).sub_(1.0)   # [N,3,32,32] in [-1,1]
    val_imgs = vl.x.float().div_(127.5).sub_(1.0)
    sch = make_schedule(args.T, device)

    model = TinyUNet(ch=args.ch).to(device)

    if args.mode == "concord":
        layers, npacked = wrap_with_concord(
            model, device, args.concord_lr, args.alpha, 0.0, args.concord_wd,
            args.eps, args.step_cap, args.precond_p, args.v_scale,
            args.gf_trust_delta_sq)
        if args.coh_gate:
            set_fixed_coh(True)
            for m in layers:
                m.enable_cohpre()
            print(f"[{args.tag}] FIXED coherence gate ENGAGED on {len(layers)} layers",
                  flush=True)
        aux = [p for p in model.parameters() if p.requires_grad]
        aux_opt = torch.optim.AdamW(aux, lr=args.aux_lr, betas=(0.9, 0.95),
                                    weight_decay=0.0)
        peak_lr = args.concord_lr
        disp_params = npacked + sum(p.numel() for p in aux)
        print(f"[{args.tag}] Concord on {len(layers)} conv/linear "
              f"({npacked/1e6:.2f}M packed)  aux AdamW "
              f"{sum(p.numel() for p in aux)/1e3:.0f}K (GroupNorm+bias)  "
              f"concord_lr={args.concord_lr} wd={args.concord_wd} eps={args.eps} "
              f"precond_p={args.precond_p} v_scale={args.v_scale} "
              f"gf_trust={args.gf_trust_delta_sq}", flush=True)
    else:
        layers = []
        aux_opt = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr,
                                    betas=(0.9, 0.95),
                                    weight_decay=args.weight_decay)
        peak_lr = args.adamw_lr
        disp_params = model.num_params()
        print(f"[{args.tag}] AdamW over {model.num_params()/1e6:.2f}M  "
              f"lr={args.adamw_lr} wd={args.weight_decay}", flush=True)

    print(f"[{args.tag}] TinyUNet ch={args.ch} {disp_params/1e6:.2f}M params  "
          f"CIFAR {train_imgs.shape[0]} imgs  bsz={args.bsz}  T={args.T}", flush=True)

    def lr_at(it):
        if it < args.warmup_iters:
            f = (it + 1) / args.warmup_iters
        else:
            p = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
            f = args.lr_min_frac + 0.5 * (1 - args.lr_min_frac) * (1 + math.cos(math.pi * p))
        return peak_lr * f

    train_gen = torch.Generator(device=device); train_gen.manual_seed(args.data_seed)
    eval_gen = torch.Generator(device=device)

    def draw(imgs, gen):
        idx = torch.randint(0, imgs.shape[0], (args.bsz,), device=device, generator=gen)
        x0 = imgs[idx]
        t = torch.randint(0, args.T, (args.bsz,), device=device, generator=gen)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        return q_sample(x0, t, noise, sch), t, noise

    @torch.no_grad()
    def eval_mse():
        model.eval()
        eval_gen.manual_seed(args.data_seed + 1)   # FIXED eval set every call
        tot = 0.0
        for _ in range(args.eval_iters):
            x_t, t, noise = draw(val_imgs, eval_gen)
            tot += F.mse_loss(model(x_t, t).float(), noise).item()
        model.train()
        return tot / args.eval_iters

    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    best_val = 1e9
    run_loss, run_n = 0.0, 0
    for it in range(args.max_iters):
        lr = lr_at(it)
        if args.mode == "concord":
            for m in layers:
                m.lr = lr
            for g in aux_opt.param_groups:
                g['lr'] = args.aux_lr * (lr / peak_lr)
        else:
            for g in aux_opt.param_groups:
                g['lr'] = lr

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            v = eval_mse()
            best_val = min(best_val, v)
            tr = run_loss / max(run_n, 1)
            run_loss, run_n = 0.0, 0
            print(f"[{args.tag}] iter {it:>5}/{args.max_iters}  lr={lr:.5f}  "
                  f"train {tr:.4f}  val {v:.4f}  best_val {best_val:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        x_t, t, noise = draw(train_imgs, train_gen)
        aux_opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x_t, t).float(), noise)
        loss.backward()
        aux_opt.step()
        if args.mode == "concord" and (it + 1) % args.rebalance_every == 0:
            for m in layers:
                m.rebalance()
        run_loss += loss.item(); run_n += 1

    v = eval_mse()
    print(f"\n[{args.tag}] DONE {(time.time()-t0)/60:.1f} min  "
          f"final val {v:.4f}  best val {best_val:.4f}  "
          f"peak_mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
