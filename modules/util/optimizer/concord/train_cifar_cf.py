"""CIFAR-10 ablation driver for the Concord packed-B optimizer (the cf-discount version).

Builds the core's validated FusedConvNet architecture (foliated_sgd/train_cifar_fused.py)
from the FORK's ConcordConv2dPackedB / ConcordLinearPackedB -- the packed int32 layers that
carry the v_slow anchor + Wiener coherence gate + dissipation + the NEW cf-discount -- so we
can A/B the cf-discount (`set_coh_vhat`) on a fast, quantitative benchmark.

KEY FACTS (verified against prototype_packed_b.py):
  * The optimizer is FUSED INTO backward(): for the packed weights, loss.backward() IS the
    step (the custom autograd Function mutates packed_w in place). No opt.step()/zero_grad()
    for the packed weights.
  * The packed weight is a *buffer*, not a Parameter. The per-layer `bias` (nn.Parameter when
    bias=True, the default) anchors the first layer in the autograd graph so the fused backward
    fires; we keep bias=True everywhere and step biases with a side torch.optim.SGD (Concord
    computes bias.grad but does not apply it), exactly as the core harness does.
  * The coherence on/off flags (set_coh_vhat / set_fixed_coh / set_ratio_coh) bake into Triton
    constexprs at first kernel launch -> ONE ARM PER PROCESS. (set_coh_kappa is a runtime
    scalar, but we still set it pre-construction for cleanliness.) Use run_ablation.py.

EVAL PARITY (important): production DEPLOYS consolidated_weight() (s_slow+v_slow, DROPS s_fast),
not the live weight (which includes s_fast). Since the cf-discount specifically protects the
s_fast residual, cf-on vs cf-off can RANK DIFFERENTLY on live vs deployed weights. So we report
BOTH: live_acc (live forward) and deploy_acc (a plain net loaded from consolidated_weight()).
The deploy_acc is the load-bearing metric.

REGIME: dense balanced CIFAR-10 barely exercises cf (which protects diverse/scattered/long-
tailed residual). On standard CIFAR this is a correctness/stability/regression check. Use
--longtail to subsample the TRAIN set to exponential class imbalance (test stays balanced) and
report per-class / worst-class deploy accuracy -- that is the arm that actually stresses cf.

GPU + Triton required. Nothing here touches the GPU on import; do not launch while another
training job holds the GPU.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --- the FORK optimizer (sibling module; imports standalone, no OneTrainer deps) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import prototype_packed_b as pb  # noqa: E402
# --- the dual-dissipation reference + its nn.Module wrappers (CPU-capable; the smoke runs
#     here with CUDA_VISIBLE_DEVICES="") ---
from dual_dissipation_nn import (  # noqa: E402
    DualDissipationConv2d, DualDissipationLinear)
from dual_dissipation_ref import (  # noqa: E402
    SUBSTRATE_KAIMING, unpack_dual)

# --- the CORE's CIFAR data/eval + the nn.Conv2d reference net (used for AdamW and as the
#     plain "deploy" net we load consolidated weights into) ---
_CORE = r"C:\fisher\foliated_sgd"
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
from train_cifar import get_loaders, BaselineConvNet  # noqa: E402

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ConcordConvNet(nn.Module):
    """FusedConvNet's architecture (3 conv + 2 fc, ReLU + 2x2 maxpool), built from the
    fork's Concord packed layers. 32->16->8->4 spatial, so fc1 sees 64*4*4."""

    def __init__(self, device='cuda', lr=0.05, alpha=0.1, init_gap=0.0):
        super().__init__()
        kw = dict(device=device, alpha=alpha, lr=lr)  # bias=True (default) -> graph anchor
        self.conv1 = pb.ConcordConv2dPackedB(3, 32, 3, padding=1, **kw)
        self.conv2 = pb.ConcordConv2dPackedB(32, 64, 3, padding=1, **kw)
        self.conv3 = pb.ConcordConv2dPackedB(64, 64, 3, padding=1, **kw)
        self.fc1 = pb.ConcordLinearPackedB(64 * 4 * 4, 128, **kw)
        self.fc2 = pb.ConcordLinearPackedB(128, 10, **kw)
        if init_gap > 0.0:
            # FROM-SCRATCH direction: re-pack each layer's random init so s_slow leads v_slow by
            # d_sv = init_gap*coarse, giving the coherence gate an axis from step 0 (gap-zero -> coh=0).
            # deploy = s_slow+v_slow = the same random init, so the net's forward is unchanged.
            with torch.no_grad():
                for m in self.concord_layers():
                    m.load_weights(m.get_weight().float(), gap=init_gap)

    def concord_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]

    def set_lr(self, lr):
        for m in self.concord_layers():
            m.lr = lr

    @torch.no_grad()
    def load_pretrained(self, baseline):
        """Preload each packed layer's DEPLOY weight (s_slow+v_slow) from a trained BaselineConvNet via
        load_weights -- the fine-tune regime where dissipation works: the deploy is correct from step 0,
        evap only trims residual (never has to build across the 128-LSB from zero), and cf protects the
        diverse adjustments. Conv weights reshape [out,in,kh,kw] -> [out, in*kh*kw] for the 2D pack."""
        pairs = [(self.conv1, baseline.conv1), (self.conv2, baseline.conv2),
                 (self.conv3, baseline.conv3), (self.fc1, baseline.fc1), (self.fc2, baseline.fc2)]
        for dst, src in pairs:
            W = src.weight.data
            if W.dim() == 4:
                W = W.reshape(W.shape[0], -1)
            dst.load_weights(W.float())
            if dst.bias is not None and src.bias is not None:
                dst.bias.data.copy_(src.bias.data.to(dst.bias.dtype))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DualDissipationConvNet(nn.Module):
    """Same FusedConvNet architecture (3 conv + 2 fc, ReLU + 2x2 maxpool) built from the
    DUAL-DISSIPATION packed nn.Modules. CPU-capable (no Triton): forward decodes each layer's
    live_weight(); the per-layer bias Parameter anchors autograd; the optimizer step is the
    reference two-gradient step, driven explicitly by the random-microbatch-split training
    loop (NOT fused into backward)."""

    def __init__(self, device="cpu", grad_accum_M=8, bracket_d=0.5, seed=0,
                 substrate_mode=SUBSTRATE_KAIMING, substrate_scale=0.1):
        super().__init__()
        kw = dict(device=device, grad_accum_M=grad_accum_M, bracket_d=bracket_d,
                  substrate_mode=substrate_mode, substrate_scale=substrate_scale)
        # distinct substrate seeds per layer so the from-scratch priors break symmetry
        self.conv1 = DualDissipationConv2d(3, 32, 3, padding=1, seed=seed + 1, **kw)
        self.conv2 = DualDissipationConv2d(32, 64, 3, padding=1, seed=seed + 2, **kw)
        self.conv3 = DualDissipationConv2d(64, 64, 3, padding=1, seed=seed + 3, **kw)
        self.fc1 = DualDissipationLinear(64 * 4 * 4, 128, seed=seed + 4, **kw)
        self.fc2 = DualDissipationLinear(128, 10, seed=seed + 5, **kw)
        for m in self.dd_layers():
            m.load_weights()                  # from-scratch: zero offset, seeded substrate

    def dd_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]

    def zero_weight_grads(self):
        for m in self.dd_layers():
            m.zero_weight_grad()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def configure_dualdis(model, lam, lr, steps_per_epoch):
    """Per-layer dual-dissipation step hyperparameters. Reuses the REFERENCE's own validated
    step() defaults (the ones its CPU suite — test 7 deploy-ratchet, test 4 conservation —
    passes with): the noise-proxy preconditioner uses eps=1.0 + v_scale=1.0 (NOT the prod
    Concord eps=1e-10/v_scale=0 cf recipe, which assumes the vhat trust-region denom the
    pure-torch _dual_tick does not implement — that combo gives denom=1e-5 -> a ~1/1e-5 step
    gain that clamps to step_cap and saturates the fine register, so the deploy never learns).
    gf_consol = lam/lr so the dimensionless dissipation lam = lr*gf_consol tracks the cosine."""
    b2 = 1.0 - 1.0 / max(steps_per_epoch, 1)
    return dict(
        alpha=0.1, drift_cancel_C=0.02, alpha_v_fast=0.001, coh_kappa=1.0,
        v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0, min_leak=0.1,
        evap_build_min=128.0, beta1=0.0, beta2=b2, use_coh_vhat=True,
        mass_preserve=True, chase_floor=0.1, leak_floor=0.05, consf=1.0)


def cosine_lr(step, total, lr_max, lr_min_frac, warmup=0):
    lr_min = lr_max * lr_min_frac
    frac = min(max(step / max(total, 1), 0.0), 1.0)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * frac))
    if warmup > 0:
        lr = lr * min(1.0, (step + 1) / float(warmup))   # linear LR warmup: caps the cold-start overshoot
    return lr


def ratio_floors(step, horizon, chase0=0.9, chase_min=0.1, leak0=0.999, leak_min=0.1):
    """Bootstrap chase/leak floors: cosine-anneal (0.9, 0.999) -> (0.1, 0.1) over ~1 epoch. High floors
    early IGNITE the chase (gate >= floor regardless of coh) so s_fast consolidates into s_slow before
    coherence establishes -- the from-scratch ignition production gets from winner_step. Lets dissipation
    run from scratch without the evap draining s_fast below the consolidation LSB. (Mirrors concord_winner
    WINNER floors + the winner_step schedule.)"""
    p = min(max(step / max(horizon, 1), 0.0), 1.0)
    f = 0.5 * (1.0 + math.cos(math.pi * p))   # 1 -> 0 over the horizon
    return chase_min + (chase0 - chase_min) * f, leak_min + (leak0 - leak_min) * f


def configure_concord(model, lam, lr, ratio_coh, disable_cohpre, steps_per_epoch):
    """Per-layer recipe mirroring concord_winner.swap_unet_to_winner, plus dissipation.
    gf_consol is FIXED at lam/lr so the dimensionless dissipation lam = lr(t)*gf_consol tracks
    the cosine, exactly as production. Also pins the v_hat EMA to a 1-epoch window (production's
    beta2_epoch_window), so the bias-correction driver's 1/(1-b2^t) matches. Returns gf_consol."""
    gf = lam / max(lr, 1e-12)
    b2 = 1.0 - 1.0 / max(steps_per_epoch, 1)
    for m in model.concord_layers():
        m.set_optimizer_kind('adamw', weight_decay=0.0, eps=1e-10, step_cap=10.0)
        m.precond_p = 0.5
        m.v_scale = 0.0
        m.gf_trust_delta_sq = 1.0
        m.gf_consol = gf                      # dissipation; 0.0 => no gf-evaporation
        m.adafactor_beta2 = b2                # 1-epoch v_hat window (matches production)
        if disable_cohpre:
            m.disable_cohpre()                # production uses ratio_coh in place of coh_pre EMA
    return gf


def get_longtail_train_loader(batch_size, data_dir, num_workers, imb_factor, seed):
    """Exponential long-tailed CIFAR-10 TRAIN set (test stays balanced via get_loaders).
    imb_factor = n_min/n_max (0.01 = 100x). Class 0 keeps all ~5000; class 9 keeps
    imb_factor*5000. This is the regime that actually exercises the cf-discount (rare-class
    gradients are the scattered/diverse residual cf is meant to protect)."""
    import numpy as np
    tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
    targets = np.array(train.targets)
    C = 10
    n_max = int((targets == 0).sum())
    rng = np.random.default_rng(seed)
    keep, counts = [], []
    for c in range(C):
        n_c = max(1, int(round(n_max * (imb_factor ** (c / (C - 1))))))
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        keep.extend(idx_c[:n_c].tolist())
        counts.append(n_c)
    print(f"  longtail(imb={imb_factor}) train counts/class: {counts} total={len(keep)}", flush=True)
    persistent = num_workers > 0
    return DataLoader(Subset(train, keep), batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, persistent_workers=persistent)


@torch.no_grad()
def eval_full(net, loader, device, C=10):
    """Overall acc, mean test loss, and per-class acc list. On the balanced CIFAR test set,
    overall acc == balanced acc; per-class exposes the rare-class behaviour for long-tail arms."""
    net.eval()
    correct = torch.zeros(C, device=device)
    total = torch.zeros(C, device=device)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = net(x)
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        pred = logits.argmax(dim=1)
        n += y.size(0)
        for c in range(C):
            mc = y == c
            total[c] += mc.sum()
            correct[c] += (pred[mc] == c).sum()
    pc = (correct / total.clamp(min=1)).tolist()
    overall = correct.sum().item() / max(n, 1)
    return overall, loss_sum / max(n, 1), pc


@torch.no_grad()
def sync_deploy_net(deploy_net, model, dither=False):
    """Load each Concord layer's DEPLOY weight into the plain nn deploy net.
    dither=False: consolidated_weight() = (s_slow+v_slow)*128*scale -- TRUNCATED int8, drops s_fast.
    dither=True : ADDITIVELY-DITHERED int8 -- stochastically round the FULL live weight onto the
                  int8*128 grid so the sub-LSB (s_fast) is represented as intermediate values:
                  deploy = round(W_live/lsb + u)*lsb, lsb = 128*scale (the deploy grid step). The
                  dither error is white noise that averages out in the matmul -> the int8 deploy
                  tracks the full weight, and the deploy reflects s_fast WITHOUT any chase."""
    pairs = [(deploy_net.conv1, model.conv1), (deploy_net.conv2, model.conv2),
             (deploy_net.conv3, model.conv3), (deploy_net.fc1, model.fc1),
             (deploy_net.fc2, model.fc2)]
    for dst, src in pairs:
        if dither:
            W = src.get_weight().float()              # full live weight (incl s_fast), [out, in*kh*kw]
            exp = (src.row_exp[:, None].float() + src.col_exp[None, :].float() - src.MANTISSA_BIAS)
            lsb = 128.0 * torch.pow(2.0, exp)         # deploy-grid step per (row,col)
            W = torch.floor(W / lsb + torch.rand_like(W)) * lsb
        else:
            W = src.consolidated_weight()             # [out, in] or [out, in*kh*kw]
        dst.weight.data.copy_(W.reshape(dst.weight.shape).to(dst.weight.dtype))
        if dst.bias is not None and getattr(src, 'bias', None) is not None:
            dst.bias.data.copy_(src.bias.data.to(dst.bias.dtype))


@torch.no_grad()
def sync_deploy_net_dualdis(deploy_net, model, step_salt=0):
    """Load each dual-dissipation layer's DEPLOY weight (deploy_weight() = substrate +
    consolidated offset; drops the live fine register) into the plain nn deploy net."""
    pairs = [(deploy_net.conv1, model.conv1), (deploy_net.conv2, model.conv2),
             (deploy_net.conv3, model.conv3), (deploy_net.fc1, model.fc1),
             (deploy_net.fc2, model.fc2)]
    for dst, src in pairs:
        W = src.deploy_matrix(step_salt=step_salt, deterministic=True)   # [out, in*kh*kw]
        dst.weight.data.copy_(W.reshape(dst.weight.shape).to(dst.weight.dtype))
        if dst.bias is not None and src.bias is not None:
            dst.bias.data.copy_(src.bias.data.to(dst.bias.dtype))


def _collect_half_grads(model, x_half, y_half):
    """One forward+backward over a microbatch half. Returns {layer: weight-space grad [N,K]}
    captured from each layer's decoded weight proxy, plus the half's loss (float)."""
    model.zero_weight_grads()
    for m in model.dd_layers():
        if m.bias is not None and m.bias.grad is not None:
            m.bias.grad = None
    loss = F.cross_entropy(model(x_half), y_half)
    loss.backward()
    grads = {m: m.weight_grad() for m in model.dd_layers()}
    return grads, float(loss.item())


def run_dualdis(args):
    """Dual-dissipation CPU driver: the REAL two-gradient mechanism via RANDOM MICROBATCH
    SPLIT. Each batch is split into two random halves; arm L integrates half-1's FULL
    per-layer gradient and arm H half-2's FULL per-layer gradient (each arm its own half's
    whole gradient, NOT half a shared one). Deploy-eval reads deploy_weight()."""
    device = getattr(args, 'device', None) or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.batch_size, data_dir=args.data_dir,
                                            num_workers=args.num_workers)
    if args.longtail > 0.0:
        train_loader = get_longtail_train_loader(args.batch_size, args.data_dir,
                                                 args.num_workers, args.longtail, args.seed)
    if getattr(args, 'subset', 0) and args.subset > 0:
        from torch.utils.data import Subset as _Subset, DataLoader as _DL
        _ds = train_loader.dataset
        train_loader = _DL(_Subset(_ds, list(range(min(int(args.subset), len(_ds))))),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    total_steps = args.epochs * len(train_loader)

    model = DualDissipationConvNet(device=device, grad_accum_M=args.grad_accum_M,
                                   bracket_d=args.bracket_d, seed=args.seed,
                                   substrate_scale=args.substrate_scale).to(device)
    step_kw = configure_dualdis(model, args.lam, args.lr, len(train_loader))
    for _k in ('precond_p', 'v_scale', 'alpha', 'chase_floor', 'min_leak', 'beta1'):   # ablation overrides
        _v = getattr(args, _k, None)
        if _v is not None:
            step_kw[_k] = _v
    deploy_net = BaselineConvNet().to(device)
    gf_base = args.lam / max(args.lr, 1e-12)         # gf_consol = lam/lr (dimensionless lam)

    print(f"[{args.arm}] opt=dual_dissipation lam={args.lam} gf_consol={gf_base:.4g} "
          f"bracket_d={args.bracket_d} grad_accum_M={args.grad_accum_M} "
          f"epochs={args.epochs} lr={args.lr} warmup={args.warmup} bs={args.batch_size} "
          f"seed={args.seed} steps/epoch={len(train_loader)} total_steps={total_steps} "
          f"DEVICE={device}", flush=True)

    g = torch.Generator().manual_seed(args.seed ^ 0x5D17)   # the RANDOM-SPLIT permutation RNG
    step = 0
    history = []
    nan_hit = False
    for epoch in range(args.epochs):
        model.train()
        run_loss = seen = 0
        t0 = time.time()
        last_lr = args.lr
        for x, y in train_loader:
            last_lr = cosine_lr(step, total_steps, args.lr, args.lr_min_frac, args.warmup)
            x = x.to(device); y = y.to(device)
            B = x.size(0)
            if B < 2:
                continue
            # ---- RANDOM MICROBATCH SPLIT: two disjoint random halves of THIS batch ----
            perm = torch.randperm(B, generator=g)
            h = B // 2
            idx1, idx2 = perm[:h].to(device), perm[h:2 * h].to(device)
            grads_L, loss1 = _collect_half_grads(model, x[idx1], y[idx1])   # arm L <- half-1
            grads_H, loss2 = _collect_half_grads(model, x[idx2], y[idx2])   # arm H <- half-2
            # ---- TWO-GRADIENT STEP: route half-1's grad into e_L, half-2's into e_H ----
            kw = dict(step_kw); kw['gf_consol'] = gf_base   # lam = lr*gf tracks lr via gf_base
            if getattr(args, 'recipe', False):
                # Exp 8 IGNITION recipe: anneal chase/leak floors high->low so the chase fires
                # from step 0 (force consolidation before coherence establishes) -- the
                # from-scratch ignition the dual path's fixed low floors were missing.
                _pp = min(1.0, step / max(1, total_steps))
                _co = 0.5 * (1.0 + math.cos(math.pi * _pp))
                kw['chase_floor'] = 0.1 + (0.9 - 0.1) * _co       # 0.9 -> 0.1 (chase ignition)
                kw['leak_floor'] = 0.1 + (0.999 - 0.1) * _co      # 0.999 -> 0.1 (leak ignition)
            arm_div = []
            for m in model.dd_layers():
                gL, gH = grads_L[m], grads_H[m]
                if gL is None or gH is None:
                    continue
                info = m.step_two_grad(gL, gH, last_lr, **kw)
                arm_div.append((info['coh_L'], info['coh_H']))
            if args.bias_correct:
                for m in model.dd_layers():
                    if m.bias is not None and m.bias.grad is not None:
                        m.bias.data.add_(m.bias.grad, alpha=-last_lr)   # plain SGD on biases
            lv = 0.5 * (loss1 + loss2)
            if not math.isfinite(lv):
                print(f"[{args.arm}] NON-FINITE loss at step {step} (epoch {epoch}) -> abort", flush=True)
                nan_hit = True
                break
            run_loss += lv * B
            seen += B
            step += 1
            if args.log_every and step % args.log_every == 0:
                cL = sum(a for a, _ in arm_div) / max(len(arm_div), 1)
                cH = sum(b for _, b in arm_div) / max(len(arm_div), 1)
                print(f"[{args.arm}]   step {step} lr={last_lr:.4f} loss={lv:.4f} "
                      f"avg={run_loss / max(seen, 1):.4f} coh_L={cL:.4f} coh_H={cH:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if nan_hit:
            break
        train_loss = run_loss / max(seen, 1)
        sync_deploy_net_dualdis(deploy_net, model, step_salt=step)
        deploy_acc, deploy_loss, deploy_pc = eval_full(deploy_net, test_loader, device)
        live_acc, _live_loss, _ = eval_full(model, test_loader, device)   # LIVE forward (incl fine register)
        # diagnostics: deploy offset/substrate ratio (gate b), arm content + divergence (gate c)
        off_ratio, eL_mean, eH_mean, arm_div = _dualdis_diag(model)
        dt = time.time() - t0
        history.append({'epoch': epoch, 'lr': last_lr, 'train_loss': train_loss,
                        'live_acc': live_acc, 'deploy_acc': deploy_acc, 'deploy_loss': deploy_loss,
                        'deploy_offset_ratio': off_ratio, 'e_L_abs_mean': eL_mean,
                        'e_H_abs_mean': eH_mean, 'arm_divergence': arm_div, 'sec': dt})
        print(f"[{args.arm}] ep{epoch} lr={last_lr:.4f} tl={train_loss:.4f} "
              f"live_acc={live_acc:.4f} deploy_acc={deploy_acc:.4f} deploy_off/sub={off_ratio:.3f} "
              f"|e_L|={eL_mean:.2f} |e_H|={eH_mean:.2f} arm_div={arm_div:.3f} ({dt:.1f}s)",
              flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    if getattr(args, 'dump_weights', None):
        torch.save({i: {'packed_w': m.dd.packed_w.detach().cpu(),
                        'row_exp': m.dd.row_exp.detach().cpu(),
                        'col_exp': m.dd.col_exp.detach().cpu(),
                        'substrate': m.dd.substrate.detach().cpu(),
                        'N': m.dd.N, 'K': m.dd.K, 'name': type(m).__name__}
                    for i, m in enumerate(model.dd_layers())}, args.dump_weights)
        print(f"[{args.arm}] dumped packed weights -> {args.dump_weights}", flush=True)
    best_deploy = max((h['deploy_acc'] for h in history), default=0.0)
    final_deploy = history[-1]['deploy_acc'] if history else 0.0
    print(f"[{args.arm}] DONE best_deploy={best_deploy:.4f} final_deploy={final_deploy:.4f} "
          f"nan={nan_hit}", flush=True)
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, 'w') as f:
            json.dump({'arm': args.arm, 'args': vars(args), 'nan': nan_hit,
                       'best_deploy_acc': best_deploy, 'final_deploy_acc': final_deploy,
                       'history': history}, f, indent=2)
        print(f"[{args.arm}] wrote {args.results_json}", flush=True)


@torch.no_grad()
def _dualdis_diag(model):
    """Returns (deploy_off_ratio, eL_mean, eH_mean, arm_div):
      deploy_off_ratio  mean |consolidated deploy offset| / mean |substrate| — gate (b): the
                        deploy weight builds a learned offset ABOVE the random-prior baseline.
      eL_mean, eH_mean  mean |e_L|, |e_H| (arm content).
      arm_div           mean |e_L - e_H| / mean(|e_L|+|e_H|) — gate (c): the random split
                        differentiates the arms (0 == identical == even-split, the wrong build)."""
    from dual_dissipation_ref import _scale_fwd
    off_sum = sub_sum = elem_n = 0.0
    eL_sum = eH_sum = ediff_sum = emag_sum = e_n = 0.0
    for m in model.dd_layers():
        dd = m.dd
        e_H, e_L, s_slow, v_slow = unpack_dual(dd.packed_w)
        off = ((s_slow.to(torch.float32) + v_slow.to(torch.float32)) * 128.0
               * _scale_fwd(dd.row_exp, dd.col_exp))            # deploy offset, weight units
        off_sum += float(off.abs().sum())
        sub_sum += float(dd.substrate.abs().sum())
        elem_n += off.numel()
        eL = e_L.abs().float(); eH = e_H.abs().float()
        eL_sum += float(eL.sum()); eH_sum += float(eH.sum())
        ediff_sum += float((e_L.float() - e_H.float()).abs().sum())
        emag_sum += float((eL + eH).sum())
        e_n += e_L.numel()
    off_ratio = (off_sum / max(elem_n, 1)) / max(sub_sum / max(elem_n, 1), 1e-30)
    arm_div = ediff_sum / max(emag_sum, 1e-30)
    return (off_ratio, eL_sum / max(e_n, 1), eH_sum / max(e_n, 1), arm_div)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', type=str, default='B_cf_on_k1')
    ap.add_argument('--optimizer', choices=['concord', 'adamw', 'dual_dissipation'],
                    default='concord')
    # --- the cf-discount ablation switches ---
    ap.add_argument('--coh_vhat', type=int, default=1, help='THE switch: 1=cf-discount on, 0=off')
    ap.add_argument('--coh_kappa', type=float, default=1.0, help='cf knee: discount=kappa/(cf+kappa)')
    ap.add_argument('--lam', type=float, default=0.025, help='dimensionless dissipation; gf_consol=lam/lr. 0=no gf-evap')
    ap.add_argument('--init_gap', type=float, default=0.5, help='from-scratch: seed s_slow ahead of v_slow by d_sv=gap*coarse so the coh gate has a direction at step 0 (0=gap-zero fine-tune init)')
    ap.add_argument('--deploy_dither', type=int, default=0, help='deploy = additively-dithered int8 (stochastic-round the full weight incl s_fast onto the int8 grid) instead of truncating s_fast')
    # --- SDXL-production parity extras (off by default for a clean cf A/B) ---
    ap.add_argument('--ratio_coh', type=int, default=0, help='production ratio-coh dissipation gate')
    ap.add_argument('--disable_cohpre', type=int, default=0, help='drop coh_pre EMA (production uses ratio_coh)')
    ap.add_argument('--sigmag_peak', type=float, default=0.0, help='isotropic grad-noise (||g|| units). 0=off; 0.6=production')
    # --- regime ---
    ap.add_argument('--longtail', type=float, default=0.0, help='>0: long-tail train imb factor n_min/n_max (e.g. 0.01=100x)')
    # --- training recipe (mirrors core train_cifar_fused defaults) ---
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--adamw_lr', type=float, default=1e-3, help='separate peak lr for adamw_ref (0.05 diverges for Adam)')
    ap.add_argument('--warmup', type=int, default=200, help='linear LR warmup steps (cold-start overshoot guard)')
    ap.add_argument('--bias_correct', type=int, default=1, help='Adam v_hat bias-correction 1/(1-b2^t) on concord arms (cold-start fix)')
    ap.add_argument('--lr_min_frac', type=float, default=0.01)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--data_dir', type=str, default=os.path.join(_CORE, 'cifar_data'))
    ap.add_argument('--rebalance_every', type=int, default=8)
    ap.add_argument('--pretrained', type=str, default='', help='path to a trained BaselineConvNet state_dict to preload into s_slow+v_slow (FINE-TUNE regime)')
    ap.add_argument('--save_path', type=str, default='', help='(adamw) save the trained BaselineConvNet state_dict here for later --pretrained')
    ap.add_argument('--results_json', type=str, default='')
    ap.add_argument('--max_steps', type=int, default=0, help='debug: stop after N steps (smoke test)')
    ap.add_argument('--log_every', type=int, default=0, help='debug: print batch + running train loss every N steps')
    ap.add_argument('--subset', type=int, default=0, help='use only the first N train images (short epochs for fast A/B)')
    ap.add_argument('--device', type=str, default='', help='dual_dissipation device override (cuda/cpu); other optimizers use cuda')
    # --- dual-dissipation (CPU-capable, no Triton) ---
    ap.add_argument('--bracket_d', type=float, default=0.5, help='dual-dissipation arithmetic bracket half-spread d: lam_{L,H}=lam*(1-+d)')
    ap.add_argument('--grad_accum_M', type=int, default=8, help='dual-dissipation grad-accum M (>=4 keeps the dual feature enabled)')
    ap.add_argument('--substrate_scale', type=float, default=0.1, help='dual-dissipation from-scratch substrate (random prior) scale; <<1 lets the trained offset dominate the deploy weight')
    args = ap.parse_args()

    if args.optimizer == 'dual_dissipation':
        return run_dualdis(args)       # CPU-capable random-microbatch-split path

    if not torch.cuda.is_available():
        raise SystemExit("Concord requires CUDA+Triton; no GPU visible. Aborting.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True   # SR is stochastic regardless; multi-seed gives error bars
    device = 'cuda'

    # Module-global coherence flags MUST be set before constructing the layers
    # (the on/off flags bake into Triton constexprs at the first kernel launch).
    if args.optimizer == 'concord':
        pb.set_fixed_coh(True)                          # Wiener coherence gate (the cf block nests here)
        pb.set_coh_kappa(args.coh_kappa)               # set BEFORE set_coh_vhat so the ON-canary prints the right knee
        pb.set_coh_vhat(bool(args.coh_vhat))           # <<< THE ablation switch
        pb.set_min_leak(0.1)
        pb.set_evap_build_min(128.0)
        pb.set_ratio_coh(bool(args.ratio_coh))
        sig_on = args.sigmag_peak > 0.0
        pb.set_sigmag_noise(sig_on, isotropic=True)
        if sig_on and hasattr(pb, 'set_sigmag_sigma'):
            pb.set_sigmag_sigma(args.sigmag_peak)

    # data: balanced test always; train balanced or long-tailed
    train_loader, test_loader = get_loaders(args.batch_size, data_dir=args.data_dir,
                                            num_workers=args.num_workers)
    if args.longtail > 0.0:
        train_loader = get_longtail_train_loader(args.batch_size, args.data_dir,
                                                 args.num_workers, args.longtail, args.seed)
    if args.subset and args.subset > 0:
        from torch.utils.data import Subset as _Subset, DataLoader as _DL
        _ds = train_loader.dataset
        train_loader = _DL(_Subset(_ds, list(range(min(int(args.subset), len(_ds))))),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    total_steps = args.epochs * len(train_loader)

    deploy_net = None
    if args.optimizer == 'adamw':
        model = BaselineConvNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        bias_opt = None
        gf = 0.0
    else:
        model = ConcordConvNet(device=device, lr=args.lr, alpha=args.alpha, init_gap=args.init_gap).to(device)
        gf = configure_concord(model, args.lam, args.lr,
                               bool(args.ratio_coh), bool(args.disable_cohpre), len(train_loader))
        if args.pretrained:
            base = BaselineConvNet().to(device)
            base.load_state_dict(torch.load(args.pretrained, map_location=device))
            model.load_pretrained(base)
            del base
            print(f"[{args.arm}] preloaded s_slow+v_slow from {args.pretrained} (FINE-TUNE)", flush=True)
        if args.bias_correct:
            pb.set_bias_correct_v(True)
            for m in model.concord_layers():
                pb._v_bc_buf(m.packed_w.device)        # pre-create so step 0 is already bias-corrected
        biases = [m.bias for m in model.concord_layers() if m.bias is not None]
        bias_opt = torch.optim.SGD(biases, lr=args.lr) if biases else None
        opt = None
        deploy_net = BaselineConvNet().to(device)   # plain net we load consolidated weights into

    print(f"[{args.arm}] opt={args.optimizer} coh_vhat={args.coh_vhat} kappa={args.coh_kappa} "
          f"lam={args.lam} gf_consol={gf:.4g} ratio_coh={args.ratio_coh} "
          f"disable_cohpre={args.disable_cohpre} sigmag={args.sigmag_peak} longtail={args.longtail} "
          f"epochs={args.epochs} lr={args.lr} adamw_lr={args.adamw_lr} warmup={args.warmup} "
          f"bias_correct={args.bias_correct} bs={args.batch_size} seed={args.seed} "
          f"steps/epoch={len(train_loader)} total_steps={total_steps}", flush=True)

    step = 0
    history = []
    nan_hit = False
    for epoch in range(args.epochs):
        model.train()
        run_loss = seen = 0
        t0 = time.time()
        last_lr = args.lr
        for x, y in train_loader:
            peak = args.adamw_lr if args.optimizer == 'adamw' else args.lr
            last_lr = cosine_lr(step, total_steps, peak, args.lr_min_frac, args.warmup)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.optimizer == 'adamw':
                for pg in opt.param_groups:
                    pg['lr'] = last_lr
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
            else:
                if args.ratio_coh:                        # IGNITION: anneal the chase/leak floors so the chase
                    pb.set_ratio_coh_floors(*ratio_floors(step, len(train_loader)))  # fires before coh establishes
                if args.bias_correct:                     # drive 1/(1-b2^t) into the v_hat buffer (kernel :808)
                    _b2 = model.concord_layers()[0].adafactor_beta2
                    pb.set_v_bias_correction(pb.bias_correction_factor(step, _b2))
                model.set_lr(last_lr)                     # write lr into every layer before backward
                if bias_opt is not None:
                    for pg in bias_opt.param_groups:
                        pg['lr'] = last_lr
                    bias_opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)       # forward
                loss.backward()                           # <<< the FUSED packed step happens here
                if bias_opt is not None:
                    bias_opt.step()                       # biases only (Concord doesn't apply bias grad)
                if step % args.rebalance_every == 0:
                    for m in model.concord_layers():
                        m.rebalance()                     # mantissa-overflow guard (rarely fires)
            lv = loss.item()
            if not math.isfinite(lv):
                print(f"[{args.arm}] NON-FINITE loss at step {step} (epoch {epoch}) -> abort arm", flush=True)
                nan_hit = True
                break
            run_loss += lv * x.size(0)
            seen += x.size(0)
            step += 1
            if args.log_every and step % args.log_every == 0:
                print(f"[{args.arm}]   step {step} lr={last_lr:.4f} loss={lv:.4f} "
                      f"avg={run_loss / max(seen, 1):.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if nan_hit:
            break
        train_loss = run_loss / max(seen, 1)
        # live-weight eval (includes s_fast)
        live_acc, live_loss, _ = eval_full(model, test_loader, device)
        # deploy-weight eval (consolidated: drops s_fast == what production ships)
        if deploy_net is not None:
            sync_deploy_net(deploy_net, model, dither=bool(args.deploy_dither))
            deploy_acc, deploy_loss, deploy_pc = eval_full(deploy_net, test_loader, device)
        else:
            deploy_acc, deploy_loss, deploy_pc = live_acc, live_loss, None
        worst = min(deploy_pc) if deploy_pc else float('nan')
        dt = time.time() - t0
        history.append({'epoch': epoch, 'lr': last_lr, 'train_loss': train_loss,
                        'live_acc': live_acc, 'deploy_acc': deploy_acc,
                        'deploy_loss': deploy_loss, 'deploy_worst_class': worst,
                        'deploy_per_class': deploy_pc, 'sec': dt})
        wc = f" worst_cls={worst:.4f}" if args.longtail > 0.0 else ""
        print(f"[{args.arm}] ep{epoch} lr={last_lr:.4f} tl={train_loss:.4f} "
              f"live_acc={live_acc:.4f} deploy_acc={deploy_acc:.4f}{wc} ({dt:.1f}s)", flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    best_live = max((h['live_acc'] for h in history), default=0.0)
    best_deploy = max((h['deploy_acc'] for h in history), default=0.0)
    best_worst = max((h['deploy_worst_class'] for h in history
                      if math.isfinite(h['deploy_worst_class'])), default=float('nan'))
    final_deploy = history[-1]['deploy_acc'] if history else 0.0
    print(f"[{args.arm}] DONE best_deploy={best_deploy:.4f} best_live={best_live:.4f} "
          f"best_worst_cls={best_worst:.4f} final_deploy={final_deploy:.4f} nan={nan_hit}", flush=True)
    if args.save_path and args.optimizer == 'adamw':
        torch.save(model.state_dict(), args.save_path)
        print(f"[{args.arm}] saved pretrained baseline -> {args.save_path}", flush=True)
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, 'w') as f:
            json.dump({'arm': args.arm, 'args': vars(args), 'nan': nan_hit,
                       'best_deploy_acc': best_deploy, 'best_live_acc': best_live,
                       'best_worst_class': best_worst, 'final_deploy_acc': final_deploy,
                       'history': history}, f, indent=2)
        print(f"[{args.arm}] wrote {args.results_json}", flush=True)


if __name__ == '__main__':
    main()
