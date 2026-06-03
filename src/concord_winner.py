"""Concord — the WINNING configuration, as a reusable integration module.

Per WINNING_CONFIG.md (verified against tools/run_sigma_fine.sh : sf_060 and
src/prototype_packed_b.py), the winner is a **fluctuation-dissipation pair on
rank-1 v-hat AdamW**, 32 b/param, deployed off consolidated_weight():

  bare recipe  — baked ConcordLinearPackedB defaults:
      optimizer_kind='adamw', eps=1e-10, v_scale=0, gf_trust_delta_sq=1,
      precond_p=0.5, alpha=0.1 (chase), alpha_v_fast=0.001 (leak),
      fixed Wiener coherence gate (S/(S+noise^2)).
  + DISSIPATION ("split") — ratio_coh ON, coh_pre dropped (stays 32 b/param),
      chase/leak floors cosine-decay 0.9 -> 0.1 / 0.999 -> 0.1 over ~1 epoch,
      gf_consol = 50 (coherence-gated evaporation of incoherent s_fast mass).
  + FLUCTUATION (noise) — isotropic white noise, sigma_peak = 0.6, rising-late
      sigma = 0.6 * (1 - lr/lr_peak), injected in the fused backward via a
      device-tensor sigma (CUDA-graph-safe).
  deploy — consolidated_weight() = (s_slow + v_slow)*128*2^exp  (drop s_fast).

Bench (nanoGPT char, same-seed, deployed-sv): bare 1.5404 -> +dissipation 1.5180
-> +fluctuation 1.4967 (WINNER), vs AdamW ~1.534 / Muon ~1.578, all at 32 b/param.
Honest caveat (from the doc): the fluctuation -0.021 is single-seed / nanoGPT-only,
jitter ~= half the effect -> the *mechanism* is validated, the *magnitude* on a new
task (e.g. SDXL) is not yet load-bearing. The dissipation -0.022 is deterministic.

IMPORTANT: the two halves live in different files. Dissipation is in
concord/packed_b.py, but the NOISE kernel branch exists ONLY in
src/prototype_packed_b.py. This module imports the modules + setters from
prototype_packed_b (the superset) so BOTH halves are expressible.
"""
import math

import torch
import torch.nn as nn

import prototype_packed_b as ppb
from prototype_packed_b import (
    ConcordLinearPackedB, ConcordConv2dPackedB,
    set_ratio_coh, set_ratio_coh_floors, set_fixed_coh,
    set_sigmag_noise, set_sigmag_sigma,
)

# The exact winner constants (== sf_060 arm). The recipe values are also the
# baked __init__ defaults; we set them explicitly so the config is unambiguous
# and self-documenting (and robust if prototype defaults ever drift).
WINNER = dict(
    # --- bare recipe (rank-1 v-hat AdamW + fixed coherence gate) ---
    alpha=0.1, alpha_v_fast=0.001, weight_decay=0.0, eps=1e-10,
    step_cap=10.0, v_scale=0.0, precond_p=0.5, gf_trust_delta_sq=1.0,
    # --- dissipation (the "split") ---
    gf_consol=50.0,
    ratio_chase_floor=0.9, ratio_chase_floor_min=0.1,
    ratio_leak_floor=0.999, ratio_leak_floor_min=0.1,
    # --- fluctuation (the noise) ---
    sigmag_iso=True, sigmag_peak=0.6,
    # --- schedule ---
    lr_min_frac=0.2,
)


def _scalar(v, what):
    """Collapse a symmetric 2-tuple to a scalar (the packed conv kernel uses
    stride/padding/kernel as scalars). Asymmetric -> clear error."""
    if isinstance(v, (tuple, list)):
        if len(v) == 2 and v[0] == v[1]:
            return int(v[0])
        raise ValueError(f"ConcordConv2dPackedB needs symmetric {what}, got {v}")
    return int(v)


def swap_unet_to_winner(unet, device, lr, gf_consol=None, verbose=True):
    """Swap every nn.Linear / nn.Conv2d in `unet` (in place) to
    Concord{Linear,Conv2d}PackedB with the validated recipe + dissipation, load
    the pretrained weights, and engage the global winner flags (ratio_coh,
    fixed_coh, isotropic noise). Returns the list of created Concord modules.

    The caller still drives the per-step schedule via winner_step() and calls
    rebalance() on each returned module every step.
    """
    if gf_consol is None:
        gf_consol = WINNER["gf_consol"]
    layers, n_lin, n_conv = [], 0, 0
    for parent in list(unet.modules()):
        for name, child in list(parent.named_children()):
            c, W2d = None, None
            if isinstance(child, nn.Linear):
                c = ConcordLinearPackedB(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, device=device,
                    alpha=WINNER["alpha"], lr=lr)
                W2d = child.weight.data
                n_lin += 1
            elif isinstance(child, nn.Conv2d):
                k = _scalar(child.kernel_size, "kernel_size")
                c = ConcordConv2dPackedB(
                    child.in_channels, child.out_channels, k,
                    stride=_scalar(child.stride, "stride"),
                    padding=_scalar(child.padding, "padding"),
                    bias=child.bias is not None, device=device,
                    alpha=WINNER["alpha"], lr=lr)
                W2d = child.weight.data.reshape(
                    child.out_channels, child.in_channels * k * k)
                n_conv += 1
            if c is None:
                continue
            # Validated recipe (== baked defaults; set explicit for clarity).
            c.set_optimizer_kind('adamw', weight_decay=WINNER["weight_decay"],
                                  eps=WINNER["eps"], step_cap=WINNER["step_cap"])
            c.precond_p = WINNER["precond_p"]
            c.v_scale = WINNER["v_scale"]
            c.gf_trust_delta_sq = WINNER["gf_trust_delta_sq"]
            c.gf_consol = gf_consol            # dissipation
            with torch.no_grad():
                c.load_weights(W2d.float())
                if child.bias is not None:
                    c.bias.data.copy_(child.bias.data.to(c.bias.dtype))
            c.disable_cohpre()                 # ratio-coh: drop coh_pre -> 32 b/param
            setattr(parent, name, c)
            layers.append(c)
    # Global winner flags (module-level switches in prototype_packed_b).
    set_fixed_coh(True)                        # Wiener coherence gate
    set_ratio_coh(True)                        # dissipation: live ratio-coh gate
    set_sigmag_noise(True, isotropic=WINNER["sigmag_iso"])  # fluctuation
    if verbose:
        print(f"[winner] swapped {n_lin} Linear + {n_conv} Conv2d -> Concord "
              f"(gf_consol={gf_consol}, ratio_coh ON, isotropic noise ON)")
    return layers


def winner_step(it, total_iters, layers, peak_lr, warmup=100, floor_horizon=None,
                sigmag_peak=None, lr_min_frac=None, noise=True):
    """Advance the per-step winner schedule (call BEFORE backward each step):
      - lr: warmup * cosine(1 -> lr_min_frac)        -> m.lr on every layer
      - sigma: rising-late sigmag_peak * (1 - f)      (f = cosine factor)
      - ratio floors: cosine 0.9 -> 0.1 / 0.999 -> 0.1 over floor_horizon
    All three are pushed to device tensors (CUDA-graph-safe). Returns the lr.
    """
    if sigmag_peak is None:
        sigmag_peak = WINNER["sigmag_peak"]
    if lr_min_frac is None:
        lr_min_frac = WINNER["lr_min_frac"]
    if floor_horizon is None:
        floor_horizon = max(1, total_iters)

    p = min(1.0, it / max(1, total_iters))
    f = lr_min_frac + 0.5 * (1 - lr_min_frac) * (1 + math.cos(math.pi * p))  # cosine factor
    warm = min(1.0, (it + 1) / warmup) if warmup > 0 else 1.0
    lr = peak_lr * f * warm
    for m in layers:
        m.lr = lr

    # rising-late isotropic noise: ~0 early (f~1), grows late (f->lr_min_frac)
    set_sigmag_sigma(sigmag_peak * (1.0 - f) if noise else 0.0)

    # ratio-coh bootstrap floors cosine-decay to their mins
    def cos_floor(start, end):
        if it >= floor_horizon:
            return end
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * it / floor_horizon))
    set_ratio_coh_floors(
        cos_floor(WINNER["ratio_chase_floor"], WINNER["ratio_chase_floor_min"]),
        cos_floor(WINNER["ratio_leak_floor"], WINNER["ratio_leak_floor_min"]))
    return lr


def make_aux_optimizer(params, lr, momentum=0.9, weight_decay=0.0):
    """The aux optimizer for the NON-Concord params (norms/biases/embeddings)
    is plain SGD, NOT AdamW (project owner's call, overriding the README).

    Rationale: AdamW normalizes every step, so on the tiny GroupNorm scale/bias
    set it slams those scales toward 0 -> the network's output collapses to ~0
    (the trivial predict-the-mean solution), which both hurts training and, in a
    single-batch overfit, MASKS whether the Concord weight step is doing anything.
    Plain SGD is proportional to the gradient and won't manufacture that collapse.
    """
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def active_config():
    """Snapshot of the live module-level winner switches (to PROVE the config is
    engaged, not silently off). Reads the globals in prototype_packed_b."""
    return dict(
        ratio_coh=ppb._RATIO_COH,
        fixed_coh=ppb._USE_FIXED_COH,
        noise_on=ppb._SIGMAG_NOISE,
        noise_isotropic=ppb._SIGMAG_ISO,
        sigma_now=round(ppb._SIGMAG_SIGMA, 4),
        chase_floor_now=round(ppb._RATIO_CHASE_FLOOR, 4),
        leak_floor_now=round(ppb._RATIO_LEAK_FLOOR, 4),
    )


def consolidated_state_dict(layers, names):
    """Deploy path: materialize each Concord module's consolidated_weight()
    (drop s_fast) for .safetensors export. `names` parallels `layers`. Conv
    weights are reshaped back to (out, in, k, k)."""
    out = {}
    with torch.no_grad():
        for name, m in zip(names, layers):
            W = m.consolidated_weight()
            if isinstance(m, ConcordConv2dPackedB):
                W = W.reshape(m.out_channels, m.in_channels, m.kh, m.kw)
            out[name + ".weight"] = W
            if getattr(m, "bias", None) is not None:
                out[name + ".bias"] = m.bias.detach()
    return out
