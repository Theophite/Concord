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

from dataclasses import dataclass


@dataclass
class ConcordConfig:
    """The optimizer-picker entry for Concord: every winner knob, configurable.
    Defaults == the validated sf_060 winner. `kind` lets a picker choose among
    optimizers; the rest are the Concord details the trainer reads -- nothing is
    hardcoded in the loop."""
    kind: str = "concord"               # picker selector: "concord" | "adamw" | "sgd"
    lr: float = 5e-4                    # peak lr (SDXL UNet finetune wants ~1e-5; nanoGPT used 5e-4)
    alpha: float = 0.1                  # chase (s_fast -> s_slow)
    alpha_v_fast: float = 0.001         # leak  (s_slow -> v_slow)
    weight_decay: float = 0.0
    eps: float = 1e-10
    step_cap: float = 10.0
    v_scale: float = 0.0
    precond_p: float = 0.5
    gf_trust_delta_sq: float = 1.0
    # dissipation (the "split")
    gf_consol: float = 50.0
    ratio_coh: bool = True
    ratio_chase_floor: float = 0.9
    ratio_chase_floor_min: float = 0.1
    ratio_leak_floor: float = 0.999
    ratio_leak_floor_min: float = 0.1
    # fluctuation (the noise)
    noise: bool = True
    sigmag_iso: bool = True
    sigmag_peak: float = 0.6
    # schedule
    lr_min_frac: float = 0.2
    warmup: int = 100
    # aux optimizer for the non-swapped params (norms/biases)
    aux: str = "sgd"                    # "sgd" | "adamw" | "none"


WINNER_CONFIG = ConcordConfig()


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


def winner_step(it, total_iters, layers, peak_lr=None, warmup=None, floor_horizon=None,
                sigmag_peak=None, lr_min_frac=None, noise=None, config=None):
    """Advance the per-step winner schedule (call BEFORE backward each step):
      - lr: warmup * cosine(1 -> lr_min_frac)        -> m.lr on every layer
      - sigma: rising-late sigmag_peak * (1 - f)      (f = cosine factor)
      - ratio floors: cosine chase/leak -> their mins over floor_horizon
    Config-driven (pass a ConcordConfig); explicit args still override it, and the
    old lr-based call signature keeps working. All three are device tensors
    (CUDA-graph-safe). Returns the lr.
    """
    cfg = config if config is not None else WINNER_CONFIG
    peak_lr = cfg.lr if peak_lr is None else peak_lr
    warmup = cfg.warmup if warmup is None else warmup
    sigmag_peak = cfg.sigmag_peak if sigmag_peak is None else sigmag_peak
    lr_min_frac = cfg.lr_min_frac if lr_min_frac is None else lr_min_frac
    noise = cfg.noise if noise is None else noise
    if floor_horizon is None:
        floor_horizon = max(1, total_iters)

    p = min(1.0, it / max(1, total_iters))
    f = lr_min_frac + 0.5 * (1 - lr_min_frac) * (1 + math.cos(math.pi * p))  # cosine factor
    warm = min(1.0, (it + 1) / warmup) if warmup > 0 else 1.0
    lr = peak_lr * f * warm
    for m in layers:
        m.lr = lr

    set_sigmag_sigma(sigmag_peak * (1.0 - f) if noise else 0.0)   # rising-late noise

    def cos_floor(start, end):
        if it >= floor_horizon:
            return end
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * it / floor_horizon))
    set_ratio_coh_floors(
        cos_floor(cfg.ratio_chase_floor, cfg.ratio_chase_floor_min),
        cos_floor(cfg.ratio_leak_floor, cfg.ratio_leak_floor_min))
    return lr


class GatedRebalance:
    """Fire rebalance only when a layer's mantissa actually crosses MAX_M.

    Per-step rebalance launches one kernel per Concord layer (~840 ms for SDXL's
    794 layers) -- but rebalance only *does* anything when a row/col max exceeds
    MAX_M, which at a finetune lr is ~0% of steps (measured). The launches are
    almost pure waste.

    The winner's apply kernel already writes each layer's per-row/col mantissa
    maxima (atomic_max) and the backward zeros+repopulates them every step. So we
    back every layer's max buffers with slices of two SHARED tensors, and one
    reduction over them tells us whether ANY layer needs a tick. Common case: no
    fire -> skip all 794 launches. Rare fire -> run the exact same per-layer
    rebalance. Math-identical (same MAX_M trigger, same kernel); touches no kernel
    code -- only repoints the bookkeeping buffers and gates the dispatch.

    Drop-in for `for m in layers: m.rebalance()` -- just call the instance.
    """

    def __init__(self, layers):
        self.layers = [m for m in (layers or []) if hasattr(m, "rebalance")]
        self.MAX_M = max((getattr(m, "MAX_M", 24000) for m in self.layers), default=24000)
        self.row_all = None
        self.col_all = None
        self.fires = 0
        self.calls = 0

    def _wire(self):
        """Repoint the per-layer max buffers at slices of two shared tensors.
        Called once, after the first backward has created the buffers."""
        dev = self.layers[0].packed_w.device
        rs = [m._row_max_buf.shape[0] for m in self.layers]
        cs = [m._col_max_buf.shape[0] for m in self.layers]
        self.row_all = torch.zeros(sum(rs), dtype=torch.int32, device=dev)
        self.col_all = torch.zeros(sum(cs), dtype=torch.int32, device=dev)
        ro = co = 0
        for m, n, k in zip(self.layers, rs, cs):
            m._row_max_buf = self.row_all[ro:ro + n]; ro += n
            m._col_max_buf = self.col_all[co:co + k]; co += k

    def __call__(self):
        """Call after backward (in place of the per-layer rebalance loop)."""
        if not self.layers:
            return False
        self.calls += 1
        if self.row_all is None:
            if not hasattr(self.layers[0], "_row_max_buf"):
                return False                  # no apply has run yet
            self._wire()                      # first wired step: buffers just zeroed, safe at init
            return False
        peak = torch.maximum(self.row_all.max(), self.col_all.max())
        if bool((peak > self.MAX_M).item()):  # one reduction + one tiny sync gates 794 layers
            for m in self.layers:
                m.rebalance()
            self.fires += 1
            return True
        return False


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


def make_aux(params, config):
    """Aux optimizer for the non-Concord params, chosen by the config."""
    if config.aux == "sgd":
        return torch.optim.SGD(params, lr=config.lr, momentum=0.9)
    if config.aux == "adamw":
        return torch.optim.AdamW(params, lr=config.lr)
    return None


def configure_optimizer(unet, device, config):
    """THE OPTIMIZER PICKER. Given a ConcordConfig, set up the optimizer for `unet`
    and return (concord_layers, aux_opt, config). For kind='concord': swap to the
    winner, push every config knob onto the layers + the global flags, build the aux.
    For a baseline kind: a plain torch optimizer over the UNet (concord_layers=None).
    The trainer then drives winner_step(config=...) + rebalance + aux_opt.step()."""
    if config.kind != "concord":
        opt = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[config.kind]
        return None, opt(unet.parameters(), lr=config.lr), config
    layers = swap_unet_to_winner(unet, device, config.lr, gf_consol=config.gf_consol,
                                 verbose=False)
    for m in layers:                                   # push the rest of the config
        m.set_optimizer_kind('adamw', weight_decay=config.weight_decay,
                             eps=config.eps, step_cap=config.step_cap)
        m.precond_p = config.precond_p
        m.v_scale = config.v_scale
        m.gf_trust_delta_sq = config.gf_trust_delta_sq
        m.alpha_v_fast = config.alpha_v_fast
    ppb.set_ratio_coh(config.ratio_coh)                # global flags from the config
    ppb.set_sigmag_noise(config.noise, isotropic=config.sigmag_iso)
    aux = [p for p in unet.parameters() if p.requires_grad]
    print(f"[picker] concord: {len(layers)} layers, lr={config.lr}, gf_consol="
          f"{config.gf_consol}, ratio_coh={config.ratio_coh}, noise={config.noise}, "
          f"aux={config.aux} ({sum(p.numel() for p in aux)/1e6:.1f}M)")
    return layers, make_aux(aux, config), config


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
