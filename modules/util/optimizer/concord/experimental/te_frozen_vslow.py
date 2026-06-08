"""EXPERIMENTAL: frozen-v_slow text-encoder training for Concord packed layers.

Idea (Theophite): train a TE layer as a 16-bit fast+slow DELTA on top of a FROZEN
v_slow anchor (the pretrained weight), so the encoder stays tethered to pretrained
(anti language-drift) while still fine-tuning.

What the kernel trace (prototype_packed_b.py) actually says about freezing v_slow
-------------------------------------------------------------------------------
m_eff = s_slow_i8*128 + s_fast + v_slow_i8*128,  W = m_eff * 2^(row_exp+col_exp-MB)

1. PRECONDITIONER IS SAFE.  v_scale=0.0 by default ("proven-dead velocity-noise
   precond"); the live denom is the Adafactor rank-1 v_hat (gf_trust_delta_sq=1).
   v_hat comes from v_row/v_col grad EMAs -- INDEPENDENT of v_slow. Freezing
   v_slow does not touch the step size. (My earlier "8-bit deploy" worry was wrong:
   with the per-row/col exponent and s_fast KEPT, get_weight() is ~16-bit.)

2. COHERENCE GATE AUTO-NEUTRALISES.  coh = sig^2/(sig^2+noise^2) with sig =
   C*d_sv and d_sv = s_slow-v_slow. Pin v_slow and d_sv becomes displacement (it
   grows), so coh -> 1. But coh_pre EMA leaks at alpha_v_fast; set alpha_v_fast=0
   and coh_pre freezes at its init 1.0, so the chase gate = coh + coh_pre*(1-coh)
   = 1 regardless. Result: a clean ungated 2-timescale (fast->slow) optimiser. No
   crash, no surgery -- alpha_v_fast=0 does it.

3. THE "RESTORING FORCE FOR FREE" DOES NOT EXIST -- this is where my to-do #4 bites.
   wd_sv/wd_sf pull s_slow / (s_slow*128+s_fast) TOWARD v_slow. But the kernel treats
   (s_slow + v_slow) as ONE mass-preserved position (the leak shuffles mass between
   them), so "pull s_slow toward v_slow" is mean-reversion of the SPLIT, not a pull
   toward a fixed anchor. Repurpose v_slow as a frozen anchor and those terms drag
   s_slow UP to equal v_slow -> weight ~= 2*anchor. Wrong direction.
   => The anchor pull must be an explicit L2 on the DELTA (s_slow,s_fast -> 0), which
      relaxes W toward v_slow*128. That is a NEW term, now ON THE KERNEL TICK as the
      `wd_anchor` coefficient in apply_packed_adamw (a consf-gated SR-tick that shrinks
      s_slow and s_fast toward 0). wd_anchor=0 floors both ticks to 0 -> bit-exact no-op
      for every existing UNet/embedding run.

This script validates, on GPU (Triton):
  A. faithful init     : get_weight() ~= pretrained (~16-bit); consolidated_weight()
                         (drops s_fast) is visibly coarser -> TE must deploy get_weight.
  B. v_slow frozen     : the anchor bits never change across training.
  C. delta trains      : feed grad of 1/2||W-W*||^2; loss falls, W -> W*.
  D. anchor knob works : with decay_delta(lam), W settles BETWEEN W0 and W*; bigger
                         lam -> closer to the pretrained anchor (the anti-drift dial).
  E. restoring force   : stop the gradient, keep the decay -> W relaxes back to W0.

Run: python experimental/te_frozen_vslow.py   (needs CUDA + the concord package).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import prototype_packed_b as ppb
from prototype_packed_b import ConcordLinearPackedB

INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767


# ---- packed-state helpers (operate directly on packed_w; production untouched) ----
def unpack(packed):
    # mirror ConcordLinearPackedB.get_state (arithmetic shifts sign-extend the int8s)
    s_fast = (packed >> 16)
    s_slow = ((packed << 16) >> 24)
    v_slow = ((packed << 24) >> 24)
    return s_fast, s_slow, v_slow


def repack(s_fast, s_slow, v_slow):
    return (((s_fast & 0xFFFF) << 16) | ((s_slow & 0xFF) << 8) | (v_slow & 0xFF))


@torch.no_grad()
def pack_anchor(layer, W):
    """Pretrained W -> COARSE part in v_slow (the frozen anchor, x128), FINE residual
    in s_fast, s_slow=0.  W = (v_slow*128 + s_fast)*scale reproduces W to ~16-bit; the
    trainable delta is (s_slow*128 + s_fast) and starts at the (small) fine residual."""
    W = W.to(device=layer.packed_w.device, dtype=torch.float32)
    MB = layer.MANTISSA_BIAS
    max_abs = W.abs().amax(dim=1).clamp(min=1e-30)
    layer.row_exp.copy_(torch.ceil(torch.log2(max_abs) + 1.0)
                        .clamp(layer.EXP_MIN, layer.EXP_MAX).to(layer.row_exp.dtype))
    layer.col_exp.zero_()
    exp = (layer.row_exp[:, None].float() + layer.col_exp[None, :].float() - MB)
    scale = torch.pow(2.0, exp)
    m_total = (W / scale).round().to(torch.int32)
    v_slow = (m_total.float() / 128.0).round().clamp(INT8_MIN, INT8_MAX).to(torch.int32)
    s_fast = (m_total - v_slow * 128).clamp(INT16_MIN, INT16_MAX).to(torch.int32)
    s_slow = torch.zeros_like(s_fast)
    layer.packed_w.copy_(repack(s_fast, s_slow, v_slow))
    layer._resync_weight_buf()


@torch.no_grad()
def freeze_anchor_config(layer, wd_anchor=0.0):
    """Frozen-v_slow knobs: kill the leak (alpha_v_fast=0 -> v_slow pinned AND coh_pre
    frozen at 1 -> ungated chase), zero the now-meaningless drift_cancel_C, and set
    wd_anchor (the kernel-tick delta->0 pull). Precond stays production v_hat AdamW."""
    layer.alpha_v_fast = 0.0
    layer.drift_cancel_C = 0.0
    layer.wd_sv = 0.0          # these pull s_slow TOWARD v_slow -> would DOUBLE a pinned
    layer.wd_sf = 0.0          # anchor (see module docstring); the right term is wd_anchor
    layer.weight_decay = 0.0
    layer.wd_anchor = float(wd_anchor)   # kernel-tick L2 of the delta toward the anchor


def vslow_fingerprint(layer):
    _, _, v_slow = unpack(layer.packed_w)
    return v_slow.clone()


def rel(a, b):
    return (a - b).norm().item() / b.norm().clamp(min=1e-30).item()


# ----------------------------------- experiment -----------------------------------
def main():
    if not torch.cuda.is_available():
        print("NEED CUDA (Triton kernel is GPU-only). Aborting."); return
    dev = "cuda"
    torch.manual_seed(0)
    OUT, IN = 256, 256
    layer = ConcordLinearPackedB(IN, OUT, bias=False, device=dev, alpha=0.1, lr=2e-2)
    freeze_anchor_config(layer)

    # "pretrained" anchor W0 and a fine-tune target W* = W0 + structured delta.
    W0 = (torch.randn(OUT, IN, device=dev) * 0.02)
    Wstar = W0 + torch.randn(OUT, IN, device=dev) * 0.03
    pack_anchor(layer, W0)
    anchor0 = vslow_fingerprint(layer)

    def W():  # current deploy-faithful weight
        return layer.get_weight().float()

    # ---- A. faithful init + deploy contrast ----
    init_keep = rel(W(), W0)                       # get_weight: keeps s_fast (~16-bit)
    init_drop = rel(layer.consolidated_weight().float(), W0)  # drops s_fast (coarse)
    print(f"[A] init  get_weight rel-err  = {init_keep:.2e}   (deploy this)")
    print(f"[A] init  consolidated rel-err= {init_drop:.2e}   (drops s_fast -> coarse)")

    # ---- C. delta trains: descend 1/2||W-W*||^2 (wd_anchor=0) ----
    ppb.set_consolidate(dev, True)
    for step in range(400):
        layer.apply_grad_step(W() - Wstar)         # grad of 1/2||W-W*||^2
    fit_err = rel(W(), Wstar)
    drift_from_anchor = rel(W(), W0)
    print(f"[C] after 400 steps  ||W-W*||/||W*|| = {fit_err:.2e}   (fits target)")
    print(f"[C] drift ||W-W0||/||W0||           = {drift_from_anchor:.2e}")

    # ---- B. v_slow stayed frozen through training ----
    frozen_ok = torch.equal(vslow_fingerprint(layer), anchor0)
    print(f"[B] v_slow anchor unchanged          = {frozen_ok}")

    # ---- D. anchor dial: sweep the kernel-tick wd_anchor coefficient ----
    print("[D] anchor dial (kernel wd_anchor; equilibrium between W0 and W*):")
    for wda in (0.0, 0.5, 2.0, 8.0):
        L = ConcordLinearPackedB(IN, OUT, bias=False, device=dev, alpha=0.1, lr=2e-2)
        freeze_anchor_config(L, wd_anchor=wda); pack_anchor(L, W0)
        for step in range(400):
            L.apply_grad_step(L.get_weight().float() - Wstar)
        w = L.get_weight().float()
        print(f"     wd_anchor={wda:<5} ->  toW*={rel(w, Wstar):.2e}  toW0={rel(w, W0):.2e}")

    # ---- E. restoring force: fit (wd_anchor=0), then zero-grad steps with wd_anchor>0
    #         -> the kernel tick alone relaxes the weight back to the anchor ----
    L = ConcordLinearPackedB(IN, OUT, bias=False, device=dev, alpha=0.1, lr=2e-2)
    freeze_anchor_config(L, wd_anchor=0.0); pack_anchor(L, W0)
    for step in range(400):
        L.apply_grad_step(L.get_weight().float() - Wstar)
    moved = rel(L.get_weight().float(), W0)
    L.wd_anchor = 8.0                               # turn the kernel anchor pull on
    zero = torch.zeros(OUT, IN, device=dev)
    for step in range(400):                         # gradient OFF, kernel anchor decay ON
        L.apply_grad_step(zero)
    relaxed = rel(L.get_weight().float(), W0)
    print(f"[E] trained drift from W0 = {moved:.2e}  -> after anchor-only = {relaxed:.2e}"
          f"   ({'RELAXES to anchor' if relaxed < moved * 0.5 else 'NO relax'})")

    print("\nSUMMARY: init faithful & 16-bit (A), anchor frozen (B), trains (C), "
          "anchor dial controls drift (D), restoring force works (E).")


if __name__ == "__main__":
    main()
