# Concord — the winning configuration

The validated, best-on-bench Concord optimizer, distilled to exactly what works.
**32 bits/param**, the full optimizer state in one int32, the weight update fused into
the backward pass.

## What "winning" means (tiny-shakespeare overfit bench, 5000 iters, wd=0, shared seed)

Best validation loss, head-to-head, same bench:

| optimizer | best deployed val |
|---|---|
| AdamW (lr 1e-3, fp32 master + m + v = ~96 b/param) | 1.534 |
| Muon (faithful: NS5 + Nesterov, native fp) | 1.578 |
| **Concord (bare recipe) — deploy the consolidated weight** | **1.526** |

Concord at **32 b/param beats both AdamW and Muon** on this bench. The shipped default is
the **bare recipe** (no knobs); the win comes from two pieces, both included here:

> **Tentatively best (not the shipped default): the "split" un-stranding config.** A separate
> overfit run found that the ratio-coherence variant with `chase_floor_min=0.1` +
> `gf_consol=50` ("split-the-difference": a small chase floor banks borderline-coherent mass
> losslessly, light evaporation trims clearly-dead noise) deploys at **sv 1.517** — the lowest
> deployed val in the study, and ~0.03 below *that run's* bare-nogate (1.550). CAVEAT: it has
> NOT been A/B'd same-seed against the 1.526 headline above (different run/seed; 1.517 vs 1.526
> is within the run-noise band). So it is **promising, not confirmed**. The machinery is shipped
> (`set_ratio_coh`, `set_ratio_coh_floors`, `--gf_consol`, off by default); to reproduce:
> `--ratio_coh --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1 --gf_consol 50`. A clean
> same-seed A/B vs bare-nogate is the open task to promote it from tentative to default.

1. **The validated recipe** = a bare `ConcordLinearPackedB`. No knobs to set. It is
   rank-1 v̂ AdamW (Adafactor row×col E[g²]; `v_scale=0`, `gf_trust_delta_sq=1`,
   `eps=1e-10`, `precond_p=0.5`) + a fixed Wiener coherence gate, all baked as the defaults.
2. **Deploy the consolidated (slow) weight** — for inference/export use
   `layer.consolidated_weight()` = `(s_slow + v_slow)·128·2^exp`, **dropping the transient
   `s_fast`**. `s_fast` carries the noisiest, most overfit-prone recent velocity; the slow
   accumulators are the denoised position. This is worth **~0.04–0.05 val nats**, stable
   from 10.8M → 49M params, and is what makes Concord beat AdamW (whose deployed weight has
   no such denoised slow copy).

## The format (it is NOT "int8" — it's block-float)

One int32 per weight = a **~17-bit signed integer mantissa** on a **shared per-row+col
block-floating-point exponent**:

```
s_fast    int16  (bits 31:16)  fine/low mantissa bits — the velocity (momentum)
s_slow    int8   (bits 15:8)   coarse mid bits   (×128)  — position bearer (chase α~0.1)
v_slow    int8   (bits  7:0)   coarse high bits  (×128)  — long anchor    (leak α_v~0.001)
m_eff  = s_slow·128 + s_fast + v_slow·128                # ONE 17-bit signed mantissa
weight = m_eff · 2^(row_exp + col_exp − bias)            # block-float, shared row+col exp
```

vs bf16 (8-bit mantissa, per-element exponent): Concord trades a per-element exponent for a
**shared** one, and spends the saved bits on a **finer (17-bit) mantissa**. The two int8
fields are the *high bits of one integer*, not three separate int8 quantities — "int8" never
describes the live weight.

## Usage

```python
from concord import ConcordLinearPackedB

layer = ConcordLinearPackedB(in_features, out_features, bias=True)  # drop-in nn.Linear
layer.lr = 5e-4

y = layer(x)
loss = criterion(y, target)
loss.backward()        # the Concord step is FUSED here (no optimizer.step() for this layer)
layer.rebalance()      # per-step block-float envelope retune

# ...after training, DEPLOY the consolidated slow weight (drop s_fast):
W_deploy = layer.consolidated_weight()   # (s_slow + v_slow)·128·2^exp   <-- ship this
```

Non-Linear params (embeddings, LayerNorm) take a small standard optimizer (e.g. AdamW) —
Concord goes on the 2D Linear weights, which is where the parameters (and the win) are.

## Why it works (one paragraph, corrected 2026-06-01)

The slow cascade is **mass-preserving redistribution, not a momentum buffer**: the chase
(`s_fast→s_slow`) and the leak (`s_slow→v_slow`, `mass_preserve_v=True` here) move mantissa
between accumulators with the live weight `m_eff` invariant, so the per-step *update* is the
instantaneous preconditioned gradient (`−lr·g/√v̂`) and the chase rate α is a redistribution
timescale, **not a β1**. (`d_sv = s_slow − v_slow` does correlate +0.87 with `EMA_0.9(grad)`,
but that is a *readable* momentum-like signal — consumed by the Wiener coherence gate and the
consolidated-weight deploy — not a momentum term in the step.) The win is two things, neither
of them momentum: a per-coordinate rank-1 v̂ (Adam-style scaling) + the coherence-gated
**consolidated deploy weight** (drop `s_fast`), which denoises the shipped weight — and it does
so at 1/3 the storage, beating Adam's per-coord step and Muon's orthogonalization on this
(low-rank-gradient) regime. Momentum *can* be injected (non-mass-preserving leak via
`mass_preserve_v=False`, explicit β1, or the d_sv blend) but is OFF here. (Orthogonalization,
nwv coherence-weighting, 2×v_slow deploy, and the d_sv momentum blend were all tested and
rejected; see the project log.)

## Files

- `concord/packed_b.py` — the optimizer (Linear + Conv2d) + fused Triton kernels. The bare
  class IS the validated recipe; `consolidated_weight()` is the deploy path.
- `concord/__init__.py` — public surface.
- `test_baked_defaults.py` — asserts the bare layer = the validated config, and that it trains.
- `requirements.txt` — torch ≥ 2.1 + Triton (CUDA).

## Requirements

PyTorch ≥ 2.1 + Triton, CUDA GPU. Verify with `python test_baked_defaults.py`.
