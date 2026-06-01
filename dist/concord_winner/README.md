# Concord — the winning configuration

The validated, best-on-bench Concord optimizer, distilled to exactly what works.
**32 bits/param**, the full optimizer state in one int32, the weight update fused into
the backward pass.

## What "winning" means (tiny-shakespeare overfit bench, 5000 iters, wd=0, shared seed)

Best validation loss, head-to-head, same bench:

| optimizer | best val |
|---|---|
| AdamW (lr 1e-3, fp32 master + m + v = ~96 b/param) | 1.534 |
| Muon (faithful: NS5 + Nesterov, native fp) | 1.578 |
| **Concord — deploy the consolidated weight** | **1.526** |

Concord at **32 b/param beats both AdamW and Muon** on this bench. The win comes from
two pieces, both included here:

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

## Why it works (one paragraph, earned this session)

Concord is a **momentum optimizer in disguise**: the slow cascade reconstructs a β1≈0.9
gradient momentum as a difference of EMAs — `d_sv = s_slow − v_slow` aligns +0.87 with a
true `EMA_0.9(grad)`. The two int8 accumulators at different leak rates *are* a momentum
buffer. On this (low-rank-gradient) regime, that denoised momentum + per-coordinate v̂
scaling beats both Adam's per-coord-only step and Muon's spectral orthogonalization — and it
does so at 1/3 the storage. (Orthogonalization, nwv coherence-weighting, and 2×v_slow
deployment were all tested and rejected; see the project log.)

## Files

- `concord/packed_b.py` — the optimizer (Linear + Conv2d) + fused Triton kernels. The bare
  class IS the validated recipe; `consolidated_weight()` is the deploy path.
- `concord/__init__.py` — public surface.
- `test_baked_defaults.py` — asserts the bare layer = the validated config, and that it trains.
- `requirements.txt` — torch ≥ 2.1 + Triton (CUDA).

## Requirements

PyTorch ≥ 2.1 + Triton, CUDA GPU. Verify with `python test_baked_defaults.py`.
