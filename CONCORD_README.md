# Concord AdamW

Storage-efficient AdamW: **40 bits per parameter** of optimizer state vs
standard AdamW's 96 bits/param, with comparable or slightly better
accuracy on the configs we've tested.

The state is three int accumulators per weight element (16+16+8 bits)
plus shared per-row and per-col int8 exponents amortised to ~0 bits/param.
The bf16 weight is never persisted — it's reconstructed each forward
from the int state into a transient buffer, fed to cuBLAS/cuDNN for the
matmul, and freed after backward.

## Memory-profile menu

Three paths, picked per layer:

| Path | State / param | Notes |
|---|---|---|
| **SGD chase** (two-accumulator) | 32 bits | Just s_slow + s_fast. No AdamW preconditioning. Use when you don't need a variance signal — the cheapest option. |
| **AdamW `v_rank1`** | 32 bits + O(N+K) fp32 / layer | s_slow + s_fast for the weight; Adafactor-style rank-1 v_row[N] / v_col[K] for the variance. Lowest-memory AdamW; preferred for very large layers where a full per-element v_slow_i8 buffer is too costly. **Don't combine with `enable_v_slow_i8`** — the two variance sources compete on the same timescale and the combination ran below baseline. |
| **AdamW `three_accum`** (default) | 40 bits | s_slow + s_fast + int8 v_slow_i8. The drift-cancelled noise residual is the variance source. Best accuracy in our CIFAR runs; the path the headline number below is from. |

## The three accumulators

For each weight matrix `W ∈ ℝ^(N×K)`, the layer holds:

| Buffer | Shape | dtype | What it is |
|---|---|---|---|
| `s_slow` | (N, K) | int16 | Medium-time-constant mantissa |
| `s_fast` | (N, K) | int16 | Fast mantissa (gets the gradient tick) |
| `v_slow_i8` | (N, K) | **int8** | Long-time-constant mantissa, shifted scale |
| `row_exp` | (N,) | int8 | Per-row binade |
| `col_exp` | (K,) | int8 | Per-col binade |

The live bf16 weight is

```
w[i,j] = (s_slow[i,j] + s_fast[i,j] + v_slow_i8[i,j] · V_SLOW_FACTOR)
         · 2^(row_exp[i] + col_exp[j] − mantissa_bias)
```

`V_SLOW_FACTOR = 128 = 2^7` is the **shifted scale**: each int8 unit of
v_slow_i8 represents 128 mantissa units, so its dynamic range is
±128·128 = ±16384 in mantissa-units terms — matching s_slow/s_fast at
int16 (which work at MAX_M = 24000) for a 1/3-of-the-bits storage cost.

Total state per param: 16 + 16 + 8 = **40 bits**. Compare AdamW's
3×fp32 (param + m + v) = 96 bits/param.

## The update mechanics

### Per step (inside one Triton kernel)

1. **gradient → s_fast**: SR-rounded tick of `−lr · step_live · scale_inv`
   onto s_fast.
2. **chase**: SR-rounded tick of `α · (s_fast − s_slow)` onto s_slow, with the
   same amount taken back out of s_fast — i.e. **mass-preserving redistribution**
   (the live weight is invariant). Default α=0.1 → redistribution time constant
   ≈10 steps; this is NOT a β1 momentum (the chase accumulates no gradient).
   Any momentum in this recipe comes from the *non*-mass-preserving v_slow leak
   (next), not the chase; explicit β1 is left at 0.
3. **v_slow leak**: SR-rounded tick of `α_v_fast · (s_fast − v_slow·F) / F`
   onto v_slow_i8. Default α_v_fast=0.001 → time constant ≈1000 steps.
   Non-mass-preserving (chase-style: the live weight grows by the leak).
4. **Bayesian-anchored weight decay** (optional, see below).
5. Saturate everything to int range and store.

### Per `rebalance_every` steps (default 8)

A separate kernel pass: if `|s_slow+s_fast|` exceeds MAX_M on any row or
column, tick the corresponding `row_exp` / `col_exp` up by 1 and
SR-right-shift `s_slow`, `s_fast`, **and** `v_slow_i8` to keep the live
weight invariant. A defensive int8-specific check fires the same
rebalance if `|v_slow_i8|` exceeds `V_SLOW_I8_MAX=96` (75% of int8 cap),
so the int8 buffer can't silently saturate when s_slow+s_fast hasn't yet
hit MAX_M.

### `step_live`: the AdamW preconditioned step direction

The three-accumulator design's signature contribution. The variance
estimator is a **drift-cancelled noise residual**:

```
noise   = (s_fast − s_slow) − C · (s_slow − v_slow·F)
v_proxy = (noise · scale_fwd)² · v_scale
step    = grad / sqrt(v_proxy + eps) + weight_decay · current_weight
step    = clamp(step, ±step_cap)
```

`(s_fast − s_slow)` alone is the velocity, but in a drift regime
(constant gradient pressure) it tracks `⟨g⟩`, not `⟨g²⟩` — so squaring
it gives a *biased* variance estimate. The high-pass term
`(s_slow − v_slow·F)` carries the same drift component at a longer time
constant; subtracting `C` times it removes the drift from velocity,
leaving the high-frequency *noise* component. `noise²` is then the
unbiased per-element second-moment estimator.

`C` is auto-derived from the steady-state drift balance. Let
`L = (1−α)/α` (the d_fs drift-lag ratio) and `ρ = α_v_fast + α_v_slow/T_r`
(effective per-step v-leak). Solving E[noise]=0 under pure drift:

```
C* = L·ρ / (1 − L·α_v_fast)
   = (1−α)·ρ / (α − (1−α)·α_v_fast)
   ≈ 0.0204 at canonical OneTrainer defaults (T_r=8, α_v_fast=0.001,
                                                α_v_slow=0.01, α=0.1)
   ≈ 0.0091 at prototype-B defaults (α_v_slow=0, α_v_fast=0.001, α=0.1)
```

(Previous shipped default 0.1 was 5×–11× too large depending on whether
the periodic v-refit was active. The previous README formula above had
`T_r·α_v_fast` in both numerator and denominator where it should have
`(1−α)` and `(1−α)·α_v_fast` respectively, which produced ≈ 0.196 —
10× too large. With C off, drift-cancel fails: the noise estimate
picks up the signal as well, flattening the per-weight variance map
and breaking the "is this weight converged?" diagnostic the estimator
is structurally capable of producing.)

## Two flavours of weight decay

The standard wd term `weight_decay · current_weight` shrinks the **live
weight** toward 0 each step. `current_weight` includes v_slow_i8's
contribution, so wd shrinks all three accumulators proportionally
(s_fast directly via the step, s_slow and v_slow via chase + leak).
That's standard decoupled AdamW wd, applied at the mantissa level.

The **Bayesian-anchored** wd is novel. Two extra knobs:

```
s_slow −= SR(lr · wd_sv · (s_slow − v_slow_full))
s_fast −= SR(lr · wd_sf · (s_fast − v_slow_full))
```

Rationale: `v_slow_full = v_slow_i8 · V_SLOW_FACTOR` is the long-time
averaged gradient signal — the part of the weight that is *supported by
the training distribution* (averaged over many gradients, low variance).
The offsets `(s_fast − v_slow_full)` and `(s_slow − v_slow_full)` are
the *less-confirmed transients* — high-frequency signal that hasn't yet
been integrated into the long-time average. A small decay of these
offsets toward `v_slow_full` shrinks the part of the weight not yet
justified by the long-time gradient history, while leaving the
well-supported part alone. Per-element confidence-weighted weight decay:
less decay where the data has spoken, more decay where it hasn't.

`wd_sv = wd_sf = 1e-5` was the empirical optimum on bigger CIFAR. At
1e-4 the damping is too strong (caps the peak by anchoring too tightly).
At 1e-5 it slightly raises the peak accuracy while leaving the chase
dynamics intact.

## Forward / backward path

```
forward(x):
    1. Reconstruct bf16 weight from
       (s_slow, s_fast, v_slow_i8, row_exp, col_exp)
       into a transient HBM buffer (one Triton kernel pass).
    2. y = F.linear(x, weight, bias)        # cuBLAS
       or F.conv2d(x, weight, bias, ...)    # cuDNN
    3. Save weight + x in ctx for backward.

backward(grad_y):
    1. grad_x = matmul(grad_y, weight)              # cuBLAS
       or conv2d_input(...)                          # cuDNN
    2. Fused Triton grad_W + update kernel:
       computes grad_W = grad_y.T @ x inline (no HBM grad_W)
       applies the full three-accumulator chase update on
       (s_slow, s_fast, v_slow_i8) in place.
```

The bf16 weight is transient. cuDNN and cuBLAS expect a real bf16 tensor
as input — Triton fused-recon-and-matmul kernels are competitive at
SDXL-scale tile sizes but slower than cuDNN/cuBLAS at small-batch
CIFAR-scale work.

## Results

CIFAR-10 / `WiderConvNet` (3.2M params, BatchNorm), 80 epochs, bsz=32,
seed 0:

| Optimizer | wd | Best | Final | Bits/param |
|---|---|---|---|---|
| Vanilla `torch.optim.AdamW` lr=1e-3 | 0 | 90.25 | 90.17 | 96 |
| Vanilla `torch.optim.AdamW` lr=1e-3 | 0.01 | 90.65 | 90.49 | 96 |
| Concord int8 three_accum | 0 | 90.21 | 90.21 | 40 |
| Concord int8 three_accum | 0.01 | 90.84 | 90.71 | 40 |
| **Concord + wd=0.01 + wd_sv=wd_sf=1e-5** | **0.01** | **90.91** | **90.66** | **40** |

Same seed across all rows; differences are stable in our runs but small
enough that the relative ordering between the top two concord configs
is within seed noise.

At bsz=128 the per-launch overhead amortises and concord matches vanilla
AdamW wallclock (~2.3 vs ~2.25 s/ep).

## Quick-start

The canonical reproducible example:

```bash
# Set CIFAR_DATA_DIR to your CIFAR-10 dir (or pass --data_dir).
export CIFAR_DATA_DIR=/path/to/cifar_data    # one-time, or use --data_dir
python cifar_concord_adamw.py
# → ~16 min on a 4090; reproduces 90.91/90.66 from the results table.
# Try `--batch_size 128` for the ~6 min variant.
```

Read `cifar_concord_adamw.py` for the minimum boilerplate needed in
your own training loop. The substantive part is just two lines per
wrapped Linear/Conv2d:

```python
m.enable_v_slow_i8()
m.set_optimizer_kind("adamw", weight_decay=0.01, eps=1.0)
# set_optimizer_kind installs the canonical three-accumulator defaults
# (v_scale=1, drift_cancel_C=C* via compute_drift_cancel_C,
# alpha_v_fast=0.001, alpha_v_slow=0.01, wd_sv=wd_sf=1e-5).
```

For OneTrainer with TrainConfig:

```python
config.optimizer.optimizer = Optimizer.CONCORD_SGD
config.optimizer.concord_aux_optimizer = 'adamw'
config.optimizer.weight_decay = 0.01
```

For pure SGD-chase (no AdamW preconditioning, 32 bits/param state):
don't call `set_optimizer_kind` and don't `enable_v_slow_i8` —
defaults give the classic two-accumulator chase.

## Knob reference

Read from `config.optimizer.concord_*` (or set directly on each layer).
Defaults shown are the empirical champion on bigger CIFAR.

| Field | Default | Notes |
|---|---|---|
| `concord_alpha` | 0.1 | chase rate; mass-preserving redistribution time constant ≈10 steps (NOT a β1 — the chase carries no momentum) |
| `concord_aux_lr` | base | lr for biases/norms/embeddings (the aux optimizer) |
| `concord_aux_optimizer` | `'adamw'` | or `'sgd'`; for SDXL use adamw |
| `concord_rebalance_every` | 8 | steps between exponent rebalances |
| `concord_refit_every` | 0 | 0 = off; periodic per-row/col exponent refit |
| `concord_refit_target` | 16384 | target |mantissa| for refit |
| `concord_alpha_v_fast` | 0.001 | per-step v_slow ← s_fast leak rate |
| `concord_alpha_v_slow` | 0.01 | per-rebalance v_slow ← s_slow leak |
| `concord_drift_cancel_C` | auto (C*) | high-pass coefficient. Auto = compute_drift_cancel_C from rates (≈0.0204 at OneTrainer defaults, ≈0.0091 at prototype-B defaults). Pass an explicit float to override. |
| `concord_v_scale` | 1.0 | temperature on the preconditioner |
| `concord_v_lr_scale` | 0.2 | lr multiplier for AdamW linears (per-layer) |
| `concord_v_eps` | 1.0 | denominator floor; O(1) is the mantissa-units natural floor |
| `concord_v_step_cap` | 10.0 | per-element step magnitude cap |
| `concord_wd_sv` | 1e-5 | Bayesian-anchored wd on (s_slow − v_slow_full) |
| `concord_wd_sf` | 1e-5 | Bayesian-anchored wd on (s_fast − v_slow_full) |
| `concord_qtridiag` | True | Q-aware tridiagonal coupling on MLP up/down boundaries |
| `concord_qtridiag_pairs` | None | regex; default filters to ff.net.* + linear_1→linear_2 |
| `concord_qt_refresh` | 3000 | steps between Q rebuilds |
| `concord_target_modules` | `'.*'` | regex of paths to wrap |
| `weight_decay` | 0.01 | standard decoupled wd on live weight |

## Training text encoders + embeddings

The shipped `ConcordTrainer` supports four targets, picked independently
via the standard OneTrainer config fields:

| `config.…` field | What it does |
|---|---|
| `text_encoder.train = True` | Wrap **all** of CLIP-L's Linears in Concord. Biases / norms / token_embedding → aux AdamW. |
| `text_encoder_2.train = True` | Wrap **all** of OpenCLIP-G's Linears in Concord. Same aux split. |
| `text_encoder.train_embedding = True` (and `.train = False`) | TE1 stays frozen; only `text_model.embeddings.token_embedding.weight` is unfrozen and goes to aux AdamW. Cheapest TE-side path. |
| `text_encoder_2.train_embedding = True` (and `.train = False`) | Same for TE2. |
| `text_encoder.learning_rate` / `text_encoder_2.learning_rate` | Per-component LR. If unset, falls back to `learning_rate` (UNet LR). For SDXL TE training, ~1e-5 is usually appropriate vs. ~1e-4 for the UNet. |

When any of these flags is true, the trainer flips into **live-TE
mode**: the latent / text cache stores tokenised `input_ids` instead of
TE outputs, and the train step runs the TE forward each iteration with
gradients enabled. Frozen-TE caches and live-TE caches don't collide on
disk — the cache key includes the mode flag.

**Memory at SDXL scale** (CLIP-L 123M + OpenCLIP-G 695M Linear params):

| Setup | TE state size | Notes |
|---|---|---|
| TEs frozen | 0 | The headline UNet-only case. |
| Both TEs full-trained, three_accum | ~4.1 GB | vs. ~9.8 GB fp32 AdamW. |
| Both TEs full-trained, v_rank1 | ~3.3 GB + small | The cheapest TE-train path. |
| Embeddings only (both TEs), default | ~810 MB aux | Full-vocab `(49408, 768) + (49408, 1280)` ≈ 101M-param matrix at fp32 m+v through aux AdamW. |
| Embeddings only (both TEs), `concord_wrap_embeddings=True` | ~400 MB | Same matrix on Concord int storage (32 bits/param). Default OFF; set the flag to opt in. |
| Embeddings only, `concord_wrap_embeddings=True` + `enable_v_slow_i8()` | ~500 MB | Three-accumulator variant; same drift-cancel variance as the Linear three_accum path. |

**Embedding-only caveat**: the v1 wiring trains the **entire**
`token_embedding` matrix (every token, not just newly added ones). True
textual inversion (only the new-token rows train) needs
`AdditionalEmbeddingWrapper` plumbing that ConcordTrainer doesn't port
yet. If you set `train_embedding=True` you'll get useful results, but
on a per-token-row basis it's "all tokens drift" rather than
"only these tokens drift".

**Concord-on-Embedding** (`concord_wrap_embeddings = True`): the
`(vocab, dim)` matrix lives in the same int16 s_slow + s_fast + per-row
exponent format as a Concord Linear, at **32 bits/param**. Forward is
a gather kernel that reconstructs only the indexed rows to bf16
(transient, ~`batch · seq · dim · 2` bytes — tiny). Backward groups
per-occurrence gradients by unique token (`torch.unique` +
`index_add_`) and then runs a sparse SR-tick + chase update over the
touched rows only. Untouched rows are byte-identical after backward
(verified by `test_concord_embedding.py:test_sparse_backward`).

For SDXL with `concord_wrap_embeddings = True`, the combined ~101M-param
token-embedding matrix drops from ~810 MB fp32 AdamW state to **~400 MB
int** (concord-storage at 32 bits/param). With three-accumulator
(`enable_v_slow_i8`) it's **~500 MB** at 40 bits/param. Either way,
1.5–2× cheaper than the aux-AdamW path it replaces. Default is `False`
so existing OneTrainer configs that train embeddings keep the standard
fp32 path until the user opts in.

**Don't combine** `train=True` with `train_embedding=True` on the same
encoder — it's redundant (the full TE includes the embedding). The
trainer treats `train=True` as authoritative.

## Gradient accumulation (Concord-native)

Concord has no `param.grad` to accumulate into — the SR-tick on
`s_fast` happens inside the backward kernel. Standard PyTorch grad
accumulation (`K * loss.backward()` then `opt.step()`) doesn't map
directly, so Concord has its own primitive:

```python
opt.set_accum_steps(K)        # once at training start
for effective_step in range(N):
    opt.zero_grad()
    for k in range(K):
        loss = compute(microbatch) / K
        loss.backward()
        opt.advance_accum()    # bumps the cycle position
    opt.step()                 # aux AdamW + rebalance/refit/etc.
```

On microbatches 0..K-2 the backward kernel SR-ticks `s_fast` and
skips the chase + v_slow leak. On the K-th microbatch it fires the
full update — chase + leak + Bayesian-anchored wd — over the
accumulated drift. `zero_grad()` resets the cycle position; `step()`
runs the aux AdamW step and the periodic rebalance/refit/BMA
cadences once per effective step.

**The noise property.** SR-rounding is unbiased per draw:
`E[round_sr(f)] = f` with `Var ≤ 0.25` regardless of `|f|`. So K
ticks of `g/K` versus one tick of `g`:

| | E[sum] | Var[sum] |
|---|---|---|
| K ticks of `g/K` | `g` | `K × 0.25` |
| 1 tick of `g` | `g` | `0.25` |

Same mean; K× the SR variance. Each microbatch's individual SR
decision — whether each weight ticks up or down by an LSB — is
preserved in the `s_fast` bit pattern. A vanilla fp32 grad
accumulator averages that structure away.

**Memory benefit.** Concord has no per-parameter grad buffer (the
SR-tick mutates `s_fast` in place), so K-accumulation costs the same
as K=1 on the optimizer-state side. For a UNet that would otherwise
need 10 GB of fp32 `.grad` to run vanilla-AdamW grad accumulation,
this is the savings.

**Coverage.** `APPLY_CHASE` is wired through the SGD-chase Linear /
Conv2d (both the fused and apply-update kernel paths), the
three-accumulator AdamW Linear path, and the embedding sparse-update
path. The `v_rank1` AdamW variant currently chases on every
microbatch (no chase-skip wiring); v_rank1 + grad accumulation falls
back to "K full small steps at lr/K", which is still valid SGD-with-
preconditioner but doesn't get the Concord-native noise-preservation
property.

Smoke test: `test_concord_grad_accum.py`.

## Notes for SDXL

- **Use cuDNN/cuBLAS forward path** (the default in the current code).
  The fused-recon-and-matmul Triton kernels are competitive on SDXL UNet
  tile sizes but cuDNN/cuBLAS are still faster at typical bsz=1-4.
- **Bsz dominates wall**. At bsz=32 the per-launch overhead (~18 µs per
  Triton kernel × ~50 launches per step) is ~30% of wallclock. At
  bsz≥128 it's negligible and concord matches vanilla AdamW wall.
- **HBM transient cost**: the materialised bf16 weight buffer per
  wrapped layer. For SDXL UNet, ~5 GB peak transient on top of ~10 GB
  of int state — fits a 24 GB card.
- **`concord_aux_optimizer = adamw`** for SDXL. The SGD-aux finding is
  CIFAR-specific (5 bias params at lr=0.05 don't generalise).
- **`concord_qtridiag_pairs` default filter is important**. Auto-discovery
  on a UNet would have caught ~573 spurious conv/QKV pairs.

## File index

| File | What's in it |
|---|---|
| `concord_triton_fused.py` | Triton kernels + autograd Functions; the optimizer's hot path |
| `concord_triton.py` | Rebalance kernel + bf16 weight recon kernel + helper ticks |
| `concord_linear_fused.py` | `ConcordLinearFused` / `ConcordConv2dFused` modules |
| `concord_optimizer.py` | OneTrainer-shaped wrapper, qtridiag, BMA, Polyak (optional) |
| `onetrainer_concord_patch.py` | `install()` + monkey-patches for OneTrainer integration |
| `concord_polyak.py` | PolyakHypothesis + BoxVelocityMean (off by default) |
| `cifar_concord_adamw.py` | **Canonical reference** — runs with no flags, reproduces 90.91 |
| `_tmp_cifar_bigger_vanilla.py` | Vanilla `torch.optim.AdamW` baseline on the same network |
