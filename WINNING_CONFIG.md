# Concord — the winning configuration (exact)

Single source of truth for "what is the validated/confirmed best Concord config." Every
number below is read from committed source (`concord/packed_b.py`, `notebook/src/train_nanogpt.py`,
`notebook/tools/run_split_ab.sh`) and the same-seed A/B log — not from memory. File:line refs are to
`concord/packed_b.py` unless noted.

---

## 0. TL;DR

The winner is a **fluctuation–dissipation pair on top of rank-1 v̂ AdamW**: the baked
`ConcordLinearPackedB` defaults + **split dissipation** (the four gate/evap flags) +
**isotropic noise injection** (the fluctuation), deployed off `consolidated_weight()`.

Ship invocation (the validated full recipe — both halves, all at `--concord_lr 5e-4`):

```
# dissipation (the "split" — friction: coherence-gated evaporation + decaying floors)
--ratio_coh  --ratio_chase_floor_min 0.1  --ratio_leak_floor_min 0.1  --gf_consol 50
# fluctuation (noise — wired into the kernel backward, survives CUDA-graph replay)
--sigmag_iso  --sigmag 0.6
```

Measured on the SAME seed and LR (nanoGPT char; deployed-sv is the decision metric):

| arm                                              | deployed-sv ↓ | Δ vs prev |
|--------------------------------------------------|---------------|-----------|
| ab_nogate — bare recipe (gf_consol=0, no noise)  | 1.5404        | —         |
| ab_consol — **+ split dissipation**              | 1.5180        | −0.022    |
| sf_060 — **+ isotropic noise σ=0.6  (WINNER)**   | **1.4967**    | −0.021    |
| AdamW (32 b/param baseline)                      | ~1.534        |           |
| native Muon                                      | ~1.578        |           |

The full pair lands **1.497 vs AdamW 1.534 / Muon 1.578**, all at **32 bits/param**. The
dissipation step (−0.022) is confirmed deterministic same-seed; the fluctuation step (−0.021)
is single-seed with ~0.01 trajectory jitter (≈ half the effect) — the *mechanism* is wired in
and CUDA-graph-correct, the *magnitude* wants multi-seed validation on the target task.

---

## 1. Storage format — one int32 per weight (the 32 b/param win)

The weight **is** the optimizer state. One `int32` packs three signed integer fields that
share a per-row + per-col block-float exponent (`row_exp + col_exp`, `MANTISSA_BIAS = 15`):

| bits   | field      | type   | role                                    | unpack (L126–129)            |
|--------|------------|--------|-----------------------------------------|------------------------------|
| 31:16  | `s_fast`   | int16  | velocity / fast momentum                | `packed >> 16`               |
| 15:8   | `s_slow`   | int8   | position (chase target, α = 0.1)        | `(packed << 16) >> 24`       |
| 7:0    | `v_slow`   | int8   | long anchor (leak, α_v = 0.001)         | `(packed << 24) >> 24`       |

Effective mantissa and live weight:

```
m_eff  = s_slow·128 + s_fast + v_slow·128        # L129
weight = m_eff · 2^(row_exp + col_exp − 15)
```

Repack (L263–265): `((s_fast & 0xFFFF) << 16) | ((s_slow & 0xFF) << 8) | (v_slow & 0xFF)`.

**Format note (do not call this "int8"):** `m_eff` is a ~17-bit *signed integer* mantissa on a
shared per-row+col exponent — **finer** than bf16's 8-bit mantissa. The int8 fields are the
high bits of one integer, not an independent int8 cascade.

**Mechanism note (the chase is redistribution, not a β1 term):** the α=0.1 chase is a
mass-preserving redistribution between `s_fast`/`s_slow` (the live `m_eff` is invariant across
it; α is a timescale, not a learning gain). The readable momentum-like signal is
`d_sv = s_slow − v_slow` (corr +0.87 with EMA₀.₉(grad)); the gate/deploy *consume* it, the step
does not *add* it.

---

## 2. The bare recipe — baked `__init__` defaults

`ConcordLinearPackedB(in_features, out_features, bias=True, device='cuda', alpha=0.1,
beta1=0.0, lr=0.01)` (L1359). The validated recipe is "rank-1 v̂ AdamW + fixed Wiener
coherence gate," realized entirely by these defaults — **no flags needed for the bare arm**:

| knob                  | value                                  | meaning                                        |
|-----------------------|----------------------------------------|------------------------------------------------|
| `optimizer_kind`      | `'adamw'`                              | AdamW-style step (not SGD-chase)               |
| `weight_decay`        | `0.0`                                  | none                                           |
| `_eps_value`          | `1e-10`                                | preconditioner epsilon                         |
| `step_cap`            | `10.0`                                 | per-step trust clip                            |
| `v_scale`             | `0.0`                                  | rank-1 v̂ only (no separate v accumulator)      |
| `precond_p`           | `0.5`                                  | RMS preconditioner power (Adam = 0.5)          |
| `adafactor_beta2`     | `0.999`                                | row×col E[g²] (Adafactor factoring)            |
| `gf_trust_delta_sq`   | `1.0`                                  | trust-region scale                             |
| `alpha` (chase)       | `0.1`                                  | s_fast→s_slow redistribution timescale         |
| `alpha_v_fast` (leak) | `0.001`                                | s_slow→v_slow long-anchor leak                 |
| `beta1`               | `0.0`                                  | **off** (momentum WIP rides here, default off) |
| `drift_cancel_C`      | `compute_drift_cancel_C(alpha, α_v)`   | analytic leak/chase drift correction           |
| `mass_preserve_v`     | `True`                                 | chase conserves `m_eff`                         |
| `apply_chase`         | `True`                                 | chase enabled                                  |
| `track_rebalance`     | `True`                                 | overflow-rebalance bookkeeping                 |
| `gf_consol`           | `0.0`  *(default — split overrides ↓)* | consolidation evaporation off by default       |
| `_USE_FIXED_COH`      | `True` (module-level, L782)            | Wiener gate `coh = S/(S+noise²)`               |
| `_coh_pre`            | allocated ON (L1447)                   | per-param coherence EMA buffer                 |

Constants: `MANTISSA_BIAS=15`, `S_SLOW_FACTOR=V_SLOW_FACTOR=128`, `MAX_M=24000` (rebalance
threshold). Harness LR: `--concord_lr 0.05`, cosine to `lr_min_frac 0.1` (the `__init__`
`lr=0.01` default is overridden by the harness).

---

## 3. The split = dissipation — the four opt-in flags

The "split" is the **dissipative half** of the recipe: coherence-gated friction that bleeds
transient/noisy mass out of the fast field while protecting confident params. Four changes on
the bare recipe (from `notebook/tools/run_split_ab.sh`):

```
run_arm ab_consol $CONC --ratio_coh \
        --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1 --gf_consol 50
```

1. **`--ratio_coh`** — switch the gate from the `coh_pre` EMA buffer to the **live coherence
   ratio** computed from the packed word each step (kernel `USE_RATIO_COH`, L440/596/619). This
   gates *chase and leak* directly and **drops the `coh_pre` buffer** (`disable_cohpre()`,
   `train_nanogpt.py:395`) → stays at a true **32 bits/param** (no side buffer).

2. **`--ratio_chase_floor_min 0.1`** — the chase ratio-gate floor. The chase gate cosine-decays
   from `ratio_chase_floor=0.9` **down to 0.1** over `ratio_coh_floor_epochs=1.0`
   (`cos_floor(start, it, min)`, `train_nanogpt.py:449`, scheduled per-step L535–536). Floor of
   0.1 (not 0) keeps a minimum chase alive late in training.

3. **`--ratio_leak_floor_min 0.1`** — same, for the leak gate: `ratio_leak_floor=0.999` → **0.1**
   over one epoch.

4. **`--gf_consol 50`** — consolidation evaporation. Each step, low-coherence mass evaporates
   from the `s_fast`/`s_slow` gap back toward the anchor:
   ```
   evap_mantissa = lr · gf_consol · (1 − coh) · d_fs        # L568
   delta_t       = delta_grad − beta1·d_fs − evap_mantissa  # L573
   ```
   i.e. confident (high-coh) params are untouched; noisy (low-coh) params shed transient mass.

**Floors/ratios are device tensors** (`set_ratio_coh_floors` `.fill_()`s them, L837) — required
so a CUDA-graph replay sees the live per-step schedule instead of an iter-0-frozen float.

---

## 4. Deploy / save path — `consolidated_weight()` (drop s_fast)

The weight you **export and evaluate** is the deploy-slow weight, **not** the live `m_eff`:

```
consolidated_weight = (s_slow + v_slow)·128 · 2^(row_exp + col_exp − 15)   # drops s_fast
```

The A/B metric "deployed-sv" = exactly this. Dropping the fast velocity at deploy is what
makes the split win (live `m_eff` for ab_consol is 1.5563; its deploy-sv is **1.5180**). For
SDXL `.safetensors` export, materialize each Concord module via `consolidated_weight()`.

---

## 5. Per-step driver requirements (whoever owns the optimizer slot)

- Call **`rebalance()`** on every Concord module each step (overflow guard at `MAX_M=24000`).
- Advance the **lr device tensor** (`m.lr = lr` / `_lr_buf.fill_()`); for the split, advance the
  **ratio-floor device tensors** per step via the cosine schedule (`set_ratio_coh_floors(...)`).
- For noise, advance the **σ device tensor** per step (`set_sigmag_sigma(σ·(1−lr/lr_peak))`, or
  the constant `σ` if you ablate the schedule) — device tensor so it survives graph replay.
- The swapped layers **self-step in their autograd `Function.backward`** (L1237) — the optimizer
  only steps the **non-swapped** params (norms/biases/embeddings) = the aux-AdamW split.
- All four per-step scalars (lr, the two ratio floors, σ) MUST be device tensors before any
  CUDA-graph capture, or replay freezes them at their iter-0 values (the bug that burned us).

---

## 6. Noise injection — the fluctuation half (wired in)

The fluctuation term that pairs with §3's dissipation. **Isotropic** white noise, peak
`σ = 0.6`, default rising-late schedule (`σ·(1 − lr/lr_peak)`, ≈ constant per the ablation),
injected in the backward off the deploy weight. Flags: `--sigmag_iso --sigmag 0.6`.

**Wired in, not bolted on:** the noise lives in the fused backward kernel (`_SIGMAG_NOISE`
branch, `notebook/src/prototype_packed_b.py` L1421) reading a **device-tensor σ** (`_SIGMAG_SIGMA_T`) so
it survives **CUDA-graph replay** — the per-step schedule `.fill_()`s the tensor; the captured
graph reads the live value, not an iter-0 freeze. Verified graph-correct vs eager (SR-floor gate).

Isotropic σ-sweep on the split config (same seed, `--concord_lr 5e-4`; anchor = split, no-noise
= **1.5180**):

| σ_peak |  0.10  |  0.20  |  0.25  |  0.35  |  0.40  |  0.50  |   0.60   |  0.70  |
|--------|--------|--------|--------|--------|--------|--------|----------|--------|
| dep-sv | 1.5216 | 1.5097 | 1.5145 | 1.5076 | 1.5042 | 1.5158 | **1.4967** | 1.5098 |

Minimum at **σ = 0.6 → 1.4967 (−0.021 vs the 1.5180 anchor)**; the 0.35–0.60 band is all ≤ 1.508.

**Honest caveats (do not drop these):** single-seed, nanoGPT-only; the ~0.01 trajectory jitter
is ≈ half the −0.021 effect, so the magnitude is not yet load-bearing — multi-seed it before
trusting σ on SDXL. The ablation **refuted** structured Σ_g noise's claimed necessity:
isotropic ≥ Σ_g, and rising ≈ constant — so use `--sigmag_iso` (never Σ_g) and don't bother
tuning the schedule. (Likely the noise-doc's CIFAR win was BatchNorm-mediated; nanoGPT is
BN-free, which is why only the plain isotropic kick survives here.)

---

## 7. Provenance

- Storage layout / `m_eff` / evap: `concord/packed_b.py` L126–129, L263–265, L568, L573.
- Baked defaults: `concord/packed_b.py` `__init__` (L1359 sig; defaults L1365–1423).
- Split (dissipation) flags + bench: `notebook/tools/run_split_ab.sh`; schedule `notebook/src/train_nanogpt.py`
  L449, L535–536; live-coh / evap kernel `concord/packed_b.py` L440, L568, L596–619.
- Noise (fluctuation): flags `notebook/src/train_nanogpt.py` L243–249, schedule L529–532; kernel branch
  `notebook/src/prototype_packed_b.py` L1421 (`_SIGMAG_NOISE`/`_SIGMAG_ISO`, device-tensor `_SIGMAG_SIGMA_T`).
- Numbers — all same-seed, `--concord_lr 5e-4`, deterministic:
  - dissipation A/B: `[ab_consol] deployed-sv=1.5180` vs `[ab_nogate]=1.5404` (`compare_out/split_ab.log`).
  - fluctuation sweep on the split config: `[sf_060] deployed-sv=1.4967` (best), full curve in
    `compare_out/sigma_sweep.log` + `compare_out/sigma_fine.log`. Noise ablation:
    `compare_out/noise_ablation.log`.
  Concord is bit-deterministic at fixed seed, so the dissipation Δ is real, not noise; the
  fluctuation Δ carries ~0.01 trajectory jitter (single-seed). Full record: `notebook/notes/CONTROL_PLANE.md`.
- The clean importable winner package: `concord/` (committed main). The split + noise flags are
  applied by the driver/harness; the package defaults are the bare recipe.
