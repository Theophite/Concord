# How Concord works

The mechanism deep-dive. For the condensed, winning-configuration-centered summary see
[`SDXL_WINNER_REPORT.md`](SDXL_WINNER_REPORT.md); for the exact validated numbers and
their provenance see [`WINNING_CONFIG.md`](WINNING_CONFIG.md).

Code references: `concord/packed_b.py` is the distilled winner package on `main`;
the production kernel in the OneTrainer fork is
`modules/util/optimizer/concord/prototype_packed_b.py` on `concord-integration`
(same math, plus the gradient-accumulation `consf` gate, `wd_anchor`, and the noise
branch). Line references below are to `concord/packed_b.py` unless marked
*(integration)*.

---

## 1. The problem it solves

A standard AdamW full fine-tune stores, per parameter: the weight (2 bytes in bf16), an
fp32 master copy (4), and two fp32 moments (8) — ~14–16 bytes/param. For the SDXL UNet
(~2.6 B params) that is far beyond a 24 GB card, which is why consumer fine-tuning
defaults to LoRA.

Concord stores **4 bytes/param, total** — the weight and the entire optimizer state in
one int32 — and gets Adam-class adaptive behavior out of it. A full SDXL UNet fine-tune
runs in ~15 GB.

## 2. The storage format

One int32 per weight, three signed integer fields sharing a per-row + per-column
block-float exponent (unpack L461–465, repack L656–660):

```text
bits 31..16   s_fast   int16   velocity — catches each step's gradient
bits 15..8    s_slow   int8    consolidated position (×128)
bits  7..0    v_slow   int8    long-time anchor (×128)

m_eff  = 128·s_slow + s_fast + 128·v_slow            # ~17-bit signed mantissa
weight = m_eff · 2^(row_exp + col_exp − 15)          # MANTISSA_BIAS = 15
```

Three derived quantities organize everything that follows:

```text
live weight   W = (s_fast + 128·s_slow + 128·v_slow) · scale    # what the forward uses
deploy weight P = (         128·s_slow + 128·v_slow) · scale    # what gets saved
velocity      u = W − P = s_fast · scale
telescope     d = 128·(s_slow − v_slow)                          # drift record (gate input)
```

Notes on the format:

- `m_eff` is a ~17-bit integer mantissa on a shared exponent — *finer* than bf16's 8-bit
  mantissa. Do not read the int8 fields as an "int8 optimizer"; they are the high bits of
  one integer.
- The exponents are per-row + per-column (`row_exp[N]`, `col_exp[K]`, int8 each) — the
  same factorization trick AdaFactor uses for the second moment, applied to dynamic
  range. Cost: O(N+K), negligible.
- The only optimizer state outside the word is the AdaFactor pair `v_row[N]`, `v_col[K]`
  (fp32) per layer.

## 3. The update rule

Per weight, the optimizer is a noisy, damped, driven particle — a fluctuation–dissipation
pair around a preconditioned gradient flow:

```text
g̃   = g + σ·‖g‖·ξ ,   ξ ~ N(0, I)                   # fluctuation
v̂   ← rank-1 EMA of g̃²   (AdaFactor, β₂ = 0.999)    # preconditioner
coh = μ² / (μ² + (u − μ)²) ,   μ = C*·d              # Kalman gain: SNR of the velocity

u ← u − lr·clip( g̃ / √(v̂ + ε), ±c )                 # drive
      − lr·κ·(1 − coh)·u                             # dissipation
      + β1·coh·u                                     # optional momentum (default β1 = 0)

P ← P + α·gc·u ,   u ← (1 − α·gc)·u                  # consolidation: continuous Lookahead
      gc = φc + (1 − φc)·coh
```

Winner constants: α = 0.1, α_v = 0.001, κ = 50, c = 10, ε = 1e-10, σ peak 0.6,
floors φc 0.9→0.1 and φl 0.999→0.1, C\* ≈ 0.00908. See `SDXL_WINNER_REPORT.md` for the
line-by-line commentary and the variational-inference reading; the rest of this document
is how the kernel realizes this rule in integers, fast.

## 4. One training step, end to end

### Forward (per swapped layer)

The layer holds no fp32 or bf16 master weight. Two paths:

- **Fused** (`concord_fused_matmul`, default on): a Triton kernel computes
  `y = x · m_effᵀ`, applying `2^(row_exp+col_exp)` as separable per-N / per-K factors
  *inside* the matmul (L156–305 region in the integration file) — the full bf16 weight is
  never materialized. Linear layers only; Conv2d dequantizes into one shared scratch
  buffer per forward (cuDNN needs the 4D tensor). Eliminating the per-layer bf16 caches
  saves ~5 GB on SDXL (~15 GB vs ~20 GB footprint).
- **Cached** (fallback, and used when gradient accumulation needs a frozen weight): the
  forward reads `weight_buf`, a bf16 copy that the *backward kernel* of the previous step
  emitted (see materialize-merge below).

### Backward (per swapped layer) — where the optimizer lives

Each layer is an autograd `Function` (`PackedLinearFn`); its `backward`
(L1237–1318; integration L1640–1720) does, in order:

1. **Standard grads**: `grad_x = grad_y @ W`; `grad_W = grad_yᵀ @ x` (flattened over
   batch dims), bf16.
2. **Noise injection** *(integration L1651–1668)*: `grad_W += ξ · σ_t·‖grad_W‖/‖ξ‖`,
   `ξ ~ N(0,I)` (isotropic; the Σ_g-shaped variant exists but lost the ablation). σ_t is
   read from a device tensor so a captured CUDA graph sees the live schedule. Injected
   **before** the second moment, so the noise rides through the preconditioner.
3. **AdaFactor EMA** (L1269–1284): `v_row ← 0.999·v_row + 0.001·Σ_k g²`, `v_col`
   likewise, plus precomputed `1/Σv_row`.
4. **The apply kernel** — one Triton launch, `_apply_packed_adamw_kernel` (L404–685),
   everything per-element in registers:
   - unpack the int32; reconstruct `v̂ = v_row ⊗ v_col / Σv_row` (L498–503);
   - compute the gate `coh` (§7);
   - **drive + dissipation**: `Δ = −lr·clip(g/√(v̂+ε), ±10)/scale − lr·κ(1−coh)·s_fast`,
     SR-ticked into `s_fast` (L539–581). These are the only writes that move `W`;
   - **chase**: transfer `α·gc·s_fast` from `s_fast` into `s_slow` (SR-ticked at /128
     granularity; mass-preserving, so `W` is invariant) (L586–611);
   - **leak**: relax `v_slow` toward `s_slow` at `α_v·gl`, subtracting the same mass from
     `s_slow` so `P` is invariant — this only advances the telescope `d` (L613–629);
   - clamp (int16/int8), repack, store the int32;
   - **materialize-merge** (L663–671): emit the new bf16 `W` into `weight_buf` for the
     next forward — one extra cast+store replaces a separate per-layer kernel launch;
   - **rebalance watermarks** (L675–685): `atomic_max` of `|m_eff|` into per-row/col
     buffers (skippable via `TRACK_REBALANCE`).

There is no `optimizer.step()` for swapped layers. The visible optimizer is plain SGD
(momentum 0.9) over the non-swapped leftovers — norms, biases, embeddings.

### Host, between steps

`winner_step` (`concord_winner.py` L231–266 *(integration)*) advances four scalars
**as device tensors** — lr (warmup × cosine to 0.2), σ (rising-late `0.6·(1−f)`), and the
two gate floors. Device tensors because a CUDA graph freezes Python scalars at capture
time; `.fill_()` outside the graph propagates to every replay (a bug class this codebase
hit and documents).

`GatedRebalance` (`concord_winner.py` L269–325) then asks, with **one** reduction over
two shared buffers, whether *any* layer's mantissa watermark crossed `MAX_M = 24000`. At
fine-tune learning rates the answer is essentially always no — skipping ~794 per-layer
rebalance launches per step (~1.8× iteration speedup). When it fires, the rebalance
kernel ticks the offending row/col exponent up by 1 and stochastically-rounds all three
fields right by one bit, migrating the sub-bit residue of `s_slow`/`v_slow` into `s_fast`
so the live weight is preserved in expectation (L940–1180). Tick-*down* exists but is off:
it oscillates against the v̂ chase's exponent ratcheting and measurably hurt (L371–375).

## 5. The arithmetic — what is computed in which format

The design rule: **integers persist, fp32 exists only in registers, bf16 only on the
wire.** Per-element fp32 state is never stored.

| Quantity | Format | Where |
|---|---|---|
| weights + optimizer state | int32 (3 packed fields) | persistent, global memory |
| shared exponents | int8 per row + per col | persistent |
| AdaFactor `v_row`/`v_col`, `1/Σv` | fp32, O(N+K) | persistent |
| activations, `grad_y`, `grad_x`, `grad_W` | bf16 (fp32 accumulate in the matmuls) | transient |
| all apply-kernel math (gate, denom, ticks) | fp32 | registers only |
| noise draw, v̂ EMA marginals | fp32 | transient |
| step scalars (lr, σ, floors, consf, salt) | fp32/int32 device tensors | persistent, 1 elem |

The details that make low-bit arithmetic behave:

- **Scale conversion is exact.** `scale = 2^(row_exp + col_exp − 15)` is an integer power
  of two computed by `exp2` of an int sum — no rounding. In the fused matmul the row and
  column factors are applied separably (`2^(r+c) = 2^r · 2^c`), so factoring the exponent
  out of the inner product is exact, not an approximation; the fused kernel is validated
  to match cuBLAS to bf16 precision.
- **The preconditioner power** is computed as `denom = exp2(p·log2(v_proxy + ε))`
  (L539) — safe because ε > 0, and one transcendental instead of `pow`.
- **Stochastic rounding is the precision mechanism, not a detail.** Every fractional
  update Δ becomes `floor(Δ) + Bernoulli(frac(Δ))`, with the uniform drawn from an
  xorshift hash of (field value, element position, step salt) (L91–99) — five independent
  streams per element (gradient tick, chase, leak, two anchored-wd terms), distinguished
  by salt. Consequences: `E[tick] = Δ` exactly, so **sub-LSB gradient signal is not lost**
  — a step of 0.01 LSB lands as a full LSB 1% of the time and accumulates correctly over
  steps; quantization error is zero-mean with magnitude ≤ 1 LSB and averages down as
  1/√K over K steps.
- **What an LSB is worth.** One `s_fast` LSB in weight units is `2^(row_exp+col_exp−15)`.
  Rebalance holds `|m_eff|` below 24000, so relative resolution at the watermark is
  ~1/24000 ≈ 2⁻¹⁴·⁵ — roughly 14 bits of relative mantissa, versus bf16's 8. The int8
  fields tick in quanta of 128 fast-LSBs; the SR residue of every coarse tick stays
  behind in `s_fast`, so granularity mismatch between fields never loses mass.
- **Exponent changes are integer surgery.** A rebalance tick-up right-shifts all three
  fields by one bit; the shifted-out residue of `s_slow`/`v_slow` (bounded ±192 mantissa
  units) is SR-migrated into `s_fast`, so the live weight is preserved exactly in
  expectation through a range change.
- **The step salt** is a device counter incremented inside the apply wrapper — fresh SR
  randomness on every step, including CUDA-graph replays (the increment is captured).

## 6. Where the passes are, and how gradient accumulation works

### Pass inventory

Per swapped layer, per optimizer step (accumulation 1), the complete list of GPU work:

```text
forward     1 matmul          fused: one Triton dequant-matmul over packed_w
                              cached: one cuBLAS matmul over the bf16 weight_buf
backward    1 matmul          grad_x = grad_y @ W        (fused: dequant-gradx kernel)
            1 matmul          grad_W = grad_yᵀ @ x
            2 reductions      v̂ row/col EMA update (+ 1/Σv)
            1 apply kernel    optimizer step + repack + next-forward weight + watermarks
```

There is **no optimizer pass** (fused into the apply kernel), **no materialize pass**
(the apply kernel's final store *is* next forward's weight), **no `.grad` storage** for
swapped layers (`grad_W` is consumed inside the same `backward` call, never attached to
a parameter), and **no rebalance pass** in the common case (one global reduction over
all layers' watermarks decides; per-layer rebalance kernels launch only on overflow).
With gradient checkpointing on (the SDXL preset), each forward additionally replays once
inside backward. Under the CUDA graph, this entire chain — predict, loss, backward,
self-step — is a single graph replay.

### Gradient accumulation: `s_fast` is the accumulator

Accumulation is structurally awkward here: the optimizer steps *inside* backward, so
"sum gradients, then step once" has no natural home — there is no `.grad` to sum and no
deferred `step()`. The solution (commit `f4efb0e`): split the kernel into an
unconditional **tick** and a gated **consolidation**, and let the int16 fast field double
as the accumulation buffer.

A per-device int32 flag (`set_consolidate`, integration L483 — a stable device tensor,
so a captured graph keeps one pointer while the trainer toggles the value) selects the
mode; the trainer sets it 0 for micro-steps 1…A−1 and 1 on the update step:

```text
every micro-step (consf = 0 or 1):
  noise injection, v̂ EMA                              # per micro-batch
  s_fast += SR( −lr · clip(gᵢ/√(v̂ᵢ+ε), ±c) / scale )  # the tick: ALWAYS fires

only on the update step (consf = 1):
  evaporation, optional β1                             # dissipation terms
  chase (s_fast → s_slow), leak (s_slow → v_slow)      # consolidation
  anchored wd terms                                    # prior pull
  weight_buf store + rebalance watermarks              # emit next cycle's weights
```

(Kernel: `delta_t = delta_grad + consf·(β1·coh·d_fs − evap)`, integration L796; chase,
leak, and the wd terms each carry `·consf`, L830–892; the `weight_buf` store and the
watermark `atomic_max` are masked by `cons == 1`, integration L915–933.)

Mid-cycle, the **cached** forward reads `weight_buf` — which the kernel did *not*
re-emit — so every micro-step's forward sees identical weights: textbook frozen-weight
accumulation. The marginal cost of accumulation is zero bytes (the field already exists)
and zero extra kernel launches.

Three things worth being precise about:

1. **What accumulates is the sum of preconditioned micro-steps, not the preconditioned
   sum.** Each micro-batch gradient is individually noised, individually clipped, and
   preconditioned by that micro-step's v̂ before its tick lands:
   `Σᵢ lr·clip(gᵢ/√(v̂ᵢ+ε))` rather than AdamW's usual `lr·(Σᵢgᵢ)/A/√v̂`. To first order
   at fine-tune learning rates these agree; the difference (per-micro-batch clipping and
   a fresher v̂) is a deliberate semantic of the design, not an accident. Headroom is
   ample: each tick is bounded by `lr·c/scale` mantissa units against int16's ±32768.
2. **The `fast_gain` guard.** `fast_gain` scales `s_fast`'s share of the *forward*
   weight (1.0 = full; → 0 would make the forward run on the deploy weight). Any value
   < 1.0 forces the forward to re-materialize from the live `packed_w` every call —
   which mid-cycle is *moving* (its `s_fast` is absorbing ticks) — breaking the frozen
   forward. Nothing schedules it below 1.0 today; the setup raises `ValueError` under
   accumulation if anything ever does (`StableDiffusionXLFineTuneSetup.py:199–215`),
   failing loudly instead of corrupting silently.
3. **Fused + accumulation: the freeze is relaxed, deliberately.** The fused path has no
   `weight_buf` to freeze — it dequantizes `packed_w` inside the matmul, so mid-cycle
   forwards see `s_fast` ticking through the cycle. The original integration guarded
   this off (forcing the cached path under accumulation); commit `0a47271` dropped the
   guard as "bogus," on the grounds that the consolidate gate, not the bf16 cache, is
   what defines accumulation. The honest characterization: **cached + accumulation is
   frozen-weight accumulation; fused + accumulation is closer to A small online steps
   whose consolidation fires once per cycle.** Both produce the same end-of-cycle
   consolidation; they differ in where mid-cycle gradients are evaluated.

The non-swapped parameters (norms, biases, embeddings) accumulate the ordinary way —
their `.grad` sums across micro-steps and the aux SGD steps on the update step — and
under the CUDA graph `zero_grad(set_to_none=False)` keeps those grad buffers at stable
addresses across replays.

## 7. The gating mechanism

The gate answers, per weight, per step: *how much of the current velocity is signal?*

1. **Drift estimate.** The telescope `d = 128·(s_slow − v_slow)` is the gap between two
   EMAs of the same trajectory at rates α and α_v — a long-window record of consolidated
   motion. Under a pure-drift gradient stream every field rides a ramp; solving the
   steady state gives the lag ratios, and `C* = L·ρ/(1 − L·α_v)` with `L = (1−α)/α`
   (L52–84) is the coefficient that makes the prediction `μ = C*·d` match `E[s_fast]`
   exactly. At winner rates C\* ≈ 0.00908. (The derivation comment records that the old
   shipped value 0.1 was ~11× too large, which flattened the per-weight variance map.)
2. **Signal/noise split.** `n = s_fast − μ` is the innovation residual. Both `μ` and `n`
   come from the *same* decomposition of the same quantity, so their ratio is
   dimensionless and units-correct (the legacy gate compared mismatched units and read
   ~0 — `_USE_FIXED_COH` exists because of that scar).
3. **The gain.** `coh = μ²/(μ² + n²) ∈ [0,1]` (L512–524) — a Wiener gain, i.e. the
   steady-state Kalman gain / MMSE shrinkage factor. `coh → 1`: the velocity is a
   coherent trend. `coh → 0`: it is noise.
4. **Three consumers** (all in the same kernel pass):
   - the **chase gate** `gc = φc + (1−φc)·coh` (L596–597) — how much velocity
     consolidates into `P` this step;
   - the **leak gate** `gl = φl + (1−φl)·coh` (L619–620) — how fast the telescope
     advances;
   - the **evaporation** `lr·κ·(1−coh)·s_fast` (L568) — the dissipation: incoherent
     velocity is drained before it can consolidate. lr-proportional, so the cosine
     schedule self-fades the friction and late, small signal isn't over-skimmed.
   - (optional) the **momentum** `β1·coh·s_fast` — reinforce only the coherent fraction;
     ungated heavy-ball diverges here because the velocity is part of the live weight and
     feeds back through the preconditioner. Off in the winner.
5. **Bootstrap floors.** At init the telescope is empty and `coh` reads 0, which would
   freeze consolidation. The floors start high (chase 0.9, leak 0.999 ≈ ungated) and
   cosine-decay to 0.1 over ~one epoch as the telescope fills — coherence has to *earn*
   control. The floors live in device tensors for graph-replay correctness (L837–843).

The cost of all this: zero state. Every input to the gate is already in the packed word
or in v̂.

## 8. Keeping integers honest

- **Stochastic rounding everywhere.** Every fractional tick is floored, then rounded up
  with probability equal to the fraction, using an xorshift hash of
  (value, position, step-salt) (L91–99) — three independent streams per element (distinct
  salts for tick/chase/leak, L577/606/621). E[integer update] = the real-valued rule;
  quantization error is unbiased and averages out over steps.
- **Mass preservation.** The chase and leak move mass *between* fields; the live weight
  (chase) and deploy weight (leak) are invariant by construction. Clamps are applied
  before the mass-preserve subtraction so saturation can't silently destroy mass
  (L399–402 comment records the bug).
- **Why int16 for `s_fast`.** The first design (int8 fast field) saturated within 5–15
  steps in an MLP smoke test (header comment, L19–20). The fast field needs headroom of
  ~±128 chase quanta.
- **Gradient accumulation** never touches fp32 buffers either: the micro-step sums live
  in the same int16 field as everything else, SR-rounded per micro-batch (§6).

## 9. Making it fast: the CUDA-graph step

With batch size 1, kernel-launch overhead dominates an SDXL step. The fork manually
captures *UNet predict → loss → backward (+ the fused self-steps)* into one CUDA graph
(`modules/util/optimizer/concord_graph.py`):

- `torch.cuda.make_graphed_callables` NaN'd on the first real step (static-buffer
  backward × self-stepping layers × checkpointing); the capture is manual, with eager
  fallback on any failure (L37–42, L356–382 *(integration)*).
- Everything that changes per step crosses the graph boundary as a device tensor: lr,
  σ, gate floors, the consolidate flag, the step salt. Fresh diffusion noise and
  timesteps are injected into static buffers before each replay, so the capture doesn't
  pin the RNG.
- Aspect-ratio bucketing would force a re-capture per shape change; the fork adds
  `ContiguousAspectBatchSorting` (default on) so each resolution bucket is one contiguous
  run per epoch, and per-shape v̂ buckets persist across switches.
- The graph's private memory pool fragments around sampling/backup (`torch_gc`); the
  graph is explicitly released before those, and a wrapper can checkpoint-and-relaunch
  the process after each sample (`CONCORD_RESTART_ON_SAMPLE`) where releasing isn't
  enough.

## 10. Save, resume, deploy

- **Deploy** (final SAFETENSORS/DIFFUSERS save): every packed layer is *consolidated*
  back into a standard `nn.Linear`/`nn.Conv2d` holding `P = consolidated_weight()`
  (drop `s_fast`). Saved checkpoints are ordinary SDXL — loadable anywhere, no Concord
  code needed. Shipping `P` rather than `W` is part of the validated win (deploy-sv
  1.518 vs live 1.556 on the split arm).
- **Backup/resume**: backups carry the full packed state. Resume rebuilds a standard
  UNet, re-swaps, then overwrites with the backed-up packed tensors and resyncs
  `weight_buf` — bit-exact continuation, including the frozen-anchor TE's original
  `v_slow`.
- **Fine-tune init**: `load_weights` places the pretrained weight so the residual sits in
  the fine field; `load_weights_anchor` (frozen-anchor TE) pins the pretrained weight in
  `v_slow` as the prior and trains `s_fast`/`s_slow` as an elastic delta with `wd_anchor`
  pulling home. This also zeroes the telescope at init, so the gate starts unbiased
  instead of reading the pretrained weight as "drift".

## 11. Three readings of the same mechanism

- **Mechanical**: EMA Lookahead AdaFactor, fused in-register — chase = Lookahead's
  slow-weight update run continuously; v̂ = AdaFactor's factored second moment; the whole
  step is one load→compute→store pass over the packed word.
- **Physical**: a fluctuation–dissipation pair — injected noise pumps incoherent
  coordinates, coherence-gated friction drains them, and only statistically coherent
  signal accumulates into the slow weight.
- **Statistical**: variational inference — the word stores an implicit Gaussian posterior
  `q(W) = N(μ, τ²Σ)` (mean = slow fields, fluctuation = fast field); the dynamics explore
  it at temperature σ, the Wiener gain shrinks each observation to its conditional mean,
  the anchor terms regularize toward the (pretrained) prior, and deploy ships the
  posterior mean.

These are not three features; they are one update rule (§3) viewed at three altitudes.

## 12. What's validated

- Same-seed A/B on nanoGPT-char (deterministic at fixed seed): bare recipe 1.5404 →
  +dissipation 1.5180 → +fluctuation (σ = 0.6) **1.4967**, vs AdamW ~1.534 and Muon
  ~1.578, all at 32 bits/param. The dissipation Δ is deterministic; the fluctuation Δ is
  single-seed with ~0.01 trajectory jitter — multi-seed before trusting the magnitude on
  a new task. (`WINNING_CONFIG.md` for provenance.)
- SDXL full-UNet fine-tune (the integration): functional, validated samples, ~15 GB
  fused / ~20 GB cached; regression tests assert the swap (794 layers), no-NaN training,
  standard-SDXL checkpoints, and token sanitization
  (`modules/util/optimizer/concord/tests/` *(integration)*).
