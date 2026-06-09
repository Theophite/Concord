# The Muonified optimizer: Triton and CUDA-graph implementation design

Blueprint for implementing the NS drive ([`MUON_DRIVE.md`](MUON_DRIVE.md)) in the
canonical package (`concord/packed_b.py`) and the SDXL fork. **Status: design doc.**
The rule is CPU-validated (exps 9/9b/9c); no GPU code exists yet; the adoption gates
of `MUON_DRIVE.md` §8 apply to everything here.

---

## 1. The optimizer, complete

**Persistent state, total:** the int32 word (`s_fast` int16 | `s_slow` int8 | `v_slow`
int8) + per-row/col int8 exponents. Nothing else — no `v_row`/`v_col`/`sum_v_inv`, no
ε, no step cap, no trust region. The v̂-arm's only out-of-word state is deleted.

**Per layer, per step** (everything not named here — gate, friction, chase, leak,
anchored decay, deploy, schedules, rebalance, SR — is byte-identical to the v̂ arm):

```text
O   = √max(N,K) · NS5( g / ‖g‖_F )            # the drive: transient, bf16, no state
[σ:  O ← O + σ_t·‖O‖·ξ/‖ξ‖]                    # fluctuation, POST-NS, default OFF
coh = μ²/(μ²+(u−μ)²),  μ = C*·d                # unchanged; reads only packed fields
u  ← u − lr·O − lr·κ(1−coh)·u                  # tick + friction; NO clip, NO denom
P  ← P + α·gc·u ;  u ← (1−α·gc)·u              # chase (unchanged)
leak, wd terms, repack, weight-emit, watermarks: unchanged
```

**Constants and envelopes:**
- `C*` unchanged — `compute_drift_cancel_C(α, α_v, mass_preserve=True)`. The fixed
  point assumes only per-element tick stationarity under drift; NS of a stable matrix
  is stable (CLAUDE.md invariant #3 discharged, with that argument recorded).
- `γ = √max(N,K)` — makes the step per-element RMS ≈ 0.7 (measured), spectral norm
  exactly γ. Shape constant: bake at construction.
- κ is **per-drive**: ≈ 4× lower than the v̂ arm at matched noise (NS ticks are
  unit-RMS, friction works harder per unit κ). The autotuner table must be calibrated
  for this drive; the gate-coh meter itself (`measure_coherence`) is drive-agnostic.
- Stability: capless-stable across lr ∈ [1e-3, 1e-1] on the CPU bench; the
  **lr·κ < 2 friction ceiling is drive-independent and still applies** — keep the
  setup guard, and note it binds earlier at Muon-arm lr (lr=0.1 → κ < 20).
- β1: untested under NS (exp 7 was v̂-arm). Default 0; autotuner `beta1_on=0`.

**NS5** (the standard Muon iteration; bf16 throughout, per Muon practice — the
quintic coefficients are tuned to tolerate bf16):

```text
a, b, c = 3.4445, −4.7750, 2.0315
X = G / (‖G‖_F + 1e-7);  transpose if rows > cols
repeat 5×:  A = X Xᵀ;  X = a·X + (b·A + c·A·A)·X
transpose back
```

Input is `grad_W`, which the backward already holds in bf16 — **no dequantization is
needed anywhere** (the c = 0 result killed the only path that would have orthogonalized
packed state). The exponents never enter the NS pass.

## 2. Where the NS pass lives: inside `PackedLinearFn.backward`

The self-stepping contract (grad consumed in the same backward call, never stored)
pins the placement. The backward becomes:

```text
1. grad_x = grad_y @ W           (fused gradx kernel / cuBLAS — unchanged)
2. grad_W = grad_yᵀ @ x          (unchanged)
3. [DELETED: σ pre-injection]    (the _SIGMAG branch moves to step 5)
4. [DELETED: v̂ row/col EMA + 1/Σv]
5. O = γ·ns5(grad_W)  [+ post-NS σ from the device tensor, if enabled]
6. apply kernel(O, ...)          (USE_MUON path, §3)
```

Step 5 is **per-layer, in eager order** — each layer's backward fires separately
during autograd's sweep, so the NS matmuls run unbatched on cuBLAS. That preserves the
memory contract exactly (no gradient is ever resident beyond its own backward call).
The batched alternative — group same-shape layers, stash their grads, run one `bmm`
NS chain per shape group — buys GPU utilization at the cost of holding that group's
gradients (a partial return of the ~5 GB the fusion exists to avoid) and a trainer
restructure. **Baseline: per-layer. Batching is an optimization to consider only if
§6's measured overhead demands it.**

Workspace: one transient bf16 `[N, K]` (can reuse the fused-matmul shared scratch —
it is idle during backward) plus one `[m, m]` (m = min(N,K)) accumulator, sized to the
largest swapped layer, allocated once at swap time. No per-step allocation (this also
keeps the CUDA-graph memory pool stable, §4).

**Conv2d:** NS operates on the packed 2-D view `[C_out, C_in·kh·kw]` — the layout the
packed conv already uses for its row/col exponents. Standard Muon practice.

## 3. Kernel changes: one constexpr, three dead branches

`_apply_packed_adamw_kernel` gains `USE_MUON: tl.constexpr`. Diff, in full:

```text
if USE_MUON:
    step_live = grad           # grad_W_ptr carries O = γ·NS5(ĝ) [+ post-NS σ]
else:
    denom_p   = tl.exp2(precond_p * tl.log2(v_proxy + eps))
    step_live = grad / denom_p
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
```

plus: the `v_proxy` / `v_hat` loads are guarded so the USE_MUON ∧ USE_FIXED_COH path
never touches `v_row_ptr`/`v_col_ptr`/`sum_v_inv_ptr` (stub them with `packed_w`, the
existing pattern — note the fixed Wiener gate reads **only** `d_fs`, `d_sv`, and
`drift_cancel_C`; it never needed v̂). Everything downstream — `delta_grad =
−lr·step_live·scale_inv`, the consf gating, evaporation, chase, leak, wd terms, clamp
order, repack, materialize-merge, watermarks — is untouched.

**SR streams:** unchanged. The gradient tick is still one tick (same salt); chase,
leak, and wd terms keep theirs. No new fractional write is introduced, so no new
stream is needed (CLAUDE.md invariant #6 discharged).

**Dead arguments** (`eps_ptr`, `step_cap`, `v_scale`, `precond_p`,
`gf_trust_delta_sq`) stay in the signature, branched out by the constexpr — signature
stability keeps the wrapper and graph-capture code paths single-sourced.

**Layer surface:** `ConcordLinearPackedB` gets `drive = "vhat" | "muon"` (constructor
arg + `set_drive()`), gating: buffer allocation (skip `v_row`/`v_col`/`sum_v_inv` and
the backward EMA when `"muon"`), the backward branch (§2), and the kernel constexpr.
`swap_unet_to_winner(..., drive=...)` plumbs it; the config field is one more
`concord_*` entry in the fork.

## 4. CUDA-graph story

Better than the v̂ arm's, not worse:

- **Capture:** the NS pass is static-shape tensor ops inside
  `PackedLinearFn.backward`, which is already inside the Stage-3 capture
  (`concord_graph.py` captures predict → loss → backward). It records and replays with
  no new machinery. NS is deterministic → **bit-determinism at fixed seed is
  preserved** (the validation currency survives).
- **Per-step scalars:** lr, floors, consf, salt already cross as device tensors. γ is
  a shape constant (baked at capture is correct). σ, if enabled, must read the
  existing `_SIGMAG_SIGMA_T` device tensor *at its new post-NS call site* — the
  current pre-injection branch is Python-level in the backward and already
  graph-safe in the magnitude (tensor) but baked in the on/off (bool); same contract,
  new location. κ under graph keeps the standing caveat (`INSTALL_SDXL.md` §3d):
  launch-time scalar — probe eagerly then capture, or port κ to a device tensor.
- **Memory pool:** the NS workspaces are allocated once at swap time (§2), so the
  graph's private pool sees fixed addresses; no interaction with the
  release/recapture machinery or `CONCORD_RESTART_ON_SAMPLE`.
- **What gets simpler:** the v̂ EMA (two reductions + a `fill_` per layer per
  backward) disappears from the capture entirely.

## 5. Gradient accumulation

Unchanged in structure, with the same documented semantics shift as the v̂ arm:
micro-steps tick `−lr·γ·NS5(ĝᵢ)` into `s_fast` (each micro-gradient individually
orthogonalized — the accumulated quantity is the *sum of Muon micro-steps*, not the
Muon step of the summed gradient), and `consf` gates consolidation to the update step
exactly as now. The cached-path weight freeze, the fused-path relaxation, and the
`fast_gain == 1` guard are all drive-independent. NS cost note: the pass runs per
micro-step (it sits in backward), so accumulation multiplies NS flops along with the
GEMMs — ratio unchanged.

## 6. Cost model (the one open risk)

Per layer with m = min(N,K), k = max(N,K), T = tokens in the batch:

```text
NS5 flops      ≈ 20·m²·k          (5 iterations × {X Xᵀ, A·A, B·X})
layer GEMMs    ≈ 6·m·k·T          (forward + two backward matmuls)
overhead ratio ≈ 3.3 · m / T
```

- **LM scale** (T ~ 10⁵–10⁶): < 5% — Muon's known economics.
- **SDXL at batch 1**: T is 1k–4k per attention block and m up to 1280, so square
  projections at low resolution can see ratios approaching 1 — the NS pass could
  rival the layer's own GEMMs there. Mitigations, in order of preference: bf16 NS
  (assumed), NS3 instead of NS5 (test on the bench first), per-shape batching (§2,
  with its memory tradeoff), orthogonalizing only above a size threshold (tiny
  layers gain least from spectral preconditioning).
- The deleted v̂ EMA buys back two reductions per layer per step; memory is a wash
  (O(N+K) was negligible). **The win is quality + lr robustness + state simplicity;
  the price is flops — and the SDXL-bs1 price is an empirical question that must be
  measured before the fork adopts the drive.** (`CONCORD_GRAPHMEM`/step-timing under
  the existing graph harness answers it in one run.)

## 7. Controller and autotuner integration

- `winner_step` schedules: unchanged (lr/floors; σ schedule only if post-NS σ is
  enabled).
- `DissipationAutoTuner`: works as-is mechanically (the meter reads packed state,
  which is drive-agnostic), but **the (coh → κ) table and the probe default κ are
  per-drive** — calibrate with the Muon arm (expect the κ axis compressed ~4×; the
  exp-9 sweep suggests probing at κ ≈ 25–50 and a table topping out near 100–200
  where the v̂ table reached 400). `beta1_on = 0` until exp-7 is repeated under NS.
- Stability guard (`INSTALL_SDXL.md` §2): keep; it binds earlier at Muon-arm
  learning rates.

## 8. Build order and validation

1. **Package** (`concord/packed_b.py`): `ns5()` helper, `drive=` plumbing, backward
   branch, `USE_MUON` constexpr. CPU parity test extends the exec-from-source pattern:
   the shipped `ns5` against `experiments/cpu_dynamics/exp9_muon.py::ns5`
   (semi-orthogonality: `‖NS5(X)·NS5(X)ᵀ − I‖` small; idempotence; transpose path).
2. **CUDA smoke**: `python concord/packed_b.py` extended with a `drive="muon"` MLP
   arm — converges, no NaN, deterministic across two same-seed runs.
3. **The bench**: same-seed nanoGPT A/B, NS vs v̂ at each one's (κ\*, lr\*),
   deployed-sv — the adoption gate, and the rematch on the bench native Muon lost.
4. **Fork port** (after 3 passes): the three files per `INSTALL_SDXL.md` discipline +
   `tests/regression.py` before/after + the §6 timing measurement under
   `concord_cuda_graph` on/off.
5. **Recalibrate**: Muon-arm autotune table on the target task; re-derive nothing
   (C\* is untouched by construction).
