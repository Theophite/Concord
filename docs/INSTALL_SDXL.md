# Installing the fixes and the autotuner in the SDXL fork (`concord-integration`)

Step-by-step port of this branch's changes into the OneTrainer fork. Three pieces, in
order of importance:

| piece | what | risk |
|---|---|---|
| 1. C\* fix | the drift-cancel coefficient, ~2× too small for the mass-preserving leak | low (one function; legacy preserved behind a flag) |
| 2. stability guard | fail loudly if `lr·gf_consol ≥ 2` (divergence ceiling) | none |
| 3. coherence meter + autotuner | probe-then-commit κ (and optionally β1) | opt-in; needs per-domain calibration + graph-capture handling |

Reference implementations all live on this branch in `concord/packed_b.py`
(search markers given below); the evidence is in [`RESULTS.md`](RESULTS.md) and
[`experiments/cpu_dynamics/EXPERIMENTS.md`](../experiments/cpu_dynamics/EXPERIMENTS.md).

---

## 1. The C\* fix (the bug)

**File:** `modules/util/optimizer/concord/prototype_packed_b.py` on
`concord-integration`. Three edits — identical in shape to the ones already applied to
`concord/packed_b.py` here (commit `2e427ae`).

**(a)** Replace the body of `compute_drift_cancel_C` (the function at ~L52; its current
last three lines are `L = ...; rho = ...; return L * rho / (1.0 - L * alpha_v_fast)`).
Add a `mass_preserve=True` keyword and the corrected branch:

```python
def compute_drift_cancel_C(alpha, alpha_v_fast,
                              alpha_v_slow=0.0, refit_period=1,
                              mass_preserve=True):
    ...existing docstring...
    L = (1.0 - alpha) / max(alpha, 1e-12)
    rho = alpha_v_fast + alpha_v_slow / max(refit_period, 1)
    if mass_preserve:
        # The MASS_PRESERVE leak debits s_slow as well as crediting v_slow,
        # so the telescope d = s_slow - v_slow relaxes at 2*rho per step:
        #   d' = (1 - 2*rho)(d + mu)  ->  E[d] = mu*(1 - 2*rho)/(2*rho)
        # with E[d_fs] = mu*L unchanged. C* = L*2*rho/(1 - 2*rho).
        # Validation: experiments/cpu_dynamics (drift coh 0.29 -> 0.96,
        # noise control unchanged).
        rho2 = 2.0 * rho
        return L * rho2 / (1.0 - rho2)
    return L * rho / (1.0 - L * alpha_v_fast)
```

**(b)** Wrapper call site (~L1235, inside `apply_packed_adamw`): pass the wrapper's own
flag through —

```python
        drift_cancel_C = compute_drift_cancel_C(alpha, alpha_v_fast,
                                                mass_preserve=mass_preserve)
```

**(c)** Layer `__init__` (~L1808): make the default explicit —

```python
        self.drift_cancel_C = compute_drift_cancel_C(
            alpha, self.alpha_v_fast,
            mass_preserve=True)   # matches mass_preserve_v default below
```

**Notes:**
- The frozen-anchor TE path is unaffected: `swap_text_encoder_to_anchor` runs
  `alpha_v_fast = 0`, and C\* = 0 under both formulas.
- C\* is computed at construction and is **not** stored in checkpoints, so resuming an
  old backup with the fix simply continues with the honest gate — bit-exact state, new
  dynamics. For clean A/Bs, prefer fresh runs.
- **The validated SDXL preset was tuned around the dull gate.** With the honest gate,
  friction exempts coherent signal more cleanly (good) and the operating point shifts
  (CPU evidence: clean tasks improve at matched κ; heavy-memorization regimes may want
  κ above the default 50). Gate adoption on a same-seed A/B against your current runs,
  ideally with a small `gf_consol` sweep — that knob is exposed in the optimizer panel.

## 2. The stability guard (free)

The evaporation term diverges past `lr·gf_consol ≈ 2` (CPU-verified bracketing:
1.5 trains, 2.5 NaNs; the int16 clamp saturates rather than NaNs on GPU, but training
is equally dead). The shipped configs sit two orders inside the bound — guard it
anyway. In `StableDiffusionXLFineTuneSetup.__setup_optimizations` (next to the
existing `fast_gain` accumulation guard, ~L199):

```python
        if config.optimizer.optimizer == Optimizer.CONCORD:
            gf = float(getattr(model.concord_controller.config, "gf_consol", 0.0))
            if config.learning_rate * gf >= 2.0:
                raise ValueError(
                    f"lr*gf_consol = {config.learning_rate * gf:.2f} >= 2: the "
                    f"dissipation term is linearly unstable (u <- u - lr*k*(1-coh)*u). "
                    f"Lower the learning rate or gf_consol.")
```

## 3. The coherence meter + autotuner (opt-in)

### 3a. Copy the code

Append the block between the markers
`# Dissipation autotuning: the gate's own coherence as a noise meter` and
`# Smoke test` from this branch's `concord/packed_b.py` (three objects:
`gate_coherence_from_fields`, `measure_coherence`, `DissipationAutoTuner`) into
`modules/util/optimizer/concord/prototype_packed_b.py`. It is pure torch over existing
layer state (`packed_w`, `drift_cancel_C`, and the `gf_consol`/`beta1` attributes the
integration layers already have) — no kernel changes, no new buffers. The CPU parity
test (`experiments/cpu_dynamics/test_autotuner_parity.py`) execs these objects from
source and can be pointed at the integration copy as a check.

### 3b. Wire it into the controller

`modules/util/optimizer/concord_ot.py`, `ConcordController`:

```python
    # __init__, after self.layers / self.gate are built:
    self.autotuner = None
    if autotune_table is not None:                 # plumb from config / optimizer_defaults
        from prototype_packed_b import DissipationAutoTuner
        self.autotuner = DissipationAutoTuner(
            self.layers,                            # UNet layers ONLY — see note below
            probe_start=int(0.04 * self.total_steps),
            probe_end=int(0.10 * self.total_steps),
            table=autotune_table,
            probe_kappa=self.config.gf_consol)

    # before_step(), before the winner_step call:
    if self.autotuner is not None:
        self.autotuner.step(self.step_idx)
```

Two hard constraints:

- **UNet layers only, never `te_layers`.** The frozen-anchor TE runs `alpha_v_fast = 0`
  — its telescope never advances, its coherence reads ~0 regardless of data quality,
  and it would drag the probe mean toward "maximum noise."
- **The probe window must clear the init-consolidation transient** (the first ~1/α
  steps after warmup, while `load_weights` mass chases into the slow fields) and must
  match the window used for calibration. The 4–10%-of-run window above mirrors the
  validated CPU setup (epochs 3–8 of 25); recalibrate if you move it.

### 3c. Calibrate the table (required — the MNIST numbers do not transfer)

The table maps *probe-window coherence* → κ for **your** task/architecture/schedule:

1. Run the probe window at the default κ on a few datasets of known quality (e.g. your
   clean set, and copies with deliberately corrupted/shuffled captions) and record the
   probe coherence for each — `measure_coherence` averaged over the window, every
   ~10 steps.
2. For each quality level, sweep κ (the `gf_consol` field; e.g. {0, 25, 50, 100, 200,
   400}) to the deployed-metric optimum — this is the exp-5 procedure.
3. Table = [(coh_i, κ\*_i)] sorted by descending coherence. Expect the same *shape* as
   the CPU result (κ\* rising with incoherence, then plateauing), not the same numbers.

**Calibrate in F = lr·κ, not κ.** κ is defined per unit lr, so the dimensionless
friction F = lr·κ is the real knob: F·(1−coh) is the per-step velocity decay (and, via
the κ-identity in `MIXUP.md` §6, the per-step self-distillation weight toward the EMA
teacher), F(1−coh)/(α·gc) the deleted-vs-consolidated split, F < 2 the stability
ceiling. Reference points from the campaign: MNIST heavy-memorization optimum
F ≈ 1.0–1.5; nanoGPT validated winner F = 0.025; **the current SDXL preset runs
F = 0.00375 — near-frictionless in these units**, while diffusion fine-tuning (noisy
ε/t-sampled gradients, small datasets, many steps) is plausibly the heaviest-
memorization regime Concord faces, and the field's reliance on aggressive weight-EMA is
independent evidence it wants a strong pull toward the average. Sweep
F ∈ {0.004, 0.025, 0.1, 0.5, 1.0}. Committing F (not κ) also keeps the friction sweep
orthogonal to any upward lr sweep — and the package tuner supports it natively:
`DissipationAutoTuner(..., peak_lr=lr)` interprets the table and `probe_kappa` in F
units and derives raw `gf_consol` itself. (The C\* calibration survives high F: at the
pure-drift fixed point coh → 1 and the friction term self-vanishes.)

β1 selection (`beta1_on=0.1, beta1_coh_threshold=...`) rides the same probe; set the
threshold between your cleanest and next quality level's probe coherence, or pass
`beta1_on=0` to leave momentum off (the conservative default — the β1 evidence is one
task deep).

### 3d. CUDA-graph compatibility (the one real integration constraint)

`gf_consol` and `beta1` enter the kernel as **launch-time scalars**: eager backward
picks the commit up on the next step, but the Stage-3 captured graph (`ManualUNetGraph`,
`concord_cuda_graph: true`) bakes them at capture. Three options, best first:

1. **Capture after the commit.** Delay `install`/first capture of the UNet graph until
   `step_idx >= probe_end` (run the probe eagerly). The trainer already knows how to
   run eagerly — this is a gating condition in the setup/trainer where
   `concord_graph_v2` is created/replayed.
2. **Port κ/β1 to device tensors** (the clean fix, same pattern as lr/σ/floors/consf):
   add `kappa_ptr`/`beta1_ptr` 1-elem fp32 tensor args to `_apply_packed_adamw_kernel`,
   `tl.load` them at the top, have the tuner `.fill_()` them. Then the tuner works
   under replay with no capture ordering constraints.
3. **Run autotuned jobs with `concord_cuda_graph: false`.** Costs the batch-size-1
   speedup; zero code risk.

## 4. Validation checklist

1. `python modules/util/optimizer/concord/prototype_packed_b.py` (the built-in CUDA
   smoke test) still converges.
2. `tests/regression.py` (the Stage-1/2 net): 794 layers swap, no NaN, checkpoint is
   standard SDXL, sanitize works.
3. A short fine-tune with `CONCORD_GRAPHMEM` on: confirm the probe prints its commit
   line (`[concord] dissipation autotune: probe coh=... -> kappa=..., beta1=...`) and
   that eager-vs-graph behavior matches your chosen option from 3d.
4. The adoption gate, as everywhere in `RESULTS.md`: a same-seed A/B (old C\* + fixed
   winner) vs (new C\* + autotuner) on your real task, deployed metric, before making
   either the default.
