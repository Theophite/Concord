# Concord dynamics, recalibration, and autotuning — results

Findings from a CPU experimental campaign on the Concord optimizer (June 2026), run
against the real-valued reference of the kernel's update rule. Companion documents:
[`SDXL_WINNER_REPORT.md`](SDXL_WINNER_REPORT.md) (what the optimizer is),
[`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) (how the kernel realizes it), and
[`experiments/cpu_dynamics/EXPERIMENTS.md`](../experiments/cpu_dynamics/EXPERIMENTS.md)
(the full lab log; every number below is reproducible from the scripts there).

## TL;DR

1. **A real calibration bug was found and fixed**: the drift-cancel coefficient C\* was
   ~2× too small for the mass-preserving leak, leaving the coherence gate half-blind to
   genuine signal (pure-drift coherence 0.29 instead of ~1). The fix is derived,
   simulated, and shipped on this branch.
2. **The fluctuation–dissipation machinery is regime-dependent, exactly as the repo's
   notes claim** — it costs accuracy on clean tasks and earns it back with interest
   under noise — and the optimal dissipation follows a measurable curve:
   **κ\*(noise) ≈ min(1000·ρ, 400)**.
3. **The gate doubles as a free noise meter**, which closes the loop: a
   **probe-then-commit autotuner** (now shipped in `concord/packed_b.py`) reads the
   gate's own mid-training coherence and commits κ — and β1 — without being told the
   noise level, recovering the oracle settings almost exactly.
4. **End to end, autotuned Concord beats AdamW at every noise level tested**
   (+0.75 to +1.74 test accuracy, margin growing with noise), at identical lr and
   schedule, while AdamW's weight decay does nothing for it.

**Validity bounds, up front**: all results are from a CPU fp32 reference of the kernel
rule (`/experiments/cpu_dynamics/concord_ref.py` — mirrors the kernel's order, gating,
schedules, and init line-by-line; stochastic rounding makes the integer kernel equal it
in expectation), on small MLPs, MNIST-scale tasks, 3 seeds, no per-arm lr tuning. They
characterize the *dynamics* of the update rule, not SDXL-scale behavior. The adoption
gate for everything here is the same-seed nanoGPT A/B in the real GPU harness.

---

## 1. Dynamics relative to AdamW

*(exp 1–2: lr sweep on noisy regression; controlled gradient streams)*

- **Same loss, bigger steps.** At fine-tune lrs Concord moves weights ~1.5× more than
  AdamW for the same final loss — no first moment (β1 = 0) and a step cap of 10, versus
  Adam's ~lr-bounded bias-corrected step.
- **A stability ceiling AdamW doesn't have.** The dissipation term
  `u ← u − lr·κ·(1−coh)·u` diverges past **lr·κ ≈ 2** (verified bracketing: 1.5 trains,
  2.25 marginal, 2.5 NaN; κ=0 control survives). The validated configs sit two orders
  of magnitude inside it (nanoGPT 0.025, SDXL 0.004). In the integer kernel the int16
  clamp saturates instead of NaN-ing, but the dynamics would be equally broken.
- **Warmup is load-bearing, not cosmetic.** `load_weights` puts the entire initial
  weight in the velocity; the chase consolidates it over ~1/α steps while warmup holds
  the (lr-proportional) friction off. Remove warmup and the friction eats the model.
- **Selectivity is real but was mild as shipped** (~15% less random-walk displacement
  than AdamW under a pure-noise gradient stream, equal drift tracking) — the gate was
  the bottleneck, which led directly to:

## 2. The C\* recalibration (the bug)

*(steady-state analysis + simulation; fix shipped)*

Under a pure-drift gradient stream the gate should read coherence → 1. It read
**0.26–0.29** (0.55 even under the exact assumptions of the C\* derivation). Root
cause: the mass-preserving leak debits `s_slow` as well as crediting `v_slow`, so the
telescope `d = s_slow − v_slow` relaxes at **2α_v** per step — but
`compute_drift_cancel_C` derived with ρ = α_v. Fixed point of the kernel recursion
(tick → chase → leak) under drift μ:

```text
E[u] = μ·L,   L = (1−α)/α            (unchanged)
d' = (1−2α_v)(d + μ)  →  E[d] = μ·(1−2α_v)/(2α_v)
C* = L·2α_v/(1−2α_v) ≈ 0.018036       (shipped: 0.00908 — half)
```

The steady-state analysis predicts the measured coherence (0.42 predicted vs 0.55
simulated at open gates; v̂ rank-1 inexactness is the gap), and the refit reads
**0.96 on pure drift / 0.12 on pure noise** in the full winner config. With the honest
gate, drift tracking becomes *faster than AdamW* (friction shuts off on coherent
motion) and displacement selectivity (drift/noise) improves from 45 to 58 (AdamW: 43).

Same genus as the bug already recorded in that docstring's history (the constant was
previously 11× off in the other direction). **Shipped** in `concord/packed_b.py`,
`dist/concord_winner/concord/packed_b.py`, `notebook/src/prototype_packed_b.py`
(`mass_preserve=True` default, matching the layer; legacy formula preserved under
`mass_preserve=False`). Porting to `concord-integration` is the same one-line edit.

One subtlety the downstream tests exposed: consumers were tuned around the dull meter.
At matched κ the honest gate wins wherever coherent = generalizing and loses slightly
where coherent = *memorizing* (the gate's structural blind spot: memorization drift is
temporally coherent). The fix sharpens the meter; the operating points around it then
want retuning — which motivated everything below.

## 3. The regime axis: when the machinery pays

*(exp 3–4: MNIST clean and with label noise)*

| regime | ordering | numbers (deploy acc) |
|---|---|---|
| clean MNIST (2 ep, 60k) | **bare > dissip ≈ winner > AdamW** | 96.55 > 96.0 > 95.8 > 95.08 |
| label noise, no overfit budget | same as clean | memorization ≈ chance for all arms |
| 30% noise + overfitting room (4k × 25 ep) | **dissip ≈ winner > AdamW > bare** | 90.06 ≈ 90.00 > 89.15 > 88.15 |

Every Concord arm beat AdamW at matched settings in every regime. But the *internal*
ablation flips exactly as the repo's session notes predict: on a near-zero-Bayes-error
task the dissipation is pure cost (−0.7%), while under noise **with time to memorize
it**, bare Concord memorizes 43% of the wrong labels and collapses, and the dissipation
cuts that to 24% and takes the lead. Deploy ≥ live everywhere, with the margin
appearing exactly when gradients are noisy. The fluctuation half (σ-noise) added
nothing beyond the dissipation on this task — consistent with the repo's own caveats
(small, single-seed, possibly BatchNorm-mediated; these MLPs have no BN).

## 4. The dissipation curve: κ\*(noise)

*(exp 5: 33-cell grid, ρ × κ, fixed-C\* gate, σ = 0)*

| label noise ρ | κ\* | acc at κ\* | acc at κ=0 | gain |
|---|---|---|---|---|
| 0% | 0 | 93.66 | 93.66 | — |
| 10% | 100 | 92.72 | 92.42 | +0.30 |
| 20% | 200 | 91.72 | 90.53 | +1.19 |
| 30% | 400 | 90.77 | 88.27 | +2.50 |
| 45% | 400 | 89.53 | 83.01 | +6.52 |

- **κ\* ≈ min(1000·ρ, 400)**: linear rise, then a plateau. The prettier odds-law
  extrapolation (κ\* ∝ ρ/(1−ρ), predicting ~740 at 45%) was tested and **falsified** —
  κ = 600/800 at 45% both score below 400. There is a maximum useful friction, and it
  is loss-driven, not stability-driven (the lr·κ < 2 ceiling sits at κ ≈ 2000 here).
- **κ sets a memorization-rate budget, not a noise response**: at fixed κ the fraction
  of wrong labels memorized is nearly independent of ρ (κ=100 → ~19–23% everywhere;
  κ=400 → ~12%).
- **The risk is asymmetric**: over-damping clean data costs −1.5% (κ=400); under-damping
  noisy data costs −6.5% (κ=0 at 45%). When in doubt, err high.

## 5. Autotuning: the gate is the noise meter

*(exp 6: three iterations, each informative)*

1. **Raw-gradient coherence fails as a meter** (saturates at 0.995–0.999 — at batch 128
   the gradient stream is almost all minibatch noise) — yet the closed loop matched the
   oracle at ρ ≤ 30% anyway, by accidentally discovering a **time profile**: κ low
   early, high late. Standalone finding: κ=400 after epoch 3 costs a clean task −0.2 vs
   −1.5 from step 0 — the dissipation's clean-task tax is paid early.
2. **Schedule alone is insufficient**: meter-free κ ramps reproduce the low-noise wins
   but lose badly at 45% (87.6 vs 89.5) — heavy noise needs friction *early*, which
   requires sensing.
3. **The right meter is the gate itself.** Mid-training gate coherence (velocity-side —
   the telescope has already integrated out minibatch noise) separates noise levels
   cleanly: coh(epochs 3–8) = 0.387/0.314/0.288/0.274/0.256 for ρ = 0/10/20/30/45%,
   spreads 10–50× below the separations. **Probe-then-commit** (default κ=50 probe,
   read coh, commit from the piecewise-linear table) recovers the oracle almost
   exactly — committed κ = 2/103/205/381/400 vs oracle 0/100/200/400/400 — interpolates
   sensibly at held-out noise levels (15%, 38%), and stays within ~0.2% of the oracle
   frontier through ρ = 30% while paying only −0.14 on clean. Known residual: at 45%
   the probe's eight κ=50 epochs lock in memorization the commit can't undo (−1.5).

**Shipped** in `concord/packed_b.py`: `gate_coherence_from_fields` /
`measure_coherence` (host-side, scale-invariant — the exponents cancel — zero kernel
changes, zero per-step cost) and `DissipationAutoTuner` (probe-then-commit), with CPU
parity tests (`test_autotuner_parity.py`) and documented caveats: the table is
task-calibrated (the *procedure* transfers); probe window must match the calibration
window and clear the init-consolidation transient; under a captured CUDA graph,
gf_consol is baked at capture — probe eagerly, then capture, or port κ to a device
tensor.

## 6. Coherence-gated momentum (β1)

*(exp 7: sweep under autotuned κ)*

The winner ships β1 = 0; both that and the "ungated momentum diverges" note predate the
C\* rescale. With the honest gate:

- **β1 = 0.10 works on clean streams**: 93.68 ± 0.04 vs 93.53 (best clean result of the
  campaign). Not arbitrary — it sits at the critical-damping boundary
  `(1+β1)(1−α) ≈ 1`: coherent velocity sustained, not amplified.
- **No β1 > 0 survives label noise** (even 0.05 at ρ = 10%): momentum is a coherence
  *amplifier*, and the gate's one blind spot is that memorization drift is coherent —
  β1 = 0.1 at 10% noise drives memorization 23% → 43%. A sharper β1·coh² gate softens
  but doesn't fix it. The β1 = 0 default is vindicated for Concord's target regimes.
- **Nothing diverged through β1 = 0.8**: partly the gate's self-limiting, mostly the
  autotuner acting as an emergent **stability governor** — runaway momentum reads as
  velocity incoherence at the probe, so the tuner commits κ = 400 and contains it.
- **The probe selects β1 too**: one probe commits κ from the table *and* β1 = 0.1 iff
  probe coh ≥ 0.35. Validated at ρ = 0/5(held-out)/10/30%: momentum on only at ρ = 0
  (+0.13), correctly off elsewhere, zero regressions. Shipped in
  `DissipationAutoTuner` (`beta1_on`/`beta1_coh_threshold`, probe runs at β1 = 0,
  `beta1_on=0` disables).

## 7. Head-to-head: autotuned Concord vs AdamW, variable noise

*(exp 8: the complete shipped package — fixed C\*, one probe committing κ and β1 —
vs AdamW at wd = 0 and wd = 0.01, identical lr/schedule/model/data, 3 seeds)*

| ρ | AdamW (wd=0 / 0.01) | Concord autotuned (deploy) | margin | committed |
|---|---|---|---|---|
| 0% | 92.78 / 92.76 | **93.66 ± 0.10** | +0.88 | κ=3, β1=0.1 |
| 5% | 92.14 / 92.19 | **93.11 ± 0.11** | +0.92 | κ=67 |
| 10% | 91.97 / 91.94 | **92.72 ± 0.21** | +0.75 | κ=101 |
| 20% | 90.73 / 90.72 | **91.63 ± 0.44** | +0.90 | κ=202 |
| 30% | 89.15 / 89.19 | **90.53 ± 0.21** | +1.34 | κ=381 |
| 45% | 86.25 / 86.24 | **87.99 ± 0.68** | +1.74 | κ=400 |

- **Every cell, margin growing with noise.** Decoupled weight decay does nothing for
  AdamW here (wd = 0.01 ≡ wd = 0), so the gap at high noise is structural: AdamW has no
  mechanism that distinguishes coherent from incoherent gradient signal.
- The autotuner committed sensible knobs at every level without being told ρ — κ
  tracking the noise, momentum only on the clean stream.
- **The win is not only noise suppression**: at 5–10% noise Concord memorizes *more*
  wrong labels than AdamW (25% vs 19%) yet generalizes ~+0.9 better — the AdaFactor
  preconditioner, the β1 = 0 step geometry, and deploying the slow (Polyak) weight
  contribute independently of the friction.

## What shipped on this branch

| change | where |
|---|---|
| C\* mass-preserve correction (2× rescale, legacy preserved) | `concord/packed_b.py`, `dist/concord_winner/concord/packed_b.py`, `notebook/src/prototype_packed_b.py` |
| host-side coherence meter (`gate_coherence_from_fields`, `measure_coherence`) | `concord/packed_b.py` |
| `DissipationAutoTuner` — probe-then-commit κ + β1 | `concord/packed_b.py` |
| CPU reference, exps 1–8, parity tests, figures, lab log | `experiments/cpu_dynamics/` |
| reports | `docs/SDXL_WINNER_REPORT.md`, `docs/HOW_IT_WORKS.md`, this file |

## Threats to validity, and the path to adoption

- **CPU fp32 reference, not the Triton kernel**: equal in expectation by construction
  (SR unbiasedness), and the formula-level parity is tested, but integer/SR
  interactions at scale are not exercised here.
- **One task family**: MNIST-scale MLPs with synthetic label noise. The κ table and the
  β1 threshold are calibrated in this domain's probe-coherence units; the transferable
  objects are the procedures and the curve *shapes*, not the constants.
- **3 seeds, no per-arm lr tuning** (single shared peak lr; AdamW gets its standard
  MNIST value).
- **The gate's blind spot is structural**: temporal coherence cannot distinguish
  generalizing drift from memorizing drift; only the friction level manages that
  tradeoff. Any consumer of coh (β1 above all) inherits this.

Adoption path, in order: (1) same-seed nanoGPT A/B of the C\* rescale in the real
harness, with a small κ sweep (the consumers were tuned around the dull meter);
(2) calibrate a probe table on the target task and A/B the autotuner against the fixed
winner; (3) only then β1, behind the probe threshold. For the SDXL fork, additionally
port κ to a device tensor so the tuner can operate under the Stage-3 captured graph.
