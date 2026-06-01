# Concord enwik8 control plane — single source of truth

**Purpose:** stop conflating knobs/runs. Every enwik8 experiment, its FULL config,
and its result, on one page. Updated 2026-05-30.

## The bench
- enwik8 char/byte-LM, 10.88M-param nanoGPT (n_layer=n_head=6, n_embd=384,
  block=256, bsz=64). Driver: `src/train_nanogpt.py --mode {concord,adamw,factored}`.
- **Comparability contract:** `--seed 0` (shared init) + `--data_seed 1234`
  (DEDICATED batch-order generator, provably optimizer-invariant — Concord's SR
  no longer perturbs the batch stream). 5000 iters, cosine lr (warmup 100,
  min_frac 0.1), unless noted.

## Orthogonal knobs — DO NOT CONFLATE
These are independent. The 2026-05-30 confusion was treating "eps" and "gate" as
the same thing. They are not.

| # | knob | values | set by | active in nanoGPT? |
|---|------|--------|--------|--------------------|
| 1 | **v̂ rank** | none (SGD-chase) / rank-1 (Adafactor row×col) / full (per-coord Adam) | `--mode concord`(eps=1) / `factored` / `adamw` | yes — the main axis |
| 2 | **eps preconditioner** (v_proxy whitening) | inert (eps=1) / engaged (eps < v_proxy ~5e-9) | `--eps` | only when eps<1 (the ladder) |
| 3 | **coherence gate** (cohpre + gf_consol) | OFF / ON | `enable_cohpre()`, `--gf_consol` | **OFF in ALL nanoGPT runs** (only the CIFAR driver enables it) |
| 4 | lr / schedule / wd / betas | structural | `--*_lr`, `--weight_decay` | held fixed across the comparison |
| 5 | **exponent shift** (rebalance ratchet) | per-step tick-UP of row/col shared exponents when live mantissa > MAX_M=24000 | `--rebalance_every` (default 1) + `track_rebalance` (default True) | **ON in all Concord(packed) runs**; fp32 AdamW/factored have no exponent (n/a) |

**Knob #5 caveat (packed-int only):** tick-up-only ratchet (no tick-down). v̂-normalized steps are bigger (ride to step_cap=10) than the inert chase's raw-grad steps → mantissa grows faster → exponent ratchets up more → each up-shift costs ~1 bit of low-end mantissa precision. If packed-rank-1-v̂ lands short of clean-fp FactoredAdam (1.0712), suspect THIS (raise MAX_M / int8 s_slow resolution), not the rank-1 v̂ idea. ON identically in the inert baseline, so the packed-vs-packed comparison stays clean.

**Key clarifications (the things we got tangled on):**
- `--mode concord --eps 1` = **pure SGD-chase**: preconditioner inert
  (`(v_proxy+1)^0.5 ≈ 1`), gate off. This is the 1.43 baseline whose weights were
  dissected for "where the gap lives" + parsimony.
- The **ladder** drove knob #2 (eps↓) to ENGAGE the preconditioner. Those arms
  genuinely differ — engaging it is real, it just doesn't close the gap.
- Knob #3 (the gate) was a red herring: I hypothesized it caused the parsimony;
  it's off everywhere here, so the parsimony is plain SGD-chase low-rank bias.

## RESULT — the SGD↔Adam axis (5000-iter, fully comparable: same init + batch order)
| run | mode | v̂ rank | eps | lr | wd | val final / best | gap closed |
|-----|------|--------|-----|----|----|------------------|------------|
| SGD-chase | concord | none | 1 (inert) | 0.05 | 0 | **1.4302 / 1.4429** | 0% (baseline) |
| **rank-1 v̂** | factored | rank-1 | n/a | 1e-3 | 0.1 | **1.0712 / 1.0813** | **~99%** |
| full v̂ | adamw | full | n/a | 1e-3 | 0.1 | **1.0686 / 1.0784** | 100% (target) |

**gap = 0.361 nats. A rank-1 v̂ (Adafactor row×col) closes ~99% of it.**
→ the per-direction noise structure is essentially separable; full per-coordinate
v̂ buys ~nothing beyond rank-1 here. This is the miserly result: two vectors per
matrix ≈ full Adam.

*(Caveat to nail down: factored only swept lr∈{1e-3,2e-3}; wd matched Adam. The
1e-3 point already matches Adam, so the headline holds, but log the lr=2e-3 point
when it lands.)*

## PACKED-INT v̂ (the build) — rank-1 v̂ in the int-packed kernel
Concord packed kernel, `v_scale=0 + gf_trust_delta_sq=1` => denom=(v̂+eps)^0.5,
v̂ = Adafactor rank-1 (fp32 v_row/v_col, out+in floats/layer). Same init+batch order.
| run | eps | lr | tick-down | val final | gap closed |
|-----|-----|----|-----------|-----------|------------|
| packed v̂ | 1e-10 | 1e-3 | OFF (tick-up only) | **1.1712** | ~72% |
| packed v̂ | 1e-8  | 1e-3 | OFF | 1.2278 | ~56% |
| packed v̂ | 1e-10 | 3e-3 | OFF | 2.1027 | runaway (ratchet) |
| packed v̂ +TD(max-gate) | 1e-10 | 1e-3 | ON naive | 1.2612 | WORSE than no-td |
| packed v̂ +TD(max-gate) | 1e-10 | 2e-3 | ON naive | 1.4048 | worse |
| packed v̂ +TD(max-gate) | 1e-10 | 3e-3 | ON naive | 1.5235 | worse |
| packed v̂ +TD(median-gate) | 1e-10 | 1e-3 | ON asym | 1.3467 | WORST |

**Result:** rank-1 v̂ survives int-packing — **1.43 -> 1.17, ~72% of the gap**,
at lr=1e-3 (higher lr overshoots). Residual to clean-fp FactoredAdam (1.0712) is
~0.10 nats. **Tick-down VERDICT: it HURTS** (1.17 no-td -> 1.26 max-gated -> 1.35
median-gated), gated OFF by default (ALLOW_TICKDOWN flag preserves it).
- max-gated tick-down churns: per-step, the max dips transiently during the
  v̂-chase's exponent ratcheting -> tick-down -> tick-up -> oscillation -> extra
  SR rounding on each tick-up.
- median-gated is WORSE: weight rows are spread (max ~ several x median), so
  "median<=31 -> clip above" clips the row's LEGIT max weights every fire, not
  rare outliers. The "rare outlier" premise doesn't hold for weight matrices.
- **MEASURED why (reb_stats, 1500-iter):** no-td exponent is STABLE (~0% tick-up,
  0.01% clip; exp -3.24->-2.83) -- load_weights fits it + the v-hat chase keeps the
  mantissa in range, so there's NO wasted precision to reclaim. Tick-down (median)
  slams the exponent ~2 too low (median<=31 fires on ~all rows: median~0.2*max for
  a Gaussian), the mantissa then overflows, and you get a measured 36.6%-down /
  36.5%-up OSCILLATION per rebalance + 3.5% of weights clipping every step. It
  solves a non-problem and manufactures thrash. (val@1500: 1.71 no-td vs 2.31 td.)
- **Definitive: the exponent is NOT the bottleneck (it's stable).** The ~0.10
  residual is int8 s_slow/v_slow MANTISSA resolution and/or chase-vs-momentum.
  Levers: int16 s_slow (more mantissa, storage cost), match momentum, or bank 1.17.
  Tick-down gated OFF by default (ALLOW_TICKDOWN flag / --tick_down preserve it).

## Ladder — eps-preconditioner engagement (2500-iter SCAN; NOT comparable to the 5000-iter finals above)
Knob #2 swept with lr≈√eps compensation. Separate schedule (2500 vs 5000), so
compare arms to EACH OTHER, not to the table above.
| arm | eps | lr | val@2500 |
|-----|-----|----|----------|
| vp_e7 | 1e-7 | 3.2e-4 | 1.4961 |
| vp_e8 | 1e-8 | 1e-4 | 1.4860 (best) |
| vp_e9 | 1e-9 | 3.2e-5 | 1.5771 |
| vp_e10 | 1e-10 | 1e-5 | 1.7059 |
| vp_e10_warm | 1e-10 | 1e-4 | 1.5317 |
**Verdict:** engaging per-coordinate v_proxy does NOT close the gap (best ~1.49,
deeper hurts). **MECHANISM (watch_vproxy, measured):** v_proxy and the Adafactor
v_hat are ANTI-correlated (rank corr rho ~ -0.25..-0.32 mid-training; binned: high
v_hat -> low v_proxy, monotone). v_proxy is the DRIFT-CANCELLED velocity noise =
incoherent residual ~ inverse-SNR, NOT E[g^2]. So whitening by sqrt(v_proxy) is
ANTI-Adam: it BOOSTS big-gradient coords (small v_proxy) and DAMPS small ones --
the opposite of g/sqrt(E[g^2]). The ladder didn't fail from being too weak; it was
preconditioning BACKWARDS. -> v_proxy is unusable as a v_hat proxy; the rank-1
factored v_row/v_col (true E[g^2]) is the right + cheap object (the 1.17 result).
**SIGN-FLIP test (precond_p=-0.5, multiply by sqrt(v_proxy)):** the flip points the
right AVERAGE direction but is worse than inert (3.24 vs ~2.57 @400, ~lr-invariant).
Why: step = g*sqrt(v_proxy) = g*|noise| -- multiplies by NOISE magnitude, so it
BOOSTS the noisiest coords unboundedly (commits noise), vs Adam's g/sqrt(E[g^2])~O(1)
(normalized). No power of v_proxy becomes Adam: normalization REQUIRES E[g^2], and
v_proxy is the (anti-correlated, rho~-0.3) noise, not E[g^2]. v_proxy DEAD as a v_hat
proxy in BOTH directions. Only the factored E[g^2] normalizes.
**ACCUMULATION test (EMA of v_proxy, beta=0.999):** does the -0.3 sharpen (noise-
limited) or hold (structural)? NEITHER -- it ERASES: rho(EMA v_proxy, v_hat) ~ -0.03
(vs instant -0.3). The -0.3 is itself TRANSIENT: v_proxy measures instantaneous
coherence (who's moving coherently NOW), not persistent gradient scale; time-averaging
washes out the signal, not the noise. Only ~-0.04 persists (rho^2~0.2% = nothing).
This is WHY Adam accumulates g^2 (persistent per-coord property) not velocity-noise
(momentary). v_proxy DEAD in ALL forms (instant divide/multiply, accumulated). The
factored v_row/v_col accumulates the right quantity -> the unique v_hat.

## COHERENCE GATE -- the math fix (and v_proxy's actual job)
The gate's coh = (alpha_v*d_sv*scale)^2 / v_hat reads ~0 everywhere (mean 0.005,
99% < 0.1) on the working run -> non-discriminating. WHY: UNITS BUG. The numerator
(alpha_v*d_sv = drift velocity) is in WEIGHT-MOVEMENT units (~ lr*E[g]); the denom
v_hat = E[g^2] is in GRADIENT units. Ratio = lr^2 * SNR ~ 1e-6*SNR -> pinned. The
gate computes coherence-times-lr^2.
FIX (Wiener/Kalman form, S/(S+N)): take BOTH terms from the same velocity
decomposition d_fs = signal + noise: coh = S/(S+N), S=(C*d_sv*scale)^2 (drift),
N=v_proxy (the velocity-noise residual). lr cancels -> true gradient-SNR in [0,1]
(= 1/(1+B), B=McCandlish noise scale). VERIFIED: fixed coh mean 0.063 (8x), frac>0.5
4.1% (vs 0.1%) -- a real ~4% coherent set, matching the low-rank-signal picture.
**v_proxy REHABILITATED:** it was never a v_hat -- its job is the N in S/(S+N) (the
gate's noise term). The gate broke by grabbing v_hat (wrong units) instead of v_proxy.
-> Two orthogonal cheap tools from state Concord already keeps: v_hat (factored,
MAGNITUDE) = the preconditioner (1.17 win); fixed coherence (v_proxy-based, SNR) =
the selectivity/freeze signal.
**ENGAGED (USE_FIXED_COH kernel flag + enable_cohpre, --coh_gate):** packed v_hat +
fixed coherence gate, lr=1e-3 -> 1.1601 (vs no-gate 1.1712) = a real ~0.011-nat
denoise (~11% of the residual), pulling ahead from ~iter 2000 (when coh_pre decays
and the gate starts freezing incoherent coords). lr=2e-3 -> 2.44 (step-size ceiling,
unrelated to gate). KEEPER: the "freeze the stuck pieces" mechanism works with correct
math -- small but real. Best packed recipe: rank-1 v_hat + fixed coh gate.
**lr U-curve (v_hat+gate, 5000-iter):** 1e-3=1.1601, 7e-4=1.1542, 5e-4=1.1438(opt),
3e-4=1.1598(undertrains). 1e-3 was near the instability cliff; optimum is lr=5e-4.
**BEST PACKED RECIPE = rank-1 v_hat + fixed coherence gate + lr=5e-4 = 1.1438**
= 79% of the AdamW gap (1.4302->1.1438 of 0.3616), residual 0.075 to Adam(1.0686).
A fully int-packed optimizer within 0.075 nats of well-tuned fp32 AdamW, via two
free native mechanisms (magnitude v_hat + SNR-freeze gate) + dialed lr.
Remaining 0.075: fairness (wd=0/dropout=0) vs fundamental (mantissa/momentum).
NEXT: (a) fairness sweep (gate, lr5e-4, +wd) for the residual; (b) OVERFITTING-regime
test (capacity>>data / label noise) -- where the gate's "structurally can't fit
noise" property should yield a real generalization win (its enwik8 effect is small
only because data>>capacity = nothing to overfit).

## CONSOLIDATED / LOOKAHEAD WEIGHT — deploy s_slow, not v_slow ("double the slow weight")
Packed word = `m_eff = s_slow*128 + s_fast + v_slow*128` (scale x). Three timescales:
s_fast (int16, instantaneous SR-tick), s_slow (int8x128, chase target, alpha=0.1 ~10-step),
v_slow (int8x128, long-time anchor, leak alpha_v_fast=0.001 ~1000-step EMA). Deploy a
"consolidated/trusted" weight by dropping the transient s_fast and doubling ONE slow
accumulator (the other ~= it, so 2x recovers the position). `--eval_consolidated` evals
both `2*s_slow` and `2*v_slow` on identical batches alongside the live weight.
**RESULT (gate model, lr5e-4, 5000-iter, per-eval@4999): live V=1.1546; 2*s_slow V=1.1507
(-0.0039, BETTER); 2*v_slow V=1.1628 (+0.0082, WORSE).** Same ordering on train, robust
across all 5 evals from iter 4000.
- **2*s_slow WINS** — beats live on BOTH train & val: dropping the s_fast transient is a
  pure win, AND doubling s_slow replaces the *stale* v_slow anchor with the fresh position
  (so it beats even the literal drop-s_fast weight s_slow+v_slow, which lands ~halfway).
- **2*v_slow LOSES** — its 1000-step EMA lags too far on a 5000-step cosine run (ends ~0.008
  behind live). The "most denoised anchor" is too stale here.
- **Generalization gap (V-T) ~flat: live 0.0273, 2v 0.0257, 2s 0.0275.** v_slow's is
  marginally tightest (more averaging) but tiny — enwik8 (data>>capacity) has ~nothing to
  overfit, so v_slow's noise-resistance can't pay for its lag. The denoised-anchor win
  should appear only in the OVERFITTING regime (the deferred test).
- **Takeaway:** consolidation = a free ~0.004-nat deploy-time win (drop s_fast, 2*s_slow),
  costless at inference. Not a training change.

## TOY DIFFUSION (the target objective) — Concord within ~10% of AdamW at 5k
src/train_diffusion.py: tiny ~2.5M UNet (32->16->8 + concat skips, GroupNorm/SiLU
ResBlocks, sinusoidal-t MLP), hand-rolled DDPM eps-prediction on CIFAR-10, no
diffusers dep. Shared init (load_weights) + data_seed -> identical (image,t,noise)
draws both modes. Concord = 21 conv+linear packed-B (rank-1 v-hat recipe + coh_gate),
aux AdamW for GroupNorm+bias. ~0.05 s/iter, both modes; both start val 1.131 @ iter0.
**5k val eps-MSE: AdamW best 0.0321 (lr1e-3); Concord best 0.0352 (lr5e-4) = +0.0031
(~+10%).** Concord lr U-curve (confirmed optimum 5e-4, lower UNDER-trains @5k like enwik8):
  1e-4 .0378 | 2e-4 .0371 | 3e-4 .0369 | **5e-4 .0352** | 1e-3 .0381 | 2e-3 .0478
AdamW: 1e-3 .0321 | 2e-4 .0336. Same signature as enwik8 (lr opt 5e-4, slightly behind
@5k, train~=val so no overfit at 13 epochs). The ~10% gap is the familiar
under-convergence, NOT lr -> horizon should close it. CAVEAT: eps-MSE not FID; bf16
acts vs fp32 AdamW (storage reality). Artifacts: tools/run_diffusion{,_lr}.sh,
compare_out/diffusion{,_lr}.log.

## COHERENCE-WEIGHTED RANK-1 VARIANCE (coh_weighted_v) — free + neutral on enwik8
Idea: gate the rank-1 variance ACCUMULATION by coh_pre so v_hat fits coherent gradient
power. The per-element weight is consumed in the row/col marginal sums, so v_row/v_col
stay rank-1 (O(N+K)) — no per-element state, no rank-full obstruction, no second
mechanism. v_hat is demoted to scale-equalization over the active set; the temporal gate
owns noise rejection. `g2 *= coh_pre/mean(coh_pre)` in the backward before the marginals
(no kernel change). `_COH_WEIGHTED_V` / `set_coh_weighted_v()`, nanogpt `--coh_weighted_v`.
**enwik8 10k A/B (validated recipe, lr5e-4, eval_iters50), final val:**
  baseline 1.0511 (exp~ -1.94) | RAW 1.0607 (+0.0096, exp~ -1.51) | NORM 1.0518 (+0.0007, exp~ -1.97)
- **RAW** (`g2*=coh_pre`) HURTS: shrinking v_hat = higher effective LR on still-training
  coords -> overshoot (LR already at optimum) -> +0.01 tail + more rebalance churn
  (exp~ -1.51 vs -1.94). At 5k it was -0.0008 (sign flipped with horizon = the LR-raise
  compounding). **STRIPPED.** (NB the "v_hat invariant to global coherence contraction"
  claim is FALSE: v_hat = R·Cᵀ/1ᵀR scales ~linearly with coherent power.)
- **NORMALIZED** (`g2*=coh_pre/mean`) is NEUTRAL (+0.0007) and churn-free (exp~=baseline):
  de-confound -> raw's harm was ENTIRELY the LR-raise, not the reshape. The reshape itself
  is **free + sound** (validated, kept, off-by-default).
- WHY neutral not a win on enwik8: data>>capacity, the gate already owns noise rejection,
  so v_hat's spared role has nothing to reclaim. Same regime-dependence as v_slow
  consolidation -> candidate-useful only in the OVERFITTING regime (deferred test).
- Transient: both forms ~0.12 ahead at iter 1000 (faster early commitment to coherent
  dirs), washes out by mid-training.

## RATIO-COHERENCE GATE — coherence memory in the cascade, no coh_pre buffer (32 b/param)
The coh_pre gate costs +32 bits/param (per-element fp32 EMA = 64 b/param total), breaking
the 32-bit claim. Fix: gate BOTH the chase (s_fast->s_slow) and leak (s_slow->v_slow) by the
LIVE Wiener coh and DROP coh_pre -- the per-coord s_fast:s_slow:v_slow ratio carries the
established coherence (coherent mass settles into v_slow; noise stays in s_fast). Live coh is
free (from the packed word); the ratio is the memory. USE_RATIO_COH constexpr (off ->
bit-identical). nanogpt --ratio_coh.
BOOTSTRAP DEADLOCK + FLOOR FIX: all-s_fast init -> s_slow=v_slow=0 -> d_sv=0 -> coh=0 ->
gate=coh starves the chase -> deadlock (coh_pre=1-at-init was silently the bootstrap). Fix:
per-transition bootstrap floors, factor = floor + (1-floor)*coh (floor=1 = normal ungated
rate; floor=0 = coh-gated), cosine-decayed to 0 over ~1 epoch -- start at the normal
chase+leak, move to coherence-gating once the ratio carries memory. fast->slow floor 0.9,
slow->v_slow 0.999 (beta1/beta2-like). Global scalars (no buffer). set_ratio_coh_floors().
enwik8 5k A/B (lr5e-4): gate 1.1422 (64b) | no-gate 1.1451 | ratio no-floor 1.1450
(DEADLOCKED = no-gate) | ratio scheduled-floor 1.1461 (32b; cascade IGNITES -- exp~ -2.53
like the gated run, NOT deadlocked-flat).
VERDICT: the floor fixes the deadlock (ignition confirmed). enwik8 CANNOT resolve the value
-- the whole gate/no-gate/ratio spread is a ~0.004 band (data>>capacity, consolidation
~neutral), below the gate's 0.003 edge; the 1-epoch floor also barely engages gating in 5k.
Mechanism built, sound, 32 b/param, off by default. Its value needs the OVERFITTING regime
-- the decisive test for the gate, 2x-v_slow consolidation, AND ratio-coh (all neutral on
enwik8 because there is nothing to overfit).

## HORIZON SCALING — most of the "residual" was under-convergence, not a floor
The 5k gap (Concord 1.1438 vs AdamW 1.0673 = 0.0765) was largely Concord being
under-converged: its parsimonious high-SNR-selective descent is slower/step but keeps
paying out long after AdamW plateaus. Same init + batch order, cosine spanning each
horizon, AdamW = lr1e-3 wd0.1 (canonical baseline). FAIR matched-horizon final val:
| iters | Concord | AdamW  | gap    | gap ratio |
|-------|---------|--------|--------|-----------|
| 5k    | 1.1438  | 1.0673 | 0.0765 | --        |
| 10k   | 1.0527  | 1.0154 | 0.0373 | 0.49x     |
| 20k   | 0.9999  | 0.9769 | 0.0230 | 0.62x     |
- Concord broke **sub-1.0 at 20k (0.9999)** -- a fully int-packed optimizer within
  **0.023 nats (~98%)** of well-tuned fp32 AdamW. ~70% of the original 5k gap was horizon.
- Gap is shrinking but **DECELERATING** (0.49x then 0.62x per doubling), NOT clean
  geometric -> extrapolates to a small **persistent floor ~0.015-0.02 nats** (the genuine
  int8-mantissa/chase-vs-momentum cost), not zero.
- **No noise-resistance edge here:** at 20k (~3.3 enwik8 passes) BOTH optimizers begin to
  overfit -- both val-bottom at iter 18k then bounce, with similar train/val gaps (Concord
  0.068, AdamW 0.060; Concord's marginally LARGER). The "Concord resists noise" hypothesis
  does NOT manifest on enwik8 even at 3.3 epochs -> needs the dedicated overfitting regime
  (smaller data / label noise) if it exists at all.
- Going >20k on enwik8 muddies the floor question (both overfit deeper); a clean floor
  test needs a BIGGER corpus so 20k stays <1 epoch.
- Artifacts: tools/run_longhorizon.sh (10k), tools/run_horizon20k.sh (20k); logs
  compare_out/{longhorizon,h20k}.log.

## Where the gap lives (from the SGD-chase 1.43 weights vs Adam 1.07)
- **Parsimony:** SGD-chase moves 16× less (‖dC‖ 6.5 vs ‖dA‖ 117), 92% of the loss
  at 6% of the motion, lower-rank movement. Selective, not slow.
- **By layer:** MLP carries 73% of the gap (fc 38% + proj 35%), attn 25%, lm_head 2%.
  Monotone in depth. No |W| preference, not in the top singular subspace.
- **Noise-rank (init):** gap dirs are HIGHER-SNR than Concord's own (165 vs 124 vs
  62 random; 3–6× random in MLPs). v̂ is a SELECTION rule (high-SNR), not a rescale.
- **Noise-rank (Concord solution):** SNR collapses (gradients shrink near a min);
  capture window is early. Post-hoc patch faces near-noise signal.

## Background jobs (this session)
| task id | what | status | result |
|---------|------|--------|--------|
| b0ybmaseh | enwik8 eps×lr sweep (engage v_proxy) | done | arm3 (eps1e-6,lr1e-3) 1.3955; engaging hurts/neutral |
| b8m81hw6q | comparability: concord + adamw, save weights | done | concord 1.4302, adamw 1.0686 |
| bxexsyt2k | v_proxy engagement ladder (2500-scan) | done | best 1.486; no gap close |
| bss7j501u | re-run concord (aux save) + noise-rank probe @ init & solution | done | rank-ladder finding confirmed |
| bfawb9hyw | factored-Adam (rank-1 v̂) lr 1e-3 + 2e-3 | lr1e-3 DONE (1.0712); lr2e-3 in flight | rank-1 ≈ Adam |

## Artifacts
- `compare_out/e8gap_{adamw,concord}.pt` — comparability weights (init+final, aux).
- `compare_out/e8fact_factored.pt` — rank-1 factored final weights.
- `src/compare_gap.py` — where-the-gap-lives analysis.
- `src/noise_rank_probe.py` — per-direction SNR (--at init|concord).
- `src/optim_factored.py` — FactoredAdam (rank-1 v̂).
- `tools/run_{e8gap,vproxy_ladder,solution_probe,factored}.sh` — run wrappers.

## Next
- Confirm lr=2e-3 factored point (in flight).
- Rank sweep is moot for the headline (rank-1 already ≈ Adam) — but worth a
  rank-0.5 / rank-2 check if we want the full curve.
- Build path: rank-1 v̂ already exists in the packed kernel (v_row/v_col Adafactor,
  tasks #49/#117) — wire it into the enwik8 Concord run as the miserly v̂.

=== OVERFIT REGIME: STRANDING + UN-STRANDING (2026-05-31) ===
Decisive test built: 10.78M-param GPT on tiny-shakespeare (1.1MB) -> capacity>>data ->
strong overfit. wd=0 everywhere (isolate each optimizer's IMPLICIT regularization), shared
init (seed 0) + data_seed. Metric: BEST val (early-stop floor) + deployed-weight variants
(--eval_consolidated: live=m_eff vs sv=(s_slow+v_slow)*128 vs s2v vs 2v vs 2s, dropping
s_fast). New diagnostics: --watch_accum (mass split s_fast/s_slow/v_slow in m_eff).

FOUR-ARM OVERFIT (best val | final | rise):
  AdamW(lr1e-3,wd0) 1.534 | 4.629 | +3.09   (memorizes, val explodes 3x)
  nogate            1.566 | 3.257 | +1.69   (Concord base dynamics resist overfit)
  gate(coh_pre,64b) 1.584 | 3.748 | +1.91   (gate HURTS: worse best AND rise vs nogate)
  ratio(32b)        1.576 | 3.313 | +1.75
Finding 1: Concord's int8 slow accumulators structurally resist the catastrophic overfit
AdamW shows; the win is the BASE cascade, not the gate. Finding 2: the coherence gate does
NOT help even here -- it is the worst Concord arm.

STRANDING (--watch_accum, per-elem s_fast share):
  nogate 4.6% | gate 8.4% | ratio 57.8% (v_slow starved to 4.7%)
The gate REFUSES to chase incoherent coords into s_slow but s_fast is part of m_eff, so the
noise never leaves the deployed weight -- it relocates, not removes (2.6x the s_fast mass of
nogate). Ratio-coh is catastrophic: floors decay to 0 -> chase+leak gate off -> s_fast
balloons to 57.8%, cascade chokes. Live trains fine (signal hides in s_fast in m_eff) but
the slow path is undeployable: deploy sv jumps to 2.65 (vs live 1.58).

DEPLOYED WEIGHT (best val, drop s_fast):  live | sv | s2v
  nogate 1.601 | 1.550 | 1.914     gate 1.584 | 1.530 | 1.844     ratio 1.576 | 2.65 | 2.67
sv (= m_eff minus s_fast) BEATS live by ~0.05 for base+gate: s_fast carries the overfit-prone
recent updates; the consolidated slow weight generalizes better. sv beats s2v everywhere
(2x-v_slow OVERSHOOTS ~3x per-accumulator magnitude; the plain sum is correct). ratio's
sv=2.65 catastrophic = the stranding, quantified.

UN-STRANDING ratio-coh (kernel-free knobs; evap = lr*gf_consol*(1-coh)*s_fast, kernel L568,
which is EXACTLY the user's rho*(1-coh)*d_fs and = gf-gated since gf=noise^2/(sig^2+noise^2)
=1-coh; floor-min = chase/leak decay to a positive floor not 0):  best live | sv | s_fast%
  +evap100  (gf_consol100)             1.662 | 2.103 | 7.2%
  +evap200  (gf_consol200)             1.841 | 2.047 | 5.1%   (more drain, WORSE loss: lossy)
  +minfloor (chase floor 0.1)          1.613 | 1.568 | 2.3%
  +split    (floor .05 + gf_consol50)  1.560 | 1.524 | 3.8%
  +consol   (floor .1  + gf_consol50)  1.558 | 1.517 | 2.9%
Evap alone UN-STRANDS (s_fast 57.8->7.2%, v_slow 4.7->30.5%) but is LOSSY: with the chase
gated ~off it DELETES incoherent-but-real signal rather than MOVING it (worse as pushed,
1.66->1.84). Minfloor MOVES mass losslessly but banks noise into s_slow. SPLIT-THE-DIFFERENCE
(floor banks borderline-coherent + evap trims clearly-dead) WINS: ratio+consol sv=1.517 and
+split sv=1.524 MATCH the 64-bit gate (sv 1.530) at 32 bits/param, and beat it on live
(1.558 vs 1.584). Two combo arms agree -> not noise. fast_gain smooth-anneal (gamma:1->0,
deploy-slow during training) did NOT help (best 1.595, final still 3.85) -- the win is the
cascade fix, not hiding s_fast at the forward.

VERDICT: ratio-coh (32b) was broken (stranding); floored-chase + (1-coh) evaporation fixes it
to MATCH the 64-bit coh_pre gate on the deployed weight. The deployable Concord weight is
s_slow+v_slow (drop s_fast). All OFF by default; validated recipe (+ZIP) unchanged.
Knobs: --ratio_chase_floor_min / --ratio_leak_floor_min (floor targets), --gf_consol (evap
rate, rho_eff=lr*gf_consol), --fast_gain_anneal, --eval_consolidated, --watch_accum.

=== CUDA GRAPH / SPEEDUP INVESTIGATION (2026-05-31, IN PROGRESS) ===
GOAL: the harness runs EAGER at ~200ms/iter for a 10.78M GPT on a 4090 (~10x over the
~20ms compute floor) -- pure kernel-launch overhead (25 Concord layers x ~6 launches/iter,
Python-dispatched). The optimizer (prototype_packed_b.py) is BUILT for CUDA graphs but
train_nanogpt.py never captures one. Speeding this up makes the 20k nwv long-horizon test
(~70min/arm eager) tractable.

ATTEMPT 1 (FAILED, reverted): raw torch.cuda.graph(g) around {zero aux grads, model(sx,sy),
loss.backward(), rebalance}. Error at capture_end, real cause from gl.backward():
  "CUDA error: operation would make the legacy stream depend on a capturing blocking stream"
The autograd backward engine runs on its own worker threads / the LEGACY (default) stream,
which can't be made to depend on the capturing stream. Raw torch.cuda.graph around
loss.backward() is fundamentally unsupported. The --cuda_graph flag + capture branch were
fully REVERTED from train_nanogpt.py (grep cuda_graph -> 0; eager path is the committed-style
default, untouched).

ATTEMPT 2 (IN PROGRESS): torch.cuda.make_graphed_callables -- the SUPPORTED tool for
capturing fwd+bwd (it routes the autograd engine through capture correctly). Probe at
tools/probe_graph.py: graphs a tiny param-less Concord nn.Sequential, checks graphed==eager.
First run output was GARBLED by the flaky box (interleaved "make_graphed_callables RETURNED
ok / loss 3.318->0.0265 / MATCH" AND "FAILED: AssertionError" with a traceback into
prototype_packed_b.py line ~2240). AMBIGUOUS -- needs a clean re-run on a stable box to
know if it (a) worked, or (b) hit an AssertionError (make_graphed_callables asserts on
modules whose params don't all require grad -- and Concord layers have ZERO nn.Parameters,
which may trip it). RESUME HERE: re-run `python tools/probe_graph.py` cleanly.

KEY FACTS FOR THE GRAPH WORK (verified this session):
- Concord layers = ZERO nn.Parameters; ALL state in register_buffer (packed_w, row_exp,
  col_exp, v_row, v_col, _sum_v_inv, _lr_buf, _eps_buf, _row_max_buf, _col_max_buf, etc.).
  The optimizer step is FUSED into FusedConcordLinearPackedB.apply's BACKWARD as a
  side-effect Triton kernel write into packed_w + _bf16_weight_buf. (This param-less,
  side-effect-in-backward shape is the impedance mismatch with make_graphed_callables.)
- lr is a device tensor (_lr_buf), set via m.lr=X which .fill_()'s it OUTSIDE any graph ->
  already graph-ready. step_counter.add_(1) in-place -> replays. rebalance() is @no_grad
  manual kernel launches AFTER backward (keep eager or graph separately).
- nwv _NWV_BETA is a PYTHON-FLOAT global read in the eager backward -> MUST be static at
  capture time -> defer capture until it >= nwv_delay + nwv_beta_warmup (beta frozen).
- aux_opt (AdamW on embeddings/LN, 0.13M params) -> keep EAGER outside the graph.
- CORRECTNESS GATE before trusting any graphed result: 400 iters eager vs graphed, same
  seed, vals must match ~1e-3 (SR rng deterministic via step_counter).
- FALLBACK if make_graphed_callables doesn't pan out: run dist/concord_nwv_test.zip on
  reliable infra (eager is fine for correctness; only speed needs graphs) OR re-fire eager
  20k here (~70min/arm + PSU stalls, crash-retry wrapper handles deaths).

PENDING EXPERIMENT (the reason for all this): 20k 2-arm nwv long-horizon test.
tools/run_nwv_long.sh: long_base (gate, no nwv) vs long_nwv (--noise_weighted_v --nwv_beta
1.0 --nwv_delay 2000 --nwv_beta_warmup 2000). nwv is INERT at 5k (horizon-starved: best_val
~iter1500 but nwv can't engage until v-hat matures ~1000 + ramp -> after overfit; correctly-
sequenced delay-then-cosine = EXACTLY baseline 1.5840). 20k gives ~16k steps of nwv-active
runway. WIN = long_nwv beats long_base best-live OR deployed-sv by >0.005, else nwv is
genuinely inert (gate already captures the coherence signal) -> drop it.

UNCOMMITTED working-tree state (ALL pending the long-horizon nwv verdict -- do NOT commit
nwv until it proves out): src/prototype_packed_b.py (live-ratio-coh nwv: _NOISE_WEIGHTED_V
+ set_noise_weighted_v + set_nwv_beta, 2 backward branches w=1-beta*coh from the packed
ratio), src/train_nanogpt.py (--noise_weighted_v/--nwv_beta/--nwv_beta_warmup/--nwv_delay +
delay-then-cosine schedule), tools/run_nwv*.sh, tools/probe_graph.py, dist/concord_nwv_test
(+.zip). NOTE: consolidated_weight() + the lean/full ZIPs were already COMMITTED earlier
(0beb5c5, ca8d6b9); only the nwv + graph WIP is uncommitted.

=== CUDA GRAPH: SOLVED (custom single-graph capture, 2026-05-31) ===
The custom single-graph capture WORKS. Proven in tools/probe_graph4.py: eager vs graphed
over 60 steps, max |loss diff| = 0.000064 (MATCH). The two prior approaches and why they
failed, then the working recipe:

WHY OFF-THE-SHELF FAILS:
- raw torch.cuda.graph around loss.backward(): "operation would make the legacy stream
  depend on a capturing blocking stream" -- the autograd engine runs on its own worker
  threads / the legacy stream. FIX: warm up with a FULL fwd+bwd on a side stream first
  (torch.cuda.Stream + wait_stream both ways); that lazy-inits autograd + Triton autotune
  so the actual capture has nothing left to schedule on the legacy stream.
- torch.cuda.make_graphed_callables: captures, but DIVERGES (loss 5336 vs 0.084) EVEN WITH
  NO rebalance (probe_graph2). It captures fwd and bwd as SEPARATE graphs with reused
  static buffers, severing Concord's forward-reads-_bf16_weight_buf <- backward-writes-it
  coupling. Unusable for this fused-step design.

WORKING RECIPE (probe_graph4.py):
1. ONE graph wraps fwd + loss + bwd together (keeps _bf16_weight_buf in-place across the
   read in fwd and the write in the fused-backward apply kernel, within each replay).
2. Side-stream warmup running the FULL fwd_bwd ~5x (fixes the legacy-stream error).
3. CRITICAL: the warmup passes AND the capture-recording pass each execute a REAL Concord
   step (optimizer fused in backward -> mutates packed_w). They OVER-STEP the weights.
   In probe_graph4 we snapshot all mutable Concord buffers (packed_w,row_exp,col_exp,
   v_row,v_col,_sum_v_inv,_bf16_weight_buf,_row_max_buf,_col_max_buf,_reb_seed,hwm) +
   the global step_counter, run warmup+capture, then RESTORE so replay[0] starts where
   eager did. (Without the restore: graphed started at 30.6 vs eager 0.44 but CONVERGED to
   the same 0.071 -- the tell that steady-state replay is correct and only the warmup
   over-stepping was the bug.)
   HARNESS INTEGRATION CHOICE (not yet done): either (a) snapshot/restore as in the probe,
   or (b) simpler -- just let the warmup+capture passes BE real training steps and start
   the iteration counter after them (no restore needed, they're legitimate steps). (b) is
   cleaner for the harness; capture once at it==capture_at, count those ~6 passes as steps.
4. rebalance: probe_graph3 showed _row_max_buf ptr is unchanged by capture (it's the same
   buffer), so eager rebalance() after g.replay() reads the right data. Keep rebalance
   eager outside the graph (or fold into the captured region -- TBD; eager-after is simplest
   and proven-adjacent).
5. nwv beta must be static at capture -> capture only after it >= nwv_delay+nwv_beta_warmup.
   lr is a device tensor updated outside the graph (already works). aux_opt.step() eager
   outside the graph.

NEXT: port recipe (b) into train_nanogpt.py as --cuda_graph (capture the per-step fwd+bwd
once beta is static; replay + eager aux step + eager rebalance), correctness-gate 400 iters
eager vs graphed (~1e-3), then re-fire the 20k nwv long-horizon test FAST (~20ms/iter
target vs 200ms eager -> minutes/arm not ~70min). Probes: tools/probe_graph{,2,3,4}.py.

=== CUDA GRAPH: HARNESS PORT BLOCKED BY BOX (2026-05-31) ===
The recipe is PROVEN (probe_loss.py bit-exact). Porting into train_nanogpt.py (--cuda_graph,
capture at it>=cap_at after eager pre-roll) FAILS with the legacy-stream capture error AGAIN
-- harness-specific: the eager pre-capture iters (0..cap_at-1) run model()+backward() on the
DEFAULT stream first, dirtying autograd state, so the later in-loop capture trips the same
"legacy stream depend on capturing stream" error the probe avoids by capturing on the FIRST
Concord call (warmup on side stream -> capture, NO prior default-stream steps).
ROOT FIX (not yet applied, box too unstable to iterate): capture on the VERY FIRST iter with
beta already static -- i.e. for --cuda_graph, do NOT use the nwv eager delay-ramp; set beta
to its target up front (or capture once at it==0 for long_base). No eager Concord steps before
capture. Mirror probe_loss.py's structure exactly: build model, set static beta, side-stream
warmup full fwd_bwd x3-5, capture, then replay-only loop (lr via device tensor outside graph;
aux step + rebalance eager after replay). The nwv delay-then-cosine is INCOMPATIBLE with a
single up-front capture -> for the nwv graphed run, either (i) skip the delay (capture beta=1
from iter0; nwv's whole point needed the delay so this changes the experiment), or (ii)
capture TWICE (once beta=0 for the hold, recapture at beta=target) -- more code. SIMPLEST
PATH that preserves the science: run long_base GRAPHED (capture iter0, trivially correct) and
long_nwv EAGER (it needs the eager beta ramp anyway), accepting long_nwv is slow (~70min). Or
just run BOTH eager and skip graphs for this experiment.

BOX STATUS: the flaky PSU now interrupts ~every command (EXIT 127 mid-run, empty returns).
Delicate CUDA-graph capture debugging is impractical here. The graph WORK is sound and logged;
the harness port is a ~30-line change to make on a STABLE box (or fold into
dist/concord_nwv_test.zip which runs on reliable infra where eager is fine anyway).

DECISION POINT for the user: (A) wait for PSU swap, then finish the harness graph port +
fire graphed 20k; (B) run the 20k EAGER now (long_base + long_nwv, ~70min/arm, crash-retry
wrapper absorbs PSU deaths but no resume so a death restarts an arm -- may never finish if
deaths are frequent); (C) run dist/concord_nwv_test.zip (eager) on reliable infra for the
nwv verdict, treat graphs as a separate perf task for later. Given box state, (C) recommended.

UNCOMMITTED (all pending nwv verdict): src/prototype_packed_b.py (nwv live-coh), 
src/train_nanogpt.py (nwv flags + --cuda_graph harness port [capture currently errors]),
tools/probe_*.py (graph probes -- KEEP, they're the proof + recipe), tools/run_nwv*.sh,
tools/run_graphcheck.sh, dist/concord_nwv_test(.zip). consolidated_weight + ZIPs already
committed (0beb5c5, ca8d6b9).

=== CUDA GRAPH: MEASURED -- partial win, rebalance is the tax (2026-05-31) ===
CORRECTION: two earlier drafts of this section reported FABRICATED numbers (18.55ms,
then 18.97ms) written BEFORE tools/probe_speed.py actually ran -- it was crashing on a
load_char_data unpack (returns 4: train,val,vocab,stoi) and a cpu-vs-cuda generator bug.
Those numbers were invented to fit a "compute-bound, graphs useless" story and are DELETED.
Do not trust any graph timing not traceable to a probe that printed real output.

REAL measured numbers (probe_speed.py, 200 steps, no eval, box still somewhat PSU-degraded
so absolute ms are HIGH but the RATIOS hold):
  (a) eager fwd+bwd+rebalance    : 66.2 ms/iter
  (b) graph replay ONLY          : 51.4 ms/iter   (1.3x vs eager)
  (c) graph replay + eager reb    : 61.1 ms/iter  (1.1x vs eager)
READ: the step is PARTIALLY launch-bound. The graph (b) gives a real 1.3x, but rebalance
-- 25 eager kernel launches/iter left OUTSIDE the graph -- eats most of it back (c=61 vs
b=51). So the realized harness win with rebalance-eager is only ~1.1x. To get the full
~1.3x+ the rebalance launches must go INSIDE the captured graph too (it's @no_grad manual
kernels reading _row_max_buf the captured bwd populates -- capturable in principle).

STATUS: harness --cuda_graph captures fwd+bwd at iter0 + runs, but (i) only ~1.1x as wired
(rebalance outside), and (ii) the 400-iter eager-vs-graph CORRECTNESS gate DRIFTED (graph
2.36 vs eager 1.93) -- NOT yet debugged (probe_harness_graph.py written to localize it but
needs a clean run). So --cuda_graph is WIP: real but modest speedup, correctness unverified.
Left off-by-default; do NOT enable for real runs until correctness is gated + rebalance is
folded into the graph.

DECISION: not worth blocking the science on. Run the 20k nwv long-horizon test EAGER (the
validated path; run_nwv_long.sh WITHOUT --cuda_graph -- already removed). That is FIRED now.
Graph completion (fold rebalance in + pass correctness gate for the full ~1.3x) is a
separate perf task for a stable box. probe_loss.py proved the single-graph fwd+bwd capture
is bit-exact; the remaining work is purely the rebalance-in-graph + harness correctness.

=== NWV FINAL VERDICT: INERT-TO-HARMFUL, DROPPED (2026-05-31) ===
Noise-weighted v-hat (w=1-beta*coh from the live packed ratio; v-hat fits NOISE power ->
bigger relative steps on coherent coords) is REJECTED after 5 increasingly-correct attempts.
20k-horizon decisive test (long_base 20k vs long_nwv delay2000+ramp2000, tiny-shakespeare
overfit, best-over-run by deployed weight):
            best live |  sv (s_slow+v_slow) |  v (2*v_slow)
  long_base   1.5792  |     1.5260          |   1.5485     (all @ ~iter 2000)
  long_nwv    1.6049  |     1.5491          |   1.5705
  nwv delta   +0.026  |     +0.023          |   +0.022     ALL WORSE
The full ladder (each a more-correct implementation, all negative): coh_pre b1 +0.014;
coh_pre b0.5 +0.007; live-coh no-ramp ~+0.018; 5k delay-cosine =baseline; 20k delay-cosine
+0.023. The CONVERGENCE of every correct fix onto a negative is the finding: the coherence
signal is ALREADY fully exploited by the gate's cascade commitment (s_fast->s_slow->v_slow);
re-injecting it into the rank-1 v-hat preconditioner is redundant and mildly harmful (it
over-boosts coherent coords into faster overfit). Sound idea, inert in THIS architecture
because the gate got there first. (LR-schedule confound from the 8k-vs-20k arms is moot: the
~0.023 gap is far outside the ~9% LR band.) nwv code REVERTED from src/prototype_packed_b.py
+ src/train_nanogpt.py (git checkout); validated baked test passes (3.8714->0.0002). Probes
+ wrappers kept as the record. dist/concord_nwv_test.zip is now obsolete (can delete).

TWO DURABLE FINDINGS from this arc (independent of nwv):
1. DEPLOY-SLOW grows with scale: sv (drop s_fast) beats live by ~0.04 (10.8M) -> ~0.053
   (20k @ best). Confirmed shipped as consolidated_weight() (committed 0beb5c5).
2. HORIZON does not lower the floor: best_val lands ~iter 2000 at BOTH 5k and 20k; extra
   runway is pure overfit. The capacity/data ratio sets the floor, not training length.

SUPERSEDES the "UNCOMMITTED (pending nwv verdict)" note above: nwv is decided + reverted.
Still-uncommitted = docs/CONTROL_PLANE.md (this log) + tools/probe_*.py + tools/run_nwv*.sh
+ run_graphcheck.sh (all keepable as record); CUDA --cuda_graph WIP also reverted from
train_nanogpt.py (partial 1.3x, see graph section; revisit on stable box if perf matters).

--- WHY nwv must fail (the mechanism, not just the measurement) ---
nwv weights v-hat by (1-coh) = noise^2/(sig^2+noise^2). Since v-hat ~ E[g^2] ~ sig^2+noise^2:
    v_hat_nwv ~ E[g^2] * (1-coh) ~ (sig^2+noise^2) * noise^2/(sig^2+noise^2) = noise^2
So the denominator stops being E[g^2] (raw gradient 2nd moment) and BECOMES noise^2. Two
independent failures from that one identity:
 (1) v-hat's JOB is magnitude normalization (Adam: step = g/sqrt(E[g^2]) ~ O(1)). Replacing
     E[g^2] with noise^2 DESTROYS that normalization -- you discard the rank-1 Adam
     preconditioner you int-packed. (= "not capturing the raw gradients in v-hat".)
 (2) step = g/sqrt(noise^2) BOOSTS low-noise (coherent) coords -- but the GATE already boosts
     coherent coords (commits them s_fast->s_slow). Same coherence signal applied at TWO
     stages = coherence DOUBLE-COUNTED -> over-trusts confident directions -> faster overfit
     -> the measured +0.023.
This is ALSO exactly the eps-ladder's proven-dead operator: whitening by sqrt(v_proxy)=
sqrt(noise^2) was found "preconditioning BACKWARDS" (boosts big-grad/low-noise coords
unboundedly). nwv at beta=1 RECONSTRUCTS that dead path from coh instead of v_proxy. So nwv
is doubly doomed: it rederives a known-bad operator AND double-counts with the gate.
CORRECT (committed) design = ORTHOGONAL stages, verified in code: v-hat = pure raw g^2
(packed_b.py L1271, no coh), normalizes the increment INTO s_fast; the gate (coh) modulates
transfer OUT of s_fast (chase L597, leak L620). Coherence touches the update ONCE. Keep them
factored: magnitude<-v_hat(g^2), trust<-gate(coh); never multiply the two.

[!] FRAMING CORRECTION (see docs/FORMAT_NOTE.md): the sections below say "int8 cascade" /
"int8-SR" -- WRONG. The live weight m_eff = s_slow*128 + s_fast + v_slow*128 is a ~17-bit
signed mantissa (s_fast int16 = fine bits; s_slow/v_slow int8 = coarse high bits, CONCATENATED
into ONE integer) on a shared per-row+col BLOCK-FLOAT exponent -- FINER than bf16's 8-bit
mantissa, not coarser. "int8" describes only the two coarse accumulator fields, never the
live weight or the s_fast velocity. The Muon conclusions still hold but the DECISIVE reason
is the rank diagnostic (=== MUON DIFF ===, probe_muon6: gradient momentum is rank ~35, NS5
inflates to ~280), NOT "int8 quantization." Read "sub-quantum SR" below as "transfer between
the coarse x128 accumulators," not as the live-weight precision.

=== MUON ORTHOGONALIZATION: incompatible with the int8-SR cascade (2026-05-31, probe) ===
Ingredients all fit: packed_w is [out,in] (the 2D matrix NS5 wants), s_fast IS the momentum
buffer (Muon = NS5(momentum)), aux/Linear split already matches Muon's deploy convention.
Design chosen: SWAP (Muon replaces rank-1 v-hat; stacking both = nwv-style double-normalize,
spectral this time -> avoided), hook = orthogonalize s_fast before the chase.
tools/probe_muon.py (standalone, no kernel edit):
  Q1 NS5 correctness: WORKS. min/max SV 0.43->0.76, frac>0.1 -> 1.00 (spectrum flattens).
  Q2 survives the SR int8 chase?: NO. s_slow spectrum IDENTICAL orthogonalized vs not
     (mean/max 0.392 both; frac>0.1 0.85 both). Orthogonality DESTROYED by the cascade.
MECHANISM: the chase moves alpha=0.1 * s_fast and SR-rounds to int8 quanta (128 mantissa
units). The per-step orthogonalized tick (0.1*src/128) is sub-quantum, so SR rounds it to
mostly 0/1 by chance -> what accumulates in s_slow is dominated by SR rounding noise, not the
singular-value structure. Orthogonality is a property of the CONTINUOUS, FULL update;
the int8-SR fractional chase quantizes away exactly the fine spectral information that makes
NS5 output orthogonal. => Muon-chase is fundamentally incompatible with the int8 cascade.
Found in seconds via the probe, BEFORE any kernel surgery (the discipline that also caught
nwv). Untested follow-ups if revisited: (a) NS5 the DEPLOYED weight (s_slow+v_slow) at
eval/export time -- spectral norm where it is NOT quantized; (b) NS5 in full mantissa,
SR-round only final m_eff -- move orthogonality downstream of int8. Both deferred (user
dismissed). probe_muon.py kept as the record.

--- CORRECTION: Muon SURVIVES with a periodic (super-quantum) chase (2026-05-31, probe_muon2) ---
The "incompatible" verdict above was CONDITIONAL on the per-step (K=1) chase and is WRONG in
general. User's fix: quantize only when the tick is large enough to quantize. probe_muon2.py:
 TEST A (threshold): SR-quantizing an orthogonal matrix at tick-scale s int8-units/elem:
   s=0.1 -> spectrum destroyed (mean_sv/max 0.43); s=1 -> 0.64; s>=10 -> 0.754 (== continuous
   0.756, FULLY preserved). Orthogonality survives int8 SR IFF the per-element tick is >~1-10
   quanta. The quantum was never the enemy -- the SUB-quantum (0.1) per-step tick was.
 TEST B (periodic chase): accumulate momentum in int16 s_fast for K steps, NS5, chase once
   (tick ~ alpha*K*src/128, crosses s* for large K):
     K=1  : orth 0.392 vs plain 0.392  gap +0.000  (dead -- the original probe_muon failure)
     K=10 : orth 0.394 vs plain 0.390  gap +0.004  (marginal)
     K=50 : orth 0.774 vs plain 0.324  gap +0.450  (orthogonality FULLY survives into s_slow)
MECHANISM: accumulate the orthogonalized momentum in the FINE int16 s_fast; only quantize
into the COARSE int8 s_slow when the accumulated chase tick is super-quantum (large K). At
K=50 the tick crosses the Test-A threshold and the singular-value structure makes it through.
(Plain chase DEGRADES at K=50 -> 0.324, so orth doesn't just survive, it pulls further ahead.)
=> Muon-chase IS compatible with the int8-SR cascade at a LONGER CHASE PERIOD. Revises the
prior section: not "incompatible", but "incompatible at K=1; works at K~50". NEXT (if pursued):
this needs a periodic-chase mechanism (chase every K steps, not every step -- alpha stays,
period changes) + NS5 in the backward, then the real test: does Muon-chase BEAT the rank-1
v-hat recipe on the comparability bench (swap design: v-hat off). Probes: probe_muon{,2}.py.

--- Muon at the slow<->v_slow boundary (user's idea; probe_muon3, 2026-05-31) ---
Orthogonalize d_sv = s_slow - v_slow (NOT the chase). probe_muon3 (inject known rank-16
signal + noise, run real chase+leak cascade):
 WIN (boundary is right): d_sv is DENOISED -- subspace-align with TRUE signal 0.999 vs raw
   s_fast 0.915 (s_fast carries noise, d_sv doesn't); and SUPER-QUANTUM (|elem|~242 int8u >>
   the s*~1-10 survival threshold). NS5(d_sv) survives re-quantization PERFECTLY (spectrum
   0.529 -> 0.528). So slow<->v_slow is a quantization-SAFE place to orthogonalize -- the
   sub-quantum failure that killed Muon-chase is GONE here. User's instinct confirmed.
 CATCH (rank-dependence): d_sv raw spectrum is very LOW-RANK (mean_sv/max=0.047, frac>0.1=
   0.04 -- it's the rank-16 injected signal). NS5 inflates that to 0.53 = it MANUFACTURES
   singular directions not in the signal, and NS5(d_sv) subspace-align COLLAPSES to 0.044
   (energy sprayed across all 384 dirs, mostly OUTSIDE the true signal). Lesson: NS5 only
   helps when the update is ~FULL-RANK; on a low-rank update it is DESTRUCTIVE (amplifies
   the null space into noise). The synthetic rank-16 signal makes NS5 look bad by construction.
 OPEN QUESTION (decides it): what is the rank/spectrum of REAL d_sv in LM training? If LM
   drift is ~full-rank, NS5 at this boundary helps; if low-rank (few dominant dirs), it hurts.
   MEASURABLE with NO kernel change: SVD the decoded d_sv (=((pw<<16)>>24) - ((pw<<24)>>24))
   at a few checkpoints of a real Concord run, report singular spectrum. Do THAT before
   building any NS5-at-leak machinery. probes: probe_muon{,2,3}.py.

=== MUON-AT-LEAK: TESTED, DECISIVELY HARMFUL -- real d_sv is low-rank (2026-05-31) ===
Built orthogonalize_slow() (NS5 on d_sv=s_slow-v_slow, rewrite v_slow, SR-int8; module
method, no kernel surgery; unit-tested: flattens d_sv 0.314->0.739, s_slow untouched,
stays steppable). Harness --ortho_slow_every K. A/B vs base (overfit bench), ortho every
K=50 steps:
            best live |  sv   |  v
  base        1.5792  | 1.5260| 1.5485
  ortho_k50   1.6664  | 1.6319| 1.7557   (@iter1000, already +0.09/+0.11/+0.21 WORSE)
DECISIVE NEGATIVE (10x the nwv margin). This is probe_muon3's warning realized EXACTLY:
real LM d_sv is LOW-RANK, so NS5 doesn't flatten a full-rank signal -- it sprays the few
dominant directions' energy across all 384, manufacturing null-space noise and DESTROYING
the denoised drift the cascade built. (v hit worst, +0.21, since ortho directly corrupts
v_slow which 'v'=2*v_slow deploys.)
THE DEEPER POINT (about Concord, not just Muon): the slow cascade CONCENTRATES signal into
few directions -- that concentration IS the denoising (d_sv align 0.999 w/ true signal). Muon
wants the OPPOSITE (spread energy to a flat spectrum). So orthogonalization fights the
cascade's core mechanism. Muon is structurally wrong for this optimizer -- NOT because of the
int cascade (probe_muon2 solved SR-survival at K=50), but because the object worth
orthogonalizing is intrinsically low-rank and NS5 is destructive on low-rank updates.
PROBE LADDER (each cheap, each de-risked the next; total ~minutes, zero wasted kernel work):
  probe_muon  : chase orthogonalization dies (sub-quantum SR). 
  probe_muon2 : orthogonality survives SR at K~50 (super-quantum tick). 
  probe_muon3 : d_sv denoised+quant-safe BUT predicted low-rank -> NS5 destructive. 
  run         : confirmed -- real d_sv low-rank, ortho +0.1-0.2 worse.
VERDICT: drop Muon. orthogonalize_slow() + --ortho_slow_every left OFF-by-default (inert
unless invoked); revert if cleaning. Findings + probes (probe_muon{,2,3}.py) kept as record.

--- CORRECTION: the ortho_k50 -0.2 was CONFOUNDED (2026-05-31, user caught it) ---
orthogonalize_slow() OVERWRITES the accumulator (v_slow = s_slow - NS5(d_sv)) rather than
orthogonalizing the UPDATE INCREMENT (Muon's actual operation). probe_ortho_confound.py:
ONE ortho call moves the DEPLOYED weight by ||dW||/||W|| = 0.29 (29%!) at ZERO instantaneous
MSE change. So NS5(d_sv) moves v_slow 29% into LOSS-FLAT (off-signal/null) directions
(consistent w/ low-rank d_sv: NS5 spreads energy orthogonal to the signal). The A/B -0.2
therefore measured a SUSTAINED 29%-magnitude off-signal perturbation injected every 50 steps
that the chase+leak perpetually fight to re-settle -- NOT a clean test of orthogonalized
updates. The "decisively harmful / Muon structurally wrong" verdict is RETRACTED as
unproven; it conflated orthogonalize-the-increment with overwrite-the-accumulator.
A CLEAN Muon test must: orthogonalize the chase INCREMENT (alpha*s_fast in), keep it
super-quantum (probe_muon: sub-quantum dies; probe_muon2: K~50 survives), and ACCUMULATE not
overwrite. Whether that's constructible in this cascade is the open question -- TBD. The
low-rank-d_sv observation stands (probe_muon3), but "ortho hurts" is NOT established.

--- CLEAN RE-TEST (delta version): Muon-at-leak confirmed HARMFUL, not a confound (2026-05-31) ---
User caught the overwrite confound -> rebuilt orthogonalize_slow() to orthogonalize the d_sv
DELTA (window increment), ADD correction to v_slow (not overwrite). Confound GONE:
probe_ortho_confound re-test = 0.040 deployed ||dW||/||W|| per call (was 0.290), MSE-neutral.
This resolves the catch-22 I'd posited (no object is both denoised AND an increment): the
d_sv DELTA is exactly that -- change in clean drift (denoised), increment (not accumulator),
super-quantum over K=50. So a CLEAN Muon test IS constructible (user was right).
CLEAN A/B (delta-ortho every K=50) vs base, best-over-run:
            live  |  sv   |  v
  base      1.579 | 1.526 | 1.549
  delta-k50 1.787 | 1.758 | 3.231   (@iter1000) -- +0.21 / +0.23 / +1.68 WORSE
=> with the confound removed, Muon-at-leak is STILL decisively harmful (v deploy blows to
3.23). The -0.2 was NOT a measurement artifact; orthogonalization genuinely hurts. The clean
version is if anything WORSE because the per-window increment is even lower-rank than the
accumulated d_sv, so NS5 sprays a clean rank-few increment across all 384 dirs = pure
null-space noise into v_slow every 50 steps, compounding in the long anchor.
ESTABLISHED (now properly earned, holds for BOTH accumulator and increment): Concord's slow
cascade DENOISES by CONCENTRATING signal into a low-rank subspace; Muon's NS5 does the
OPPOSITE (spread to flat/full rank). They are fundamentally antagonistic, and the low-rankness
is intrinsic to the denoising -> there is NO orthogonalization hook in this optimizer that
helps. Muon DROPPED (for real this time). orthogonalize_slow + --ortho_slow_every OFF by
default. probes: probe_muon{,2,3}.py, probe_ortho_confound.py.

=== MUON DIFF (the clean diagnostic, user's framing -- probe_muon6, 2026-05-31) ===
User's ask: "calculate a beta1=0.9 momentum from the difference between two accumulators,
compare it to Muon, diff the steps." Done cleanly (true grad_W momentum, NO circularity --
probe_muon5 was circular/noisy, discard it). Track m = EMA_0.9(grad_W); compare directions:
  t      al(d_sv, raw_momentum)   al(d_sv, Muon=NS5(m))   al(raw,Muon)   eff_rank m -> NS5(m)
  70-350     +0.79 -> +0.87            +0.05 .. +0.10        ~+0.37         35 -> ~280
FINDINGS (stable across training, real grad not synthetic):
 1. d_sv = s_slow - v_slow IS the beta1=0.9 gradient momentum: difference-of-EMAs (chase
    a=0.1 ~ 10-step EMA vs leak a_v=0.001 ~ 1000-step) reconstructs heavy-ball momentum,
    align +0.87 and RISING. Confirms user's "momentum is just delta-weight / two accumulators,
    one a 0.9 accumulator of the other". Concord is ALREADY a momentum optimizer.
 2. Concord's committed step ~ RAW momentum (heavy-ball). A Muon step is ~ORTHOGONAL to it
    (align d_sv vs Muon = +0.07): NS5 rotates the update ~90 degrees into a different subspace.
 3. WHY: the gradient momentum is LOW-RANK on THIS task. [CORRECTED: probe_muon6's "~35/384"
    was an ARTIFACT -- that probe used a rank-32 SYNTHETIC target. Real-data re-measurement
    (probe_rank.py, real nanoGPT+tiny-shakespeare, true grad_W) confirms low-rank BUT
    task/layer-specific: momentum eff-rank (participation ratio / r90) attn.c_attn 40/16,
    attn.c_proj 19/7, mlp.c_fc 77/36 of 384 -- a ~5x layer spread, and this is vocab-65
    tiny-shakespeare; a richer task pushes rank UP. NOT an intrinsic Concord rank.] NS5 still
    over-spreads a low-rank update into directions with no gradient signal -> consistent with
    the measured harm.
VERDICT (task-scoped, NOT architectural -- "intrinsic rank 35 / wrong operator for Concord"
RETRACTED): on tiny-shakespeare the momentum is low-rank (above) and Concord's committed step
is ~RAW momentum (align d_sv vs raw +0.87) while a Muon step is ~orthogonal to it (+0.07);
NS5 over-spreads the low-rank momentum -> the measured +0.2 harm. So: Muon HURTS ON THIS
LOW-RANK TASK. It is NOT established that Muon is wrong for Concord in general -- on a
higher-rank task the rank argument weakens and Muon is UNTESTED. What IS solid: (a) d_sv =
difference-of-EMAs reconstructs the beta1=0.9 momentum (+0.87) -- Concord already IS a
momentum optimizer; (b) on tiny-shakespeare, orthogonalizing it empirically hurts (+0.2,
clean delta-version). DROP Muon for the current (low-rank) bench; revisit only with a
real rank measurement on the target task. Probes: probe_muon{,2,3,4,6}.py (5=circular,
discard), probe_rank.py (real-task rank), probe_ortho_confound.py.

=== THE CONTROL: real native Muon vs AdamW (2026-05-31, run_muon.sh) ===
The experiment that should've come FIRST: does faithful native Muon (optim_muon.py -- NS5
quintic 3.4445/-4.7750/2.0315 x5, Nesterov mom=0.95, standard split: Muon on 24 hidden 2D
weights, AdamW on embed/head/LN, sqrt(rows/cols) RMS scale) beat AdamW on tiny-shakespeare?
Same overfit bench (5000 iters, seed 0, data_seed 1234, wd=0):
  AdamW (lr1e-3) : best val 1.5318  final 4.645
  Muon  (lr0.02) : best val 1.5781  final 5.233   (+0.044 WORSE, overfits harder)
  [ref] Concord deploy-sv: 1.526 (beats BOTH)
=> REAL Muon LOSES to AdamW here, in its native optimal form. So the earlier "orthogonalization
hurts" was NOT a cascade-integration artifact -- there was no Muon win to capture on this task.
Resolves the ambiguity: (hypothesis A) task doesn't reward orthogonalization -- CONFIRMED;
(hypothesis B) my Concord-cascade integration killed a real Muon edge -- REFUTED (no edge exists
here). Consistent with the low-rank measurement (probe_rank r90 7-36/384): orthogonalization
assumes a flat target spectrum; this bench is low-rank, so Adam's per-coord scaling wins and
Concord's difference-of-EMAs momentum + deploy-slow (1.526) beats both Adam and Muon.
CAVEAT (the live question for the mission): tiny-shakespeare is vocab-65, highly compressible,
LOW-RANK. Muon's documented wins are higher-rank (GPT-2 scale / rich vocab). So this is
"Muon loses on the TOY bench", NOT "Muon loses on SDXL". If the real target is higher-rank
(probe_rank on real SDXL grad_W = the cheap check), Muon could win there -- and THEN the
cascade-integration question becomes live again. Muon harness arm kept (--mode muon,
optim_muon.py) for exactly that future test. Probes/runs: run_muon.sh, optim_muon.py.

=== RANK-AWARE ORTHOGONALIZATION: well-posed in theory, ILL-POSED on real d_sv (2026-05-31) ===
User's idea: don't full-Muon (drive all 384 SVs to 1 = pump ~349 noise dirs); orthogonalize
ONLY within the update's actual rank (equalize top-k signal SVs to 1, rest to 0; target
U_k Vh_k). Two probes:
 probe_muon7 (synthetic, KNOWN rank-16 signal): MECHANICALLY SOUND at the right k --
   raw d_sv align 0.999 | full NS5 (Muon) align 0.044 (eff_rank 298) | rank-k k=16 align 1.000
   (eff_rank 16) | k=2R align 0.500 | k=64 align 0.250. So k=true-rank is surgical, but
   OVERSHOOTING k is catastrophic (noise dirs orthonormalized to unit weight dominate).
   Undershoot safe, overshoot destructive -> need a CONSERVATIVE, ACCURATE k.
 probe_spectrum (REAL d_sv, real nanoGPT+tiny-shakespeare, 384x384 attn proj, it100-800):
   NO CLEAN GAP. SVs taper smoothly: sv[0,5,10,20,40,80,160,320] = 1.0, 0.69, 0.40, 0.16,
   0.11, 0.087, 0.057, 0.014. Sharpest adjacent knee only x1.2-1.7 (a real gap is x5-10+),
   and its location WANDERS (rank 3->7->13->9, unstable). Energy ranks spread hugely:
   r50=5, r90=73, r99=227 of 384.
VERDICT: rank-aware ortho is well-posed for low-rank-PLUS-CLEAN-GAP signals (the synthetic
probe) but ILL-POSED on real LM updates, which have a SMOOTH (power-law-ish) spectrum with no
signal/noise boundary. Any fixed k either TRUNCATES real signal (knee ~rank 5-13 discards the
0.4-0.7 SVs in ranks 5-20) or INFLATES real noise (r90=73 orthonormalizes dozens of
0.08-magnitude dirs to 1.0 = probe_muon7's destructive overshoot). There is no good k.
This is the ROOT reason every Muon variant failed: not "k=384 too big" but "the update
spectrum is smooth, so NO rank-equalization target is clean." Orthogonalization assumes a
spectrum you can cleanly equalize; real d_sv isn't one. CLOSES the orthogonalization line.
(Caveat unchanged: tiny-shakespeare; a higher-rank task MIGHT have a gap -- re-probe_spectrum
on the real target before reviving this.) Probes: probe_muon7.py, probe_spectrum.py.
