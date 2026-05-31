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
