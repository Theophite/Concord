# The Muon drive: spectral preconditioning from the packed state

Results of the exp 9/9b/9c series (`experiments/cpu_dynamics/`, June 2026): replacing
Concord's rank-1 v̂ preconditioner with Newton–Schulz orthogonalization of the gradient,
using no state beyond the packed word. Same validity bounds as the rest of the CPU
campaign (`RESULTS.md` §"Threats to validity"): fp32 reference, MNIST-scale, 3 seeds —
these results elect the design; the GPU nanoGPT A/B remains the adoption gate.

## TL;DR

| | clean (own κ\*, own lr\*) | 30% label noise (own κ\*) | wrong labels memorized |
|---|---|---|---|
| AdamW | 92.78 | 89.15 | 19.3% |
| Concord, v̂ drive | 94.68 | 90.77 | 12.4% |
| **Concord, NS drive** | **96.07 ± 0.05** | **92.02 ± 0.28** | 12.2% |
| native Muon | 95.40 | 82.32 | **100.0%** |

The NS drive is the best Concord arm in both regimes, **deletes every piece of
optimizer state outside the int32 word** (v̂ and its O(N+K) vectors, the EMA pass, ε,
the step cap, the trust region), is stable capless across **two orders of magnitude of
learning rate**, and converts native Muon's catastrophic label-noise failure (100% of
wrong labels memorized) into the campaign's best noisy-regime result.

## 1. The idea

Muon's entire memory cost is its momentum buffer — one fp32 matrix per layer, the thing
that makes it "half of Adam's state." Concord already carries a momentum matrix inside
the weight: the velocity `u = s_fast` is an EMA of applied steps with decay
`(1−α·gc)`. So the spectral preconditioner comes free: orthogonalize via NS5 (a
transient per-layer pass — ~10 small matmuls, no resident state), and Muon's one buffer
costs zero bytes. **Muon at 32 bits/param**, which native Muon cannot reach.

## 2. The manifold-Lookahead resolution (and a failed variant, kept per house style)

The first design blended the velocity into the NS input — `NS5(c·û + ĝ)` — on the
NS-of-EMA-of-NS argument: NS is a retraction onto the polar manifold, EMA-then-retract
is the canonical momentum-on-a-manifold pattern, and idempotence makes the scheme
degrade to exact Muon when directions are stable. The argument was right; the
implementation target was wrong. The c-sweep (clean regime):

| c | deploy acc | late gate coh | velocity magnitude |
|---|---|---|---|
| 0 | **94.88 ± 0.01** | 0.483 | 0.006 |
| 1 | 93.49 | 0.377 | 0.116 |
| 3 | 91.11 | 0.365 | 0.143 |

Blending `u` into the drive is a self-reinforcing direction loop: the velocity grows
~20×, the telescope reads the recycled direction as incoherent, consolidation
throttles. The resolution is the Lookahead idiom taken at full strength: **the chase
already is the EMA**. Retraction belongs on *directions*; positions need none; so
per-step `NS5(ĝ)` ticks plus the chase **is** the manifold-Lookahead composition, and
no Euclidean momentum belongs in the drive. (`exp9_muon.py` ships `C_BLEND = 0` with
this finding recorded at the constant.)

## 3. The rule

One line changes; everything else — gate, friction, chase, leak, deploy, schedules —
is untouched:

```text
v̂ drive:  u ← u − lr·clip( g/√(v̂+ε), ±10 ) − lr·κ(1−coh)·u
NS drive: u ← u − lr·√max(N,K)·NS5( g/‖g‖ )  − lr·κ(1−coh)·u
```

C\* needs no re-derivation: the drift-cancel fixed point assumes only that the
per-element tick has a stationary mean under drift, and NS of a stable matrix is
stable (invariant #3 of `CLAUDE.md` satisfied for free). The gate keeps its meaning —
exp 9's coherence traces stay healthy at c = 0 — with one watch-item: NS mixes
elements within a layer, so the noise reaching `u` is spatially correlated.

## 4. Why native Muon fails under noise, and why the cascade fixes it

Native Muon under 30% label noise memorizes **100.0%** of the wrong labels and
collapses to 82.3. The mechanism is its own design principle inverted: spectral
whitening gives every direction equal weight, and wrong-label gradients live in
exactly the rare, small-magnitude directions that whitening promotes. (On clean data
the same property makes it the best optimizer at this protocol — both halves of its
reputation confirmed at once.)

The identical NS5 drive inside Concord's gate/friction/deploy machinery memorizes
10–14% and *leads* the noisy regime: the fluctuation–dissipation cascade is exactly
the noise-rejection layer that spectral preconditioning lacks. The dissipation curve
is drive-dependent — κ\* for the NS drive is **100** at 30% noise vs 400 for v̂ (the
NS tick is unit-RMS, so friction works ~4× harder per unit κ); the κ sweep:
92.02 (κ=100) > 91.05 (200) > 89.68 (400) > 87.58 (800). Autotune tables are
per-drive, like everything else about it.

## 5. The fluctuation goes after NS — if anywhere (exp 9b)

Pre-NS noise isn't a fluctuation: it rotates the NS input and the rotation is
re-amplified to unit spectral weight. Measured: pre-NS σ=0.6 costs −0.27/−0.31
(clean/noisy, ~2× seed spread); post-NS σ ∈ {0.3, 0.6} is exactly neutral in both
regimes. Consistent with the whole campaign (σ has been inert on every MNIST test)
and with where σ originally earned its keep (the nanoGPT regime). Design decision:
**σ is post-NS and default-off in this arm; whether it returns is a bench question.**

## 6. The trust region and step cap are obsolete in this arm (exp 9c)

The v̂ denominator IS the winner's trust region (v_proxy = δ²·v̂, δ²=1), and the ±10
cap guards its tails. The NS drive ran without either from the start; the receipts:

**Per-element |step| tails over training:**

| drive | rms | p99.9 | max | cap=10 binds |
|---|---|---|---|---|
| NS (clean / 30% noise) | 0.66 / 0.75 | 2.9 / 3.1 | **6.7 / 6.6** | — (no cap) |
| v̂ pre-clamp (clean / 30%) | 1.5 / 2.2 | 9.1 / 13.0 | **323 / 406** | 1.6% / 1.7% of elements |

The NS step is self-bounded by the spectral constraint — its all-training max never
reaches the old cap and is *noise-level-independent* — while the v̂ drive's cap does
real work every step. Required by v̂; dead weight for NS.

**Capless lr envelope (κ=0, clean, deploy):**

| lr | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---|---|---|---|---|
| v̂ (capped) | 93.66 | 94.59 | 94.68 | 93.06 | 78.1 ± 10.3 |
| **NS (capless)** | 94.84 | 95.89 | **96.07 ± 0.05** | 95.38 | 95.14 ± 0.30 |

The capless NS drive is flat across two orders of magnitude while the *capped* v̂
drive breaks at lr = 1e-1 (the cap prevents NaN but not collapse). **The spectral
bound is a better trust region than the trust region** — and at its own lr the NS arm
sets the protocol best (96.07), revealing exp 9's numbers as an lr handicap, not a
ceiling. The lr·κ < 2 friction ceiling is drive-independent and still applies when
κ > 0 (at lr = 0.1 it caps κ at 20).

## 7. What the kernel becomes

The NS-arm apply kernel sheds `v_row`/`v_col`/`sum_v_inv`, the v̂ EMA pass, `eps`,
`step_cap`, and the trust region: **tick + gate + friction + chase + leak**, full
stop. The word is now the entire optimizer state without exception. Pipeline per
layer per step: backward grad → NS5 pass (transient, per-layer matmuls on the shared
scratch; Frobenius normalization makes it scale-free after one dequant) → apply
kernel with a `USE_MUON` constexpr consuming the orthogonalized step in place of the
gradient. Graph-capturable (static shapes); Conv2d flattens to 2D for NS, standard
Muon practice. Full implementation design — kernel diff, NS pass placement, graph
story, cost model, build order: [`MUON_IMPLEMENTATION.md`](MUON_IMPLEMENTATION.md).

## 8. Adoption gates

In order: (1) the same-seed nanoGPT A/B — NS drive vs v̂ drive at each one's κ\*/lr\*,
deployed-sv the metric (the bench where native Muon previously lost; the cascade may
change that verdict); (2) multi-seed on the MNIST grid; (3) Conv2d flattening + the
kernel `USE_MUON` path; (4) per-drive autotune calibration (the coh→κ table and the
lr\* shift together). Native Muon's clean-protocol showing (95.40, lr unswept) marks
the remaining headroom a pre-NS gradient EMA might buy — the one piece of state this
design refuses to purchase.

## 9. Gate 1 verdict: the nanoGPT A/B (2026-06-09) — not passed

The drive is implemented (package + notebook bench; `ns5` bit-exact vs the exp-9
reference, CUDA smoke + determinism green on triton 3.1/3.5) and the same-seed
char-nanoGPT A/B ran at the contemporaneous code state (fixed mass-preserve C\*,
which moves the v̂ control from the historical 1.4967 to **1.5157** — the
operating-point shift `compute_drift_cancel_C`'s fix predicted). Deployed-sv,
seed 0, 3000 iters, each arm's own knobs:

| arm | deployed-sv | final live val |
|---|---|---|
| v̂ winner (κ=50, σ=0.6) | **1.5157** | 1.537 (stable) |
| NS κ=25 | 1.5387 | 2.58 |
| NS κ=50 | 1.5385 | 2.02 |
| NS κ=100 | 1.5376 | 1.76 |
| NS κ=50, lr 1.5e-3 | 1.5669 | 1.87 |
| NS κ=50 + post-NS σ=0.6 | 1.5373 | 2.02 |

Findings, in order of interest:

1. **The NS plateau is κ-flat**: best-checkpoint quality is identical across a
   4× friction range (1.5387/1.5385/1.5376); κ controls only the *post-peak
   heating rate* (final live val 2.58 → 1.76). Dissipation is not the binding
   constraint — the drive reaches its ceiling mid-schedule and then degrades.
2. **The late-heating mechanism**: NS5 is scale-free, so once late-training
   gradients go noise-dominated, orthogonalization re-amplifies them to unit
   spectral weight — the train loss plunges (0.42/0.69/0.91 across κ; the v̂
   control sits at 1.20) while val climbs. Memorization-by-whitening: native
   Muon's §4 failure mode, damped by the cascade but not eliminated, on the
   bench regime where it matters.
3. **Post-NS σ is neutral on the bench too** (−0.001), extending exp 9b's
   MNIST result. σ does not return in this arm.
4. **Higher lr hurts** (1.5669 at 3×): the MNIST capless-lr robustness does
   not transfer to this regime.
5. **vs native Muon: decisively better** (1.538 vs 1.578 historical) — the
   gate/friction/deploy cascade converts Muon's failure into a usable arm, at
   32 b/param. The MNIST election (96.07, protocol best) did not transfer to
   char-LM overfit — consistent with the earlier probe-campaign finding that
   spectral whitening's rank-democratic prior is wrong for this low-rank task,
   and exactly what this gate exists to catch.

Caveat for any rematch: both arms ran at the OLD κ\*=50 — the honest-gate v̂
κ\* is itself unswept since the C\* fix (1.5157 may not be the v̂ optimum).
But the NS plateau's κ-flatness means no plausible NS κ\* closes the 0.022.

**Status:** the drive stays in (opt-in `drive="muon"`, default `"vhat"`),
gates 2–4 (multi-seed MNIST grid, Conv2d/kernel path beyond smoke, per-drive
autotune calibration) and the fork port do NOT proceed for adoption. The
remaining headroom marked in §8 — a pre-NS gradient EMA, the one state this
design refused to buy — is now the obvious next experiment, since the failure
is late-phase noise in the NS *input*, precisely what an input EMA filters.
