# Dual-Dissipation Packed-B Optimizer — Design & Status

*As of 2026-06-22: CPU reference + CIFAR integration GREEN (independently verified). Not yet ported to Triton / run on SDXL.*

## TL;DR

The fine accumulator of the packed-B optimizer is split into **two competing hypotheses** at different dissipation (forget) rates. They are fed **different random halves of each batch**, rated by **coherence**, and the one whose forget-timescale better fits its data **consolidates** into the shipped weight while the other dissipates. This makes dissipation **self-tuning** — it subsumes the cf-discount heuristic and the EpochDissipationServo — and is **conservation-correct** (no mass leaks into the deployed weight). Everything stays in the existing 32-bit packed word; **no floating-point sidecars**.

## 1. Mental model — weight = substrate + offset; hypotheses → theories

A weight is **substrate + offset**:

- **Substrate** — the init (Xavier/Kaiming from-scratch, or the pretrained weight for finetune). A fixed random **seed**, regenerated on the fly, **never stored, never in the accumulator**. It breaks symmetry and sets the per-row/col scale but carries no learned information — *neither believed nor disbelieved*. Dissipation decays the offset, i.e. returns the weight toward this prior, not toward zero. This dissolves the old from-scratch deploy-build problem outright: nothing is grown from zero.
- **Offset** — everything the optimizer has learned, an epistemic hierarchy (fast/tentative → slow/firm):

| register | role | scale |
|---|---|---|
| `e_L`, `e_H` | two competing **hypotheses** (the fine accumulators) | ×1 |
| `s_slow` | **weakly-held theory** (a hypothesis that earned consolidation) | ×128 |
| `v_slow` | **consolidated theory** (slow anchor; `s_slow` leaks into it) | ×128 |

The **deploy** (shipped) weight = substrate + the two theories `(s_slow+v_slow)`; the hypotheses are **dropped**. Near convergence the theories hold ~96–98% of the weight; the hypotheses are the ~2–4% un-integrated residual. Over training the hypotheses explain less and less — the integral carries the weight.

## 2. Mechanism — two timescales competing on coherence

- `e_L` and `e_H` are two **co-equal int8** accumulators (fine value = `e_L + e_H`, same scale — **not** high/low bytes of one int16).
- They run at **bracketed dissipation rates** `λ_L = λ·(1−d)`, `λ_H = λ·(1+d)` (arithmetic, so they average to the legacy rate).
- **Each gets the full gradient from its own random half of the batch** — two forward+backward per step (on CIFAR the random split stands in for the diffusion timestep-matched split). They diverge on different data; that divergence is what makes the competition real. An even-split of one shared gradient is useless (the arms stay identical).
- Each is rated by its **own per-weight coherence** (the existing Wiener machinery: shared `sig` from the `s_slow−v_slow` gap, per-arm noise). Each **soft-consolidates** its coherent part (coh-gated chase/leak) into the shared theories and **dissipates** its incoherent part at its own rate. The coherence-weighted blend **recenters** on the legacy single-accumulator rate, so the effective LR is unchanged.

Dissipation thus stops being a hand-set knob: the timescale actually producing consistent learning wins, per weight, measured empirically.

Only the **hypotheses** decay under dissipation; the **theories persist** — earned knowledge does not evaporate just because the gradient went quiet.

## 3. Scale, denormals, conservation

- Scale is a per-row/col rank-1 block-float exponent (`row_exp + col_exp`). The fine register is ×1 (128× finer than the ×128 coarse) and holds sub-coarse value; the existing per-row/col **rebalance** tracks the exponent.
- **Denormals** — single elements far below their row/col scale, which the per-row/col rebalance cannot fix — are handled by an **exponent-claim on `e_H`**: because the high-dissipation arm is statistically near-zero, its high bits are claimed by default as a per-element downward **octave/significand** (log reach *below* one mantissa unit — reach the int16 never had), with a `|e_H|`-guard falling back to plain magnitude when the byte carries real velocity. A denormal graduates into the fine register conservingly (never a +128 deploy credit). `is_denormal` reads the **coarse word only**, so it cannot self-nullify.
- **Conservation ledger** (the gating invariant): every step, `Δ(deploy mantissa) + Δ(fine mantissa) == inflow`, dissipation booked as a sink. The deployed weight only ever gains what the fine register loses — no one-directional leak (the failure mode of every rejected draft).

## 4. What it is NOT (the corrections that shaped it)

- **No fp sidecars.** An early draft kept the int16 and bolted on `err_s`/`den_frac` float buffers — rejected (violates int-only storage; leaked a DC bias).
- **Not high/low bytes.** The two int8s are co-equal at the same scale, not a positional split of one int16 (that broke the recenter math and the denormal channel).
- **No deploy dissipation.** Only the hypotheses decay; `s_slow`/`v_slow` (the earned theory) do not.
- **No even-split.** Each arm integrates its own random data half's full gradient, not half of a shared one.

## 5. Status — CPU-verified

`dual_dissipation_ref.py` + 6 test modules + the CIFAR integration are all GREEN, independently verified:

- conservation closes to residual 0 every step (including with the denormal channel live, under independent re-derivation);
- `disabled` mode is bit-exact to the legacy single-int16 kernel;
- under zero gradient the hypotheses drain (fine 0.72 → 0.10) while the theory persists (1.9 → 2.35);
- on a CIFAR conv net the loss drops, the deploy builds (consolidated offset / substrate 0 → 0.11), and the **two arms genuinely diverge** — two-grad arm-divergence 0.14 vs the dead even-split's 0.03 (~5×).

**Caveat:** absolute CIFAR accuracy is untuned/slow at the defaults. The pure-torch tick needs the reference preconditioner (`eps=1.0`, `v_scale=1.0`); the production cf recipe (`eps=1e-10`, `v_scale=0`) saturates the fine register here. A config-tune is the next step before a meaningful A/B.

## 6. Next steps

1. **GPU** — tune the dual-dissipation config to a respectable CIFAR accuracy, then **A/B** vs the legacy single-int16 (clean isolation of whether the two-timescale competition helps) + an AdamW anchor.
2. **Triton port** (SDXL scale) — the binding requirement is making the per-row/col rebalance **denormal-aware** (shift the octave field, do not integer-halve the log bits).

## 7. Files

- `dual_dissipation_ref.py` — the CPU reference (optimizer step + decode + conservation invariant).
- `dual_dissipation_nn.py` — `DualDissipationConv2d` / `DualDissipationLinear`.
- `train_cifar_cf.py` — `--optimizer dual_dissipation` (random-split training loop).
- `DUAL_DISSIPATION_DESIGN.md`, `DUAL_DISSIPATION_ENCODING.md` — the design + encoding specs.
- `tests/test_dualdis_*.py` — the 6 CPU test modules (conservation, disabled_legacy, recenter, denormal, from_scratch, substrate_conservation).
