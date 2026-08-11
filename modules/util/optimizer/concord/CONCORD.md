# Concord — Technical Reference & Why It Works

_Auto-generated from the live code by a multi-agent documentation pass (2026-06-20). Sections cite `file:line` into `modules/util/optimizer/concord/`, `concord_ot.py`, and the trainer/config. If the code and this doc ever disagree, trust the code and regenerate (`save_sidecar.py`-style pass)._

## Why it works, and how

Most optimizers have a single problem at their heart: the weight you train *is* the weight you ship. Every step of SGD or Adam writes the latest gradient — signal, noise, and overfit transient alike — directly into the parameter tensor that becomes your deployed model. You never get to ask, after the fact, "which of these updates were real, and which were the optimizer chasing a few noisy minibatches into a corner?" The denoising you do get is crude: Adam's running second moment rescales step *sizes*, but it still commits whatever direction the gradient pointed. The model that trained is the model that deploys, warts and all. This is why a learning rate that's a touch too high overfits, why late-training steps memorize, and why you babysit LR schedules so carefully — there is no separation between *exploring* a direction and *committing* to it.

**Concord's core move is to refuse that conflation.** It splits every weight into two registers and ships only one of them. Concretely, each parameter is stored as a single 32-bit word — a per-row/col block-float — holding three integer mantissae:

```
s_fast (int16)  — the recent, noisy "velocity" scratchpad
s_slow (int8)   — the committed coarse position
v_slow (int8)   — the long-time anchor
```

`s_fast` is where every gradient tick lands first. It is fast, reactive, and *disposable*. `s_slow` and `v_slow` are the slow accumulators — the position that has earned its place. The decisive line is `consolidated_weight()` (`prototype_packed_b.py:2891`): the deployed weight is `(s_slow + v_slow)·128·2^exp` — the slow path **only**. `s_fast` is dropped at export. The most recent, overfit-prone displacement never makes it into the shipped model. This isn't a heuristic that happens to help; in the overfit regime, shipping the slow path beats keeping `s_fast`. The deployed weight is, by construction, a *denoised* weight.

**So what earns a direction its way from the fast scratchpad into the committed position?** This is where Concord stops being a storage trick and becomes a filter. The fast velocity is decomposed at every step (`:783–792`) into two parts: a **drift** that the slow accumulators have already corroborated (`d_sv = s_slow − v_slow`, the gap between two time-lagged positions), and a **residual** that is whatever's left over. The drift is signal — a direction confirmed by integrating roughly a full pass over your data. The residual is the per-weight gradient *noise*. A per-coordinate Wiener/SNR gate then asks the only question that matters:

```python
coh = sig² / (sig² + noise² )          # ∈ [0,1],  signal/(signal+noise) power
```

`coh→1` means "this direction is real, commit it"; `coh→0` means "this is noise, don't." Crucially the gate is dimensionally honest — signal and noise are pushed through the same scale so the learning rate cancels and `coh` is a *pure* gradient-SNR, not something that drifts with LR or layer.

**The gate decides; dissipation enforces.** The fraction the gate rejects doesn't just sit in `s_fast` — it is actively evaporated before it can leak into the committed position (`:962`):

```python
evap_frac = lr · κ · (1 − coh)          # kill the incoherent fraction
```

This is the friction `κ` (`gf_consol`). Coherent directions (`coh≈1`) pass untouched and migrate to `s_slow`; incoherent ones are drained. The natural setting is the **Wiener point** `λ = lr·κ = 1`, where the kill exactly matches the gate. Hand-tuning a single global `κ` across hundreds of layers with wildly different SNR is hopeless, so an **opt-in per-layer servo** can autotune it (off by default). Two cheap meters — *boil* (how much coherent signal the evaporation is wrongly destroying) and *memgap* (whether dropping `s_fast` is helping or hurting the deploy loss) — drive an Rprop-style climb that finds each layer's dissipation rate, a few times per epoch.

**Now the payoff — why this gives implicit-LR robustness and overfit immunity.** When your learning rate is too large, the excess motion has to go *somewhere*. In a normal optimizer it goes into the weight and you overfit. In Concord, an oversized step lands in `s_fast`, and the gate sees it for what it is: large velocity that the slow accumulators have *not* corroborated, i.e. low coherence. That excess is gated out and evaporated; it never reaches the committed `s_slow`/`v_slow`, and it is dropped at deploy regardless. The same dimensionless `λ = lr·κ` makes the friction transfer across learning rates (`concord_ot.py:152`), so the filter behaves identically whether you run hot or cold. The too-big LR's damage is quarantined in the disposable buffer and thrown away. You ship the Wiener-denoised position, not the trajectory that produced it.

---

**Map to the technical sections.** The rest of this document drills into each subsystem: the **packed int32 format and block-float accumulator semantics**; the **update step and its preconditioner** (a rank-1 Adafactor squared-gradient second moment — *not* per-coordinate Adam, and with no momentum); the **coherence/Wiener-SNR gate** in full; the **chase/leak/consolidation cascade** that moves mass to the deploy weight; **dissipation and the boil/memgap meters**; the **per-layer servo** that autotunes `κ`; **token embeddings and the control plane**; the **`ConcordController` and the layer swap** that wire it into OneTrainer; and **persistence with the per-epoch exit-42 relaunch** that keeps the run alive on a 24 GB card.

## Contents

1. [The packed format & accumulator semantics](#the-packed-format-accumulator-semantics)
2. [Preconditioning & the update step](#preconditioning-the-update-step)
3. [The coherence / Wiener-SNR gate](#the-coherence-wiener-snr-gate)
4. [Chase, leak, consolidation & the deploy weight](#chase-leak-consolidation-the-deploy-weight)
5. [Dissipation (gf_consol) & the boil/memgap meters](#dissipation-gf_consol-the-boilmemgap-meters)
6. [The per-layer dissipation servo](#the-per-layer-dissipation-servo)
7. [Token embeddings & the control plane](#token-embeddings-the-control-plane)
8. [The controller & the layer swap](#the-controller-the-layer-swap)
9. [Persistence & the per-epoch exit-42 relaunch](#persistence-the-per-epoch-exit-42-relaunch)
10. [Config & GUI integration](#config-gui-integration)

---

## The packed format & accumulator semantics

Every trainable weight in Concord lives as a single **int32 word** — 32 bits/param total, no bf16 in the persistent state. The layout is declared at the top of `prototype_packed_b.py:3-10` and again in the class docstring (`:2472-2473`):

```
bits [31:16]  s_fast      int16   — fine SR-tick velocity, scale × 1
bits [15:8]   s_slow_i8   int8    — coarse position bearer, scale × 128
bits [ 7:0]   v_slow_i8   int8    — long-time anchor,        scale × 128
```

with `S_SLOW_FACTOR = V_SLOW_FACTOR = 128` (`:48-49`). Unpacking is pure sign-extending bit-shift (`get_state`, `:2910-2914`): `s_fast = packed_w >> 16`; `s_slow_i8 = (packed_w << 16) >> 24`; `v_slow_i8 = (packed_w << 24) >> 24`.

**Per-row/col block-float.** The three integers are **mantissae**, not the weight. The physical value is recovered through a shared exponent carried by two small int8 envelope tensors `row_exp[N]` / `col_exp[K]`:

```
m_eff  = s_slow_i8·128 + s_fast + v_slow_i8·128
weight = m_eff · 2^(row_exp + col_exp − MANTISSA_BIAS)     # MANTISSA_BIAS = 15
```

`get_weight` (`:2876-2888`) implements exactly this; the kernel forms `scale_fwd = exp2(total_exp)` and `scale_inv = exp2(-total_exp)` once per tile (`:774-776`) so the exponent never enters the dot product. Packing is the inverse (`load_weights`, `:2732`): quantize `m_total = round(W/scale)`, split the coarse part evenly across the two int8 slow accumulators (`s_slow + v_slow == coarse`, so the initial gap `d_sv ≈ 0`), leave the fine residual in s_fast, then OR the masked fields together: `((s_fast & 0xFFFF) << 16) | ((s_slow_i8 & 0xFF) << 8) | (v_slow_i8 & 0xFF)`.

**Why >bf16 effective precision.** Because the exponent is *shared* per row+col, all 16+8+8 stored bits are spent on the mantissa of one block, not split into a per-element exponent. The live mantissa `m_eff` spans int16-class magnitude (s_fast is non-saturating, `:18-20`) while the int8 slow channels add ±128-quantized coarse bits — finer than bf16's 8-bit mantissa within each block-float envelope.

**The two telescopes.** The kernel reads two differences (`:783-784`): the velocity `d_fs = s_fast` (the recent, noisy displacement) and the **drift / momentum gap** `d_sv = (s_slow_full − v_slow_full)` — s_slow leading v_slow. `d_sv` is the consolidated drift the Wiener gate treats as signal (`sig = drift_cancel_C · d_sv`); it costs nothing extra in storage — it is just the difference between two int8 channels already in the word, so momentum is *free*.

**The deploy drops s_fast.** `consolidated_weight()` (`:2891-2907`) materializes only `(s_slow_i8 + v_slow_i8)·128 · 2^exp`, discarding the s_fast field. s_fast carries the most recent, overfit-prone velocity; the slow accumulators are the denoised position. In the overfit regime the slow-path deploy beats the live `get_weight` (which keeps s_fast). The overfit transient is quarantined in s_fast and thrown away at export.

---

## Preconditioning & the update step

Despite the `optimizer_kind == 'adamw'` branch name (`apply_packed_adamw`), the **shipped update is not AdamW**: the WINNER recipe sets `beta1 = 0`, so there is **no momentum**, and the preconditioner is a **rank-1 Adafactor** factored second moment, not Adam's per-coordinate `E[g²]`. The closest standard optimizer is Adafactor; the live step is a rank-1 RMS-normalized SGD. Only the recipe defaults decide which denominator term is live. The kernel is in `prototype_packed_b.py`.

**Two candidate preconditioners; the recipe picks one.** The denominator is assembled from two terms (L794, L825):

```python
v_proxy = noise_in_w**2 * v_scale               # (A) the drift-cancel noise²
v_proxy = v_proxy + gf_trust_delta_sq * v_hat   # (B) the Adafactor rank-1 E[g²]
```

**(A)** is the drift-cancel *noise²*: from the velocity `d_fs = s_fast` and the consolidation gap `d_sv = s_slow_full − v_slow_full` (L783–784), `noise = d_fs − drift_cancel_C·d_sv` removes the expected drift so what remains is the genuine per-weight gradient variance (a real "which weights converged" signal). **(B)** is `v_hat`, the **Adafactor rank-1** reconstruction `v_row ⊗ v_col · sum_v_inv` bias-corrected by `1/(1−β2^t)` — a *factored* approximation of `E[g²]`, not Adam's full per-coordinate second moment (the kernel comment calls it "a typical-gradient-magnitude reference, not a variance estimate").

**The shipped WINNER recipe sets `v_scale = 0` and `gf_trust_delta_sq = 1`** (`concord_winner.py`), applied to every live layer. That zeroes term **(A)**, leaving `v_proxy = v_hat`: the active preconditioner is the rank-1 Adafactor `E[g²]`. So in production Concord preconditions like **Adafactor** (factored, and with no momentum), *not* Adam; the drift-cancel noise² is a real but **disabled-by-default** alternative (turn it on with `v_scale > 0`).

One thing not to conflate: the drift-cancel `noise` *also* feeds the coherence/Wiener gate (next section), and there it is load-bearing **regardless of `v_scale`**. As a *preconditioner* the noise² term is off by default; as the gate's noise estimate it is always on — the same quantity in two roles.

**The step.** Partial adaptivity comes from a Padam-style exponent `precond_p` applied via exp2/log2:

```python
denom_p   = tl.exp2(precond_p * tl.log2(v_proxy + eps))   # L889
step_live = grad_W / denom_p                              # L890
step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)  # L891
```

`precond_p = 0.5` is the usual √ (Adam-like); `0` gives `denom_p = 1`, i.e. `step = grad` (pure SGD); values in (0,0.5) interpolate between the linear and fully-smoothed regimes (L884–888). `eps>0` keeps the log safe. The result is clamped to `±step_cap`. The only live weight-decay path is the **cautious** form — it decays `s_fast` toward the slow position (not toward 0), added into `step_live` and *not* divided by `denom_p` (only the gradient term is preconditioned). The `wd·current_weight` (decay-toward-0) form is a stale comment, not a live path.

---

## The coherence / Wiener-SNR gate

The gate decides, per element, *how much of `s_fast` is signal worth keeping* versus noise to evaporate. Everything rests on one decomposition of the velocity (`prototype_packed_b.py:783-792`):

```python
d_fs = s_fast.to(tl.float32)                  # the velocity itself
d_sv = (s_slow_full - v_slow_full).to(tl.float32)   # consolidated drift
noise = d_fs - drift_cancel_C * d_sv          # velocity residual
```

So `d_fs = signal + noise`, where **signal** is the drift `drift_cancel_C * d_sv` (the part already corroborated by the slow accumulators) and **noise** is what's left. A Wiener/SNR gate then asks what fraction of the *power* is signal (`:838-861`):

```python
sig_w  = drift_cancel_C * d_sv * scale_fwd    # signal in W units
sig2   = sig_w * sig_w
coh = sig2 / (sig2 + noise_in_w * noise_in_w + 1e-30)   # S²/(S²+N²) ∈ [0,1]
```

This is the `USE_FIXED_COH` path: both `S` and `N` are pushed through the same `scale_fwd`, so the per-row/col `lr/scale` factor cancels and the bare `coh_raw` is a pure gradient-SNR Wiener coefficient. **But the shipped default `coh_vhat=True` does not gate on the bare `coh_raw`.** It cf-discounts the noise power: with `cf = d_sv²/v_hat` (the *coherent fraction* of the admitted gradient energy), the gate is `coh = sig²/(sig² + noise²·κ/(cf+κ))`, which protects diverse-but-real structure that the bare Wiener gate would kill as noise. So the formula above is `coh_raw`; production runs the cf-discounted `coh`. (The non-default legacy `else` branch computes a `(mean grad)²/v_hat` ratio against the Adafactor rank-1 `v_hat`.) Both `coh` and `coh_raw` are clamped to `[0,1]`.

The `USE_GAP_FEEDBACK` variant adds a bootstrap floor keyed to the *drift magnitude* `|d_sv|` (`:868-874`):

```python
c_pass = tl.minimum(coh + tl.exp(-tl.abs(d_sv) * gap_inv_scale), 1.0)
```

Early on, when no drift has built up (`d_sv ≈ 0`), the exponential is `≈1`, so `c_pass → 1`: almost everything passes — the "ignition" floor that lets coherence bootstrap before there is any drift to measure. As drift accumulates, the exponential decays toward 0 and `c_pass` collapses back to the bare `coh` gate.

Crucially, `coh`/`c_pass` gate the **output (evaporation)**, not the input. The non-passed fraction is what gets drained from `s_fast`: under GAP_FEEDBACK `evap_mantissa = (1 - c_pass)*alpha*d_fs` (`:937`), and under plain `USE_GF_CONSOLIDATION` `evap_frac = min(lr_eff*gf_consol*(1 - coh_evap), 1 - min_leak)`, with `lr_eff = lr·step_scale` and `coh_evap = min(coh, coh_raw + evap_slack)`. Signal (`coh→1`) is preserved and flows to `s_slow` via the unconditional chase; noise (`coh→0`) evaporates before it can consolidate. Gating the kill rather than the accumulation is the deliberate bootstrap-safe choice: a brand-new coordinate is never starved of input on the basis of a coherence estimate it hasn't had the chance to earn yet.

---

## Chase, leak, consolidation & the deploy weight

The coherent part of the velocity flows downhill through two staged transfers each step, and the deployable weight reads only what has settled at the bottom.

**The chase (s_fast -> s_slow).** After the stochastic-rounded gradient tick lands in `s_fast` (the int16 velocity, in mantissa units), a fraction `alpha` of it is moved into `s_slow` at int8 granularity (each int8 tick = 128 mantissa units). The move is *mass-preserving*: `s_fast` loses exactly what `s_slow` gains, so the live mantissa `m_eff = s_slow*128 + s_fast + v_slow*128` is conserved. In the consolidation path the chase is additionally throttled by the coherence gate and the `consf` schedule (`prototype_packed_b.py:1067-1083`):

```python
chase_mantissa = alpha * gate * gate_gain * s_fast * consf
tick_slow_i8   = SR_round(chase_mantissa / 128.0)
s_slow_i8 += tick_slow_i8;  s_fast -= tick_slow_i8 * 128
```

**The leak (s_slow -> v_slow).** `v_slow` (the int8 anchor) chases the *position* `s_slow_full = s_slow*128`, not the velocity, at rate `alpha_v_fast` (`prototype_packed_b.py:1085-1097`):

```python
gap_v_full = s_slow_full_post - v_slow_full
delta_v8   = alpha_v_fast * gap_v_full / 128.0 * consf * g_active
```

Again mass-preserving: the int8-clamped tick is debited from `s_slow`, so `d_sv = s_slow_full - v_slow_full` (the telescope gap) relaxes at `2*alpha_v_fast` per step. `alpha_v_fast` is the *telescope window*: `apply_epoch_window` (`concord_ot.py:724-788`) pins it to `1/(2*steps_per_epoch)` so the anchor integrates exactly one full dataset pass — every example votes once — before motion counts as drift.

**Mass-preserving drift cancel.** `compute_drift_cancel_C(alpha, alpha_v_fast)` (`prototype_packed_b.py:52-102`) returns the analytic `C*` that zeroes `E[noise] = E[d_fs] - C*·E[d_sv]` under a pure-drift gradient. With `L=(1-alpha)/alpha`, `rho=alpha_v_fast`, the mass-preserve branch gives `C* = L*2rho/(1-2rho)` (~0.018 at defaults). This is what makes the Wiener gate read genuine drift as coherent (coh -> ~1) instead of saturating half-blind at ~0.5.

**The deploy weight drops s_fast.** `consolidated_weight()` (`prototype_packed_b.py:2891`) returns `(s_slow*128 + v_slow*128) * 2^exp` — the *slow path only*:

```python
m_slow = s_slow_i8 * S_SLOW_FACTOR + v_slow_i8 * V_SLOW_FACTOR  # both = 128
w_fp32 = m_slow * 2.0**exp
```

`s_fast` carries the most recent, noisy, overfit-prone velocity. Discarding it is denoising: only velocity that survived the chase-and-leak through the coherence gate reaches `s_slow`/`v_slow`. In the overfit regime this slow-path deploy beats the live `get_weight()` (which keeps `s_fast`). The plain sum is correct; doubling the anchor (`s_slow + 2*v_slow`) overshoots and is worse.

---

## Dissipation (gf_consol) & the boil/memgap meters

Dissipation is the κ that *evaporates* the incoherent part of the fast velocity `s_fast` before it can consolidate into the deploy weight. It runs in the `USE_GF_CONSOLIDATION` branch of the kernel and replaces the old uniform cautious weight-decay: instead of decaying every velocity coordinate, it drains only the part the Wiener gate `coh` flags as noise.

**The evaporation fraction** (`prototype_packed_b.py`):
```python
evap_frac = tl.minimum(lr_eff * gf_consol * (1.0 - coh_evap), 1.0 - min_leak)
```
with `lr_eff = lr·step_scale` and `coh_evap = min(coh, coh_raw + evap_slack)` (a cf-aware clamp; `evap_slack` default 0.25, so the kill never shreds cf-coherent mass below the bare-Wiener level). Here `gf_consol` is κ (the rate *at unit lr*), so `lam ≈ lr*κ` is the dimensionless dissipation. `lam = 1` is the **Wiener point**: the kill exactly matches the gate. The `lr` factor means the cosine schedule auto-fades the skim in the tail (`gf_consol=0.3` at peak `lr=0.1` ≈ the old κ=0.03, `:962`). The incoherent weight `(1-coh)` is what gets drained; `coh→1` coordinates pass untouched and flow to `s_slow` via the chase.

The `min_leak` clamp (`_MIN_LEAK = 0.1`, `:1327`) is the **slam-shut guard**: it caps the per-step kill at `1 - min_leak` so at least `min_leak` of the velocity always survives. Without it, at `lam→1, coh~0` the evaporation wipes `s_fast`'s entire history every step — nothing accumulates, the slow drift freezes, `coh` stays pinned at 0, and the valve self-seals (the meter that would reopen it depends on mass flowing through it). It also kills the `lam>1` ringing and the `lam=2` instability.

Two further gates qualify the kill (`:974`, `:933`):
- **`build_ok`** (`evap_build_min`, the soft scale = one `s_slow` LSB = 128): a **stochastic, size-proportional** drain gate. A sub-LSB velocity of size `s` evaporates with probability `min(s/evap_build_min, 1)` — realized via the same SR hash as the chase — so the Wiener filter reaches into the sub-LSB band *in proportion to size* (unbiased; the smallest velocities resolve quadratically rarely, protecting infancy without a cliff). At/above the scale `p` saturates to 1 (the old above-threshold behavior); `evap_build_min=0` makes `p=1` everywhere (all-pass). The old hard `|s_fast| >= 128` gate could never fire — the chase pins `|s_fast|` below one deploy tick, which would close evaporation permanently (washout) — which is why the default had to be 0; the stochastic gate removes that, so 128 is the natural default again.
- **`g_active`** (the sighting gate, `USE_GRAD_ACTIVITY`): for sparse embedding rows, evaporation ticks only on steps where a gradient is actually present (`grad_W != 0`), so λ means the same *per-sighting* fraction regardless of token frequency.

**The meters** drive the per-layer servo. `boil`'s kill energy is realized on the consolidate step (`consf`-gated); `memgap` accumulates every micro-step:

- **boil** decomposes the *realized* kill energy: `boil_ptr[0] += sum(killed²·coh_raw)` (the coherent-waste, aligned_kill, weighted by the *un-discounted* `coh_raw`) and `[1] += sum(killed²)` (total_kill). `boil = aligned_kill / total_kill` is the fraction of evaporated energy that was actually *coherent signal* — energy the dissipation wrongly destroyed.
- **memgap** is the first-order `L_live - L_deploy`: `memgap_ptr += sum(grad_W · d_fs · scale_fwd)`, i.e. `sum(grad · s_fast_in_W)`. Since the deploy weight = live − `s_fast`, this Taylor term estimates `L_deploy ≈ L_live − memgap` (positive `memgap` ⇒ dropping `s_fast` *helps* the deploy loss).

The `boil` writes (not `memgap`, which is outside any boil guard) are gated by `write_boil`:
```python
write_boil = (abs(float(drift_cancel_C)) > 0.0) and (float(wd_anchor) <= 0.0)
```
Coherence-degenerate layers (anchored TE, `drift_cancel_C=0 → coh≡0`) are excluded — their meters would be pure noise.

---

## Per-layer dissipation control

The kappa that evaporates incoherent `s_fast` (`gf_consol`) has exactly ONE per-layer controller: the **noise-seed seeder** (`noise_seed_servo`, `NoiseScaleSeeder` in `concord_ot.py`) — set-don't-hunt, seeding each layer's kappa from its measured gradient noise-to-signal ratio. The hunting servos that preceded it (the per-layer epoch climb servo and the bracket-gap secant) are removed; their knobs no longer exist in the panel or config. `autotune_servo_per_epoch` survives as the seeder's WINDOW knob: the NSR measurement window is `steps_per_epoch // autotune_servo_per_epoch`. Without the seeder, dissipation is the fixed dimensionless `dissipation` (lam; kappa = lam/lr, lam=1 is the Wiener point), optionally one-shot table-committed via `autotune_table`.

**Per-layer meter plumbing (shared infrastructure).** The kernel atomic-adds boil/waste into a 6-vector `_boil_meter` and first-order deploy-loss into a 1-vector `_memgap_meter` — by default the SHARED per-device buffers that `read_flow_audit`/`read_memorization_gap` read and zero (with a cold-start drain on the first read). `register_layer_meters(packed_w, boil_buf, memgap_buf)` reroutes a layer's adds to its own buffers, registered *before* CUDA-graph capture so the baked pointer is graph-safe. The live consumer is the controller's TE scratch routing (winner-TE flow is kept out of the UNet audit).

---

## Token embeddings & the control plane

New-token embeddings reuse the exact same packed cascade as the UNet: `ConcordPackedEmbedding` wraps a `ConcordLinearPackedB(dim, K)` (the core class is defined in `prototype_packed_b.py:2471`; the embedding module only imports + instantiates it) whose `packed_w` is `[K, dim]` int32 -- one `s_fast|s_slow|v_slow` per element, `row_exp` per token (`concord_embedding_packed.py:99-110`). The forward is a gather of the live bf16 weight; the backward (`_PackedEmbStep`, `:25-95`) scatters the per-position grad into `G=[K,dim]` and drives the *same* fused kernel via `core.apply_grad_step(G * mod._drive, v_stats_from=G)` -- a direct kernel launch, never a nested `torch.autograd.backward()`, which would be illegal under CUDA-graph capture (`:84-89`).

**Slow-path load.** `load_weights` (`prototype_packed_b.py:2732`) packs the mantissa into the SLOW path: on *its* output `s_slow == v_slow` (gap-zero, `d_sv ≈ 0`), `deploy = (s_slow+v_slow)·128 ≈ W` from step 0, and only the fine residual (`|s_fast| ≤ 64`) is left in `s_fast`. `init_tokens` then runs `_pin_norm`, which re-rounds all three fields to hit the vocab-median norm — so *post-init* those integer bounds are only approximate (the gap can reach ~2, `|s_fast|` a little over 64); the gap-zero / small-residual property survives in the block-float-*relative* sense (`‖d_sv‖ ⁄ ‖deploy‖ ≪ 1`), which is all the deploy needs. The original re-split bug re-read `s_fast` as if it held the mantissa and collapsed deploy to ~0; the fix is to pin the already-correct `load_weights` state (`init_tokens`, `:218-225`).

**Anchor vs non-anchor** (`init_tokens`, `:193-225`). Anchor mode freezes the init in `v_slow` and sets `alpha_v_fast=0`, `drift_cancel_C=0` -- the founding semantics are immutable and everything learned accumulates as a gated `s_slow` delta (`deploy = init + delta`); norm is pinned once. Non-anchor is adaptive: `_pin_norm` runs every step, pinning each token's deploy norm to the vocab median via a power-of-2 `row_exp` plus a `[0.71,1.41]` mantissa residual (`:240-268`).

**Control plane** (`control_plane.py`). `ControlPlaneEmbedding` replaces the frozen `token_embedding` and routes each id by `kind` (0=base/frozen, 1=static zero/fixed, 2=trainable) with branch-free, static-shape `torch.where` so it is capture-safe (`forward`, `:107-123`). The `.weight` shim returns the unchanged base vocab (`:70-75`).

**Sighting gate.** `_seen` counts only gradient-bearing occurrences -- the branch-free forward routes every position through row 0 with zero grad, masked by `(ge.abs().amax(dim=1) > 0)` (`concord_embedding_packed.py:61-63`); rare tokens are clocked per-evidence via `grad_activity=True` (`concord_ot.py:524`, consumed at `prototype_packed_b.py:3138`).

**Caption-vocab.** `concord_train_caption_vocab` makes base-vocab tokens that appear in captions trainable, seeded from `base.weight[tid]` and tagged with a `(None, tid)` sentinel in `row_map` (`setup_packed_embeddings`, `:1478-1490`). The save bridge `materialize_packed_embeddings_to_vectors` writes each deploy vector back -- to `emb.vector[k]` for added tokens, or `base.weight[tid]` for caption tokens (`:1626-1633`).

---

## The controller & the layer swap

Concord is not a `torch.optim.Optimizer`. There is no `optimizer.step()` for the trained weights — each Concord layer fuses its update into the autograd backward. The **`ConcordController`** (`modules/util/optimizer/concord_ot.py:120`) owns the swapped layers, the per-step schedule, and the rebalance gate for one run. The OneTrainer-visible optimizer for the `CONCORD` choice is just a plain SGD over the *non-swapped* aux params (norms, biases); the controller carries the real work, driven by `before_step()`/`after_step()` around the existing loop.

**The swap.** `swap_unet_to_winner` (`concord_winner.py:208`) walks the UNet and replaces every `nn.Linear`/`nn.Conv2d` (in place) with a packed core, loading the pretrained weight via `c.load_weights(...)` and pushing the winner recipe (`adamw`, `precond_p=0.5`, `gf_consol`, `gf_trust_delta_sq`). OneTrainer's layer-filter is honored — unmatched layers stay standard/frozen (`:260-262`). Conv weights are reshaped `(out, in·k·k)` (`:279-280`). It also flips the module-global flags `set_fixed_coh(True)`, `set_ratio_coh(True)`, `set_sigmag_noise(...)`.

**Winner vs. anchor.** Two TE modes share the controller. `swap_text_encoder_to_winner` (`:391`) trains CLIP like the UNet: even split, `alpha_v_fast=0.001>0` so the drift-cancel `C*>0` and the coherence gate is *live*, no `wd_anchor`. `swap_text_encoder_to_anchor` (`:316`) is the opt-in frozen anchor: `alpha_v_fast=0` pins `v_slow` at pretrained W, `drift_cancel_C=0` (gate inert), `gf_consol=0`, and `wd_anchor>0` elastically pulls the delta back. The `alpha_v_fast>0` test is the universal selector that distinguishes the two everywhere downstream.

**Per-component LRs.** The setup threads `unet_lr = config.unet.learning_rate or config.learning_rate`, plus `te_lr`/`te2_lr` from each encoder's field (`StableDiffusionXLFineTuneSetup.py:171-185`). Each TE forms its own schedule *group* with its own peak lr.

**Dimensionless λ.** The GUI "dissipation" field, when set, overrides `gf_consol` at controller init: `gf_consol = lam / lr` (`concord_ot.py:152`), so the same λ means the same per-step friction `u ← u − lr·κ·(1−coh)·u` at any LR (stability ceiling reads as λ<2).

**Per-step.** `before_step()` (`:976`) optionally builds the servo, ticks autotuners, calls `winner_step(...)` (lr/sigma/coherence floors onto device tensors), then writes `gf_consol_buf = base·fill_ramp` — the warmup ramp `1−exp(−2·alpha_v·t)` that withholds friction while the anchor fills. Secondary TE/embedding groups run schedule-only. `after_step()` (`:1075`) runs the gated rebalance and ticks `step_idx`.

**gamma-SNR hook.** `on_timesteps(...)` (`:889`) modulates dissipation per batch: `kappa_t = min(base·mean(max(1, snr/knee)), LAM_MOD_CAP/lr)`, written per-layer with each layer's own LR cap. The selector `autotune_gamma_snr_on` (or `knee≤0`) turns it off, leaving the base/servo λ to stand un-capped.

**Audit.** The `[loss]` line reads `read_memorization_gap()` (`:849`, L_deploy−L_live) and `read_flow_audit()` (`:810`, boil/waste) once per logging step (`GenericTrainer.py:1176-1202`); under the per-layer servo these peek the per-layer meters non-destructively and return the delta.

---

## Persistence & the per-epoch exit-42 relaunch

On a 24 GB card the model nearly fills VRAM, and the CUDA-graph boundary work (release the pool, recommit the UNet, recapture the graph) fragments the heap *irreversibly*: `empty_cache`/reset cannot reclaim fragmented-but-committed reserved memory, so the in-process recapture overflows the ceiling and WDDM stickily demotes the tail to shared memory — observed compounding 1.08 → 2.60 s/it over a few epochs (`scripts/concord_train_restart.py:4-10`). The graph recapture itself is sound; the fix is to give each boundary a **fresh process with a clean allocator**.

**The wrapper.** `concord_train_restart.py` is a drop-in for `train.py`. It sets `CONCORD_RESTART_ON_SAMPLE=1` and `CONCORD_RESTART_ON_BACKUP=1` (line 81-82) and loops: whenever the child exits `42`, it sets `CONCORD_RESUMING=1` and relaunches resuming from the last backup (`concord_train_restart.py:96-102`). It also resumes through *native* crashes (the 0xC0000005 WDDM/CUDA-graph fault at the ceiling), capped at `CONCORD_MAX_CRASH_RETRIES` consecutive crashes; the cap resets on every clean segment so a progressing run is never starved (`:52-60, 104-117`). Clean exits (0, small non-zero, Ctrl+C) are trusted and stop.

**Where 42 fires.** In `GenericTrainer.py`, both the sample path (`:422-432`) and the per-epoch backup path (`:636-643`) write the backup, then `sys.exit(42)` *before* the in-process recommit (the model is already on `temp_device`). With sampling off (the standard config) the per-epoch backup is the boundary (`:1046-1049`). Before any backup, the captured graph is released — `_v2.release()` (`:560-562`) — because the save moves weights and stales the recorded pointers.

**Sidecars.** Next to the INTERNAL backup, the trainer writes `concord_clock.json` — `{update_steps: ctrl.step_idx, global_step, accum}` — an accum-change-proof update-step clock (deriving it as `global_step//accum` mis-seeds when accum changes between segments). The noise-seed seeder keeps its own workspace sidecar (`concord_nsr.json`, smoothed per-layer NSRs keyed by layer name) so relaunches seed immediately instead of running flat until the first window fills. (The hunting-servo kappa sidecar is gone with the servos.)

**Restore on resume.** At horizon finalize, when `continue_last_backup` is set, the trainer reads `concord_clock.json` (gated on matching `global_step`) and seeds `concord_controller.step_idx` so the fill-ramp/probe/watchdog continue; the seeder restores its smoothed NSRs from `concord_nsr.json` on construction. The model loader rebuilds a *standard* UNet (dropping packed buffers), so setup calls `__restore_concord_unet`/`__restore_concord_te` to reload the packed `packed_w`/s_fast/s_slow/v_slow into the re-swapped layers (`StableDiffusionXLFineTuneSetup.py:188-194`); embeddings re-seed from the loader-restored `base.weight` (`:206-214`). The graph (`concord_graph_v2`) is then rebuilt fresh (`:259-266`) and recaptures on the first step into a clean pool.

---

## Config & GUI integration

Concord's knobs live in three layers: **declared** as fields in `TrainConfig.py`, **defaulted** in `optimizer_util.OPTIMIZER_DEFAULT_PARAMETERS`, and **surfaced** across two GUI panels. The split matters: the *physics* knobs ride the optimizer-params plumbing, while the *embedding/data* flags are plain TrainConfig fields with concrete defaults.

**Optimizer-params knobs (physics).** These are declared in `TrainConfig.py` as `None`-defaulting *overridable* keys (the 4th tuple element `True`) — e.g. `("dissipation", None, float, True)`. `None` means "fall back to the engine winner default"; `concord_ot.make_concord_config` resolves them via `pick(...)`. The panel-visible defaults live in `optimizer_util.py`, the `Optimizer.CONCORD` block. They are surfaced in `OptimizerParamsWindow.py`:

- **`dissipation`** ("Dissipation (lam)", default `0.025`, optimizer_util.py:468): the dimensionless friction lam = lr·kappa. The engine sets `gf_consol = lam/lr` (concord_ot.py:152); clearing the field falls back to the raw `gf_consol` kappa key (TrainConfig.py:235, engine default 50). `gf_consol` itself is a config-file-only key — no panel entry.
- **`noise_seed_servo` / `autotune_servo_per_epoch` (3)**: the noise-seed seeder (the only per-layer dissipation controller) and its measurement-window knob (windows per epoch). The hunting servos' companion knobs are fully excised from the config surface.
- **`autotune_gamma_snr_on` (True) / `autotune_gamma_snr` (knee, None)** (optimizer_util.py:471-472): the master on/off plus knee for SNR-modulated dissipation. The checkbox exists precisely because `knee=0` alone could not disable it (snr/0 → inf pinned lam=1) — see the OptimizerParamsWindow.py:175-176 tooltips.

**Embedding & data flags (TrainConfig fields).** Declared at TrainConfig.py:543-553, defaulted with concrete values in `default_values` at TrainConfig.py:1164-1174, and surfaced in `TrainingTab.py`:

- `concord_packed_embeddings` (default True, :1093), `concord_embedding_anchor` (True, :1164), `concord_train_caption_vocab` (False, :1165), `concord_caption_vocab_anchor` (False, :1166).
- The divot/auto-drive cluster: `concord_embedding_delay_epochs` (1.0, :1167), `concord_embedding_auto_drive` (True, :1168), `concord_embedding_freq_exponent` (0.5, :1169).
- Quality-tag shield: `concord_embedding_quality_orthogonal` (False, :1172), `..._quality_tags` ("", :1173), `..._quality_mode` ("hard", :1174).

The embedding controls render in `TrainingTab.py:595-673`; `concord_sanitize_tokens` and the TE-anchor switches sit in the base frame (TrainingTab.py:353-355, 542-564).

**Note:** `concord_cuda_graph` (False), `concord_graph_te`, `concord_fused_matmul`, `concord_bucket_contiguous`, and `concord_packed_embeddings` are declared+defaulted (TrainConfig.py:1090-1094) but **not surfaced in any GUI panel** — they are config-file-only toggles (a grep of `modules/ui` for them returns nothing).
