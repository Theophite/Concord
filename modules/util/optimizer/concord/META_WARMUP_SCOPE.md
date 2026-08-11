# Concord Meta-Warmup — Implementation Scope

*Status: PLAN ONLY. No production file is touched by this document. A live training run is active; this is a read-only design synthesis. All line numbers are against the `concord-integration` branch as read on 2026-06-23.*

---

## 0. One-paragraph orientation

The meta-warmup is a **second-order / statistics warmup**: before committing any weight change, run repeated full-epoch "dry passes" that develop the optimizer's *statistics* (the Adafactor `v_hat`, the per-layer servo `kappa`, the `v_slow/s_slow` coherence shape) to steady state, **reverting the weights after each pass**. It fixes the cold-start over-fit: in normal training the first steps overfit because `v_hat` is cold (the Adam-style step is huge) and the LAMB trust ratio has no warmup of its own. The mechanism reuses three things that already exist in the fork: (a) the exit-42 **restart-per-segment wrapper** (`scripts/concord_train_restart.py`) as the pass host; (b) the **servo sidecar** (`concord_servo.json`) + the **controller clock** (`concord_clock.json`) as the meta-state carrier; (c) the existing **`consolidated_weight()`** + a *new* in-place repack as the per-pass weight-revert primitive. The only genuinely new running code is: a per-pass weight-revert (decompose/repack, NOT `load_weights()`), a convergence predicate over the meta-state deltas, and a small amount of orchestration glue + config knobs.

---

## 1. Design recap

### 1.1 What is wrong at cold start (the thing we are fixing)
- `v_hat` (Adafactor `v_row`/`v_col` rank-1 second moment) starts at zero. The kernel multiplies `v_hat` by a per-device bias-correction buffer `1/(1-β2^t)` (`prototype_packed_b.py:1701-1710`, `_V_BC_BUFS`), and the cf-discount block reads `d_sv²/v_hat`. At `t=1` with a 1-epoch `β2` the correction is ~125× (controller note at `concord_ot.py:956-971`). Until `v_hat` is warm, both the step size and the cf-optimism are mis-scaled.
- The LAMB trust ratio (`_LAMB_TRUST`, maintained in `before_step` at `concord_ot.py:1017-1034`) has **no warmup of its own**; its first ratios are computed against cold norms.
- The servo `kappa` is *seeded* from `config.dissipation` and climbs one-sided per epoch (`EpochDissipationServo`, `prototype_packed_b.py:3562`). For the first epoch every layer is in the "baseline" branch (`prev is None`) and does not actuate — so dissipation is at the raw seed, not yet shaped to each layer's coherence.
- The coherence gate needs a `d_sv = (s_slow-v_slow)*128` direction to score signal vs noise (`gate_coherence_from_fields`, `prototype_packed_b.py:3375-3385`). At a fresh `load_weights()` init the split is even (`d_sv≈0`), so coherence reads ~0 and the gate is initially blind.

**Net:** the earliest real gradient steps land while every statistic is in its worst, coldest state — and those steps are *committed* into the weights (overfit to the first few batches).

### 1.2 The meta-warmup contract
Run `P` warm-up passes. Each pass = a forward/backward sweep over a **fresh shuffle of the full dataset** (one MGDS epoch). Across passes:

| Quantity | Action across a pass boundary | Why |
|---|---|---|
| `v_hat` (`v_row`/`v_col`, `adafactor_beta2`) | **PRESERVE** (keep accumulating) | this is the statistic we are warming |
| bias-correction step `t` (= `step_idx`) | **PRESERVE / keep advancing** | so `1/(1-β2^t) → 1`; never re-warm |
| servo `kappa`, `step`, `last_dir`, `last_memgap`, `epoch` | **PRESERVE** | the per-layer friction shape is the statistic |
| `v_slow/s_slow` ratio = `d_sv/deploy` | **PRESERVE the normalized shape** | gives the coherence gate a direction head-start each pass |
| consolidated magnitude `(s_slow+v_slow)*128*2^exp` | **RESET to start-of-warmup** | "revert all weight changes" |
| `s_fast` (transient velocity) | **RESET to 0** (or to the tiny `|.|≤64` init residual) | hypotheses are zeroed; velocity re-earned each pass |
| `row_exp`/`col_exp` | **PRESERVE** (do NOT recompute) | recomputing rebases the block-float frame and corrupts the preserved ratio |

After a reset, `d_fs=s_fast=0` (no velocity) but `d_sv` still carries the preserved shape → the coherence gate starts each pass already pointing the right way and re-earns velocity.

### 1.3 Stop rule — "if it has stopped converging, start going"
We do **not** watch the per-pass loss (the data reshuffles every pass, so loss is not comparable across passes). We watch the **meta-state deltas**: stop when the servo `last_dir` is mostly HOLD (kappa converged), the bias-correction `t` is saturated (`1/(1-β2^t) < 1+ε`), and the coherence ratio is stable across the last two passes. Then **COMMIT**: stop resetting, advance weights normally. A hard `max_passes` cap (default 6, see §4) guarantees termination if the signals are noisy.

### 1.4 Two implementation targets
1. **PRODUCTION path** — the live cf-discount/servo packed-B optimizer (`ConcordLinearPackedB`/`ConcordConv2dPackedB` in `prototype_packed_b.py`). This is the primary implementation (§3, §6).
2. **DUAL-DISSIPATION (substrate+offset)** — the design in `DUAL_DISSIPATION.md`, where "reset all weight changes" is trivially `offset→0` then re-lay `s_slow/v_slow` at the preserved ratio (§7). Mapping only; not built here.

---

## 2. Orchestration — where the warm-up loop lives

### 2.1 Host: the restart-per-segment wrapper (natural fit)
`scripts/concord_train_restart.py` already runs `train.py` as a sequence of fresh processes: each `exit(42)` (written by `GenericTrainer.__backup` at `:636-643` when `CONCORD_RESTART_ON_BACKUP=1`) is a clean segment boundary, the wrapper relaunches with `CONCORD_RESUMING=1` (`:96-102`), and `train.py:40-47` flips `continue_last_backup=True`. **Each warm-up pass = one such segment**, with one extra rule: while warming, the per-pass boundary additionally performs the weight-revert + convergence check before the exit(42)/relaunch. The servo sidecar + clock already ride across the boundary, so the meta-state is preserved *for free* by the existing machinery.

This is strongly preferred over an in-process Python loop wrapping `trainer.train()` because:
- The whole reason the wrapper exists is that the in-process CUDA-graph recommit fragments VRAM on a near-full card (`concord_train_restart.py:1-19`). An in-process warm-up loop would hit the same wedge `P` times.
- The graph capture bakes `alpha_v_fast`, `C*`, and the fill-ramp at capture time (`apply_epoch_window` note at `:715-717`). A fresh process per pass re-captures cleanly.

### 2.2 Control flow (pseudo)

**Wrapper (`concord_train_restart.py`), augmented:**
```
phase = read_phase_marker(workspace)        # "warmup" | "train" | absent->("warmup" if meta_warmup_enabled else "train")
segment = 0
while True:
    env["CONCORD_PHASE"] = phase            # NEW: tell the child which phase it is in
    ret = run(train.py, env)
    code = ret.returncode

    if code == RESTART_EXIT_CODE (42):
        env["CONCORD_RESUMING"] = "1"
        segment += 1
        phase = read_phase_marker(workspace) # child may have flipped warmup->train (commit)
        continue
    if is_crash(code): ... (unchanged bounded-retry) ...
    else: sys.exit(code)                     # done / clean error
```
The wrapper stays almost entirely unchanged; it only **propagates** `CONCORD_PHASE` and **re-reads** the phase marker after each segment (the child owns the warmup→train transition).

**Child (`GenericTrainer`), at the per-epoch backup boundary** (the `__needs_backup` site, `:1015`, → `__backup(..., restart_after=_rob)`):
```
at epoch boundary (epoch_step==0, epoch>0), CONCORD graph live:
    if phase == "warmup":
        # 1. develop meta-state already happened during the epoch (normal fwd/bwd, servo ticked,
        #    v_hat accumulated, s_fast grew). The forward/backward ran REAL but we now revert.
        meta = controller.snapshot_meta()                # servo state + per-layer ratio shape (read-only)
        controller.meta_warmup_revert_weights()          # §3: per-layer decompose->repack
        converged = controller.meta_warmup_converged(pass_index, meta)   # §4
        pass_index += 1
        if converged or pass_index >= meta_warmup_max_passes:
            write_phase_marker(workspace, "train")       # COMMIT: next segment trains for real
            # optionally reset train_progress.epoch / global_step (see §8 "epoch continuity")
        else:
            write_phase_marker(workspace, "warmup")
        __backup(restart_after=True)                     # writes servo+clock sidecars, exit(42)
    else:  # phase == "train"
        __backup(restart_after=_rob)                     # EXISTING behavior, unchanged
```
Key point: the warm-up pass uses the **normal** train loop (real forward/backward, real `optimizer.step`, servo ticks, `v_hat` accumulates, `s_fast` grows). The *only* warm-up-specific action is the **revert at the boundary** plus the **convergence test** — both happen after the epoch's gradients have already shaped the meta-state.

### 2.3 Phase / pass persistence
A tiny `concord_warmup.json` in `workspace_dir` (sibling of the sidecars, written by the child, read by both):
```json
{ "phase": "warmup", "pass_index": 2, "start_weights_ref": "<backup-name or 'init'>" }
```
- `phase` drives the wrapper and child.
- `pass_index` is the warm-up pass counter (separate from `train_progress.epoch`).
- `start_weights_ref` records *what* "start-of-warmup magnitude" means (see §3.4: persisted vs reconstructed).

This marker is the single source of truth for "are we still warming?" and survives crash-resume (a crashed warm-up pass simply re-runs that pass; idempotent because the revert is to the same start magnitude).

---

## 3. The per-pass RESET (production path)

### 3.1 Reset vs preserve (exact)
**RESET** (per managed layer):
- consolidated magnitude `(s_slow*128 + v_slow*128)*2^exp` → back to the **start-of-warmup** weight `D_init`.
- `s_fast` → 0 (or keep the `|.|≤64` init residual; see §3.5 evaporation-gate risk).

**PRESERVE** (do NOT touch):
- `v_row`, `v_col`, `_sum_v_inv`, `adafactor_beta2` (these are *separate device buffers*, untouched by a packed-word repack — they survive automatically as long as we do **not** call `load_weights()` / `_init_weight()`).
- `row_exp`, `col_exp` (the block-float frame — preserving these is what makes the ratio re-lay exact).
- servo `_kappa`, `_step`, `_last_dir`, `_last_memgap`, `_epoch`, `_last_step_t` (carried by the sidecar across the exit-42 boundary; **also** keep live if the revert is done in-process before backup).
- `step_idx` (the bias-correction clock — keep advancing).
- the normalized coherence shape `r = d_sv/deploy` per layer.

### 3.2 Why NOT `load_weights()` for the reset
`load_weights(W)` (`:2653`) **recomputes** `row_exp` from `W.abs().max` and **even-splits** `s_slow==v_slow` (`d_sv≈0`). That throws away the developed coherence ratio *and* rebases the exponent frame. Using it as the reset primitive would destroy preserve-target (c) (the ratio) and (b) the frame. **Decision: do NOT use `load_weights()`. Add a dedicated in-place revert that repacks `packed_w` directly while leaving `row_exp`/`col_exp`/`v_row`/`v_col` alone.**

### 3.3 The ratio-preserving re-lay (the math)
Read current packed fields via `get_state()` (`:2810`): `(s_fast, s_slow_i8, v_slow_i8)`, plus `row_exp`/`col_exp`. All algebra below is **per element** (the exponent frame is per-row/col but constant through the repack, so `2^exp` cancels in every ratio).

1. Current coarse mantissa: `coarse_cur = s_slow_i8 + v_slow_i8` (this is `deploy/(128·2^exp)`).
2. Current normalized direction: `r = (s_slow_i8 - v_slow_i8) / (coarse_cur + ε)` — dimensionless, in `[-1,1]`. This is the preserved coherence shape `d_sv/deploy` (since `d_sv=(s_slow-v_slow)·128` and `deploy=(s_slow+v_slow)·128`, the `128`s cancel → `r` is exactly `d_sv/deploy`).
3. Target coarse from the start magnitude `D_init`: `coarse_tgt = round( D_init / (128 · 2^exp) )`, clamped to `[2·INT8_MIN, 2·INT8_MAX]`. Because `row_exp`/`col_exp` are preserved, `2^exp` here is the *same* frame the layer is using — `coarse_tgt` is just the start-of-warmup coarse mantissa.
4. Re-lay the split at the preserved ratio: solve `s_new - v_new = r·coarse_tgt` and `s_new + v_new = coarse_tgt`:
   - `s_new = round( coarse_tgt · (1 + r) / 2 )`, clamp `[INT8_MIN, INT8_MAX]`
   - `v_new = (coarse_tgt - s_new)`, clamp `[INT8_MIN, INT8_MAX]`
   - (This re-derives the same even-vs-skew split shape `load_weights(gap=r)` would, but at the *developed* `r`, not 0.)
5. `s_fast_new = 0` (or the small init residual; §3.5).
6. Repack: `packed_w = ((s_fast_new & 0xFFFF)<<16) | ((s_new & 0xFF)<<8) | (v_new & 0xFF)` — identical layout to `load_weights` `:2694-2699`.
7. `self._resync_weight_buf()` (`:2764`) to refresh the bf16 cache the next forward reads.

**Properties:** deploy magnitude → `coarse_tgt·128·2^exp ≈ D_init` (start magnitude restored, to coarse quantization). `r` preserved to one rounding unit. `d_fs=s_fast=0` (velocity zeroed). `v_row`/`v_col` untouched. Servo untouched.

> Note on `coh_pre`: there is no stored `coh_pre` field on the layer — coherence is *computed on demand* from the packed fields by `gate_coherence_from_fields`/`measure_coherence` (`:3375-3400`). So "preserve coh_pre" is automatically satisfied by preserving `s_slow/v_slow` ratio + `s_fast` shape; since we zero `s_fast`, the post-reset `measure_coherence` is computed from `d_sv` (the preserved direction) vs `noise=d_fs-sig=-sig`, i.e. `sig²/(sig²+sig²) = 0.5`-ish per element — a *direction head-start*, not a velocity. That is exactly the intended "gate gets a direction head-start, re-earns velocity each pass."

### 3.4 What "start-of-warmup magnitude" `D_init` is, and where it comes from
Two viable definitions; pick one (open question Q1):
- **(A) True initial weight** (recommended): snapshot `consolidated_weight()` of every layer **once**, at the very start of warm-up (pass 0, before the first gradient), to a `concord_warmup_init/` sidecar (bf16 per-layer, or just the coarse mantissa + row/col exp — compact). Every pass reverts to *this*. Deterministic, drift-free.
- **(B) Self-consistent reload**: revert to the weights as they were at the *previous* pass start. Cheaper to store (reuse the last backup) but drifts if any pass is partially applied; not recommended.

With (A), the revert reads `D_init` from the init sidecar; `coarse_tgt` in §3.3 step 3 uses it directly. The init sidecar is written once and read on every warm-up segment (it must survive the exit-42 relaunch, so it lives in `workspace_dir`, not the per-backup dir).

### 3.5 `s_fast` and the evaporation gate (must-handle)
Zeroing `s_fast` to exactly 0 trips the evaporation build gate: `|s_fast|=0 < evap_build_min` ⇒ `build_ok=0`, so the coherence gate stays inert and never re-earns velocity. The `load_weights` init deliberately keeps `|s_fast|≤64` precisely to avoid being an evaporation target while still being non-zero (`:2659-2663, 2692-2693`). **Decision:** the revert should restore the **start-of-warmup `s_fast` residual** (the `m_total - (s_slow+v_slow)·128` fine residual from §3.3 against `D_init`), NOT a hard zero. This is `|.|≤64` by construction and matches what the gate expects at a fresh layer. (If the winner default sets `evap_build_min=0`, a hard zero is also acceptable — but restoring the init residual is safe under both.)

### 3.6 Embeddings (scope decision)
The packed token-embedding cores have their own servo group (`"emb"`, `concord_ot.py:339-342`) and a sighting-clocked, per-row drive. A full-finetune freezes the base vocab, so the embedding optimizer is largely inert (per MEMORY: `concord_packed_embedding`). **Decision:** warm-up reset scope = UNet + winner-TE layers only (`controller.layers` + winner `te_layers`). Embedding cores are left live (not reverted) during warm-up; if a run trains added tokens, document that the warm-up does not revert them (open question Q5).

---

## 4. Convergence predicate + max-pass cap

Evaluated at each pass boundary (after the epoch's gradients shaped the meta-state, before the revert is final / after the revert — the signals below are read from servo state and `step_idx`, both insensitive to the revert).

**Signal 1 — servo hold fraction** (kappa converged):
```
hold = (# layers with servo._last_dir[id(m)] == 0) / len(layers)     # across ALL servo groups
```
Converged-1 when `hold ≥ meta_warmup_hold_frac` (default 0.8). `last_dir==0` means the layer neither climbed nor descended this epoch (deadband HOLD) — the per-layer friction has settled. Read from `EpochDissipationServo._last_dir` (`:3643, 3774, 3784`), aggregated over `controller.autotuners`.

**Signal 2 — bias-correction saturation** (`v_hat` warm):
```
t = controller.step_idx
bc = bias_correction_factor(t, beta2)      # prototype_packed_b.py:1708, = 1/(1-β2^(t+1))
warm = bc < (1.0 + meta_warmup_bc_eps)     # default eps=1e-3  -> warm
```
Because `step_idx` keeps advancing across passes (never re-warms — `concord_ot.py:960`), `bc→1` monotonically. With a 1-epoch `β2 = 1 - 1/spe`, one full pass already drives `t = spe` and `bc ≈ 1/(1-(1-1/spe)^spe) ≈ 1/(1-e^-1) ≈ 1.58` after pass 1, `≈1.13` after pass 2, `<1.01` by ~pass 4-5. **This signal alone roughly sets the floor on pass count.** (Note: this `bc` driver is only *active* when `config.bias_correct_v` is on; if it is off, replace this signal with a fixed `min_passes` floor — see Q3.)

**Signal 3 — coherence ratio stability**:
```
coh_now  = [measure_coherence(m) for m in layers]     # prototype_packed_b.py:3389
coh_stable = (pass_index >= 1) and max(|coh_now[i] - coh_prev[i]|) < meta_warmup_coh_tol  # default 0.01
coh_prev <- coh_now    # persist in concord_warmup.json (survives exit-42)
```
Measured **after** the revert (so it scores the preserved direction shape, comparably across passes). `coh_prev` must persist in the warmup marker because a crash mid-pass would otherwise reset the delta baseline.

**Stop predicate:**
```
converged = (hold >= hold_frac) AND warm AND coh_stable
commit = converged OR (pass_index + 1 >= meta_warmup_max_passes)
```
**Cap:** `meta_warmup_max_passes` default **6** (enough for `bc` to saturate at a 1-epoch `β2`; raise for very short epochs). Optional smoothing: require the predicate true on **2 consecutive** passes to defeat flip-flop (Q4).

---

## 5. Data reshuffle (fresh shuffle per pass — already free)

MGDS derives the per-epoch shuffle **directly from the epoch counter**, not a separately seeded RNG draw: `LoadingPipeline.start_next_epoch()` increments `__current_epoch` and pushes it as the `variation` into every module (`venv/src/mgds/.../LoadingPipeline.py:82-109`); the dataset seed is fixed once at construction (`MGDS.py:31`). Consequences:
- A warm-up pass that runs one normal epoch **already reshuffles** on the next `start_next_epoch()` (the standard epoch-boundary call at `GenericTrainer.py:864/868`). No new shuffling code is needed — each pass naturally sees a fresh variation because `__current_epoch` advances.
- **Do NOT rebuild the dataloader** per pass (the gap-noted "fresh MGDS instance" option). Reusing the instance and letting `start_next_epoch()` advance the variation is the correct, cheaper path, *provided* batch_size / drop_last / num_concepts are constant (they are, within a run).
- **Epoch-count pollution:** `train_progress.epoch` advances by `P` during warm-up. On commit, either (a) reset `train_progress.epoch`/`epoch_step`/`global_step` to 0 so the real run's LR cosine / fill-ramp / telescope start from a clean zero, or (b) carry the offset and subtract it everywhere the schedules read the clock. **Recommended: (a) reset to 0 on commit** — the warm-up explicitly committed *no* weight change, so the real run should start at step 0 of all schedules. This requires resetting `controller.step_idx` to 0 at commit **only if** we want the LR/fill-ramp warmups to run fresh on the real data; but note that resets `bc` too (re-cold `v_hat` correction). **Tension flagged as Q2** — the cleanest resolution is: reset the *schedule* clock (LR cosine, fill-ramp horizon) to 0 but keep `v_hat`/`step_idx` warm, i.e. decouple the bias-correction clock from the schedule clock at commit.

---

## 6. Changes by file (concrete)

### 6.1 `modules/util/optimizer/concord/prototype_packed_b.py`
- **NEW method** `ConcordLinearPackedB.meta_warmup_revert(self, coarse_tgt_mantissa=None, keep_residual=True)** and the same on `ConcordConv2dPackedB`:
  - Implements §3.3 exactly: read `get_state()`, compute `r`, take `coarse_tgt` from the passed start-magnitude mantissa (or recompute from a passed `D_init` weight), re-lay `s_new/v_new` at `r`, restore the init `s_fast` residual (or 0), repack `packed_w`, `_resync_weight_buf()`.
  - Must **not** touch `row_exp`/`col_exp`/`v_row`/`v_col`. (Contrast with `load_weights` `:2674-2677` which writes `row_exp`.)
- **NEW helper** `EpochDissipationServo.hold_fraction(self) -> float`: `mean(1.0 for d in self._last_dir.values() if d==0)`. (Read-only; convergence Signal 1.)
- (Optional) **NEW** `EpochDissipationServo.is_settled(self, hold_frac)` wrapping the above for readability.
- No change to `export_state`/`import_state`/`measure_coherence`/`bias_correction_factor` — reused as-is.

### 6.2 `modules/util/optimizer/concord_ot.py` (ConcordController)
- **NEW** `snapshot_meta(self) -> dict`: per-layer `r=(s_slow-v_slow)/(s_slow+v_slow)` and per-layer `measure_coherence`, plus aggregated servo `hold_fraction` and `step_idx`. Pure read.
- **NEW** `meta_warmup_revert_weights(self, init_provider)**: iterate `self.layers` (+ winner `self.te_layers`), call each layer's `meta_warmup_revert(...)` with its `D_init` coarse mantissa from `init_provider`. Skips embeddings (§3.6).
- **NEW** `meta_warmup_converged(self, pass_index, prev_coh, prev=None) -> (bool, new_coh)`: §4 predicate over `self.autotuners` hold-fraction, `bias_correction_factor(self.step_idx, beta2)`, and per-layer coherence delta vs `prev_coh`.
- **NEW** `write_warmup_init_snapshot(self, dir)` / `read_warmup_init_snapshot(self, dir)`: §3.4(A) — dump/load per-layer coarse mantissa (`s_slow_i8+v_slow_i8`) + `row_exp`/`col_exp` to `concord_warmup_init/` (compact int8/int8). This *is* `D_init`.
- The `before_step` bias-correction driver (`:961-971`) is **unchanged** but is now load-bearing for Signal 2; ensure `config.bias_correct_v` interplay is documented (Q3).

### 6.3 `modules/trainer/GenericTrainer.py`
- In `__backup` (`:536`) / the boundary site (`:1015, 1040-1049`): branch on `os.environ.get("CONCORD_PHASE")=="warmup"`. In warm-up phase, before the `restart_after` exit(42):
  - on pass 0 only: `controller.write_warmup_init_snapshot(workspace_warmup_init_dir)` (idempotent — skip if exists).
  - `meta = controller.snapshot_meta()`
  - `controller.meta_warmup_revert_weights(init_provider)`
  - `converged, coh = controller.meta_warmup_converged(pass_index, prev_coh)`
  - write `concord_warmup.json` (`phase`, `pass_index+1`, `coh_prev=coh`); flip `phase="train"` when `converged or pass_index+1>=max_passes`.
  - force `restart_after=True` (always exit(42) per warm-up pass).
- The servo + clock sidecars are **already written** by `__backup` (`:587-606`) — they carry the preserved meta-state. No change there.
- The resume path (`:937-970`) already restores `step_idx` and `_servo_resume_state` — no change; the warm-up just relies on it firing every pass (it does).
- (commit) On `phase` flip to `train`, optionally reset `train_progress` schedule clock per §5/Q2.

### 6.4 `scripts/concord_train_restart.py`
- Read `concord_warmup.json` phase at start and after each segment; set `env["CONCORD_PHASE"]`. (≈8 lines; the loop body at `:86-123` is otherwise unchanged.) A crashed warm-up segment resumes exactly like any crash (bounded retry) and re-runs that pass.

### 6.5 `scripts/train.py`
- No structural change needed (the wrapper-driven design keeps the loop in `GenericTrainer`). Optionally read `CONCORD_PHASE` for a startup log line. (If a *non-wrapper* fallback is wanted, an in-process loop could wrap `trainer.train()` at `:99` — but see §2.1 for why the wrapper path is preferred; treat the in-process path as Q6.)

### 6.6 New config knobs (`TrainConfig` / optimizer config)
| knob | default | meaning |
|---|---|---|
| `meta_warmup_enabled` | `false` | master opt-in |
| `meta_warmup_max_passes` | `6` | hard cap on warm-up passes |
| `meta_warmup_hold_frac` | `0.8` | Signal 1 servo-hold threshold |
| `meta_warmup_bc_eps` | `1e-3` | Signal 2 saturation tolerance (`bc<1+eps`) |
| `meta_warmup_coh_tol` | `0.01` | Signal 3 per-layer coherence-delta tolerance |
| `meta_warmup_min_passes` | `2` | floor (used when `bias_correct_v` is off, Q3) |
| `meta_warmup_require_consecutive` | `1` | passes the predicate must hold in a row (Q4) |
| `meta_warmup_reset_schedule_clock` | `true` | on commit, restart LR/fill-ramp at step 0 (Q2) |
| `meta_warmup_keep_sfast_residual` | `true` | restore `|.|≤64` init residual vs hard-zero (§3.5) |

---

## 7. Dual-dissipation (substrate+offset) mapping

In the `DUAL_DISSIPATION.md` model: `weight = substrate + offset`, where **substrate** is the fixed init seed (regenerated on the fly, never stored, §1 of that doc) and **offset** = `e_L + e_H + s_slow + v_slow` (hypotheses + theories). Deploy = `substrate + (s_slow + v_slow)`; the two hypotheses `e_L`, `e_H` are dropped.

The meta-warmup reset is **trivial and cleaner** here:

| production packed-B | dual-dissipation analog |
|---|---|
| revert consolidated magnitude to `D_init` | the *substrate is already the init* — "revert to start magnitude" ≡ drive **offset → 0** (or to the start-of-warmup offset). No init snapshot needed: the substrate regenerates from the seed. |
| zero `s_fast` (velocity) | zero **both** hypotheses `e_L = e_H = 0` (the §1.4 "no even-split" arms reset to empty) |
| re-lay `s_slow/v_slow` at preserved ratio `r` | re-lay the two **theories** `s_slow/v_slow` at the preserved `r` (same §3.3 math; the theories are the ×128 coarse pair, identical encoding). `resplit_anchor_to_even` (`:2724`) is the *even* case; the general case re-lays at the developed `r` exactly as §3.3. |
| preserve `v_hat`, servo kappa | **subsumed**: dual-dissipation makes dissipation self-tuning (it "subsumes the cf-discount heuristic and the EpochDissipationServo", `DUAL_DISSIPATION.md` §TL;DR). The warm-up still preserves whatever per-arm coherence statistics the two timescales have developed; the "servo kappa" preserve-target maps to the **developed `λ_L`/`λ_H` bracket split per weight**. |

So in the substrate+offset world, the per-pass reset is: **`offset → start` (≡ `e_L=e_H=0`, theories re-laid at `r`), substrate untouched (regenerated), `v_hat` + per-weight timescale stats preserved.** The conservation ledger (`DUAL_DISSIPATION.md` §3) makes "revert" exact: the deploy only ever gained what the fine register lost, so zeroing the hypotheses + re-laying the theory at the start coarse value restores the start deploy with no leak. **Note:** because the two arms are fed *different random halves* and require two fwd/bwd per step, a warm-up pass there is ~2× the compute of a production-path pass — fold into the cost estimate.

---

## 8. Risks & edge cases

1. **`load_weights()` must NOT be the reset primitive** — it recomputes `row_exp` and even-splits, destroying the preserved ratio and the exponent frame. The dedicated in-place repack (§3.3) is mandatory. *(Highest-severity correctness item.)*
2. **Evaporation gate on zero `s_fast`** — hard-zeroing `s_fast` can set `build_ok=0` and freeze the gate (§3.5). Mitigation: restore the `|.|≤64` init residual (`meta_warmup_keep_sfast_residual=true`).
3. **`v_row`/`v_col` are separate buffers** — they are preserved automatically *only if* the reset touches `packed_w`/`row_exp`/`col_exp` and nothing else. Any path that calls `_init_weight`/`load_weights`/model reswap mid-warmup would clobber them. Warm-up must not reload or swap the model between passes.
4. **Empty-meter servo guard on boundary resume** — the servo already skips a zero-accumulation boundary (`:3726-3736`) when a relaunch lands exactly on a boundary. Because each warm-up pass is a full epoch with real gradients, the meters are non-empty at the boundary → the servo acts normally. But the *first* before_step after relaunch can still hit the empty guard; the convergence read of `hold_fraction` must be taken from the **persisted** servo state (sidecar), not from a just-rebuilt-but-not-yet-ticked servo. Read it after `_build_autotuner` has imported the sidecar.
5. **Bias-correction clock vs schedule clock at commit** — resetting `train_progress`/`step_idx` to 0 on commit re-colds the `bc` driver (re-warms `v_hat` correction) — defeating the warm-up's main win. Resolution: reset only the LR/fill-ramp *schedule* horizon, keep `step_idx`/`v_hat` warm (Q2). This requires a clean separation of the two clocks at commit.
6. **Servo `import_state` layer-count gate** — `import_state` no-ops if the layer count changed (`:3821`). A warm-up that doesn't change the model is fine; flag that any architecture change voids the warm-up state.
7. **CUDA-graph capture bakes the ramp** — `alpha_v_fast`/`C*`/fill-ramp are launch-time scalars baked at capture (`apply_epoch_window` `:715-717`). The fresh-process-per-pass design re-captures cleanly; an in-process loop (Q6) would capture a stale ramp.
8. **`approximate_length()` variation drift** — MGDS length can in principle vary with the epoch `variation` (gap-4 note). Within a fixed-config run it is constant; verify the dataset length is variation-independent for the target config, else `steps_per_epoch` (servo cadence) drifts across passes.
9. **`coh_prev` persistence** — Signal 3's baseline must live in `concord_warmup.json`, else a crash mid-pass resets the delta and can spuriously report "not converged" forever (no harm beyond extra passes, but documents the cap as the real terminator).
10. **Init-snapshot durability** — the `D_init` snapshot (§3.4 A) must be in `workspace_dir` (survives exit-42), not a per-backup dir (pruned by rolling backup, `:620-621`).
11. **Crash during a warm-up pass** — bounded-retry resumes from the last backup and re-runs the pass; idempotent because the revert targets a fixed `D_init`. Confirm the crash budget (`CONCORD_MAX_CRASH_RETRIES`) is not exhausted by a deterministic warm-up fault.

---

## 9. Test plan (CPU-testable, no GPU, no live run)

All of the below run on `prototype_packed_b.py`'s CPU path (the smoke-test harness at `:3852+` instantiates layers on CPU) — **none requires the GPU, Triton, or the live run.** Build a tiny `ConcordLinearPackedB` (e.g. 16×16) on CPU.

1. **Revert round-trip — magnitude.** `load_weights(W0)`; run a few synthetic `apply` steps (mutate `s_fast`/`s_slow`/`v_slow`); call `meta_warmup_revert` against `W0`'s coarse mantissa; assert `consolidated_weight()` ≈ `W0` to coarse quantization (1 ULP of the ×128 mantissa).
2. **Revert preserves the exponent frame.** Assert `row_exp`/`col_exp` are bit-identical before/after revert (proves we did not call the `load_weights` exp-recompute path).
3. **Revert preserves `v_hat`.** Set `v_row`/`v_col` to known nonzero tensors; revert; assert they are unchanged (object identity + values).
4. **Ratio-preserving re-lay.** Construct a layer with a known skew `r≠0` (set `s_slow≠v_slow`); revert; assert `(s_slow-v_slow)/(s_slow+v_slow)` ≈ `r` to one rounding unit, and `d_sv` sign matches.
5. **`s_fast` residual rule.** With `keep_residual=true`, assert post-revert `|s_fast| ≤ 64` everywhere (never an evaporation target); with `false`, assert `s_fast==0`.
6. **Coherence head-start.** After revert, `measure_coherence(layer)` is computed from the preserved `d_sv` (nonzero where `r≠0`) and `s_fast=0` — assert it is the "direction head-start" value (≈0.5 per element where `r≠0`, not 0).
7. **Servo hold-fraction signal.** Build an `EpochDissipationServo` over CPU layers; force `_last_dir` to a known mix; assert `hold_fraction()` matches; assert the §4 predicate flips at the threshold.
8. **Bias-correction saturation.** Pure-function test of `bias_correction_factor(t, β2)` at `β2=1-1/spe`: assert it crosses `1+eps` at the pass count the cap was sized for (sanity-checks `max_passes=6`).
9. **Sidecar round-trip.** `export_state()` → JSON → `import_state()` preserves `kappa`/`step`/`last_dir`/`epoch` (already-existing behavior; pin it as a regression guard for the warm-up dependency).
10. **Init-snapshot round-trip.** `write_warmup_init_snapshot` → `read` → revert reproduces the same `D_init` as reverting against the in-memory weight.
11. **Convergence-predicate state machine.** Drive `meta_warmup_converged` with scripted (hold, bc, coh-delta) sequences; assert commit fires exactly when all three hold (and at the cap), including the `require_consecutive` smoothing.
12. **Dual-dissipation mapping (if `dual_dissipation_ref.py` is importable on CPU).** Assert `offset→0` + re-lay theory at `r` reproduces the start deploy and closes the conservation ledger (reuse the existing conservation test, `tests/test_dualdis_*` per `DUAL_DISSIPATION.md` §7).

**Not CPU-testable (defer to a guarded dry run, never against the live run):** the full exit-42 orchestration loop, real MGDS reshuffle across passes, CUDA-graph re-capture per pass, and end-to-end convergence on real data.

---

## 10. Open questions for the architect

- **Q1 — `D_init` definition.** Snapshot the true initial weight once (§3.4 A, recommended, drift-free) vs revert-to-previous-pass-start (B, cheaper storage). A picks "revert all weight changes" literally; confirm A.
- **Q2 — clock split at commit.** On commit, do we (i) reset the LR/fill-ramp **schedule** clock to 0 while keeping `step_idx`/`v_hat`/`bc` warm (decoupled clocks — recommended), or (ii) carry the full warm-up step offset into the real run, or (iii) hard-reset everything (re-colds `v_hat`, partially defeating the warm-up)? This is the one place the design has a genuine tension (§5, Risk 5).
- **Q3 — dependence on `bias_correct_v`.** Signal 2 (the cleanest convergence signal) is only *live* when `config.bias_correct_v` is on. If the production run has it off, do we (a) turn it on *for the warm-up only*, or (b) drop Signal 2 and rely on `min_passes` + Signals 1&3? Which matches the intended recipe?
- **Q4 — convergence smoothing.** Require the predicate on 2 consecutive passes (`require_consecutive=2`) to defeat flip-flop, or trust a single crossing? Given the hard cap is cheap, conservative (2) seems safe — confirm.
- **Q5 — embedding scope.** Confirm warm-up reverts UNet + winner-TE only and leaves embedding cores live (§3.6). For a run that trains added tokens, is "don't revert the embedding offset during warm-up" acceptable, or should the embedding `s_fast` also reset?
- **Q6 — orchestration host.** Confirm the restart-wrapper-as-pass-host (§2.1) over an in-process loop. The wrapper avoids the VRAM-wedge and stale-capture, at the cost of one model reload per pass (~1-2 min). Acceptable for `≤6` passes?
- **Q7 — pass = exactly one MGDS epoch?** The design says "epoch-sized fresh shuffle." Confirm one full epoch per pass (vs a fraction, e.g. a half-epoch, to make the `bc` warm-up cheaper). One epoch keeps the servo's per-epoch cadence intact (`epoch_steps`), so it is the natural unit.
- **Q8 — substrate+offset timing.** The dual-dissipation reset is cleaner (offset→0, no init snapshot) but the design is CPU-verified only, not yet on SDXL (`DUAL_DISSIPATION.md` §5). Is the meta-warmup intended to ship on the *current* packed-B production path first (§3) with the dual-dissipation mapping (§7) as the forward design, or to land together with the dual-dissipation port?
