# Local install runbook: C:\fisher\OneTrainer-clean → Concord + the June 2026 port

End-to-end instructions for a Windows box with Git Bash (paths in `/c/...` style, per
this repo's own run-record conventions). Start state: a directory at
`C:\fisher\OneTrainer-clean` that is either a vanilla OneTrainer clone or already the
Concord fork. End state: the `concord-integration` branch with the port applied
(C\* fix, coherence meter, F-units autotuner, stability guard), compiled and
smoke-tested. Companion docs: [`INSTALL_SDXL.md`](INSTALL_SDXL.md) (what the port is
and why), `patches/0001-concord-integration-cstar-tuner-guard.patch` (the diff).

## 0. Identify what you have

```bash
cd /c/fisher/OneTrainer-clean
ls modules/util/optimizer/concord/prototype_packed_b.py 2>/dev/null \
  && echo "FORK — go to step 2" || echo "VANILLA OneTrainer — go to step 1"
```

## 1. If vanilla: get the fork branch

The fork shares upstream history, so fetching into your existing clone works:

```bash
git remote add concord https://github.com/Theophite/Concord
git fetch concord concord-integration
git checkout -b concord-integration concord/concord-integration
```

(Equivalent alternative, if you prefer a separate directory:
`git clone -b concord-integration https://github.com/Theophite/Concord OneTrainer-concord`.)

## 2. Confirm the port is not already present, and the tip matches

```bash
grep -c "DissipationAutoTuner" modules/util/optimizer/concord/prototype_packed_b.py
# expect 0 (the remote branch has never carried the port)
git log --oneline -1     # expect d62931b — the commit the patch is generated against
```

If your tip has moved past `d62931b` or you have local edits to the three touched
files, use the manual steps in `INSTALL_SDXL.md` §1–§3 instead of the patch.

## 3. Apply the patch

Get it from `main` of the research repo (or use the copy you were sent):

```bash
curl -LO https://raw.githubusercontent.com/Theophite/Concord/main/patches/0001-concord-integration-cstar-tuner-guard.patch
git apply --check 0001-concord-integration-cstar-tuner-guard.patch   # must be silent
git apply         0001-concord-integration-cstar-tuner-guard.patch
```

What it changes (3 files, ~230 lines):
- `modules/util/optimizer/concord/prototype_packed_b.py` — the C\* mass-preserve fix
  (gate recalibration; see `RESULTS.md` §2) + `gate_coherence_from_fields` /
  `measure_coherence` / `DissipationAutoTuner` appended (with F-units and β1 support).
- `modules/util/optimizer/concord_ot.py` — `ConcordController` gains
  `autotune_table=...` (default `None` ⇒ **zero behavior change**),
  `autotune_table_in_F=True` (tables in dimensionless F = lr·κ), probe window
  fractions, and the per-step `tuner.step()` hook.
- `modules/modelSetup/StableDiffusionXLFineTuneSetup.py` — the F < 2 stability guard
  (refuses configs past the `lr·gf_consol` divergence ceiling at startup).

## 4. Verify

```bash
python -m py_compile modules/util/optimizer/concord/prototype_packed_b.py \
                     modules/util/optimizer/concord_ot.py \
                     modules/modelSetup/StableDiffusionXLFineTuneSetup.py

# CUDA smoke test of the packed kernels (now with the honest gate):
python modules/util/optimizer/concord/prototype_packed_b.py

# the Stage-1/2 regression net (~2 min, GPU): swap count, no-NaN, standard checkpoint
python tests/regression.py
```

Environment is unchanged — whatever ran the fork before runs it now (the patch is pure
Python over the existing Triton kernels; no new dependencies).

## 5. What is and isn't on by default

| change | default state |
|---|---|
| C\* fix | **ON** — the gate is recalibrated the moment the patch lands |
| stability guard | ON (passive; only fires on invalid configs) |
| coherence meter | available (`measure_coherence(layer)`), costs nothing until called |
| autotuner | **OFF** — dormant until a table is passed |
| β1 selection | OFF (`autotune_beta1_on=0` until you opt in) |

Because the C\* fix is on by default and the validated preset was tuned around the old
(half-blind) gate: **run a same-seed A/B against a pre-patch run before trusting new
results**, ideally with a small `gf_consol` sweep. `CONCORD_NO_RESTORE`-style
discipline applies; Concord is bit-deterministic at fixed seed, so the Δ you see is
real.

## 6. Enabling the autotuner (when you're ready)

The setup does not yet plumb a config field for the table, so enabling is a one-line
edit at the controller construction site
(`modules/modelSetup/StableDiffusionXLFineTuneSetup.py`, the
`ConcordController(...)` call at ~L160) — add:

```python
                autotune_table=[(0.43, 0.0), (0.35, 0.05), (0.31, 0.2),
                                (0.29, 0.5), (0.26, 1.0)],   # (coh -> F) — CALIBRATE, see below
```

- The table is in **F = lr·κ units** (dimensionless; F < 2 is the ceiling; reference
  points: current preset ≈ 0.004, LM winner 0.025, CPU memorization-regime optima
  0.4–1.5). The coh values above are the MNIST shape as a **placeholder** — calibrate
  per `INSTALL_SDXL.md` §3c (probe coherence on datasets of known quality + an F
  sweep) before trusting the commits.
- The probe runs at your configured `gf_consol` (converted to F internally), commits
  once at 10% of the run, and prints
  `[concord] dissipation autotune: probe coh=... -> kappa=..., beta1=...`.
- **CUDA-graph caveat** (`INSTALL_SDXL.md` §3d): `gf_consol`/`beta1` are baked at
  capture — for autotuned runs either set `concord_cuda_graph: false`, or arrange
  capture after the commit. Eager runs need nothing.

## 7. Run

GUI: optimizer = **CONCORD**, model SDXL, fine-tune; or load
`training_presets/#SDXL Concord Fused 24GB.json`. Watch `CONCORD_GRAPHMEM` lines as
usual. The first interesting experiments on this machine, in the order the research
repo's evidence ranks them (`RESULTS.md` adoption path): the same-seed A/B of the C\*
fix; the upward lr sweep (the campaign's optima sat ~10× above AdamW's — Polyak
prediction); the F sweep {0.004, 0.025, 0.1, 0.5, 1.0}.
