# The research notebook

Everything in this directory is **historical record**: the prototypes, experiment
series, run scripts, and session notes that produced the validated optimizer in
`/concord` and the findings in `/docs`. Nothing here is the product; nothing here is
maintained. It is kept because this project's culture treats provenance as part of the
result (`/WINNING_CONFIG.md` cites these files line-by-line) and scar tissue as
documentation.

Practical notes for anyone (or any agent) digging here:

- The `tools/run_*.sh` scripts are **run records, not portable runners** — they hardcode
  the original machine (`cd /c/concord`). Read them for the exact flags of a cited A/B;
  don't expect them to execute.
- `src/` files import siblings by bare name; run them with the script's own directory as
  cwd (or rely on Python putting the script dir on `sys.path`).
- One bug fix has been forward-ported into the historical tree to keep cited code
  honest: the C\* mass-preserve correction in `src/prototype_packed_b.py` (see
  `/docs/RESULTS.md` §2). Otherwise files are as the experiments left them.

## Map

### `src/` — prototypes, harnesses, experiment series

**The optimizer lineage** (chronological, roughly):

| file | what it was |
|---|---|
| `concord_optimizer.py`, `concord_linear_fused.py` | the pre-packed era: separate int accumulators (the 40 bits/param design documented in `notes/CONCORD_README.md`) |
| `concord_triton.py`, `concord_triton_fused.py` | first Triton kernels for that era |
| `prototype_packed.py`, `prototype_packed_2acc.py` | first packed-word experiments |
| `prototype_packed_e.DECOMPILED.py`, `.RECOVERED.py` | a lost variant recovered from a stale `.pyc` (toolchain in `tools/_inspect_pyc.py`, `tools/_recover_packed_e.py`, `tools/packed_e_disasm.txt`) |
| `prototype_packed_b.py` | **the lineage that won** — same family as the canonical `/concord/packed_b.py`; the SDXL fork's production kernel descends from this file |
| `optim_factored.py`, `mtopt.py` | baselines / side quests |

**Benches and harnesses:** `train_nanogpt.py` (the enwik8 validation bench — the
winner's numbers come from here), `nanogpt.py`, `train_cifar.py`, `cifar_in_memory.py`,
`train_diffusion.py`, `mtopt_cifar.py`.

**Experiment series:** `cifar_*.py` (the CIFAR era, including `cifar_vmode_fork.py` —
the label-noise/Bayes-error methodology later reused in `/experiments/cpu_dynamics`);
`sdxl_*.py` + `run_cache.py` + `onetrainer_concord_patch.py` (pre-fork SDXL integration
prototypes — superseded by the `concord-integration` branch); `concord_embedding*.py`,
`control_plane.py`, `token_init.py`, `init_token_list.py` (token control plane);
`concord_winner.py` (the controller, later vendored into the fork);
`concord_polyak.py` (the variational-posterior / Polyak hypothesis selector).

**Diagnostics and probes:** `diag_*.py`, `probe_*.py`, `profile_packed_b.py`,
`analyze_vmode.py`, `check_gf_routing.py`, `compare_gap.py`, `noise_rank_probe.py`,
`reconstruct_sweep.py`, `settle_analysis.py`, `snapshot_prod.py`, `accum_*.py`,
`prod_accum_test.py`.

**Tests** (`test_*.py`): unit and regression tests from each era — `test_ot_*` are the
OneTrainer-integration regression nets, `test_t5_*` the T5/SST-2 transfer checks,
`test_baked_defaults.py` duplicates the one shipped in `/concord`.

### `sims/` — closed-loop theory simulations

`exp1_walk.py` … `exp8_single_sample.py`: the idealized-dynamics studies (random walks,
selectivity, v_slow gating, epoch noise, gated leak, deviation factoring, closed loop,
single-sample) that motivated the mechanism before any real training run.

### `tools/` — run records and build tooling

`run_*.sh`: the cited A/B and sweep invocations (e.g. `run_split_ab.sh` and
`run_sigma_sweep.sh` are the provenance for `/WINNING_CONFIG.md`'s table).
`build_min_zip.py`: builds the distilled package snapshot (output to `/dist`).
`probe_sigmag.py`, `_inspect_pyc.py`, `_recover_packed_e.py`, `packed_e_disasm.txt`:
one-off tooling.

### `notes/` — design docs and session records

| file | what |
|---|---|
| `CONTROL_PLANE.md` | the enwik8 experiment registry — "single source of truth" for every run, config, and result on that bench; cited by `/WINNING_CONFIG.md` |
| `SESSION_NOTES_2026-05-29.md` | the session log containing, among other things, the Bayes-error regime analysis that `/docs/RESULTS.md` later confirmed out-of-domain |
| `CONCORD_README.md` | the predecessor design's full documentation (40 bits/param era) — historical, bannered |
| `OLD_README_concord_adamw.md` | the repo's previous root README — historical, bannered |

### `integration/` — the fork's design documents

`DESIGN.md` and `HANDOFF.md`: the specs that drove the `concord-integration` OneTrainer
fork. The living integration guidance is `/docs/INSTALL_SDXL.md`; these are the original
plans.
