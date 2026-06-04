# Concord integration — tests

Run with the OneTrainer venv's Python. Paths are derived from the repo (no absolute
paths); the only external dependency is an SDXL checkpoint, via env vars:

- `CONCORD_TEST_MODEL` — path to an SDXL `.safetensors` checkpoint (required for the
  regression run).
- `CONCORD_TEST_WORKDIR` — scratch dir for the synthetic dataset / config / output
  (optional; defaults to a temp dir).

### `test_stage3_assertions.py` — no model needed
Verifies the CUDA-graph design assertions: bf16 ⇒ OneTrainer makes no GradScaler; the
default RNG advances across graph replays while a custom `torch.Generator` is not
capturable (so diffusion noise must be eager-fed).

```
venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_stage3_assertions.py
```

### `setup_validation.py` + `regression.py` — the integration regression net
`setup_validation.py` builds a tiny synthetic dataset + a `optimizer=CONCORD` fine-tune
config; `regression.py` runs a real default-path (graph-off) fine-tune and asserts the
Stage 1/2 invariants the Stage-3 trainer cuts must not break: (A) UNet swaps to Concord,
(B) no NaN, (C) deploy consolidation fires, (D) the saved checkpoint is standard SDXL
(no `packed_w`), (E) sanitize zeroed the configured words and left controls intact.

```
set CONCORD_TEST_MODEL=C:\path\to\your_sdxl.safetensors
venv/Scripts/python.exe modules/util/optimizer/concord/tests/setup_validation.py
venv/Scripts/python.exe modules/util/optimizer/concord/tests/regression.py
```

For a graph-on (Stage 3 v2) run, set `concord_cuda_graph: true` in the generated config
and re-run `scripts/train.py` against it.
