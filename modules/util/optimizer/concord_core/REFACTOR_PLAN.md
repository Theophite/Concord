# Concord Optimizer Refactor — Execution Playbook

> **Durable spec.** This file is self-contained so a future session (possibly after
> context loss) can execute the reorg end-to-end. It encodes the APPROVED plan, the
> approved decisions, the full symbol→module line map, the golden gate, the two
> mandatory stress-fixes, the risks, and the deferred second-pass items. Treat it as
> the source of truth; re-read it before touching code.

---

## 0. Hard constraints (read first, every session)

- **A LIVE GPU TRAINING RUN MAY BE ACTIVE.** While it is up:
  - DO NOT touch the live `concord/` package or `concord_ot.py`.
  - DO NOT run code that imports the Concord internals (a Triton compile can crash the run).
  - DO NOT use the GPU. DO NOT run `git`.
  - L0 / L1 / L2 gates are CPU-only and SAFE while live **only** with `CUDA_VISIBLE_DEVICES=""`.
  - L3 (GPU) gates and the one-time baseline capture run **only when the run is DOWN**.
- **Allowed write root for the SETUP task that produced this file:**
  `C:\fisher\OneTrainer-clean\modules\util\optimizer\concord_core\` — new files only.
- The actual code moves (STEP 2+) happen **later, on a dedicated branch, golden-gated**.
  This playbook + the empty `__init__.py` + the 7 skeleton modules are the setup deliverable;
  no real Concord code has been moved or copied yet.

---

## 1. Approved decisions (settled — bake these in; do not re-litigate)

These are the architect's answers to the open questions O1–O8. They are FINAL.

| # | Decision |
|---|----------|
| **O1 — folder** | Target = a NEW sibling folder `concord_core/` (sibling of `concord/`, **NOT** inside it). Rationale: the live run loads `concord/` via `sys.path` injection and a second module identity; building the clean tree OUTSIDE that dir = zero risk to the live process until cut-over, and lets the golden gate diff old-vs-new side by side. |
| **O2 — shim permanence** | `concord/prototype_packed_b.py` becomes a **PERMANENT thin shim** pointing at `concord_core`. Smallest blast radius: the bare-name import, the dual-module identity, and the `sys.path` injections stay untouched. (Full replacement of `concord/` is NOT done this pass.) |
| **O3 — kernel-string doc contract** | Resolution **(A)**: repoint `test_doc_kernel.py`'s `PPB_SRC` at the file that actually holds the kernel text. This is a doc-contract edit (explicitly in scope) — the **grepped strings stay byte-identical** (no variable renames this pass; the strings ARE the contract). |
| **O4 — branch** | A **dedicated reorg branch off `concord-integration`**. Work happens there later. (You do NOT git anything during setup.) |
| **O5 — config defaults** | Config defaults stay **AST-evaluable literals** (do not break `test_doc_config.py`, whose AST eval has only the free name `Optimizer`). `CONCORD_DEFAULTS` is NOT a programmatic projection — it stays a spelled-out literal; add a CPU test asserting `literal == projection`. (See Stress-fix note / Hole 6.) |
| **O6 — Tier-B threading** | Option **(a)** this pass: keep the six mis-wired knobs as TOP-LEVEL `TrainConfig` fields and copy them onto `config.optimizer` (or pass as explicit `ConcordController` kwargs) before `make_concord_config`; add matching `pick()` lines. No config-schema migration / GUI churn. Option (b) is deferred (→ D-list). |
| **O7 — dissipation default** | `ConcordConfig.dissipation` may flip `None → 0.025` to align the test-only `configure_optimizer` picker with the live controller — **after** grepping the tests for `.dissipation is None`. If a test asserts `None`, gate the change behind the live path. |
| **O8 — scope of concord_core** | This pass moves **ONLY the `prototype_packed_b` internals** (constants / state / kernels / layers / coherence / servo / _smoke) **+ `config_defaults.py`**. `concord_winner.py`, `concord_ot.py`, and the embedding modules **stay put** (already separate files; moving them multiplies path/test churn). |

**Scope statement:** behavior-preserving reorg + config consolidation. **Semantic cleanup is DEFERRED** (see §8, D1–D14).

---

## 2. Target tree

```
C:\fisher\OneTrainer-clean\modules\util\optimizer\concord_core\
  __init__.py        # EMPTY (0 bytes). Namespace-only — mirrors concord/__init__.py (0 bytes).
                     # A NON-EMPTY __init__ would import-execute torch/triton on
                     # `import concord_core`, BREAKING the lazy-import contract. KEEP IT EMPTY.
  constants.py       # MANTISSA_BIAS, INT8/16 bounds, S/V_SLOW_FACTOR, compute_drift_cancel_C. Zero deps.
  state.py           # ALL module-global config scalars + setters + device-tensor caches + meters. (LAST move; highest risk.)
  kernels.py         # all @triton.jit kernels + launch wrappers + pack/unpack + _denom_diagnostic.
  layers.py          # FusedConcordLinearPackedB, ConcordLinearPackedB, FusedConcordConv2dPackedB, ConcordConv2dPackedB.
  coherence.py       # gate_coherence_from_fields, measure_coherence.
  servo.py           # DissipationAutoTuner, EpochDissipationServo.
  _smoke.py          # _run_one/diagnose/smoke_test/main + mlp factories, under `if __name__=="__main__"`.
  config_defaults.py # NEW (STEP 10): the canonical ConcordConfig dataclass; WINNER & CONCORD_DEFAULTS as projections.
  tests/goldens/     # NEW: baseline captures G1..G7 + source SHA + torch/triton/CUDA versions.
```

The **shim** lives at the OLD path `concord/prototype_packed_b.py` (NOT in this tree). It:
- does `import sys, pathlib; sys.path.insert(0, <concord_core dir>)` then
  `from constants import *; from state import *; from kernels import *; from layers import *; from coherence import *; from servo import *`
- explicitly re-exports every private name external code touches (the `_GLOBAL` scalars,
  `_v_bc_buf`, `_lamb_*_buf`, `_get_step_counter`, etc.).
- for LIVE-MUTABLE scalar globals, **MUST NOT snapshot** — uses **PEP 562 module `__getattr__`**
  delegating to `state` so `prototype_packed_b._COH_KAPPA` always reads state's current value.
- **STRESS-FIX SF1 exception:** `_FUSED_MATMUL` is direct-attr-assigned, so `__getattr__` is
  insufficient — see §5.
- **Stays named `prototype_packed_b.py`** so the dual-identity `rsplit('.',1)[-1]=='prototype_packed_b'`
  filter (StableDiffusionXLFineTuneSetup.py:147) still matches (R2).

---

## 3. Full symbol → module line-range map

Source: `C:\fisher\OneTrainer-clean\modules\util\optimizer\concord\prototype_packed_b.py`
(**4176 lines** — re-verified 2026-06-29; was 4150 when the map below was first written).
All line numbers below are in that file ("PB"). **No code has been moved yet** — these are
the targets for the golden-gated moves.

> **⚠️ DRIFT RECONCILIATION (2026-06-29, run-UP read-only pass).** The file grew **+26
> lines** (4150 → 4176) from the M6a / boil-buffer / servo-ceiling edits. Every range in
> this map was **re-derived from the current source** on 2026-06-29 (anchors are exact
> `def`/`class`/global starts from grep; range *ends* are bounded by the next anchor and
> are marked "re-verify exact end before moving" per the fail-closed rule). Material
> semantic drift folded in below and into §5/§6/§7:
> - **Boil meter is now 6-wide** (was 4): `torch.zeros(6,…)` at PB:1475 (shared) and
>   PB:3757 (per-layer `_boil_meter`). `[0..3]` = boil/waste; **`[4]`,`[5]` = the new M6a
>   diversity meter (num, denom)**.
> - **`boil_ptr[3]` is now weighted by `coh_evap`** (PB:993 — the clamped `min(coh,
>   coh_raw+evap_slack)`, matching the actuator), not raw `coh`. Meter/actuator mismatch
>   fix.
> - **M6a meter (PB:994–996)** is **LOG-ONLY, bit-irrelevant to the weight update**: `[4]`
>   = killed coherent mass in the hypothesis-infancy band (`|s_fast| < evap_build_min`),
>   `[5]` = consolidated `s_slow` energy; M6a = `[4]/[5]`. Written by the adamw kernel;
>   **read by `concord_ot.py` (not by `read_boil`/`read_layer_boil`)** — see the new
>   cross-module reader note under INV-B.
> - **Servo cf-ceiling gate REMOVED for `protected_boil`** (PB:3870–3877): protected mode
>   now regulates on `waste_ceiling` + `memgap` only; the **non-protected/legacy path keeps
>   the raw-boil ceiling, BIT-IDENTICAL**. Changes the G3a servo trajectory baseline for
>   `protected_boil=True` — capture from current HEAD.
> - **`TrainConfig.concord_m6a_meter`** added (TrainConfig.py:531, default **False**) —
>   gates the log-only M6a readout in `GenericTrainer`. New §6 mapping row below.

### constants.py
```
PB:45-49     MANTISSA_BIAS, INT8_MIN/MAX, INT16_MIN/MAX, S_SLOW_FACTOR, V_SLOW_FACTOR  (start UNCHANGED)
PB:52-107    compute_drift_cancel_C   (start UNCHANGED @52; ends before @triton.jit @109)
```

### state.py  (move LAST — STEP 6)  *(all ranges re-derived 2026-06-29; re-verify exact ends before moving)*
```
PB:183-213   _FUSED_MATMUL env flag (@183), _FUSED_SCRATCH (@188), _get_fused_scratch (@191)  (+ NEW set_fused_matmul, SF1)
PB:469-484   _STEP_COUNTERS (@469), _get_step_counter (@472)         (raw str(device) key — KEEP)
PB:483-508   _CONSOLIDATE_FLAGS (@483), _dev_key (@486), _get_consolidate_flag (@495), set_consolidate (@502)  (_dev_key key — KEEP asymmetry)
PB:510-527   _LR_SCALAR_CACHE (@510), _ensure_lr_tensor (@513)
PB:529-545   _EPS_SCALAR_CACHE (@529), _ensure_eps_tensor (@532)
PB:547-564   _NAMED_SCALAR_CACHE (@547), _ensure_named_scalar (@550)
PB:1286-1356 _USE_FIXED_COH (@1286)/set_fixed_coh (NOTE: set_fixed_coh moved to @1602, in the gate block below),
             _USE_COH_VHAT (@1295)/set_coh_vhat (@1297), _COH_KAPPA (@1296)/set_coh_kappa (@1306),
             _EVAP_SLACK (@1311)/set_evap_slack (@1312), _MIN_LEAK (@1327)/set_min_leak (@1348),
             _EVAP_BUILD_MIN (@1340)/set_evap_build_min (@1343)
PB:1373-1440 LAMB: _LAMB_TRUST (@1373)/_LAMB_CAP (@1374)/_LAMB_CLIP (@1375),
             _LAMB_SCALE_CACHE (@1376)/_WNORM_SQ (@1377)/_STEPNORM_SQ (@1378),
             set_lamb_trust (@1381), _lamb_scale_buf (@1393)/_lamb_wnorm_sq_buf (@1407)/_lamb_stepnorm_sq_buf (@1419)
PB:1442-1557 METERS (⚠ NOW 6-WIDE): _MEMGAP_BUFS (@1442)/_BOIL_BUFS (@1468), _boil_buf (@1471)/read_boil (@1480)/
             _memgap_buf (@1493)/read_memgap (@1502); _PERLAYER_METERS (@1523), register_layer_meters (@1526)/
             _lookup_layer_meters (@1530)/clear_layer_meters (@1534)/read_layer_boil (@1538)/read_layer_memgap (@1550).
             ⚠ `_boil_buf` allocates `torch.zeros(6)` (@1475); `[0..3]`=boil/waste, `[4],[5]`=M6a num/denom.
             `read_boil` returns only `[0:3]` (a,b,c); `read_layer_boil` returns `[0:4]` (a,b,c,d incl. coh_evap-weighted
             protected boil). M6a `[4],[5]` are read NOT here but in concord_ot.py (see INV-B note). Preserve the
             6-wide layout + index semantics as a kernel↔concord_ot CONTRACT.
PB:1560-1618 _GATE_GAIN (@1560)/set_gate_gain (@1607), _SIGMAG_NOISE (@1566)/_SIGMAG_SIGMA (@1567)/_SIGMAG_ISO (@1568)/
             _SIGMAG_SIGMA_T (@1572)/set_sigmag_noise (@1575)/set_sigmag_sigma (@1581)/_get_sigmag_sigma (@1591),
             set_fixed_coh (@1602)
PB:1616-1722 _GAP_FEEDBACK (@1616)/_GAP_SCALE (@1617)/set_gap_feedback (@1620), _COH_WEIGHTED_V (@1640)/set_coh_weighted_v (@1643),
             _RATIO_COH (@1654)/set_ratio_coh (@1667)/set_ratio_coh_floors (@1690)/_ensure_floor_tensors (@1703)
PB:1676-1687 _LAZY_GATE (@1676)/_LAZY_THRESH (@1677)/set_lazy_gate (@1680)/set_lazy_thresh (@1685)  (INTERLEAVED inside 1616-1722 — pointer)
PB:1724-1792 _BIAS_CORRECT_V (@1724)/_V_BC_BUFS (@1726)/_VHAT_MEAN_BUFS (@1738), _v_bc_buf (@1729)/_vhat_mean_buf (@1739)/
             set_bias_correct_v (@1751)/set_v_bias_correction (@1756)/bias_correction_factor (@1763);
             _GRADW_KEYS (@1776)/_GRADW_DIAG (@1777)/read_gradw_diag (@1782)
PB:2145-2172 _REB_STATS (@2145)/reset_reb_stats (@2148)/get_reb_stats (@2154), _REB_SEED_CACHE (@2158)/_ensure_reb_seed_tensor (@2161)
```

### kernels.py  (STEP 7)  *(all ranges re-derived 2026-06-29; re-verify exact ends before moving)*
```
PB:109-117   _hash_uniform (@110)
PB:120-189   _materialize_packed_bf16_kernel (@121) + materialize_packed_bf16 (@158)
PB:215-324   _FUSED_AUTOTUNE_CONFIGS, _fused_packed_linear_kernel (@216)/_fused_packed_gradx_kernel (@256)
             + fused_packed_linear (@291)/fused_packed_gradx (@309)
PB:326-470   _apply_packed_sgd_kernel (@327)   (ends before _get_step_counter @472)
PB:566-624   apply_packed_sgd (@566)
PB:626-643   _lamb_scale_kernel (@627)
PB:645-1196  _apply_packed_adamw_kernel (@646)  THE monster. ⚠ doc-asserted kernel strings are matched by CONTENT
             (`assert "<text>" in PPB_SRC` / `PPB_SRC.index`), NOT by line number (test_doc_kernel.py) — so line
             drift does NOT break them; KEEP the strings byte-identical (no renames this pass). Note the M6a /
             coh_evap boil writes now live at PB:961-996 inside this kernel (log-only [4]/[5]; coh_evap [3]).
PB:1198-1284 _DENOM_DIAG (@1198), _denom_diagnostic (@1203)   (gated enabled=False — D6: move with gate intact; ends before _USE_FIXED_COH @1286)
PB:1794-2002 apply_packed_adamw  (launcher @1794; reads S._COH_KAPPA/_EVAP_SLACK/_MIN_LEAK/_EVAP_BUILD_MIN/
             _GATE_GAIN/_LAZY_THRESH (@1956)/_GAP_*/_USE_FIXED_COH/_USE_COH_VHAT/_RATIO_COH/_GAP_FEEDBACK/
             _LAZY_GATE (@1991)/_LAMB_TRUST via the LIVE state binding; also S._GRADW_DIAG; CALLS _denom_diagnostic)
PB:2004-2143, 2174-2240   _rebalance_packed_decide_kernel (@2005) + rebalance_packed (@2174)  (reads S._REB_STATS — Hole 3;
             NOTE _REB_STATS/seed STATE globals @2145-2172 are interleaved between the kernel and its wrapper)
```

### layers.py  (STEP 8)  *(all ranges re-derived 2026-06-29; re-verify exact ends before moving)*
```
PB:2242-2470 FusedConcordLinearPackedB (@2242)
PB:2471-3200 ConcordLinearPackedB (@2471)    (class consts MANTISSA_BIAS/EXP_MIN/EXP_MAX/MAX_M ~@2488-2491 — re-verify; KEEP; concord_embedding_packed.py reads them)
PB:3201-3408 FusedConcordConv2dPackedB (@3201)
PB:3409-3472 ConcordConv2dPackedB (@3409)   (ends before gate_coherence_from_fields @3474)
```
**layers.py live module-global reads (SF2 / Hole 2 — these are BARE globals today; the refactor must
re-route them through `import state as S; S._X`, NOT kernels-only):**
`_SIGMAG_NOISE`(2347), `_SIGMAG_ISO`(2350), `_LAZY_GATE`(2365), `_LAZY_THRESH`(2372),
`_COH_WEIGHTED_V`(2391,3078,3320), `_FUSED_MATMUL`(2265,2266,2971,2978,3232,3233).
Noise draw: `torch.randn_like(gwf)` @PB:2353 (isotropic), `torch.randn(M,…)` @PB:2357 (Sigma_g eps)
— layer forward, uncovered by G1; see SF2. (The `_SIGMAG_NOISE` gate is a python bool @2347, a
control-flow branch baked under CUDA-graph capture — comment at PB:2342.)

### coherence.py  (STEP 4)  *(re-derived 2026-06-29)*
```
PB:3474-3500 gate_coherence_from_fields (@3474), measure_coherence (@3488)   (ends before class DissipationAutoTuner @3502)
```

### servo.py  (STEP 5)  *(re-derived 2026-06-29)*
```
PB:3502-3660 DissipationAutoTuner (@3502)
PB:3661-3977 EpochDissipationServo (@3661)   ⚠ cf-ceiling gate REMOVED for protected_boil @3870-3877 (2026-06-29):
             `boil_ok = True if self.protected_boil else (gate_boil < self.boil_ceiling)`. Protected mode regulates on
             waste_ceiling + memgap only; legacy (non-protected) path keeps the raw-boil ceiling BIT-IDENTICAL. The
             per-layer `_boil_meter` it allocates is now `torch.zeros(6)` (@3757). Changes the G3a baseline for
             protected_boil=True — capture from current HEAD.
```

### _smoke.py  (STEP 3)  *(re-derived 2026-06-29)*
```
PB:3978-4176 _run_one (@3978)/_packed_b_mlp* (@4014,4022)/_torch_mlp* (@4037,4048)/diagnose (@4058)/smoke_test (@4085)/main (@4166)
             (+ ADD `if __name__ == "__main__": main()` — the monolith tail still ends with a bare top-level
                sys.exit(1) and has NO guard; drop now-unused top-level import sys/import time IF only smoke used them — VERIFY first)
```

### External re-export names (the shim AND concord_winner MUST expose — verified consumers)
```
set_consolidate, register_layer_meters, compute_drift_cancel_C, read_boil, read_memgap,
read_gradw_diag, materialize_packed_bf16, ConcordLinearPackedB, ConcordConv2dPackedB,
DissipationAutoTuner, EpochDissipationServo,
set_bias_correct_v / _v_bc_buf / bias_correction_factor / set_v_bias_correction,
_lamb_scale_kernel / _lamb_wnorm_sq_buf / _lamb_stepnorm_sq_buf / _lamb_scale_buf /
    _LAMB_TRUST / _LAMB_CAP / _LAMB_CLIP,
_get_step_counter,
INT16_MIN / INT16_MAX / S_SLOW_FACTOR / V_SLOW_FACTOR / MANTISSA_BIAS,
setter family (re-exported via concord_winner): set_ratio_coh / set_ratio_coh_floors /
    set_fixed_coh / set_lazy_gate / set_lazy_thresh / set_min_leak / set_evap_build_min /
    set_lamb_trust / set_coh_vhat / set_coh_kappa / set_evap_slack / set_sigmag_noise / set_sigmag_sigma,
launch-mutable scalars read by concord_ot health/active_config:
    _FUSED_MATMUL, _EVAP_BUILD_MIN, _USE_COH_VHAT, _COH_KAPPA,
NEW (SF1): set_fused_matmul.
concord_embedding_packed.py additionally needs INT16_MIN/MAX/S_SLOW_FACTOR/V_SLOW_FACTOR
   + ConcordLinearPackedB's class consts.
```

---

## 4. Migration order (each step golden-gated; lowest coupling first; state.py LAST)

> Each step is independently committable, golden-gated, and leaves the live import
> surface byte-identical via the shim. **L0→L1→L2 after every commit; L3 batched
> before each milestone in a GPU window with the run DOWN. FAIL-CLOSED: host gates at
> atol=0, GPU gates near-bit-exact within `gpu_envelope` × margin (see §7 ⚠) — any
> exceedance → revert/fix before proceeding.**

- [x] **STEP 0 — Baseline capture (GPU window, run DOWN).** *(DONE 2026-06-29, SHA 6ecdafd…-dirty,
      torch 2.9.1+cu128 / triton 3.5.1 / RTX 4090.)* Captured G1 (32-case matrix + the per-field
      self-jitter envelope `gpu_envelope.pt`), G2 (4 sub-trajectories), G3a/G3b (CPU, exact),
      G3c (live 6-wide boil/M6a), G5 (config). **Self-equivalence compare PASS across all 41
      cases** (host atol=0; GPU near-bit-exact within envelope; weight-field dev=0 in practice).
      Gate model revised — see the §7 ⚠ block. **STILL TODO: G4 (swap), G6 (deploy/serial), G7
      (doc-constants)** were specced but not yet captured this pass. Re-baseline if the toolchain moves.
- [x] **STEP 1 — Scaffold `concord_core/`** with EMPTY `__init__.py` + skeleton modules.
      `concord/prototype_packed_b.py` NOT touched. Gate: L0 import smoke only.  *(done in setup task)*
- [ ] **STEP 2 — constants.py:** move PB:45-102; in the monolith replace those defs with
      `from constants import *`. Gate L0 + L1 + L2 (constants/compute_drift_cancel_C are
      doc-tested BY VALUE → stay green).
- [ ] **STEP 3 — _smoke.py:** move PB:3948-4150, add `__main__` guard; drop unused top-level
      `import sys`/`import time` IF only smoke used them (VERIFY first). Gate L0 + L1.
- [ ] **STEP 4 — coherence.py:** move PB:3465-3490 (leaf). Gate L0 + L1 (test_coherence_cpu).
- [ ] **STEP 5 — servo.py:** move PB:3493-3945. `import state`, `from coherence import measure_coherence`.
      Gate L0 + L1 (test_servo_cpu, test_autotuner_cpu — trajectory-snapshot diff at atol=0).
- [ ] **STEP 6 — state.py (HIGHEST RISK — snapshot hazard):** move ALL globals/setters/caches/meters.
      The monolith now `from state import *` AND re-exports live-mutable scalars via module `__getattr__`.
      Apply SF1 (`set_fused_matmul`) here. Gate: L0 (re-export chain: every setter resolves in BOTH
      `prototype_packed_b` and `concord_winner`; `set_coh_vhat(True)` → `prototype_packed_b._USE_COH_VHAT==True`)
      + L0' (live-module variant, SF/Hole-5) + L1 (full CPU suite) + targeted L3 micro-gate
      (`set_evap_slack(0.25)+set_coh_vhat(True)+set_coh_kappa(1.0)` then one apply, hash packed_w vs baseline).
- [ ] **STEP 7 — kernels.py (BIT-EXACTNESS VERDICT):** move all `@triton.jit` + wrappers + `_denom_diagnostic`.
      `import state as S`, `from constants import *`. Gate L0 + L1 + the FULL L3 GPU gate (G1 matrix + G2 trajectory at atol=0).
- [ ] **STEP 8 — layers.py:** move the 4 classes. `import state as S`, `from kernels import *`, `from constants import *`.
      Apply SF2 (S.-live reads in layers). Gate L0 + L1 (G4-CPU, test_doc_deploy/format) + L3 (G4-construct + G2 re-run).
- [ ] **STEP 9 — DOC SYNC (O3):** repoint `test_doc_kernel.py`'s `PPB_SRC` (currently
      `concord/prototype_packed_b.py:116-117`) at the file now holding the kernel text; regenerate
      CONCORD.md `:line` citations from the new tree. Grepped strings stay byte-identical.
      Gate L2 (doc tests green) + human review of the CONCORD.md diff.
- [ ] **STEP 10 — Config consolidation (separate sub-sequence; only after 1–9 green):** see §6.
      Each knob-threading change is its own commit, gated at L1 (G5 dict identical) + shipped-default L3 (G2 identical).
- [ ] **STEP 11 — Cut-over (O2 = shim-permanent):** keep `concord/prototype_packed_b.py` permanently
      as the shim pointing at `concord_core`. (Full replacement of `concord/` is option (b) — NOT this pass.)

**Parallelism:** STEP 2 (constants) and STEP 3 (_smoke) are fully independent — do first in any order.
After STEP 6 (state) lands, STEP 4 (coherence) and STEP 5 (servo) are independent leaves (either order /
concurrent). STEP 7 (kernels) then STEP 8 (layers) MUST be sequential (layers imports kernels). The config
sub-sequence (STEP 10) is independent of the reorg once the tree is stable.

---

## 5. The TWO MANDATORY STRESS-FIXES (the adversarial pass found these — must be done)

> These were the blocking findings. The plan's R1 mitigation ("read via `S.X` +
> delegate via shim `__getattr__`") is INCOMPLETE; the golden gate has a
> shipped-config-sized hole. Both MUST be in place before / during the moves.

### SF1 — `_FUSED_MATMUL` is mutated by DIRECT ATTRIBUTE ASSIGNMENT (not a setter)

- **Why the standard mitigation FAILS:** `__getattr__` fires only for MISSING attributes.
  When external code writes `ppb._FUSED_MATMUL = True` it creates a REAL attribute on the
  shim that **shadows** `__getattr__`. Meanwhile the production write loop sets the attr on
  every module named `prototype_packed_b` but **never on `state`**, so `layers.py` reading
  `S._FUSED_MATMUL` stays at its import-time env default → fused-vs-cached **silently inverts**.
  The setter-based L0 gate (`set_X(v)→assert ppb._X==v`) NEVER exercises the attr-assign path.
- **Verified writers (confirmed via grep):**
  - Production: `modules/modelSetup/StableDiffusionXLFineTuneSetup.py:146-149` — the loop
    `for _m in sys.modules.values(): if name endswith 'prototype_packed_b': _m._FUSED_MATMUL = want_fused`
    (plus the `os.environ["CONCORD_FUSED_MATMUL"]` mirror at :142-145).
    *(NOTE: the plan text cited `optimizer/SDXLFineTuneSetup.py:146-149`; the verified real path
    is `modelSetup/StableDiffusionXLFineTuneSetup.py`, write at line 149.)*
  - Tests: `concord/tests/test_autotuner_cpu.py:450` (`ppb._FUSED_MATMUL = True`) and `:459` (restore).
- **Reads after the split (6 sites in layers.py, re-derived 2026-06-29):** PB:2265,2266,2971,2978,3232,3233;
  plus `concord_ot.py:1214,1236` (was 1209,1231). Writer loop/filter re-confirmed:
  `StableDiffusionXLFineTuneSetup.py` filter `rsplit('.',1)[-1]=='prototype_packed_b'` @147, write `_m._FUSED_MATMUL=want_fused` @149, env mirror @143/145.
- **FIX (do this):**
  1. Add a real `set_fused_matmul(bool)` setter in `state.py` — the single source of truth
     (writes the one true `state._FUSED_MATMUL`). L0-gate it.
  2. Route ALL writers through it: change the SDXL setup loop and the autotuner test to call
     `set_fused_matmul(...)` (the loop must hit `state`, not just modules named `prototype_packed_b`).
  3. EITHER make the shim's `__setattr__` write THROUGH to `state` (so legacy `ppb._FUSED_MATMUL = v`
     still works AND state sees it) OR repoint the `sys.modules` write loop directly at `state`.
  4. Add an L0 gate variant: direct attr-assign on the shim → assert BOTH `layers` AND `state`
     observe the new value.
- **Treat any other direct-attr-assigned / rebound global the same way:** `_REB_STATS` (REBOUND
  via `reset_reb_stats`, PB:2139-2142) — read as `S._REB_STATS` in kernels.py, never
  `from state import _REB_STATS`. `_GRADW_DIAG` is rebind-safe (in-place mutation) but is still a
  cross-module read in the apply launcher — enumerate it as `S._GRADW_DIAG`.

### SF2 — the shipped config runs NOISE ON; bit-exact goldens can't reproduce it as-specified

- **Why:** `swap_unet_to_winner` unconditionally calls `set_sigmag_noise(True, isotropic=True)`
  (`concord_winner.py:298` — re-verify line), so the shipped path hits `torch.randn_like(gwf)` at
  **PB:2353** (isotropic) / `torch.randn(M,…)` at **PB:2357** (Sigma_g eps) EVERY step **in the layer
  forward** (not the kernel). G1's kernel-branch matrix cannot reach the layer-forward noise math, and
  G1 has NO `{noise}`/`{sigmag_iso}`/`{lazy_gate}` axis. So as written, G2 (the "integration proof")
  must run NOISE-OFF — i.e. **NOT the shipped config** — leaving the entire noise / Sigma_g / lazy-noise
  branch (the layers reads of `_SIGMAG_NOISE`@2347/`_SIGMAG_ISO`@2350/`_LAZY_GATE`@2365/`_LAZY_THRESH`@2372
  and the `torch.randn` DRAW ORDER at PB:2353/2357) with ZERO bit-exact coverage.
- **FIX (do this):** the golden harness must EITHER
  (a) **capture torch's global RNG state** before each G2 step and assert the draw order/sequence
      (so the noise-ON shipped config is actually golden-gated), OR
  (b) run a documented **noise-OFF variant** AND add a **separate draw-order check** that exercises
      `torch.randn` at PB:2353/2357 + the `_SIGMAG_*`/`_LAZY_*` reads.
  Also add a `{noise on/off, sigmag_iso, lazy_gate}` axis so the layer-forward noise path is covered.
  Document that the reorg does not touch the noise math, so noise-OFF equivalence is a sufficient
  reorg proof — but the noise path itself stays unverified unless (a) or (b) covers it.

---

## 6. Config consolidation (STEP 10 — behavior-preserving; gate every change)

**Today:** FOUR config layers with documented disagreements — WINNER dict
(`concord_winner.py:47-59`) + `ConcordConfig` dataclass (`concord_winner.py:64-188`) +
`CONCORD_DEFAULTS` (`optimizer_util.py:430-482`) + getattr-fallbacks in `ConcordController`
(`concord_ot.py`).

**Goal:** one canonical schema + defaults; WINNER and CONCORD_DEFAULTS become PROJECTIONS of it;
the six mis-wired Tier-B knobs become live; one hidden constant promoted. **No shipped numeric/bool
default changes.**

**NEW module `concord_core/config_defaults.py`:** ONE dataclass `ConcordConfig` (canonical schema),
grouped for readability, flat at runtime. Derive `WINNER` = recipe-key projection and
`CONCORD_DEFAULTS` = panel-key projection so `swap_unet_to_winner` and `optimizer_util` read the SAME
source.

**Doc-test constraints (Hole 6 / O5 — do NOT make CONCORD_DEFAULTS a computed expression):**
`test_doc_config.py` loads `ConcordConfig` via `from concord_winner import ConcordConfig` and AST-parses
`optimizer_util.OPTIMIZER_DEFAULT_PARAMETERS` with ONLY `{'Optimizer': Optimizer}` in scope.
Therefore:
- `ConcordConfig` MUST stay importable from `concord_winner` (re-export it from `config_defaults`).
- `OPTIMIZER_DEFAULT_PARAMETERS` MUST stay a top-level AST-evaluable dict literal; keep the CONCORD
  block a spelled-out literal (values sourced from the dataclass but written explicitly), and add a CPU
  test asserting `literal == projection`. **Drop the "single source" framing for the panel defaults**:
  the literal remains the source the doc-test reads.

**OLD → NEW mapping (all defaults UNCHANGED):**
```
WINNER dict literal (concord_winner.py:47-59)             -> recipe projection of ConcordConfig() (values identical)
CONCORD_DEFAULTS literal (optimizer_util.py:430-482)      -> panel projection, kept AST-evaluable literal (values identical)
ConcordConfig.dissipation None (concord_winner)           -> KEEP None (O7 RESOLVED 2026-06-29: test_autotuner_cpu.py:128 ASSERTS `cfg_default.dissipation is None`; the None-vs-0.025 split is INTENTIONAL -- None = fall back to engine gf_consol for the test-only picker, 0.025 = live panel default; controller branches on `dissipation is not None` @concord_ot:145. Flipping breaks the test AND changes picker behavior -> NOT behavior-preserving. Deferred.)
TrainConfig.concord_evap_slack  (LIVE field @TrainConfig:530/1151, read concord_ot:243) -> ConcordConfig.evap_slack via pick(); 0.25 unchanged
TrainConfig.concord_train_cond_embed (LIVE @526/1147, read :158)  -> ConcordConfig.train_cond_embed; False unchanged
TrainConfig.concord_conv_full_vhat (LIVE @527/1148, read :159)    -> ConcordConfig.conv_full_vhat; False unchanged
TrainConfig.concord_servo_protected_boil (LIVE @528/1149, read :322,326) -> ConcordConfig.servo_protected_boil; True unchanged
TrainConfig.concord_servo_waste_ceiling (LIVE @529/1150, read :327) -> ConcordConfig.servo_waste_ceiling; 0.12 unchanged
autotune_servo_per_epoch (hidden const; concord_ot.py:314 getattr default 3) -> ConcordConfig.autotune_servo_per_epoch; 3 unchanged

### ⚠ 2026-06-29 CORRECTION — the six knobs are NOT "DEAD" (they are LIVE TrainConfig fields, mis-wired)
The earlier "DEAD" label was wrong: all six are registered TrainConfig fields (TrainConfig.py:526-530 +
data-tuples 1147-1153) and are READ at runtime via `getattr(self.config, …)` in concord_ot. They are
**mis-wired** (bypass `ConcordConfig`), not dead -> CONSOLIDATE (route through the canonical schema),
do NOT delete. CONCORD_DEFAULTS path is `modules/util/optimizer_util.py:430-482` (NOT `optimizer/...`).

### PURGE (decision: consolidate + purge the PROVABLY-DEAD, 2026-06-29) — provably-dead set is SMALL
- **`concord_servo_protected_boil_ceiling` — ✅ PURGED (2026-06-29, in C:\fisher\concord-reorg-work).**
  Removed the TrainConfig decl + data-tuple and the stale "dissipation-aggressiveness dial" comment;
  collapsed concord_ot._mk_servo to `boil_ceiling=float(getattr(self.config,"autotune_boil_ceiling",0.05))`.
  Gated: py_compile OK; no dangling refs; CPU goldens (G5/G3a/G3b) PASS; `TrainConfig.default_values()`
  drops the attr AND `from_dict` of an OLD config still carrying the key does NOT crash (saved-config compat
  verified). Behavior bit-identical (the protected servo ignored the value; servo off by default). Evidence below:
- **`concord_servo_protected_boil_ceiling` — (was) PROVEN DEAD.** TrainConfig default **0.60**
  (TrainConfig.py:1153), but the concord_ot getattr fallback says **0.50** (concord_ot.py:321) — a live
  3-way disagreement (plan also said 0.50). It is read ONLY at concord_ot:321 to set the servo's
  `boil_ceiling` when `protected_boil=True`; but the 2026-06-29 servo edit made the protected path
  `boil_ok=True` (PB:3877) -> `self.boil_ceiling` is UNREAD in protected mode; when protected_boil=False it
  isn't passed at all; and `autotune_servo` defaults False (optimizer_util:479) so the servo isn't even built
  by default. Dead in every branch. FIX: delete the TrainConfig field (decl + data-tuple :1153) and collapse
  concord_ot:321-323 to `boil_ceiling=float(getattr(self.config,"autotune_boil_ceiling",0.05))` (the protected
  servo ignores it; behavior bit-identical). COUPLING NOTE: only dead while the protected path has no
  cf-ceiling — re-adding one resurrects the field. Gate: G5 dict + shipped-default G2 (no value changes).
- **NOT auto-purgeable (need a separate semantic decision — do NOT delete this pass):**
  - autotune **table vs servo** redundancy (D5): `autotune_table` is POPULATED by default + `autotune_servo`
    False -> the TABLE path is DEFAULT-ACTIVE. Deleting it CHANGES behavior. Needs your call on which
    mechanism supersedes; until then, KEEP both.
  - ConcordConfig vestigial fields (D11): per-field unreachability proof required before any deletion.
  - `momentum: 0.9` (CONCORD_DEFAULTS:431) is the AUX-SGD momentum (not a Concord-core knob) — relocate/label,
    do not delete (it is read by the aux optimizer).
concord_m6a_meter (NEW 2026-06-29; TrainConfig.py:531, default False; data-tuple :1152) -> STAYS a top-level TrainConfig
    field read DIRECTLY by GenericTrainer.py:1216-1220 (gates the log-only M6a readout). It does NOT thread into
    ConcordConfig / the optimizer (M6a is bit-irrelevant). Documented here so the consolidation does NOT absorb it
    into the dataclass projection; default False unchanged.
```

**Threading the six Tier-B knobs:** option (a) (O6) — keep them as TOP-LEVEL `TrainConfig` fields;
`StableDiffusionXLFineTuneSetup` copies them onto `config.optimizer` (or passes explicit
`ConcordController` kwargs) BEFORE `make_concord_config`; add matching `pick()` lines. No schema
migration, no GUI/doc churn.

**Invariants the consolidation MUST preserve (golden-gated; do NOT touch this pass):**
- `gf_consol = lam / max(lr, 1e-12)` (concord_ot.py:147) + the `lam < 2` stability guard.
- The D3 guard: `step_cap <= 0 AND gf_trust_delta_sq <= 0 -> restore gf_trust_delta_sq = 1.0`
  (concord_ot.py:135-138). *(Note: this "D3 guard" is the config invariant; distinct from the
  deferred item D3 in §8.)*
- The setter call ORDER + VALUES at controller init (`set_coh_kappa/_evap_slack/_min_leak/_lazy_*/
  _lamb_*` are read once at kernel launch — same order, same values).
- `gf_consol` stays the engine fallback when dissipation cleared; `ratio_coh`/`gf_consol` stay OFF the
  panel (optimizer_util.py:425-427).
- `test_doc_config.py` + CONCORD.md Section 10 are a contract: regenerate the doc's config table from
  the single source and update the test's expected defaults/line refs in lockstep.

**Gate:** L1 (G5 dict identical) + shipped-default L3 (G2 trajectory identical). (R10 — new dataclass
defaults must EQUAL current hardcoded fallbacks, else behavior changes silently.)

---

## 7. Golden gate spec (L0–L3 levels + G1–G7 captures)

> **⚠️ GATE MODEL REVISED 2026-06-29 — the kernel is NOT bit-reproducible.** STEP 0 capture
> measured (RTX 4090, torch 2.9.1+cu128, triton 3.5.1) that the apply kernel diverges
> ~70/2048 ints per step from bit-identical inputs+state, **eager AND under CUDA-graph
> replay** (graph replay does NOT rescue it). This contradicts CLAUDE.md's "bit-deterministic
> at fixed seed" (see memory `concord-kernel-nondeterminism`). Mechanism: a HW-nondeterministic
> float reduction (the kernel's only nondeterministic primitive is `tl.atomic_add`, used for
> the diagnostic meters); SR faithfully rounds the differing value. **So the GPU gates cannot be
> atol=0.** Resolution (validated, self-equivalence PASS across 41 cases):
> - **Decompose `packed_w` into its fields** — gating the raw int32 is meaningless (`s_fast` is
>   the high 16 bits; a 1-LSB velocity wobble moves the int32 by 65536).
> - **GPU gates are NEAR-bit-exact:** refactored-vs-golden ≤ `gpu_envelope.pt`[field] × margin
>   (default 2). The measured per-field self-jitter (G1, 4 draws): **`s_fast`≈4594 LSB** (the
>   velocity, DROPPED at deploy — chaotic), **`s_slow`≤14 / `v_slow`≤2 / `weight_buf`≈0.045**
>   (the DEPLOY weight — near-exact, matching CLAUDE.md "the metric is the deployed weight"),
>   meters large (log-only). The auto-calibrated per-field envelope tolerates each field to its
>   own natural jitter, so a real divergence (systematic, large) still fails the gate.
> - **HOST gates stay atol=0 exact:** G5 (config dict), G3a (servo), G3b (autotuner) — verified
>   bit-reproducible.
> - **Note:** a fresh-PROCESS first call reproduces EXACTLY (the compare run hit dev=0 on all
>   weight fields); the nondeterminism is in-process REPEAT-call jitter (reused scratch/autotune,
>   not the math). The envelope is therefore a conservative ceiling, not the typical deviation.

**Capture once** (pre-refactor HEAD, GPU window, run DOWN) → `concord_core/tests/goldens/` with source
SHA + torch/triton/CUDA versions. Re-baseline if the toolchain moves (Triton codegen is version-sensitive).

### Captures (G1–G7)
- **G1 KERNEL-OUTPUT (GPU, atol=0):** fixed `N=32,K=64` `ConcordLinearPackedB`, `torch.manual_seed(0)`,
  `load_weights(fixed W)`, FIXED grad sequence via `apply_grad_step`; snapshot the FULL int32 `packed_w`
  + bf16 `weight_buf` + `row_exp/col_exp` + `v_row/v_col/_sum_v_inv` **+ the 6-wide boil buffer
  (`[0..3]` boil/waste incl. the coh_evap-weighted `[3]`, `[4]/[5]` M6a num/denom — register a per-layer
  meter so they populate)** after EACH step. **Config MATRIX**
  exercising every constexpr branch: `{gf_consol 0 vs >0}`, `{coh_vhat on/off + coh_kappa}`,
  `{ratio_coh on/off}`, `{evap_slack 0 vs 0.25}`, `{evap_build_min 0 vs 128}`, `{min_leak}`,
  `{lamb_trust on/off}`, `{use_full_v Conv}`, `{grad_activity emb}`, `{bias_correct_v}`,
  `{step_cap<=0 no-clamp}`, `{consolidate flag 0 vs 1}`. Determinism: `step_salt == _get_step_counter`
  (single shared, +1/apply) + fixed XOR salts.
  **GAP TO FIX (Hole 5 / SF2):** add a `{noise on/off, sigmag_iso, lazy_gate}` axis — the noise math is
  in the layer forward, NOT the kernel, so G1 as specified cannot reach it.
  **GAP TO FIX (M6a, 2026-06-29):** add an `{m6a_meter}` config (register a per-layer 6-wide `_boil_meter`)
  and make `{evap_build_min 0 vs 128}` cross the infancy band so `[4]/[5]` are non-trivial. The M6a writes
  (PB:994-996) are LOG-ONLY/bit-irrelevant, so they will NOT show in `packed_w`; goldening the boil buffer is
  the ONLY way the reorg proves the M6a + coh_evap-`[3]` kernel writes were moved intact. Snapshot the raw
  6-vector directly (`read_boil` exposes only `[0:3]`, `read_layer_boil` only `[0:4]`).
  **GAP TO FIX (INV-C):** G1 is keyed on ONE device string (`'cuda'`); add a `cuda:0`/`cuda` step so
  the `_STEP_COUNTERS` (raw str) vs `_CONSOLIDATE_FLAGS` (`_dev_key` normalized) keying asymmetry is
  actually exercised (else a unification regression slips through).
- **G2 SHIPPED-DEFAULT TRAJECTORY (GPU, atol=0):** build the EXACT shipped config via
  `make_concord_config(lr, None) -> ConcordController` over a tiny synthetic UNet (few Linear + one
  Conv2d): dissipation 0.025 → `gf_consol = lam/lr`, coh_vhat True, coh_kappa 1.0, evap_slack 0.25,
  min_leak 0.1, ratio_coh True, **noise True**. Drive N steps; snapshot every layer's `packed_w` +
  device tensors + meters per step. **SF2: noise-ON requires fixing torch's global RNG state + draw
  order, OR a documented noise-OFF variant + a separate draw-order check.**
- **G3 SERVO/AUTOTUNER/METERS:** (a) `servo._kappa/_step/_last_dir/_agg_boil` trajectory over a
  scripted boil/memgap schedule (CPU, exact float). **⚠ capture from current HEAD — the cf-ceiling gate
  removal (PB:3870-3877) changed the protected_boil=True trajectory; include BOTH protected_boil True
  (new regulation: waste_ceiling+memgap only) and False (legacy raw-boil ceiling, bit-identical) schedules.**
  (b) autotuner commit/re-probe events (CPU). (c) live **6-wide** boil buffer (`[2]`,`[3]` coh_evap-weighted,
  `[4]/[5]` M6a) + memgap after a dissipating burst that crosses the `evap_build_min` infancy band so M6a is
  non-zero (GPU). Snapshot the raw 6-vector; also snapshot `concord_ot`'s derived `_last_m6a = (cur[4]-p)/(cur[5]-p)`
  (concord_ot.py:817,825) since that host re-derivation is the actual consumer.
- **G4 SWAP:** swapped-vs-frozen-vs-skipped module NAME set + per-layer config
  (`alpha/alpha_v_fast/gf_consol/precond_p/v_scale/gf_trust_delta_sq/wd_anchor/drift_cancel_C/_concord_name`)
  + `active_config()` flags (CPU for the plan; GPU for construction). Then
  `load_weights`/`load_weights_anchor` `packed_w` bit-exact + `consolidated_weight`/`get_weight`
  (CPU via the `object.__new__` stand-in from `test_doc_deploy.py`).
- **G5 CONFIG RESOLUTION (CPU, exact dict):** `make_concord_config(lr, cfg) -> ConcordConfig` as a dict
  for `cfg=None` and a GUI stub; the `gf_consol = lam/lr` conversion; the D3 guard;
  `winner_step` lr/sigma/cos_floor schedule over the horizon. The config-consolidation gate.
- **G6 DEPLOY/SERIAL:** `consolidated_state_dict`/`consolidate_into_unet` byte-identical (deploy drops
  `s_fast`); materialize/restore round-trip `packed_w` bit-exact. (CPU math; GPU only for the non-fused
  `materialize_packed_bf16` path.)
- **G7 DOC CONTRACT (CPU):** snapshot doc-asserted constants/defaults — MANTISSA_BIAS=15,
  S/V_SLOW_FACTOR=128, _MIN_LEAK=0.1, _EVAP_BUILD_MIN=128.0, dissipation 0.025, gf_consol 50, all
  `TrainConfig.concord_*` defaults; the refactor must not silently move a default.

### Levels (run L0→L1→L2 every commit; L3 batched pre-milestone)
- **L0 IMPORT + RE-EXPORT (CPU, instant, every commit):** `py_compile` every moved file; smoke
  `import prototype_packed_b; import concord_winner; import concord_ot` from the concord dir; ASSERT
  every setter in `concord_winner.py:36-42` AND `concord_ot.py:124-127` resolves in BOTH namespaces;
  ASSERT `set_X(v)`-then-read of `prototype_packed_b._X == v` (the snapshot-hazard catcher — the single
  most important automated check for STEP 6/7).
  **PLUS L0' (SF/Hole-5 — the LIVE-MODULE variant):** assert the module that HOLDS each read
  (`layers._X`, `kernels._X`) observes the setter/attr change, not just the shim namespace. A
  `from state import _X` snapshot inside layers/kernels passes the shim re-export check yet runs stale.
  **PLUS the SF1 attr-assign variant:** direct attr-assign on the shim → assert BOTH `layers` AND
  `state` observe it.
- **L1 CPU GOLDENS (CPU, seconds, every commit, SAFE WHILE LIVE):** G5/G3a/G3b/G7/G4-CPU/G6-CPU diff at
  atol=0 + the existing CPU suite (test_doc_config/deploy/format/embeddings/persistence/inspection,
  test_servo_cpu, test_autotuner_cpu, test_coherence_cpu, test_controller_wiring_cpu,
  test_controlplane_caption_cpu, test_packing_alt_cpu, test_dualdis_*CPU, test_dither_*CPU) all green.
- **L2 DOC-TESTS (CPU, on any default/constant/line move):** `test_doc_*.py` green; CONCORD.md `:line`
  diff reviewed.
- **L3 GPU GOLDENS (GPU window, run DOWN, batched pre-milestone):** G1 matrix + G2 + G3c + G4-construct
  + G6-materialize diff at atol=0. The bit-exactness verdict.

**Cycle:** L0→L1→L2 every commit; L3 batched (several CPU-gated commits, then one GPU gate before merge).

---

## 8. Risks

| ID | Risk | Mitigation |
|----|------|-----------|
| **R1 (HIGHEST)** | MODULE-GLOBAL SNAPSHOT HAZARD: `from state import _COH_KAPPA` captures the import-time value; a later setter rebinds state but the importer keeps the stale value → silent bit drift, invisible to a smoke import. | kernels.py reads launch-baked scalars ONLY as `import state as S; S._COH_KAPPA`. Shim exposes live scalars via PEP-562 `__getattr__`. L0 gate `set_X(v)→assert ppb._X==v` for every launch-baked scalar. Confirmed live reads PB:1947-1986 + backward PB:2382/3069/3311. **Extended by SF1/SF2 + INV-A taxonomy.** |
| **R2** | DUAL-MODULE-IDENTITY DESYNC: `prototype_packed_b` is loaded as TWO objects (package-qualified vs bare during swap); per-object launch globals desync; setup loops `sys.modules` filtering `rsplit('.',1)[-1]=='prototype_packed_b'` (SDXLFineTuneSetup line 147). | Keep the shim FILE NAMED `prototype_packed_b.py` so the rsplit filter matches; do NOT introduce NEW launch-baked module-global state that can desync across the two identities (concentrate all such state in state.py, single object; shim delegates). If cut-over (b) is ever chosen, repoint the rsplit filter + env-var sync. Collapsing the dual identity is DEFERRED (D13). |
| **R3** | DOC-TEST SOURCE-STRING GREP BREAKS: `test_doc_kernel.py:116-173` greps `prototype_packed_b.py` text for kernel strings + order; moving the kernel body removes them → test fails. | STEP 9 repoints `PPB_SRC` to the file holding the kernel, in lockstep with the move; grepped strings stay byte-identical (no variable renames this pass). Doc-contract edit, explicitly in scope. |
| **R4** | RE-EXPORT CHAIN BREAK: `concord_ot` imports setters FROM `concord_winner` (re-exported FROM `prototype_packed_b`); a setter missing from EITHER list fails `concord_ot.py:124`. | L0 dual-namespace resolution assert every commit; shim does `from <core> import *` + explicit re-export of the full setter list; `concord_winner.py:36-42` unchanged. |
| **R5** | DEVICE-TENSOR CACHE IDENTITY: `_LR/_EPS/_NAMED` caches, `_CONSOLIDATE_FLAGS`, `_STEP_COUNTERS`, `_REB_SEED_CACHE`, LAMB caches, `_RATIO_*_FLOOR_T`, `_SIGMAG_SIGMA_T`, `_V_BC/_VHAT_MEAN` bufs must each be ONE singleton per (name,device); duplicating a cache dict → kernel reads a different buffer than the setter fills → CUDA-graph desync / stale pointer. | ALL caches live solely in state.py; kernels/layers reach them only via state functions. Preserve the `_dev_key` vs raw `str(device)` keying ASYMMETRY (consolidate flag normalizes cuda↔cuda:0; step counter does not) — do NOT unify (PB:472 vs 486-499). |
| **R6** | SR-DETERMINISM: splitting SGD/AdamW into modules with separate step counters would desync the salt. | ONE `_STEP_COUNTERS` in state.py shared by both `apply_packed_sgd` and `apply_packed_adamw`; XOR salt constants + draw order + BLOCK_N=32/BLOCK_K=64 untouched. G1 hash catches drift. |

> The adversarial pass also enumerated R7–R10 (kept here so nothing is lost):
> **R7** Noise-on golden non-reproducibility → **see SF2** (capture torch RNG or noise-off variant).
> **R8** Fused ≠ cached bit-for-bit → golden the cached/shipped path; golden fused separately only if
> `CONCORD_FUSED_MATMUL` set; never assume cross-equivalence.
> **R9** Live-run disturbance → separate folder, no edits to `concord/` until cut-over; all L3 GPU gates
> only when the run is DOWN; L0/L1/L2 CPU + `CUDA_VISIBLE_DEVICES=""`.
> **R10** Config-consolidation silent activation → new dataclass defaults EQUAL current fallbacks
> (evap_slack 0.25, servo_* 0.50/0.12/True, train_cond_embed/conv_full_vhat False, per_epoch 3);
> G5 dict + G2 trajectory gate it.

**Stress-pass invariants (INV-A/B/C) — fold into the gate:**
- **INV-A:** classify every moved global by mutation MECHANISM (Class 1 setter / Class 2 direct-attr /
  Class 3 rebound), not "launch-baked scalar" — the protection differs per class (drives SF1).
- **INV-B:** ≥5 coherence reconstructions must stay numerically locked; the FIFTH lives outside the
  split (`concord_ot.py:_metrics ~1104-1136`, hard-codes `-15.0`, `*128.0`, `0.03 * vh.mean()`,
  reads `_ppb._COH_KAPPA/_USE_COH_VHAT/_EVAP_BUILD_MIN` via the shim) — uncovered by any golden; do NOT
  move/touch the constants without auditing it (→ D1).
- **INV-C:** the `_STEP_COUNTERS` vs `_CONSOLIDATE_FLAGS` keying asymmetry is SR-determinism-load-bearing;
  G1 must exercise a `cuda:0`/`cuda` mismatch (single-device golden won't catch a unification).
- **INV-D (NEW 2026-06-29):** the **6-wide boil-buffer LAYOUT is a kernel↔host CONTRACT**. The adamw kernel
  WRITES `[0..3]` (boil/waste; `[3]` coh_evap-weighted @PB:993) + `[4]/[5]` (M6a num/denom @PB:994-996);
  host readers index it at THREE arities — `read_boil`→`[0:3]`, `read_layer_boil`→`[0:4]`, and `concord_ot`'s
  `read_flow_audit` reads `cur[4]/cur[5]` directly (concord_ot.py:817,825) off a 6-wide scratch (:223).
  Moving the kernel MUST preserve the 6-element width AND the index semantics: a silent 4-wide / off-by-one
  regression corrupts M6a + protected-boil **without touching `packed_w`** → invisible to G1's weight hash;
  **G3c (raw 6-vector snapshot) is the only catcher.** `concord_ot.py`/`GenericTrainer.py` are OUT of the
  split (O8) but are now meter-layout consumers — do not change the layout this pass.

---

## 9. Deferred second-pass items (DOCUMENT now; do NOT do in this behavior-preserving reorg)

Each requires proving unreachability under shipped `ConcordConfig` defaults AND clearing the doc/test
contract before any deletion.

- **D1 — Multiple coherence definitions (headline defer):** unify `coh_raw` (un-cf-discounted Wiener,
  drives evap/dissipation, PB:805-867,956-962) vs cf-discounted `coh` (drives chase/leak/boil[3]) vs
  `_COH_WEIGHTED_V` (PB:1631 default False; read 3× backward at PB:2382/3069/3311) vs USE_RATIO_COH
  non-FIXED branch (PB:862-865) vs `measure_coherence`/`gate_coherence_from_fields` (host) vs the FIFTH
  host re-derivation in `concord_ot:_metrics` (INV-B). Intentional today (evap on `coh_raw` is the
  anti-lock-in friction floor; the 2026-06-22 cf/beta1 NaN guard PB:1011-1016 gates beta1 momentum on
  `coh_raw`). Unifying risks reintroducing the divergence → NaN. DEFER.
- **D2 — `_COH_WEIGHTED_V`** (False, `set_coh_weighted_v`, 3 read sites): architect-flagged dead path.
  Prove inert, then remove with its setter + reads.
- **D3 — `_GAP_FEEDBACK`/`_GAP_SCALE`** (False; USE_GAP_FEEDBACK branches PB:1607,1927,1963): alternate
  dissipation split, off by default. *(Distinct from the config-invariant "D3 guard" in §6.)*
- **D4 — `_LAZY_GATE`/`_LAZY_THRESH`** (False for UNet): embedding-relevant; keep for embeddings, prune
  only the UNet-dead branch after proving.
- **D5 — DissipationAutoTuner (PB:3493-3649):** doc says EpochDissipationServo supersedes it, but the
  shipped default has a populated table AND `autotune_servo=False` → the TABLE path is default-active.
  Resolve the table-vs-servo redundancy (which supersedes which is ambiguous today). NOT a clean removal.
- **D6 — `_denom_diagnostic` + `_DENOM_DIAG` (PB:1183-1266):** gated `enabled=False`, no setter,
  unreachable except by hand-edit, BUT CALLED at PB:1913 inside the launcher (a host sync if enabled).
  This pass MOVES it to kernels.py with the gate INTACT (do not delete yet).
- **(extra, kept for completeness)** **D7** `_GRADW_DIAG` eager block (PB:1832-1864) + `read_gradw_diag`:
  env-gated (CONCORD_GRADW_DIAG), live importer concord_ot.py:1067 — KEEP. **D8** SGD path
  (optimizer_kind=='sgd'): ablation fallback, KEEP. **D9** ALLOW_TICKDOWN rebalance branch: future
  experiment. **D10** `configure_optimizer`/`make_aux_optimizer`/`active_config`/`consolidated_state_dict`
  (concord_winner): test-only picker, candidate to fold once confirmed only tests call it. **D11** config
  triplication remnants / vestigial ConcordConfig fields (alpha/alpha_v_fast ARE load-bearing; the rest
  vestigial). **D12** extra package modules (concord_embedding*, token_init, dual_dissipation_*ref,
  dither_accum_ref, run_ablation, train_cifar_cf, experimental/*) — OUT of the prototype_packed_b split;
  keep as regression refs; second pass decides whether embedding modules move into concord_core. **D13**
  collapse the dual module identity (R2 root cause; large blast radius). **D14** CONCORD.md `:line`
  prose-citation refresh beyond what L2 enforces.

### D-extra — `coh_kappa` "cf-over-claim" calibration (the architect's open second-pass question)

The architect flagged a calibration question about `coh_kappa` and the **cf-discount over-claim**: the
cf-discounted coherence may over-credit (over-claim) coherence relative to `coh_raw`, and `coh_kappa`'s
calibration vs the cf-discount is not pinned down. **This is a SEMANTIC question, NOT a reorg change —
defer it to the second pass.** It interacts with D1 (multiple coherence definitions) and the live
telemetry in INV-B (`concord_ot:_metrics` hard-codes the 3% v_hat floor and the bias/scale constants).
Action for the second pass: characterize whether the cf-discount path over-claims coherence under shipped
defaults, and whether `coh_kappa` needs recalibration once D1's definitions are reconciled — with a golden
that snapshots BOTH `coh_raw` and cf-discounted `coh` so the over-claim is measurable. Do NOT touch
`coh_kappa` or the cf-discount in the behavior-preserving pass.

---

## 10. Open-question ANSWERS (the approved decisions — quick index)

O1 → `concord_core/` sibling. O2 → shim PERMANENT. O3 → (A) repoint `PPB_SRC`, strings byte-identical.
O4 → dedicated reorg branch off `concord-integration`. O5 → CONCORD_DEFAULTS stays a spelled-out literal
(+ `literal == projection` test). O6 → (a) top-level TrainConfig fields copied onto `config.optimizer`.
O7 → flip `dissipation None→0.025` AFTER grepping tests for `.dissipation is None`. O8 →
`concord_core` holds ONLY the prototype_packed_b split + `config_defaults`; concord_winner/concord_ot/
embedding modules stay put.

---

## 11. Setup-task status (what exists now)

Created by the setup task (NEW files only; no Concord code moved):
- `concord_core/__init__.py` — EMPTY (0 bytes).
- `concord_core/constants.py`, `state.py`, `kernels.py`, `layers.py`, `coherence.py`, `servo.py`,
  `_smoke.py` — SKELETONS (docstring + responsibility + exact PB line-range map + cross-module
  live-read / stress-fix notes; NO real code).
- `concord_core/REFACTOR_PLAN.md` — this file.

Verified against the live source (read-only): `prototype_packed_b.py` = **4176 lines** (was 4150);
`concord/__init__.py` = 0 bytes; the SF1 writer exists at
`modules/modelSetup/StableDiffusionXLFineTuneSetup.py:149` (filter @147, env mirror @143/145) — re-confirmed
2026-06-29; `concord/tests/test_autotuner_cpu.py:450,459` (re-verify); `test_doc_kernel.py` `PPB_SRC` at
lines 116-117 with the kernel-string asserts at ~162-173 matched **by content** (`in`/`.index`), not by line.

**2026-06-29 reconcile pass (run-UP, read-only + harness authoring):** all §3 line ranges re-derived from
the 4176-line source; M6a / 6-wide-boil / coh_evap-`[3]` / servo-ceiling-removal / `concord_m6a_meter` drift
folded into §3, §5 (SF1/SF2 line refs), §6 (m6a row), §7 (G1/G3 + INV-D). **No Concord code moved** (the
monolith, `concord/`, configs, and `concord_ot.py` were not touched). The golden harness in `tests/` was
EXTENDED (not run — GPU busy): `_read_meters` now snapshots the raw **6-wide** boil buffer (catches the
`[3]` coh_evap + `[4]/[5]` M6a kernel writes that never appear in `packed_w`); added **G3** —
`capture_g3_servo_cpu` (both protected_boil modes; the cf-ceiling-removal catcher), `capture_g3_autotuner_cpu`,
`capture_g3c_live` (6-wide/M6a GPU burst); added a `--cpu` SAFE-WHILE-LIVE lane (G5 + G3a/G3b, no kernel
launch) to both `golden_capture.py` and `golden_compare.py`, with the G3 names added to the re-export check.
Both files `py_compile`-clean; NOT executed (authored-blind, validate on first run-DOWN pass).

**Next session starts at STEP 0** (baseline capture, GPU window, run DOWN) on the dedicated reorg branch.
```
