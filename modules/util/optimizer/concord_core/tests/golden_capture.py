"""GOLDEN-EQUIVALENCE CAPTURE for the Concord behavior-preserving refactor.

  *** GPU + Triton REQUIRED. RUN ONLY IN A RUN-DOWN GPU WINDOW. ***

This script COMPILES and LAUNCHES the real packed Concord Triton kernels on the
GPU. A live training run shares the device; launching new Triton compiles /
kernels can crash it. Do NOT run this while a training run is active. There is
also a per-process module-global state surface (set_*/_FUSED_MATMUL) that this
script mutates -- another reason it must own the GPU alone.

It captures BIT-EXACT golden snapshots from the CURRENT code at
  modules/util/optimizer/concord/prototype_packed_b.py  (+ concord_winner / concord_ot)
so the upcoming reorg into concord_core/ can be proven behavior-preserving by
golden_compare.py (re-run the same cases, diff at atol=0).

AUTHORED BLIND (the GPU was busy with a live run when this was written). It has
NOT been executed. Treat the FIRST run as the validation run: expect to fix
small API/attr mismatches, NOT to trust the numbers until a clean PASS of a
self-consistency check (capture twice -> identical) is observed. The
self-consistency guard at the end of each capture flags non-determinism.

What it captures (see the approved plan .result.plan.golden_gate + .stress):

  G1  KERNEL-OUTPUT (bit-exact, atol=0): a fixed N=32,K=64 ConcordLinearPackedB,
      torch.manual_seed(0), fixed loaded W, a FIXED grad sequence; the FULL int32
      packed_w + bf16 weight_buf + row_exp/col_exp + v_row/v_col/_sum_v_inv after
      EACH step, across a CONFIG MATRIX that exercises every kernel constexpr
      branch. Determinism comes from step_salt == _get_step_counter (reset per
      case) + the kernel's fixed XOR salts; no torch.randn in this path.

  G2  SHIPPED-DEFAULT TRAJECTORY (bit-exact): the EXACT shipped config via
      make_concord_config(lr, None) -> ConcordController-style setup over a tiny
      synthetic UNet (a few Linear + one Conv2d). Driven N steps via the real
      autograd forward/backward. Snapshots every layer's packed_w + device
      tensors + the meters + servo kappa per step. TWO variants (HOLE-CLOSERS
      below): noise-OFF (fully reproducible) and noise-ON (torch RNG fingerprint
      + draw order captured).

  G5  CONFIG-DICT identity (CPU-checkable): the resolved WINNER / CONCORD_DEFAULTS
      projection / ConcordConfig() dataclass dict, so the config consolidation can
      be proven identity-preserving.

MANDATORY HOLE-CLOSERS (from stress.golden_gaps):

  (a) NOISE RNG. The shipped config runs noise ON: swap_unet_to_winner calls
      set_sigmag_noise(True, isotropic=True), so the layer FORWARD hits
      torch.randn_like(grad_W) EVERY step (prototype_packed_b.py:2344). That draw
      is OUTSIDE the kernel and is invisible to G1 (a kernel-branch matrix). We
      therefore (1) capture torch's global CPU + CUDA RNG state right before the
      noise-ON run, (2) snapshot the actual randn DRAW SEQUENCE (a probe tensor
      drawn from a cloned generator state, so the recorded order is the order the
      run will see), and (3) ALSO capture a noise-OFF variant (set_sigmag_noise
      (False)) that is reproducible without any RNG capture. golden_compare.py
      restores the captured RNG state before the noise-ON re-run.

  (b) _FUSED_MATMUL. This flag is mutated by DIRECT ATTRIBUTE ASSIGNMENT
      (ppb._FUSED_MATMUL = ...), NOT by a setter -- production does it in
      StableDiffusionXLFineTuneSetup.py:146-149, tests in test_autotuner_cpu.py.
      It is read at layer CONSTRUCTION time (_ensure_buffers, PB:2962/2969) and in
      the autograd forward (PB:2256/2257/3223/3224). We capture G1 + G2 goldens
      under BOTH _FUSED_MATMUL True and False, setting the attribute BEFORE
      constructing each layer.
      *** REFACTOR NOTE: the reorg MUST add a real set_fused() setter (or a
      write-through __setattr__ on the shim) AND repoint the sys.modules write
      loop -- a PEP-562 __getattr__ shim CANNOT intercept the attribute WRITE, so
      the value would desync across the dual module identity. This golden makes
      both states observable so a regression is caught. ***

Output: concord_core/tests/goldens/*.pt (+ manifest.json with source git SHA and
torch/triton/CUDA versions -- Triton codegen is version-sensitive, so a golden is
only valid against a matching toolchain; golden_compare.py warns on a mismatch).

Run:
  venv/Scripts/python.exe modules/util/optimizer/concord_core/tests/golden_capture.py
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ── script_imports-style path setup (mirror the concord/tests convention) ──────
# parents[0]=tests, [1]=concord_core, [2]=optimizer, [3]=util, [4]=modules,
# [5]=OneTrainer-clean (the OT root). Imports are BARE (concord dir on sys.path),
# matching concord_winner / concord_ot / the existing tests.
OT = Path(__file__).resolve().parents[5]
_OPT = OT / "modules" / "util" / "optimizer"   # holds concord_ot.py (bare import)
_CONCORD = _OPT / "concord"                     # holds prototype_packed_b / concord_winner
for _p in (str(OT), str(_OPT), str(_CONCORD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

import torch  # noqa: E402

# Capture-time TARGET is always the CURRENT shipped code (the baseline). The
# comparison script parameterizes its import; capture does not -- a baseline is by
# definition the current prototype_packed_b.
import prototype_packed_b as ppb  # noqa: E402
import concord_winner as cw  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================

def _device():
    if not torch.cuda.is_available():
        raise SystemExit(
            "golden_capture requires CUDA (the Concord apply kernel is Triton). "
            "Run in a run-DOWN GPU window with the OneTrainer venv python.")
    return torch.device("cuda")


def _reset_step_counter(device):
    """Reset the shared SR step counter to 0 so step_salt is deterministic per
    case. _get_step_counter keys on str(device) and the apply wrapper does
    .add_(1) BEFORE each launch, so a fresh case starting from 0 reproduces the
    exact salt sequence. (This is the SR-determinism contract -- R6.)"""
    ctr = ppb._get_step_counter(device)
    ctr.zero_()


def _snap_layer(m):
    """Bit-exact snapshot of one packed layer's full mutable state. CPU clones so
    the saved tensor is detached from the live buffer and byte-stable on disk."""
    # DECOMPOSE packed_w into its three fields. Gating the raw int32 is meaningless:
    # s_fast lives in the high 16 bits, so a 1-LSB velocity wobble moves the int32 by
    # 65536 and swamps any per-field tolerance. The near-bit-exact gate must tolerate
    # each field in ITS OWN LSB units -- s_fast (the velocity, DROPPED at deploy) runs
    # loose; the DEPLOY fields s_slow/v_slow + the bf16 weight_buf run tight (CLAUDE.md:
    # "the metric is the deployed weight"). Bit layout matches measure_coherence (PB:3494).
    _p = m.packed_w.detach().to("cpu")
    snap = {
        "s_fast": (_p >> 16).to(torch.int32).clone(),
        "s_slow": ((_p << 16) >> 24).to(torch.int32).clone(),
        "v_slow": ((_p << 24) >> 24).to(torch.int32).clone(),
        "row_exp": m.row_exp.detach().to("cpu").clone(),
        "col_exp": m.col_exp.detach().to("cpu").clone(),
        "weight_buf": (m._bf16_weight_buf.detach().to("cpu").clone()
                       if getattr(m, "_bf16_weight_buf", None) is not None else None),
        "v_row": m.v_row.detach().to("cpu").clone(),
        "v_col": m.v_col.detach().to("cpu").clone(),
        "sum_v_inv": m._sum_v_inv.detach().to("cpu").clone(),
    }
    # Conv per-element 2nd moment, when present (USE_FULL_V path).
    if getattr(m, "v_full", None) is not None:
        snap["v_full"] = m.v_full.detach().to("cpu").clone()
    return snap


def _read_meters(device):
    """The per-device shared flow audit + memgap, NON-destructively (reset=False)
    so the snapshot does not perturb the running accumulators.

    2026-06-29 DRIFT: the boil buffer is now 6-WIDE (was 4). `read_boil` still
    exposes only `[0:3]` (aligned_kill, total_kill, chase_flow), but the kernel
    also writes `[3]` = the coh_evap-WEIGHTED protected-boil (PB:993) and
    `[4]/[5]` = the M6a diversity meter (num/denom; killed coherent mass in the
    |s_fast|<evap_build_min infancy band / consolidated s_slow energy, PB:994-996).
    M6a is LOG-ONLY / bit-irrelevant to the weight update, so it NEVER shows in
    `packed_w` -- snapshotting the raw 6-vector is the ONLY way the refactor proves
    those kernel writes moved intact (INV-D; the catcher for a silent 4-wide /
    off-by-one regression). `_boil_buf` returns the SAME shared buffer the kernel
    atomic-adds into (allocs zeros(6) on first touch); clone WITHOUT zeroing."""
    a, b, c = ppb.read_boil(device, reset=False)
    g = ppb.read_memgap(device, reset=False)
    boil_raw6 = ppb._boil_buf(device).detach().to("cpu").clone()
    return {"boil_aligned": a, "boil_total": b, "boil_chase": c, "memgap": g,
            "boil_raw6": boil_raw6}


def _toolchain():
    info = {"torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version.split()[0]}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception as e:  # pragma: no cover - diagnostic only
        info["triton"] = f"<unavailable: {e}>"
    if torch.cuda.is_available():
        try:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["torch_cuda_runtime"] = torch.version.cuda
        except Exception:
            pass
    return info


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "-C", str(OT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(OT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception as e:  # pragma: no cover
        return f"<unknown: {e}>"


# ============================================================================
# G1  KERNEL-OUTPUT MATRIX
# ============================================================================
# Every dict below toggles exactly the knobs that flip a kernel constexpr branch
# (or a launch-baked module-global scalar). Each case runs from a freshly
# constructed layer with the SAME loaded W and the SAME fixed grad sequence, so a
# divergence isolates to one branch. We use apply_grad_step (the no-autograd
# direct path) so the noise/Sigma_g forward path is NOT involved -- G1 is purely
# the kernel + its scalar inputs. (Noise lives in the layer forward and is
# covered by G2.)

# Module-global flags that bake into the kernel constexprs / launch scalars. Each
# case provides any of these keys; unspecified keys take the documented default.
_G1_GLOBAL_DEFAULTS = dict(
    fixed_coh=True,       # _USE_FIXED_COH  (shipped default True)
    ratio_coh=False,      # _RATIO_COH
    coh_vhat=False,       # _USE_COH_VHAT
    coh_kappa=1.0,        # _COH_KAPPA
    evap_slack=0.0,       # _EVAP_SLACK
    min_leak=0.1,         # _MIN_LEAK
    evap_build_min=128.0,  # _EVAP_BUILD_MIN
    lazy_gate=False,      # _LAZY_GATE
    lazy_thresh=1e-4,     # _LAZY_THRESH
    lamb_trust=False,     # _LAMB_TRUST (WRITE_LAMB_NORMS)
    gap_feedback=False,   # _GAP_FEEDBACK
    coh_weighted_v=False,  # _COH_WEIGHTED_V
    sigmag_noise=False,   # _SIGMAG_NOISE (off for G1; G1 uses apply_grad_step anyway)
    sigmag_iso=False,
)

# Per-layer attributes (set on the constructed module before stepping). These
# pick the layer-owned branches: gf_consol (USE_GF_CONSOLIDATION), step_cap
# (no-clamp), gf_trust_delta_sq (USE_GF_TRUST_REGION), bias_correct_v, etc.
_G1_LAYER_DEFAULTS = dict(
    gf_consol=0.0,
    step_cap=10.0,
    gf_trust_delta_sq=1.0,
    v_scale=0.0,
    precond_p=0.5,
    alpha=0.1,
    alpha_v_fast=0.001,
    weight_decay=0.0,
    eps=1e-10,
    grad_activity=False,
)


def _g1_cases():
    """The config matrix. Each entry: (name, {global overrides}, {layer overrides},
    {extra}). `extra` carries non-knob switches:
        kind="conv"        -> use ConcordConv2dPackedB (exercises USE_FULL_V too)
        use_full_v=True     -> conv per-element v_hat
        consolidate=0       -> set_consolidate(False): tick-only micro-step gate
        bias_correct_v=True -> set_bias_correct_v + drive set_v_bias_correction
    """
    C = []
    # baseline (every default; the rank-1 v-hat AdamW + fixed Wiener coh)
    C.append(("baseline", {}, {}, {}))

    # gf_consol 0 vs >0  (USE_GF_CONSOLIDATION)
    C.append(("gf_consol_50", {}, {"gf_consol": 50.0}, {}))

    # coh_vhat on/off + coh_kappa (the cf-discount residual branch)
    C.append(("coh_vhat_on_k1", {"coh_vhat": True, "coh_kappa": 1.0},
              {"gf_consol": 50.0}, {}))
    C.append(("coh_vhat_on_k0p25", {"coh_vhat": True, "coh_kappa": 0.25},
              {"gf_consol": 50.0}, {}))

    # ratio_coh on/off  (USE_RATIO_COH; the shipped dissipation gate). ratio_coh
    # needs coh_pre dropped (disable_cohpre) -- handled in the runner via extra.
    C.append(("ratio_coh_on", {"ratio_coh": True}, {"gf_consol": 50.0},
              {"disable_cohpre": True}))

    # evap_slack 0 vs 0.25  (EVAP clamp slack)
    C.append(("evap_slack_0p25", {"evap_slack": 0.25, "ratio_coh": True},
              {"gf_consol": 50.0}, {"disable_cohpre": True}))

    # evap_build_min 0 vs 128  (hypothesis-infancy guard)
    C.append(("evap_build_min_0", {"evap_build_min": 0.0, "ratio_coh": True},
              {"gf_consol": 50.0}, {"disable_cohpre": True}))

    # min_leak (servo floor)
    C.append(("min_leak_0p5", {"min_leak": 0.5, "ratio_coh": True},
              {"gf_consol": 50.0}, {"disable_cohpre": True}))

    # lamb_trust on/off  (WRITE_LAMB_NORMS). The host scale is computed by the
    # controller's before_step in production; here it stays 1.0 (first step / off
    # value) so the kernel just accumulates the norm bufs -- the WRITE_LAMB_NORMS
    # branch itself is what we want to exercise bit-exactly.
    C.append(("lamb_trust_on", {"lamb_trust": True}, {}, {}))

    # bias_correct_v  (the v_bc multiply on v_hat)
    C.append(("bias_correct_v", {}, {"gf_consol": 50.0},
              {"bias_correct_v": True}))

    # step_cap <= 0  (no-clamp -> bounded by the trust-region denom). The D3 guard
    # lives in the controller, NOT in apply_packed_adamw, so a layer with
    # step_cap<=0 AND gf_trust_delta_sq>0 is a valid kernel case (step bounded by
    # the v_hat denom; kernel sets step_cap=1e30).
    C.append(("step_cap_off", {}, {"step_cap": 0.0, "gf_trust_delta_sq": 1.0}, {}))

    # consolidate flag 0 vs 1  (gradient-accumulation gate: tick-only micro-step)
    C.append(("consolidate_0", {}, {"gf_consol": 50.0}, {"consolidate": 0}))

    # use_full_v Conv  (USE_FULL_V; per-element v_hat). Also covers the conv apply
    # path + the device-side vhat_mean prep.
    C.append(("conv_baseline", {}, {}, {"kind": "conv"}))
    C.append(("conv_full_v", {"coh_vhat": True}, {"gf_consol": 50.0},
              {"kind": "conv", "use_full_v": True}))

    # grad_activity emb  (USE_GRAD_ACTIVITY sighting-clocked dissipation)
    C.append(("grad_activity", {"ratio_coh": True},
              {"gf_consol": 50.0, "grad_activity": True}, {"disable_cohpre": True}))

    # explicit cuda:0 vs cuda device-string (INV-C: _STEP_COUNTERS keys on raw
    # str(device); _CONSOLIDATE_FLAGS normalizes -- exercise the asymmetry the
    # refactor must preserve). Same math as baseline; only the device string moves.
    C.append(("baseline_cuda0_devstr", {}, {}, {"device_str": "cuda:0"}))

    return C


def _apply_g1_globals(g):
    """Push the per-case global overrides through the SHIPPED setters (the same
    ones the controller uses), so the kernel reads exactly what production bakes.
    Returns nothing; mutates module-global state in prototype_packed_b."""
    full = dict(_G1_GLOBAL_DEFAULTS)
    full.update(g)
    ppb.set_fixed_coh(full["fixed_coh"])
    ppb.set_ratio_coh(full["ratio_coh"])
    ppb.set_coh_vhat(full["coh_vhat"])
    ppb.set_coh_kappa(full["coh_kappa"])
    ppb.set_evap_slack(full["evap_slack"])
    ppb.set_min_leak(full["min_leak"])
    ppb.set_evap_build_min(full["evap_build_min"])
    ppb.set_lazy_gate(full["lazy_gate"])
    ppb.set_lazy_thresh(full["lazy_thresh"])
    ppb.set_lamb_trust(full["lamb_trust"], None, None)
    ppb.set_gap_feedback(full["gap_feedback"])
    ppb.set_coh_weighted_v(full["coh_weighted_v"])
    ppb.set_sigmag_noise(full["sigmag_noise"], isotropic=full["sigmag_iso"])
    # Reset the ratio floors to their canonical bootstrap values for a
    # reproducible per-case start (the schedule is a host concern; G1 fixes them).
    ppb.set_ratio_coh_floors(0.9, 0.999)


def _build_g1_layer(extra, device, N=32, K=64):
    """Construct a fresh layer for a G1 case. _FUSED_MATMUL is read at
    construction (HOLE 1), so it must already be set by the caller. Linear uses
    K=in_features, N=out_features; conv is shaped so in_channels*kh*kw == K."""
    dev = torch.device(extra.get("device_str", str(device)))
    torch.manual_seed(0)
    if extra.get("kind") == "conv":
        # 64 = in_channels*kh*kw with kh=kw=1 -> in_channels=64; out_channels=32.
        m = ppb.ConcordConv2dPackedB(in_channels=K, out_channels=N, kernel_size=1,
                                     stride=1, padding=0, bias=False,
                                     device=dev, alpha=0.1, lr=1e-3)
        if extra.get("use_full_v"):
            m.use_full_v = True
    else:
        m = ppb.ConcordLinearPackedB(K, N, bias=False, device=dev,
                                     alpha=0.1, lr=1e-3)
    return m, dev


def _fixed_W(N, K, device, dtype=torch.float32):
    """A FIXED loaded weight (seed 0) -- the same across all G1 cases."""
    g = torch.Generator(device="cpu").manual_seed(12345)
    W = torch.randn(N, K, generator=g, dtype=torch.float32) * 0.05
    return W.to(device)


def _fixed_grad_seq(N, K, device, n_steps, dtype=torch.bfloat16):
    """A FIXED, DETERMINISTIC grad sequence (no global-RNG dependence). A coherent
    drift component + a per-step deterministic ripple, so the coherence gate, the
    chase, and the leak all see non-trivial signal. Same for every case."""
    g = torch.Generator(device="cpu").manual_seed(777)
    base = torch.randn(N, K, generator=g, dtype=torch.float32)
    seq = []
    for t in range(n_steps):
        # coherent drift + a small deterministic oscillation (reproducible).
        ripple = torch.cos(torch.tensor(float(t) * 0.3)) * 0.1
        gW = (base * (0.5 + ripple)
              + 0.02 * torch.sin(base * (t + 1)))
        seq.append(gW.to(device=device, dtype=dtype))
    return seq


def capture_g1(device, n_steps=5):
    out = {}
    fused_states = [False, True]   # HOLE 1: both attr-assignment states
    for fused in fused_states:
        ppb._FUSED_MATMUL = bool(fused)   # DIRECT ATTRIBUTE ASSIGNMENT (the production write path)
        for (name, gov, lov, extra) in _g1_cases():
            key = f"fused{int(fused)}/{name}"
            N, K = 32, 64
            _apply_g1_globals(gov)
            # bias-correct_v is a global toggle + a per-step driver.
            ppb.set_bias_correct_v(bool(extra.get("bias_correct_v", False)))
            m, dev = _build_g1_layer(extra, device, N, K)
            _reset_step_counter(dev)
            # Load the fixed weight (shipped gap-zero init).
            with torch.no_grad():
                m.load_weights(_fixed_W(N, K, dev))
            if extra.get("disable_cohpre"):
                m.disable_cohpre()
            # Apply per-layer overrides.
            full_lov = dict(_G1_LAYER_DEFAULTS)
            full_lov.update(lov)
            for attr, val in full_lov.items():
                setattr(m, attr, val)
            # consolidate gate (micro-step accumulation): default 1.
            cons = extra.get("consolidate", 1)
            ppb.set_consolidate(dev, bool(cons))
            grads = _fixed_grad_seq(N, K, dev, n_steps)
            steps = []
            # Snapshot the post-load state (step 0 = before any apply).
            steps.append({"layer": _snap_layer(m), "meters": _read_meters(dev)})
            for t, gW in enumerate(grads):
                if extra.get("bias_correct_v"):
                    ppb.set_v_bias_correction(ppb.bias_correction_factor(t))
                m.apply_grad_step(gW)
                torch.cuda.synchronize()
                steps.append({"layer": _snap_layer(m), "meters": _read_meters(dev),
                              "step_counter": int(ppb._get_step_counter(dev).item())})
            out[key] = {
                "name": name,
                "fused": bool(fused),
                "globals": gov,
                "layer_overrides": lov,
                "extra": extra,
                "n_steps": n_steps,
                "steps": steps,
            }
    # Restore the default fused state for any later capture in this process.
    ppb._FUSED_MATMUL = False
    return out


# ============================================================================
# G2  SHIPPED-DEFAULT TRAJECTORY
# ============================================================================
# Build the EXACT shipped config via make_concord_config(lr, None), then mirror
# what ConcordController.__init__ does to the global flags + per-layer recipe via
# swap_unet_to_winner over a tiny synthetic UNet. Drive N real
# forward/backward steps (the autograd path -> exercises the noise forward).

class _TinyUNet(torch.nn.Module):
    """A few Linear + one Conv2d, named so swap_unet_to_winner treats them as
    ordinary trainable layers (no 'time_embedding'/'add_embedding' in the names,
    so none are frozen)."""

    def __init__(self):
        super().__init__()
        self.lin_a = torch.nn.Linear(64, 32, bias=True)
        self.lin_b = torch.nn.Linear(32, 32, bias=False)
        self.conv = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.head = torch.nn.Linear(32, 16, bias=True)

    def forward(self, x_seq, x_img):
        h = torch.relu(self.lin_a(x_seq))
        h = torch.relu(self.lin_b(h))
        y_seq = self.head(h)
        y_img = self.conv(x_img)
        return y_seq, y_img


def _shipped_config():
    """The resolved shipped ConcordConfig (cfg=None path -> all winner defaults)."""
    from concord_ot import make_concord_config
    return make_concord_config(learning_rate=1e-5, optimizer_config=None)


def _apply_shipped_globals(cfg):
    """Mirror ConcordController.__init__'s global-flag setup (concord_ot.py:229-243)
    so a standalone G2 sees the SAME launch-baked state as production. Called AFTER
    swap_unet_to_winner (which itself sets fixed_coh/ratio_coh/sigmag_noise)."""
    cw.set_lazy_gate(cfg.lazy_gate)
    cw.set_lazy_thresh(cfg.lazy_active_thresh)
    cw.set_min_leak(cfg.min_leak)
    cw.set_evap_build_min(cfg.evap_build_min)
    cw.set_lamb_trust(cfg.lamb_trust, cfg.lamb_cap, cfg.lamb_clip)
    cw.set_coh_vhat(cfg.coh_vhat)
    cw.set_coh_kappa(cfg.coh_kappa)
    # evap_slack: the controller reads getattr(config,"concord_evap_slack",0.25);
    # ConcordConfig lacks that field today, so production always gets 0.25.
    cw.set_evap_slack(float(getattr(cfg, "concord_evap_slack", 0.25)))


def _record_rng_state(device):
    """HOLE-CLOSER (a): the noise-ON shipped path draws torch.randn in the layer
    forward. Capture the global CPU + CUDA RNG state so the noise-ON trajectory is
    reproducible, AND record the actual draw SEQUENCE (a probe drawn from a CLONE
    of the current state, so we record exactly what the run is about to consume
    without perturbing it)."""
    cpu_state = torch.get_rng_state().clone()
    cuda_state = (torch.cuda.get_rng_state(device).clone()
                  if torch.cuda.is_available() else None)
    # Draw a probe from a cloned generator state to fingerprint the draw order
    # WITHOUT advancing the live RNG (we restore immediately after).
    saved_cpu = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    probe = torch.randn(64, device=device).to("cpu").clone()
    torch.set_rng_state(saved_cpu)
    if saved_cuda is not None:
        torch.cuda.set_rng_state(saved_cuda, device)
    return {"cpu_rng_state": cpu_state, "cuda_rng_state": cuda_state,
            "draw_probe": probe}


def _drive_g2(cfg, device, n_steps, noise_on, seed=0):
    """Build the tiny UNet, swap to the winner recipe, set the shipped globals,
    drive n_steps. Returns the per-step snapshots + (for noise-on) the RNG
    fingerprint captured right before stepping."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = _TinyUNet().to(device)
    layers = cw.swap_unet_to_winner(net, device, cfg.lr, gf_consol=cfg.gf_consol,
                                    step_cap=cfg.step_cap,
                                    gf_trust_delta_sq=cfg.gf_trust_delta_sq,
                                    verbose=False)
    _apply_shipped_globals(cfg)
    # Noise on/off: swap_unet_to_winner forced it ON; honor the variant here.
    cw.set_sigmag_noise(bool(noise_on), isotropic=cfg.sigmag_iso)

    rng_fp = None
    if noise_on:
        # Seed deterministically THEN fingerprint, so a re-run that re-seeds with
        # the same seed restores the same starting RNG -> the draw order matches.
        torch.manual_seed(seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 1)
        rng_fp = _record_rng_state(device)

    # Deterministic fixed inputs/targets (independent of the global RNG so the
    # ONLY global-RNG consumer is the noise draw).
    gin = torch.Generator(device="cpu").manual_seed(2024)
    x_seq = (torch.randn(4, 64, generator=gin) * 0.1).to(device)
    x_img = (torch.randn(2, 8, 8, 8, generator=gin) * 0.1).to(device)
    t_seq = (torch.randn(4, 16, generator=gin) * 0.1).to(device)
    t_img = (torch.randn(2, 8, 8, 8, generator=gin) * 0.1).to(device)

    names = [n for n, _ in net.named_modules()]  # for the snapshot keys
    layer_names = []
    for n, mod in net.named_modules():
        if mod in layers:
            layer_names.append(n)

    steps = []

    def snap_all(t):
        per = {}
        for n, mod in net.named_modules():
            if mod in layers:
                per[n] = _snap_layer(mod)
        meters = _read_meters(device)
        steps.append({"t": t, "layers": per, "meters": meters,
                      "sigma_now": round(float(ppb._SIGMAG_SIGMA), 8)})

    snap_all(-1)  # pre-step state (post-swap, post-load)
    for t in range(n_steps):
        # Advance the winner schedule like the controller's before_step (lr +
        # sigma + ratio floors). update_globals=True so sigma/floors move.
        cw.winner_step(t, n_steps, layers, peak_lr=cfg.lr, warmup=cfg.warmup,
                       sigmag_peak=cfg.sigmag_peak, lr_min_frac=cfg.lr_min_frac,
                       noise=bool(noise_on), config=cfg, update_globals=True)
        y_seq, y_img = net(x_seq, x_img)
        loss = ((y_seq - t_seq) ** 2).mean() + ((y_img - t_img) ** 2).mean()
        loss.backward()
        # No aux optimizer.step() needed: the Concord layers self-step in
        # backward. Zero any aux grads (bias params) so they don't accumulate.
        net.zero_grad(set_to_none=True)
        for m in layers:
            m.rebalance()
        torch.cuda.synchronize()
        snap_all(t)
    return {"layer_names": layer_names, "steps": steps, "rng_fingerprint": rng_fp,
            "n_steps": n_steps, "seed": seed, "noise_on": bool(noise_on)}


def capture_g2(device, n_steps=6):
    cfg = _shipped_config()
    out = {"config": _config_to_dict(cfg)}
    for fused in (False, True):   # HOLE 1: both fused states for the integration path
        ppb._FUSED_MATMUL = bool(fused)
        out[f"fused{int(fused)}_noise_off"] = _drive_g2(cfg, device, n_steps, noise_on=False)
        out[f"fused{int(fused)}_noise_on"] = _drive_g2(cfg, device, n_steps, noise_on=True)
    ppb._FUSED_MATMUL = False
    return out


# ============================================================================
# G5  CONFIG-DICT IDENTITY
# ============================================================================

def _config_to_dict(cfg):
    """Resolve a ConcordConfig dataclass to a plain dict (sorted keys) for a
    byte-stable JSON snapshot."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(cfg):
        d = asdict(cfg)
    else:
        d = dict(cfg)
    return {k: d[k] for k in sorted(d)}


def capture_g5():
    """Snapshot the three config surfaces so the consolidation can be proven
    identity-preserving on CPU:
      - WINNER dict literal (concord_winner.WINNER)
      - ConcordConfig() defaults (the dataclass)
      - the resolved make_concord_config(lr, None) (the shipped path)
      - active_config() snapshot AFTER the shipped globals are engaged (proves the
        live module switches match the config)."""
    cfg = _shipped_config()
    g5 = {
        "WINNER": {k: cw.WINNER[k] for k in sorted(cw.WINNER)},
        "ConcordConfig_defaults": _config_to_dict(cw.ConcordConfig()),
        "make_concord_config_lr1e-5_None": _config_to_dict(cfg),
        # Doc-contract constants (G7 overlap): the refactor must not move these.
        "constants": {
            "MANTISSA_BIAS": ppb.MANTISSA_BIAS,
            "INT8_MIN": ppb.INT8_MIN, "INT8_MAX": ppb.INT8_MAX,
            "INT16_MIN": ppb.INT16_MIN, "INT16_MAX": ppb.INT16_MAX,
            "S_SLOW_FACTOR": ppb.S_SLOW_FACTOR, "V_SLOW_FACTOR": ppb.V_SLOW_FACTOR,
            "default_MIN_LEAK": _module_default("_MIN_LEAK"),
            "default_EVAP_BUILD_MIN": _module_default("_EVAP_BUILD_MIN"),
        },
        # compute_drift_cancel_C at the shipped rates (a value the doc tests assert).
        "drift_cancel_C_default": ppb.compute_drift_cancel_C(0.1, 0.001, mass_preserve=True),
        "drift_cancel_C_legacy": ppb.compute_drift_cancel_C(0.1, 0.001, mass_preserve=False),
    }
    return g5


def _module_default(name):
    """Read a module-global scalar's CURRENT value (used for the doc-contract
    defaults; these may have been mutated by an earlier capture, so capture_g5 is
    called FIRST in main(), before any setter runs)."""
    return getattr(ppb, name, None)


# ============================================================================
# G3  SERVO / AUTOTUNER / METERS   (added 2026-06-29 for the M6a / 6-wide-boil /
#     servo-ceiling-removal drift -- INV-D + the cf-ceiling-removal catcher)
# ============================================================================
# G3a/G3b are PURE HOST (CPU) -- the EpochDissipationServo and DissipationAutoTuner
# act on host floats read from per-layer meters + measure_coherence(layer), which
# only needs layer.packed_w (int32) + layer.drift_cancel_C. So they run on a CPU
# MOCK layer with NO GPU/Triton kernel launch -> SAFE WHILE A LIVE RUN IS UP
# (the L1 lane; run with CUDA_VISIBLE_DEVICES=""). G3c is the live GPU burst that
# proves the kernel's 6-wide boil + M6a writes -- it runs ONLY in a run-DOWN GPU
# window. AUTHORED BLIND -- the schedules are designed to drive divergent
# protected/non-protected trajectories; validate on first run.


class _MockLayer:
    """Minimal stand-in exposing exactly what EpochDissipationServo /
    DissipationAutoTuner / measure_coherence touch on a real packed layer:
      - packed_w: a CPU int32 tensor (has .device + .data_ptr() for
        register_layer_meters; its bit-fields feed measure_coherence)
      - gf_consol / beta1: plain assignable attrs (real layer uses a property
        backed by a device buffer; the servo only writes them)
      - drift_cancel_C: float read by gate_coherence_from_fields
    The servo installs _boil_meter (6-wide) / _memgap_meter itself in __init__."""

    def __init__(self, drift_cancel_C=1.0, n=64, device="cpu"):
        self.packed_w = torch.zeros(n, dtype=torch.int32, device=device)
        self.gf_consol = 0.0
        self.beta1 = 0.0
        self.drift_cancel_C = float(drift_cancel_C)


def _pack_fields(s_fast, s_slow, v_slow, n=64, device="cpu"):
    """Build a uniform int32 packed_w with the kernel's bit layout
    (measure_coherence reads: s_fast=(p>>16); s_slow=((p<<16)>>24);
    v_slow=((p<<24)>>24)). i.e. bits [31:16]=s_fast(int16), [15:8]=s_slow(int8),
    [7:0]=v_slow(int8). Used to script a DETERMINISTIC coherence for G3b."""
    sf = int(s_fast) & 0xFFFF
    ss = int(s_slow) & 0xFF
    vs = int(v_slow) & 0xFF
    word = (sf << 16) | (ss << 8) | vs
    # interpret as signed int32
    if word >= 0x80000000:
        word -= 0x100000000
    return torch.full((n,), word, dtype=torch.int32, device=device)


def _servo_schedule(t, epoch_steps, n_layers):
    """Per-step scripted (a, b, c, d, memgap_signed) the harness writes into each
    mock layer's meters BEFORE servo.step(t). Designed so the protected_boil and
    non-protected paths DIVERGE (the 2026-06-29 cf-ceiling removal): an epoch with
    HIGH raw boil (a/b > boil_ceiling 0.05) but SHRINKING memgap climbs under
    protected (ceiling gone) yet holds under non-protected (raw-boil ceiling
    blocks the climb). a=aligned_kill, b=total_kill, c=chase_flow,
    d=coh_evap-weighted protected-kill, memgap shrinks over epochs."""
    epoch = t // epoch_steps
    # memgap shrinks 1.0 -> 0.5 -> 0.25 ... (monotone shrinking => climb-permission)
    memgap = 1.0 / (2.0 ** epoch)
    # total_kill nonzero so the empty-meter guard passes.
    b = 1.0
    c = 1.0
    # raw boil a/b: HIGH (0.6) on even epochs, LOW (0.01) on odd -> exercises the
    # boil-ceiling gate difference between modes.
    a = 0.6 * b if (epoch % 2 == 0) else 0.01 * b
    # cf-discounted protected kill d/b kept LOW (0.02 < any ceiling) so protected
    # mode is never the one blocked -- isolates the raw-boil-ceiling effect.
    d = 0.02 * b
    return (a, b, c, d, memgap)


def capture_g3_servo_cpu(n_layers=2, epoch_steps=4, n_epochs=5):
    """G3a: EpochDissipationServo kappa/step/last_dir/agg trajectory over a
    scripted boil/memgap schedule, for BOTH protected_boil True and False (the
    cf-ceiling-removal catcher). CPU-only. Snapshots are indexed by LAYER ORDER
    (servo dicts key on id(m), unstable across processes -> never golden id())."""
    out = {}
    total_t = n_epochs * epoch_steps
    for protected in (False, True):
        layers = [_MockLayer() for _ in range(n_layers)]
        servo = ppb.EpochDissipationServo(
            layers, lr=1e-5, seed_kappa=50.0, epoch_steps=epoch_steps,
            climb_rate=0.5, boil_ceiling=0.05, memgap_rel_floor=0.02,
            beta1_on=0.0, protected_boil=protected, waste_ceiling=1.0,
            verbose=False)
        steps = []
        for t in range(0, total_t + 1):
            # Write the scripted meters into each layer before the servo reads them
            # (read_layer_* zeros after reading, so re-write every step).
            a, b, c, d, mg = _servo_schedule(t, epoch_steps, n_layers)
            for m in layers:
                m._boil_meter[0] = a
                m._boil_meter[1] = b
                m._boil_meter[2] = c
                m._boil_meter[3] = d
                m._boil_meter[4] = 0.1 * b   # M6a num (log-only; carried for shape parity)
                m._boil_meter[5] = 1.0       # M6a denom
                m._memgap_meter[0] = mg
            ret = servo.step(t)
            steps.append({
                "t": t,
                "ret": (None if ret is None else float(ret)),
                "epoch": int(servo._epoch),
                "kappa": [float(servo._kappa[id(m)]) for m in layers],
                "step": [float(servo._step[id(m)]) for m in layers],
                "last_dir": [int(servo._last_dir[id(m)]) for m in layers],
                "committed_median": float(servo.committed),
                "agg_boil": [float(x) for x in servo._agg_boil],
                "agg_memgap": float(servo._agg_memgap),
            })
        ppb.clear_layer_meters()   # don't leak the mock pointers into the registry
        out[f"protected_{int(protected)}"] = {
            "protected_boil": protected, "n_layers": n_layers,
            "epoch_steps": epoch_steps, "n_epochs": n_epochs,
            "kappa_cap": float(servo.kappa_cap),
            "kappa_floor": float(servo.kappa_floor),
            "steps": steps,
        }
    return out


def capture_g3_autotuner_cpu(n_layers=2):
    """G3b: DissipationAutoTuner probe -> commit -> (one) re-probe event trace.
    Drives a DETERMINISTIC measure_coherence by swapping each mock's packed_w
    bit-fields between scripted presets (mid-coh during the probe -> a low-coh
    drop to trigger the watchdog re-probe). CPU-only. Snapshots the committed
    kappa/beta1 + re-probe bookkeeping per step."""
    # mid-coh: s_fast=256, s_slow=1, v_slow=0 -> sig=128,noise=128 -> coh=0.5
    # low-coh: s_fast=256, s_slow=0, v_slow=0 -> sig=0,noise=256   -> coh=0.0
    mid = _pack_fields(256, 1, 0)
    low = _pack_fields(256, 0, 0)
    layers = [_MockLayer(drift_cancel_C=1.0) for _ in range(n_layers)]
    for m in layers:
        m.packed_w = mid.clone()
    table = [(0.9, 0.0), (0.5, 100.0), (0.1, 300.0)]   # coh strictly descending
    tuner = ppb.DissipationAutoTuner(
        layers, probe_start=2, probe_end=6, table=table, probe_kappa=50.0,
        measure_every=2, verbose=False, beta1_on=0.0,
        reprobe_band=0.05, reprobe_beta=0.7, watchdog_min_t=0)
    steps = []
    N = 30
    for t in range(0, N):
        # Drop coherence after the first commit to trigger one watchdog re-probe.
        if t == 14:
            for m in layers:
                m.packed_w = low.clone()
        ret = tuner.step(t)
        steps.append({
            "t": t,
            "ret": (None if ret is None else float(ret)),
            "committed": (None if tuner.committed is None else float(tuner.committed)),
            "committed_beta1": (None if tuner.committed_beta1 is None
                                else float(tuner.committed_beta1)),
            "reprobes": int(tuner.reprobes),
            "probe_start": int(tuner.probe_start),
            "probe_end": int(tuner.probe_end),
            "baseline": (None if tuner._baseline is None else float(tuner._baseline)),
            "n_samples": len(tuner._samples),
            "gf_consol": [float(m.gf_consol) for m in layers],
        })
    return {"steps": steps, "table": table, "n_layers": n_layers}


def capture_g3c_live(device, n_steps=8):
    """G3c (GPU, run-DOWN only): drive a real packed layer with a PER-LAYER 6-wide
    meter through a dissipating burst that crosses the evap_build_min infancy band,
    so the kernel populates boil[3] (coh_evap) + [4]/[5] (M6a). Snapshots the raw
    6-vector via read_layer_boil semantics AND concord_ot's host re-derivation
    `_last_m6a = (cur[4]-prev)/(cur[5]-prev)` -- the actual M6a consumer (INV-D)."""
    N, K = 32, 64
    ppb._FUSED_MATMUL = False
    # Strong dissipation: ratio_coh ON, high gf_consol, evap_build_min crossing.
    _apply_g1_globals({"ratio_coh": True, "evap_slack": 0.25, "evap_build_min": 128.0})
    ppb.set_bias_correct_v(False)
    torch.manual_seed(0)
    m = ppb.ConcordLinearPackedB(K, N, bias=False, device=device, alpha=0.1, lr=1e-3)
    m.disable_cohpre()
    m.gf_consol = 50.0
    _reset_step_counter(device)
    with torch.no_grad():
        m.load_weights(_fixed_W(N, K, device))
    # Register a PER-LAYER 6-wide meter (the servo/concord_ot routing path). Set the
    # ATTRS too (read_layer_boil reads m._boil_meter) AND register the SAME objects
    # in the data_ptr registry (the kernel routes writes there) -- exactly what
    # EpochDissipationServo.__init__ does (PB:3757-3762).
    boil_meter = torch.zeros(6, dtype=torch.float32, device=device)
    memgap_meter = torch.zeros(1, dtype=torch.float32, device=device)
    m._boil_meter = boil_meter
    m._memgap_meter = memgap_meter
    ppb.register_layer_meters(m.packed_w, boil_meter, memgap_meter)
    grads = _fixed_grad_seq(N, K, device, n_steps)
    steps = []
    prev4 = prev5 = 0.0
    for t, gW in enumerate(grads):
        m.apply_grad_step(gW)
        torch.cuda.synchronize()
        raw6 = boil_meter.detach().to("cpu").clone()
        # concord_ot's read_flow_audit derivation (concord_ot.py:817,825), log-only.
        cur4, cur5 = float(raw6[4]), float(raw6[5])
        denom = cur5 - prev5
        last_m6a = ((cur4 - prev4) / denom) if denom > 0 else None
        prev4, prev5 = cur4, cur5
        steps.append({
            "t": t,
            "layer": _snap_layer(m),
            "boil_raw6_perlayer": raw6,
            "read_layer_boil4": list(ppb.read_layer_boil(m, reset=False)),
            "last_m6a": last_m6a,
        })
    ppb.clear_layer_meters()
    return {"n_steps": n_steps, "steps": steps}


# ============================================================================
# Self-consistency guard + driver
# ============================================================================

def _flatten_tensors(obj, prefix=""):
    """Yield (path, tensor) for every torch.Tensor nested in dicts/lists -- used
    by the in-process determinism check."""
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_tensors(v, f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten_tensors(v, f"{prefix}[{i}]")


def _self_consistency(label, fn):
    """Run a capture fn twice and assert byte-identical tensors. Catches
    non-determinism (e.g. an uncaptured RNG draw, autotune nondeterminism) BEFORE
    the golden is trusted. Returns the (first) result either way; prints a loud
    WARN on mismatch rather than aborting, so all goldens still get written for
    inspection."""
    a = fn()
    b = fn()
    ta = dict(_flatten_tensors(a))
    tb = dict(_flatten_tensors(b))
    mismatches = []
    keys = set(ta) | set(tb)
    for k in sorted(keys):
        if k not in ta or k not in tb:
            mismatches.append((k, "missing in one run"))
            continue
        x, y = ta[k], tb[k]
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            mismatches.append((k, f"differ (shape {tuple(x.shape)} dtype {x.dtype})"))
    if mismatches:
        print(f"  [WARN] {label}: NOT bit-reproducible in-process "
              f"({len(mismatches)} field(s) differ). First few:")
        for k, why in mismatches[:8]:
            print(f"          {k}: {why}")
        print("        -> a noise/RNG or autotune nondeterminism leaked; "
              "the golden for this section is UNSAFE for an atol=0 gate.")
    else:
        print(f"  [ok] {label}: bit-reproducible in-process ({len(ta)} tensors).")
    return a


def _capture_g3_cpu():
    """The CPU-safe (L1, run-UP-safe with CUDA_VISIBLE_DEVICES="") G3 captures:
    the servo trajectory (both protected modes) + the autotuner trace. No kernel
    launch, so these are also the determinism-checkable host goldens."""
    print("[golden_capture] G3a servo trajectory (CPU; protected on/off) ...")
    g3_servo = _self_consistency("G3a/servo", capture_g3_servo_cpu)
    print("[golden_capture] G3b autotuner trace (CPU) ...")
    g3_auto = _self_consistency("G3b/autotuner", capture_g3_autotuner_cpu)
    torch.save(g3_servo, GOLDENS_DIR / "g3a_servo_cpu.pt")
    torch.save(g3_auto, GOLDENS_DIR / "g3b_autotuner_cpu.pt")
    return g3_servo, g3_auto


def _flatten_numeric(obj, prefix=""):
    """Like _flatten_tensors but ALSO yields scalar int/float leaves (not bool/None)
    -- so the kernel-jitter envelope covers host-derived diagnostic floats
    (last_m6a, read_boil values) as well as the int32/bf16 weight tensors."""
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_numeric(v, f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten_numeric(v, f"{prefix}[{i}]")


def _field_of(path):
    """The trailing field-NAME of a flattened path, index-stripped:
    '/fused0/baseline/steps[3]/layer/packed_w' -> 'packed_w'. The jitter envelope
    is keyed per field-name (it's a property of the kernel arithmetic per field,
    ~consistent across layers/steps), not per full path (a fresh draw jitters at
    DIFFERENT elements each run, so a per-element envelope would false-fail)."""
    return path.split("/")[-1].split("[")[0]


def _capture_envelope(label, fn, reps=4):
    """Run a GPU-capture fn reps+1 times; return (reference_capture, envelope).
    The kernel is intrinsically nondeterministic per step (HW float-reduction
    order -- NOT bit-exact even under CUDA-graph replay; verified 2026-06-29), so
    the gate is NEAR-bit-exact: 'refactored deviates no more than the baseline
    deviates from ITSELF'. envelope = {field_name -> max |run_i - run_0|} over the
    reps extra draws, the measured self-jitter the compare tolerates (x a margin)."""
    runs = [fn() for _ in range(reps + 1)]
    ref = runs[0]
    ref_map = dict(_flatten_numeric(ref))
    env = {}
    for r in runs[1:]:
        for path, v in _flatten_numeric(r):
            rv = ref_map.get(path)
            if rv is None:
                continue
            if isinstance(v, torch.Tensor):
                if not isinstance(rv, torch.Tensor) or rv.shape != v.shape:
                    continue
                dev = float((v.float() - rv.float()).abs().max())
            else:
                dev = abs(float(v) - float(rv))
            f = _field_of(path)
            env[f] = max(env.get(f, 0.0), dev)
    hot = {k: v for k, v in sorted(env.items(), key=lambda kv: -kv[1]) if v > 0}
    print(f"  [envelope] {label}: self-jitter over {reps} draws, top fields: "
          + (", ".join(f"{k}={v:g}" for k, v in list(hot.items())[:6]) or "ALL BIT-EXACT"))
    return ref, env


def main_cpu():
    """SAFE-WHILE-LIVE lane (L1): CPU-only captures. Run with
    CUDA_VISIBLE_DEVICES="" while a training run owns the GPU. Captures G5 (config
    dict, no setters) + G3a/G3b (host servo/autotuner). Does NOT touch the GPU and
    launches no Triton kernel. The GPU goldens (G1/G2/G3c) still require main()
    in a run-DOWN window."""
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[golden_capture --cpu] writing CPU goldens to: {GOLDENS_DIR}")
    print(f"[golden_capture --cpu] toolchain: {_toolchain()}")
    print("[golden_capture --cpu] G5 config-dict identity ...")
    g5 = capture_g5()
    (GOLDENS_DIR / "g5_config.json").write_text(json.dumps(g5, indent=2, default=str))
    _capture_g3_cpu()
    (GOLDENS_DIR / "manifest_cpu.json").write_text(json.dumps({
        "git_sha": _git_sha(), "toolchain": _toolchain(), "lane": "cpu-only",
        "goldens": {
            "g5_config.json": "G5 config-dict identity + doc-contract constants",
            "g3a_servo_cpu.pt": "G3a EpochDissipationServo trajectory (protected on/off; cf-ceiling-removal catcher)",
            "g3b_autotuner_cpu.pt": "G3b DissipationAutoTuner probe/commit/re-probe trace",
        },
        "notes": ["CPU-only lane: safe to capture while a live run owns the GPU "
                  "(CUDA_VISIBLE_DEVICES=\"\"). GPU goldens come from the full main()."],
    }, indent=2, default=str))
    print("[golden_capture --cpu] DONE.")


def main():
    device = _device()
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[golden_capture] OT root: {OT}")
    print(f"[golden_capture] writing goldens to: {GOLDENS_DIR}")
    tc = _toolchain()
    print(f"[golden_capture] toolchain: {tc}")

    # G5 FIRST -- it reads module-default scalars before any setter mutates them.
    print("[golden_capture] G5 config-dict identity ...")
    g5 = capture_g5()
    (GOLDENS_DIR / "g5_config.json").write_text(json.dumps(g5, indent=2, default=str))

    print("[golden_capture] G1 kernel-output matrix (+ self-jitter envelope) ...")
    # The kernel is NOT bit-reproducible (HW float-reduction order; not even under
    # graph replay). Measure the per-field self-jitter envelope from reps draws and
    # use ref(run 0) as the golden. The compare gate is near-bit-exact within this
    # envelope. This ONE envelope (per field-name) is reused for G1/G2/G3c -- the
    # jitter is a property of the kernel arithmetic per field, ~constant across gates.
    g1, gpu_env = _capture_envelope("G1", lambda: capture_g1(device, n_steps=5), reps=4)
    torch.save(g1, GOLDENS_DIR / "g1_kernel_matrix.pt")
    torch.save(gpu_env, GOLDENS_DIR / "gpu_envelope.pt")

    print("[golden_capture] G2 shipped-default trajectory ...")
    # G2 noise-OFF must be bit-reproducible; noise-ON is NOT (it draws randn) and
    # is only reproducible via the captured RNG state -> we do NOT run the
    # double-capture determinism check on the combined G2 (it would flag noise-on).
    # Instead capture once; golden_compare validates noise-on via the RNG restore.
    g2 = capture_g2(device, n_steps=6)
    torch.save(g2, GOLDENS_DIR / "g2_shipped_trajectory.pt")
    # NOTE: no determinism re-check here -- the kernel is known nondeterministic
    # per step (2026-06-29). G2 is gated near-bit-exact against the shared
    # gpu_envelope; noise-ON additionally restores the captured RNG so the only
    # residual is kernel jitter (within-envelope), not a different noise draw.

    # ---- G3 (servo/autotuner/meters; 2026-06-29 drift catcher) ----
    # G3a/G3b are CPU (determinism-checked); G3c is the live GPU burst that proves
    # the kernel's 6-wide boil + coh_evap-[3] + M6a-[4]/[5] writes (INV-D).
    _capture_g3_cpu()
    print("[golden_capture] G3c live boil/M6a burst (GPU) ...")
    g3c = capture_g3c_live(device, n_steps=8)
    torch.save(g3c, GOLDENS_DIR / "g3c_live_meters.pt")
    ppb._FUSED_MATMUL = False

    manifest = {
        "git_sha": _git_sha(),
        "toolchain": tc,
        "goldens": {
            "g1_kernel_matrix.pt": "G1 kernel-output matrix (fused x {0,1} x config matrix; meters now 6-wide incl M6a)",
            "g2_shipped_trajectory.pt": "G2 shipped-default trajectory (fused x {0,1} x noise {off,on})",
            "g5_config.json": "G5 config-dict identity + doc-contract constants",
            "g3a_servo_cpu.pt": "G3a EpochDissipationServo trajectory (protected on/off; cf-ceiling-removal catcher)",
            "g3b_autotuner_cpu.pt": "G3b DissipationAutoTuner probe/commit/re-probe trace",
            "g3c_live_meters.pt": "G3c live 6-wide boil + coh_evap-[3] + M6a-[4]/[5] burst (INV-D)",
            "gpu_envelope.pt": "Per-field self-jitter envelope (max|run_i-run_0| over 4 draws); the near-bit-exact tolerance",
        },
        "notes": [
            "GATE MODEL (2026-06-29): the apply kernel is NOT bit-reproducible -- a single "
            "step from identical inputs/state diverges ~70/2048 ints, eager AND under CUDA-graph "
            "replay (HW float-reduction order). So GPU gates (G1/G2/G3c) are NEAR-bit-exact: "
            "refactored must stay within gpu_envelope (the baseline's measured self-jitter) x margin. "
            "BIT-EXACT atol=0 only for the HOST gates: G5 (config dict), G3a/G3b (servo/autotuner).",
            "G2 noise_on restores rng_fingerprint (cpu+cuda RNG) so the only residual vs golden is "
            "kernel jitter (within-envelope), not a different noise draw.",
            "_FUSED_MATMUL captured under BOTH True/False via direct attribute "
            "assignment (no setter exists -- the refactor MUST add set_fused()).",
            "G1 includes a cuda:0-device-string case to exercise the "
            "_STEP_COUNTERS vs _CONSOLIDATE_FLAGS keying asymmetry (INV-C).",
            "2026-06-29 DRIFT: boil buffer is 6-wide ([3]=coh_evap-weighted, [4]/[5]=M6a). "
            "_read_meters snapshots the raw 6-vector. G3a covers BOTH protected_boil modes "
            "(servo cf-ceiling REMOVAL PB:3870-3877 changed the protected trajectory; legacy stays bit-identical).",
        ],
    }
    (GOLDENS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[golden_capture] DONE. SHA={manifest['git_sha']}")
    print("[golden_capture] Review the [WARN] lines above before trusting the goldens.")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Concord golden capture")
    _ap.add_argument("--cpu", action="store_true",
                     help="CPU-only lane (G5 + G3a/G3b): SAFE WHILE A LIVE RUN IS "
                          "UP with CUDA_VISIBLE_DEVICES=\"\". Skips all GPU goldens.")
    _args = _ap.parse_args()
    if _args.cpu:
        main_cpu()
    else:
        main()
