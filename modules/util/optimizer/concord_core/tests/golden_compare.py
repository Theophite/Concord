"""GOLDEN-EQUIVALENCE COMPARE for the Concord behavior-preserving refactor.

  *** GPU + Triton REQUIRED. RUN ONLY IN A RUN-DOWN GPU WINDOW. ***

Re-runs the SAME cases golden_capture.py recorded, against a TARGET import path
(parameterized: the OLD monolith prototype_packed_b in concord/, or the FUTURE
concord_core package), and diffs against the saved goldens at atol=0 (bit-exact)
where the path is SR-deterministic. Prints a clear PASS/FAIL per case and, on a
FAIL, the FIRST field+step that diverged + a magnitude summary.

AUTHORED BLIND alongside golden_capture.py (the GPU was busy with a live run).
NOT executed. The first run is the validation run -- expect to fix small
import-surface mismatches in --target wiring before trusting a PASS.

USAGE:
  # default: compare the OLD prototype_packed_b against its own goldens (a
  # sanity / self-equivalence check -- should PASS trivially once goldens exist):
  venv/Scripts/python.exe modules/util/optimizer/concord_core/tests/golden_compare.py

  # compare the FUTURE refactored core (once it exists) against the OLD goldens:
  venv/Scripts/python.exe .../golden_compare.py --target concord_core

  # explicit module name / dir:
  venv/Scripts/python.exe .../golden_compare.py \
       --target-module prototype_packed_b --target-dir <abs path on sys.path>

HOW THE TARGET IS SWAPPED IN
  golden_capture.py does all its work through two module references: `ppb` (the
  packed core) and `cw` (concord_winner). This script imports golden_capture,
  rebinds golden_capture.ppb / golden_capture.cw (and the make_concord_config it
  pulls via concord_ot) to the TARGET, then re-invokes the SAME capture_g1 /
  capture_g2 / capture_g5 functions. So the cases are guaranteed identical to the
  ones that produced the goldens -- only the implementation under them moves.

  For the refactor the EXPECTATION (per the plan) is that concord_core re-exposes
  the SAME bare names (ConcordLinearPackedB, the set_* setters, read_boil, etc.)
  via the shim, so rebinding `ppb` to the new package is sufficient. If a name is
  missing the rebind raises AttributeError naming exactly the missing symbol --
  which IS the R4 re-export-chain check, surfaced early.

NOISE-ON: restored deterministically from the captured rng_fingerprint (cpu+cuda
RNG state) recorded in the golden, so the noise-ON trajectory is comparable
bit-exactly. _FUSED_MATMUL is set by direct attribute assignment on the target
module before each sub-capture (matching production's write path).
"""
import argparse
import json
import sys
from pathlib import Path

OT = Path(__file__).resolve().parents[5]
_OPT = OT / "modules" / "util" / "optimizer"   # holds concord_ot.py (bare import)
_CONCORD = _OPT / "concord"                     # holds prototype_packed_b / concord_winner
for _p in (str(OT), str(_OPT), str(_CONCORD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# Import the capture module -- we reuse its case definitions + runners verbatim.
import golden_capture as GC  # noqa: E402

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"


# ============================================================================
# Target wiring
# ============================================================================

def _load_target(target_module, target_dir):
    """Import the TARGET packed-core module and return it. `target_dir` (if given)
    is prepended to sys.path first so a bare import resolves to the refactored
    tree. Default target is the OLD prototype_packed_b already on the concord
    path."""
    if target_dir:
        td = str(Path(target_dir).resolve())
        if td not in sys.path:
            sys.path.insert(0, td)
    import importlib
    mod = importlib.import_module(target_module)
    return mod


def _rebind_capture_to(target_ppb):
    """Point golden_capture's module references at the TARGET. concord_winner +
    concord_ot import the packed core internally; for a shim-based refactor those
    keep working unchanged (the shim re-exports). We rebind GC.ppb so the G1 path
    (which calls ppb.* directly) hits the target, and assert the winner module
    resolves its names through the same target.

    NOTE (refactor): if concord_core replaces concord/ entirely (cut-over option
    b), concord_winner/concord_ot must already import from the new path; this
    script does NOT patch their internals -- it relies on the shim, exactly as the
    plan recommends (O2: shim-permanent)."""
    GC.ppb = target_ppb
    # Sanity: the winner module must expose the bare names G2 uses (R4 chain).
    required = ["swap_unet_to_winner", "winner_step", "set_sigmag_noise",
                "set_coh_vhat", "set_coh_kappa", "set_evap_slack", "set_min_leak",
                "set_evap_build_min", "set_lazy_gate", "set_lazy_thresh",
                "set_lamb_trust", "WINNER", "ConcordConfig"]
    missing = [n for n in required if not hasattr(GC.cw, n)]
    if missing:
        raise AttributeError(
            f"concord_winner is missing re-exported names {missing} -- the "
            f"R4 re-export chain is broken for the current target.")
    # The packed core must expose every symbol the capture touches.
    core_required = ["ConcordLinearPackedB", "ConcordConv2dPackedB",
                     "_get_step_counter", "set_consolidate", "read_boil",
                     "read_memgap", "bias_correction_factor", "set_v_bias_correction",
                     "set_bias_correct_v", "compute_drift_cancel_C",
                     "MANTISSA_BIAS", "INT16_MIN", "INT16_MAX",
                     "S_SLOW_FACTOR", "V_SLOW_FACTOR",
                     # G3 (2026-06-29 drift) -- servo/autotuner/meters surface.
                     "EpochDissipationServo", "DissipationAutoTuner",
                     "measure_coherence", "gate_coherence_from_fields",
                     "read_layer_boil", "register_layer_meters", "clear_layer_meters",
                     "_boil_buf"]
    miss2 = [n for n in core_required if not hasattr(target_ppb, n)]
    if miss2:
        raise AttributeError(
            f"target packed core is missing names {miss2} -- re-export incomplete.")


# ============================================================================
# Diff engine
# ============================================================================

def _tensors_equal(a, b):
    if a is None and b is None:
        return True, ""
    if (a is None) != (b is None):
        return False, "one is None"
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if torch.equal(a, b):
        return True, ""
    # bit-exact failed -> quantify.
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    n_diff = int((diff > 0).sum().item())
    return False, (f"{n_diff}/{a.numel()} elems differ, "
                   f"max|Δ|={float(diff.max().item()):.3e}")


def _walk_compare(gold, got, path=""):
    """Recursively compare two nested golden structures. Returns a list of
    (path, reason) mismatches; empty == identical."""
    out = []
    if isinstance(gold, torch.Tensor) or isinstance(got, torch.Tensor):
        ok, why = _tensors_equal(
            gold if isinstance(gold, torch.Tensor) else None,
            got if isinstance(got, torch.Tensor) else None)
        if not ok:
            out.append((path, why))
        return out
    if isinstance(gold, dict):
        if not isinstance(got, dict):
            return [(path, f"type {type(gold).__name__} vs {type(got).__name__}")]
        for k in gold:
            if k not in got:
                out.append((f"{path}/{k}", "missing in target"))
            else:
                out.extend(_walk_compare(gold[k], got[k], f"{path}/{k}"))
        for k in got:
            if k not in gold:
                out.append((f"{path}/{k}", "extra in target"))
        return out
    if isinstance(gold, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(gold) != len(got):
            return [(path, f"list len {len(gold)} vs "
                     f"{len(got) if hasattr(got,'__len__') else '?'}")]
        for i, (x, y) in enumerate(zip(gold, got)):
            out.extend(_walk_compare(x, y, f"{path}[{i}]"))
        return out
    # scalars / strings / None
    if isinstance(gold, float) and isinstance(got, float):
        # config floats etc.: require EXACT equality (atol=0 contract).
        if gold != got and not (gold != gold and got != got):  # NaN==NaN ok
            out.append((path, f"{gold!r} vs {got!r}"))
        return out
    if gold != got:
        out.append((path, f"{gold!r} vs {got!r}"))
    return out


def _report_case(case_name, mismatches, n_fields):
    if not mismatches:
        print(f"  [PASS] {case_name}  ({n_fields} fields, atol=0)")
        return True
    print(f"  [FAIL] {case_name}  ({len(mismatches)} mismatch(es)); first divergence:")
    for path, why in mismatches[:5]:
        print(f"           {path}: {why}")
    if len(mismatches) > 5:
        print(f"           ... and {len(mismatches) - 5} more")
    return False


def _count_fields(obj):
    n = 0
    for _ in GC._flatten_tensors(obj):
        n += 1
    return n


def _compare_with_env(ref, got, env, margin):
    """NEAR-bit-exact compare for the GPU gates. The kernel is nondeterministic per
    step (HW float-reduction order; 2026-06-29), so a numeric leaf PASSES when
    |refactored - golden| <= env[field]*margin, where env[field] is the baseline's
    measured self-jitter for that field-name. A field absent from env (e.g. config
    echoes, step_counter, n_steps) gets tol 0 -> exact. Returns (mismatches, worst)
    where worst maps field -> (max_observed_dev, tol) for the report."""
    rmap = dict(GC._flatten_numeric(ref))
    gmap = dict(GC._flatten_numeric(got))
    mism, worst = [], {}
    for p, a in rmap.items():
        if p not in gmap:
            mism.append((p, "missing in target"))
            continue
        b = gmap[p]
        f = GC._field_of(p)
        tol = env.get(f, 0.0) * margin
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                mism.append((p, f"shape/dtype {tuple(a.shape)}/{a.dtype} vs "
                             f"{tuple(b.shape) if isinstance(b, torch.Tensor) else type(b).__name__}"))
                continue
            d = float((a.float() - b.float()).abs().max())
        else:
            d = abs(float(a) - float(b))
        ow, _ = worst.get(f, (0.0, tol))
        worst[f] = (max(ow, d), tol)
        if d > tol:
            mism.append((p, f"max|Δ|={d:g} > tol {tol:g} (field '{f}')"))
    for p in gmap:
        if p not in rmap:
            mism.append((p, "extra in target"))
    return mism, worst


def _report_env_case(name, mism, worst):
    """Report an envelope-gated case + the headroom (worst observed dev vs tol per
    field) so a human can see how close to the envelope the refactor ran."""
    head = ", ".join(f"{f}:{d:g}/{t:g}" for f, (d, t) in
                     sorted(worst.items(), key=lambda kv: -(kv[1][0]))[:5] if t > 0)
    if not mism:
        print(f"  [PASS] {name}  (within envelope; worst dev/tol: {head or 'all exact'})")
        return True
    print(f"  [FAIL] {name}  ({len(mism)} leaf(s) exceed envelope); first:")
    for p, why in mism[:5]:
        print(f"           {p}: {why}")
    if len(mism) > 5:
        print(f"           ... and {len(mism) - 5} more")
    return False


# ============================================================================
# G2 noise-ON: restore the captured RNG before re-running
# ============================================================================

def _recapture_g2_with_rng(cfg, device, gold_g2, sub_key):
    """Re-run one G2 sub-trajectory. For noise-ON, restore the captured cpu+cuda
    RNG state recorded in the golden so the randn draw order matches bit-for-bit.
    Returns the freshly computed sub-trajectory dict."""
    sub = gold_g2[sub_key]
    noise_on = sub["noise_on"]
    fused = "fused1" in sub_key
    GC.ppb._FUSED_MATMUL = bool(fused)
    if not noise_on:
        return GC._drive_g2(cfg, device, sub["n_steps"], noise_on=False, seed=sub["seed"])
    # noise-ON: the golden captured the cpu+cuda RNG state right before stepping. We
    # RESTORE that exact state and run a variant of _drive_g2 that does NOT re-seed (so
    # the restore stands), reproducing the randn draw order bit-for-bit.
    # NOTE: the golden also stores a `draw_probe` fingerprint, but it is NOT cross-checked
    # here -- an earlier draft described a "verify the probe, else fall back to the saved
    # state" guard that was never implemented. TODO: either add the probe assert or drop
    # draw_probe from the capture so the comment and code agree.
    fp = sub["rng_fingerprint"]
    return _drive_g2_from_state(cfg, device, sub["n_steps"], fp, seed=sub["seed"])


def _drive_g2_from_state(cfg, device, n_steps, fp, seed):
    """A noise-ON G2 driver that RESTORES a captured RNG state instead of
    re-seeding, so the randn draw sequence is reproduced exactly from the golden.
    Mirrors GC._drive_g2's body but skips its internal re-seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = GC._TinyUNet().to(device)
    layers = GC.cw.swap_unet_to_winner(net, device, cfg.lr, gf_consol=cfg.gf_consol,
                                       step_cap=cfg.step_cap,
                                       gf_trust_delta_sq=cfg.gf_trust_delta_sq,
                                       verbose=False)
    GC._apply_shipped_globals(cfg)
    GC.cw.set_sigmag_noise(True, isotropic=cfg.sigmag_iso)

    # Restore the captured RNG state (the noise draw consumer).
    torch.set_rng_state(fp["cpu_rng_state"])
    if fp["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(fp["cuda_rng_state"], device)

    gin = torch.Generator(device="cpu").manual_seed(2024)
    x_seq = (torch.randn(4, 64, generator=gin) * 0.1).to(device)
    x_img = (torch.randn(2, 8, 8, 8, generator=gin) * 0.1).to(device)
    t_seq = (torch.randn(4, 16, generator=gin) * 0.1).to(device)
    t_img = (torch.randn(2, 8, 8, 8, generator=gin) * 0.1).to(device)

    layer_names = [n for n, mod in net.named_modules() if mod in layers]
    steps = []

    def snap_all(t):
        per = {n: GC._snap_layer(mod) for n, mod in net.named_modules() if mod in layers}
        steps.append({"t": t, "layers": per, "meters": GC._read_meters(device),
                      "sigma_now": round(float(GC.ppb._SIGMAG_SIGMA), 8)})

    snap_all(-1)
    for t in range(n_steps):
        GC.cw.winner_step(t, n_steps, layers, peak_lr=cfg.lr, warmup=cfg.warmup,
                          sigmag_peak=cfg.sigmag_peak, lr_min_frac=cfg.lr_min_frac,
                          noise=True, config=cfg, update_globals=True)
        y_seq, y_img = net(x_seq, x_img)
        loss = ((y_seq - t_seq) ** 2).mean() + ((y_img - t_img) ** 2).mean()
        loss.backward()
        net.zero_grad(set_to_none=True)
        for m in layers:
            m.rebalance()
        torch.cuda.synchronize()
        snap_all(t)
    return {"layer_names": layer_names, "steps": steps, "rng_fingerprint": fp,
            "n_steps": n_steps, "seed": seed, "noise_on": True}


# ============================================================================
# Driver
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="old",
                    help="'old' (prototype_packed_b in concord/, default) or "
                         "'concord_core' (the refactored package) or a module name.")
    ap.add_argument("--target-module", default=None,
                    help="explicit module name to import as the packed core "
                         "(overrides --target).")
    ap.add_argument("--target-dir", default=None,
                    help="dir to prepend to sys.path before importing the target.")
    ap.add_argument("--only", default=None,
                    help="run only one gate: g1 | g2 | g3 | g5")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU-only lane: compare G5 + G3a/G3b (servo/autotuner) "
                         "against the CPU goldens. SAFE WHILE A LIVE RUN IS UP with "
                         "CUDA_VISIBLE_DEVICES=\"\". Skips all GPU gates.")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="Near-bit-exact tolerance multiplier on the measured "
                         "self-jitter envelope for the GPU gates (default 2.0). A "
                         "single fresh draw can slightly exceed the 4-sample field "
                         "max, so >1 is expected; a REAL divergence is far larger.")
    args = ap.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit("golden_compare requires CUDA (run in a run-DOWN GPU window). "
                         "For the live-safe lane use --cpu with CUDA_VISIBLE_DEVICES=\"\".")
    device = None if args.cpu else torch.device("cuda")

    # Resolve the target module name.
    if args.target_module:
        tmod, tdir = args.target_module, args.target_dir
    elif args.target in ("old", None):
        tmod, tdir = "prototype_packed_b", None
    elif args.target == "concord_core":
        # The shim keeps the bare name; if the cut-over renames the module, the
        # caller passes --target-module explicitly.
        tmod = "prototype_packed_b"
        tdir = str((OT / "modules" / "util" / "optimizer" / "concord_core").resolve())
    else:
        tmod, tdir = args.target, args.target_dir

    target_ppb = _load_target(tmod, tdir)
    _rebind_capture_to(target_ppb)

    print(f"[golden_compare] target packed core: {target_ppb.__name__} "
          f"from {getattr(target_ppb, '__file__', '?')}")

    # Manifest / toolchain warning (Triton codegen is version-sensitive). The CPU
    # lane writes manifest_cpu.json (G5 + G3a/G3b only); the full GPU run writes
    # manifest.json. Prefer the lane-matching one.
    man_path = GOLDENS_DIR / ("manifest_cpu.json" if args.cpu else "manifest.json")
    if not man_path.exists():
        # Fall back to whichever manifest exists, so a CPU compare still works off a
        # full-run capture (and vice-versa for the shared G5/G3 goldens).
        alt = GOLDENS_DIR / ("manifest.json" if args.cpu else "manifest_cpu.json")
        if alt.exists():
            man_path = alt
        else:
            raise SystemExit(f"no goldens found at {GOLDENS_DIR} "
                             f"(looked for {man_path.name}) -- run golden_capture.py first.")
    manifest = json.loads(man_path.read_text())

    # Near-bit-exact envelope for the GPU gates (absent in the --cpu lane).
    gpu_env = {}
    env_path = GOLDENS_DIR / "gpu_envelope.pt"
    if not args.cpu:
        if env_path.exists():
            gpu_env = torch.load(env_path, map_location="cpu")
            top = ", ".join(f"{k}={v:g}" for k, v in
                            sorted(gpu_env.items(), key=lambda kv: -kv[1])[:6] if v > 0)
            print(f"[golden_compare] GPU envelope (margin x{args.margin}); top self-jitter: {top}")
        else:
            print("  [WARN] no gpu_envelope.pt -- GPU gates will demand atol=0 and "
                  "almost certainly FAIL (the kernel is nondeterministic). Re-capture.")

    cur_tc = GC._toolchain()
    gold_tc = manifest.get("toolchain", {})
    for k in ("torch", "triton", "cuda"):
        if gold_tc.get(k) != cur_tc.get(k):
            print(f"  [WARN] toolchain {k} differs: golden={gold_tc.get(k)} "
                  f"current={cur_tc.get(k)} -- Triton codegen is version-sensitive; "
                  f"a bit-exact FAIL may be a toolchain artifact, not a refactor bug.")

    all_pass = True

    # ---- G5 (CPU dict identity) ----
    if args.only in (None, "g5"):
        print("[golden_compare] G5 config-dict identity ...")
        gold_g5 = json.loads((GOLDENS_DIR / "g5_config.json").read_text())
        got_g5 = GC.capture_g5()
        # JSON round-trip the fresh capture so float/str reprs match the golden's
        # serialization exactly (the golden was JSON; compare JSON to JSON).
        got_g5 = json.loads(json.dumps(got_g5, default=str))
        mism = _walk_compare(gold_g5, got_g5, "g5")
        all_pass &= _report_case("G5 config-dict", mism, _count_g5(gold_g5))

    # ---- G1 (kernel matrix) ----
    if args.only in (None, "g1") and not args.cpu:
        print("[golden_compare] G1 kernel-output matrix ...")
        gold_g1 = torch.load(GOLDENS_DIR / "g1_kernel_matrix.pt", map_location="cpu")
        got_g1 = GC.capture_g1(device, n_steps=5)
        g1_pass = True
        for key in sorted(gold_g1):
            if key not in got_g1:
                print(f"  [FAIL] {key}: case missing in target run")
                g1_pass = False
                continue
            mism, worst = _compare_with_env(gold_g1[key], got_g1[key], gpu_env, args.margin)
            g1_pass &= _report_env_case(f"G1/{key}", mism, worst)
        for key in sorted(got_g1):
            if key not in gold_g1:
                print(f"  [WARN] {key}: extra case in target run (no golden)")
        all_pass &= g1_pass

    # ---- G2 (shipped trajectory; noise off bit-exact, noise on via RNG restore) ----
    if args.only in (None, "g2") and not args.cpu:
        print("[golden_compare] G2 shipped-default trajectory ...")
        gold_g2 = torch.load(GOLDENS_DIR / "g2_shipped_trajectory.pt", map_location="cpu")
        cfg = GC._shipped_config()
        # config dict identity first (CPU).
        mism_cfg = _walk_compare(gold_g2["config"],
                                 json.loads(json.dumps(GC._config_to_dict(cfg), default=str)),
                                 "g2/config")
        all_pass &= _report_case("G2/config", mism_cfg, 0)
        g2_pass = True
        for sub_key in [k for k in gold_g2 if k != "config"]:
            got_sub = _recapture_g2_with_rng(cfg, device, gold_g2, sub_key)
            mism, worst = _compare_with_env(gold_g2[sub_key], got_sub, gpu_env, args.margin)
            tag = "noise-off" if "noise_off" in sub_key else "RNG-restored noise-on"
            g2_pass &= _report_env_case(f"G2/{sub_key} [{tag}, envelope]", mism, worst)
        GC.ppb._FUSED_MATMUL = False
        all_pass &= g2_pass

    # ---- G3 (servo/autotuner CPU + live meters GPU; 2026-06-29 drift) ----
    if args.only in (None, "g3"):
        print("[golden_compare] G3 servo/autotuner/meters ...")
        gold_g3a = torch.load(GOLDENS_DIR / "g3a_servo_cpu.pt", map_location="cpu")
        got_g3a = GC.capture_g3_servo_cpu()
        all_pass &= _report_case("G3a servo (CPU; protected on/off)",
                                 _walk_compare(gold_g3a, got_g3a, "g3a"),
                                 _count_g5(gold_g3a))   # leaf count (G3a/b are scalars, not tensors)
        gold_g3b = torch.load(GOLDENS_DIR / "g3b_autotuner_cpu.pt", map_location="cpu")
        got_g3b = GC.capture_g3_autotuner_cpu()
        all_pass &= _report_case("G3b autotuner (CPU)",
                                 _walk_compare(gold_g3b, got_g3b, "g3b"),
                                 _count_g5(gold_g3b))   # leaf count
        g3c_path = GOLDENS_DIR / "g3c_live_meters.pt"
        if device is not None and g3c_path.exists():
            GC.ppb._FUSED_MATMUL = False
            gold_g3c = torch.load(g3c_path, map_location="cpu")
            got_g3c = GC.capture_g3c_live(device, n_steps=gold_g3c["n_steps"])
            mism_g3c, worst_g3c = _compare_with_env(gold_g3c, got_g3c, gpu_env, args.margin)
            all_pass &= _report_env_case("G3c live meters (GPU; 6-wide boil + M6a, envelope)",
                                         mism_g3c, worst_g3c)
        elif device is None:
            print("  [skip] G3c live meters: --cpu lane (GPU gate deferred to a run-DOWN window).")
        else:
            print("  [skip] G3c live meters: no g3c golden present "
                  "(run the full golden_capture.py in a run-DOWN window first).")

    print("=" * 70)
    print(f"[golden_compare] RESULT: {'PASS' if all_pass else 'FAIL'} "
          f"(target={target_ppb.__name__})")
    sys.exit(0 if all_pass else 1)


def _count_g5(d):
    # crude leaf count for the report header
    n = 0
    stack = [d]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        else:
            n += 1
    return n


if __name__ == "__main__":
    main()
