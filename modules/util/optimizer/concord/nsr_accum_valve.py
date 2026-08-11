"""NSR accumulation valve — the solvency-law batch doubler (advisory).

Derivation and receipts: the mechanics worktree's EVICTION_CONTROL_LOOP.md
addenda 3-4 and exps 62-64 [M]. The law: a distinction needs NSR/2
batches of corroboration inside the evidence interval 1/lam; the seeder
holds lam*NSR = C (a uniform solvency deficit of C/2), so as the
measured NSR grows the marginal distinction outgrows the interval. The
efficient remedy is doubling the effective batch (gain ~ tail_epochs *
log2(m), cost ~ tail*(m-1): m=2 is always the per-sample frontier —
exp 64), applied for as long a tail as affordable.

This module only ADVISES. DESIGN-STAGE — NOT YET WIRED: no call site
exists yet. The INTENDED integration is the seeder's window commit
(concord_ot._commit) calling it with the smoothed median NSR, writing
workspace/run/concord_solvency.json; the trainer would then apply the
multiplier at segment start (GenericTrainer, before the horizon
recompute), so the change lands at the run's natural RESTART_ON_SAMPLE
boundaries — the between-segment accum path is the one hardened against
mid-segment clock corruption (update_steps persisted in concord_clock.json).

Arming: the valve is INERT unless a marker file NSR_VALVE_ON exists in
the run dir (delete it to disarm; also delete concord_solvency.json to
forget the anchor/latch). The anchor may be pre-seeded in the sidecar
({"anchor": <early-run NSR median>}); otherwise the first windowed
median after arming becomes the anchor (conservative if armed late).
GROW = 1.3 (NSR growth to earn the doubling), warm-up floor t >= 1000
updates, CAP = 2 (exp 64: higher multipliers are never per-sample
optimal), and the doubling LATCHES — the law says hold it to the end.
"""
import json
import os

GROW = 1.3
CAP = 2
T_MIN = 1000


def advise(sidecar_path, nsr_med, t=None):
    """Called from the seeder commit. sidecar_path = concord_nsr.json
    path (the run dir is its parent). Never raises to the caller."""
    run_dir = os.path.dirname(str(sidecar_path))
    if not os.path.exists(os.path.join(run_dir, "NSR_VALVE_ON")):
        return None
    path = os.path.join(run_dir, "concord_solvency.json")
    try:
        state = json.load(open(path))
    except Exception:
        state = {}
    anchor = state.get("anchor")
    if not anchor or anchor <= 0:
        state["anchor"] = anchor = float(nsr_med)
    ratio = float(nsr_med) / max(float(anchor), 1e-9)
    mult = int(state.get("mult", 1))
    if mult < CAP and ratio >= GROW and (t is None or t >= T_MIN):
        mult = CAP                       # latch: hold the doubling
        state["latched_t"] = t
    state.update({"mult": mult, "nsr_med": round(float(nsr_med), 2),
                  "ratio": round(ratio, 3),
                  "t": int(t) if t is not None else None})
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)
    return state
