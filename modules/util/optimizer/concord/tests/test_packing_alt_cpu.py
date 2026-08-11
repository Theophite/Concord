"""CPU unit tests for Concord packing/deploy paths that the live WINNER recipe does
NOT use (no GPU, no Triton).

Covered (all pure-torch on a _CpuCore stand-in that binds the real unbound methods,
so the Triton-launching __init__/_ensure_buffers is never run):
  * load_weights_finetune  -- alternative pretrained init (half-to-v_slow steady state)
  * load_weights_anchor    -- the LINEAR frozen-anchor init: deploys ~= W (NOT ~0),
                              correcting the earlier conflation with the embedding bug
  * resplit_anchor_to_even -- the TE-washout fix: rebalances slow channel to the even
                              split while preserving deploy AND s_fast EXACTLY
  * the embedding anchor init (FIXED): now routes through load_weights_anchor
           (coarse -> v_slow) so deploy ~= init; was the ~0-deploy re-split bug

resplit_anchor_to_even is the off-by-default fix for the documented frozen-anchor-
resumed-under-creep washout (only a CREEP layer ever calls it; the winner's frozen
anchor has alpha_v_fast=0 and never does).

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_packing_alt_cpu.py
"""
import sys
from pathlib import Path

import torch

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import prototype_packed_b as ppb
from prototype_packed_b import ConcordLinearPackedB as CLP

results = []
xpasses = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def xfail(name, fixed, detail=""):
    # `fixed` True would mean the bug is GONE (correct behavior holds). We EXPECT
    # False (bug present) -> XFAIL. If it ever flips True the test XPASSes -> alert
    # to update the doc/test (the bug got fixed).
    xpasses.append(bool(fixed))
    tag = "XPASS!! bug fixed -> update test" if fixed else "XFAIL (known bug, expected)"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


class _CpuCore:
    """Carries only the buffers the pure-tensor methods touch; binds the REAL
    unbound methods so we test the actual arithmetic without a Triton __init__."""
    MANTISSA_BIAS = CLP.MANTISSA_BIAS
    EXP_MIN = CLP.EXP_MIN
    EXP_MAX = CLP.EXP_MAX

    consolidated_weight = CLP.consolidated_weight
    get_weight = CLP.get_weight
    get_state = CLP.get_state
    load_weights = CLP.load_weights
    load_weights_anchor = CLP.load_weights_anchor
    load_weights_finetune = CLP.load_weights_finetune
    resplit_anchor_to_even = CLP.resplit_anchor_to_even

    def __init__(self, out_features, in_features):
        self.packed_w = torch.zeros(out_features, in_features, dtype=torch.int32)
        self.row_exp = torch.zeros(out_features, dtype=torch.int8)
        self.col_exp = torch.zeros(in_features, dtype=torch.int8)

    def _resync_weight_buf(self):
        return None   # no _bf16_weight_buf -> mirrors the real None-guard, no kernel


def rel_err(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-30)


# ------------------------------------------------------------------ #
print("== load_weights_finetune (alternative half-to-v_slow init) ==")
torch.manual_seed(0)
W = torch.randn(8, 16) * 0.3
core = _CpuCore(8, 16)
core.load_weights_finetune(W)
dep = core.consolidated_weight().float()
sfw, ssw, vsw = core.get_state()
check("finetune: consolidated_weight ~= W", rel_err(dep, W) < 0.05, f"rel={rel_err(dep, W):.4f}")
check("finetune: |s_slow - v_slow| small (steady-state intent)",
      int((ssw - vsw).abs().max()) <= 2, f"max|d|={int((ssw - vsw).abs().max())}")
check("finetune: |s_fast| is a sub-128 residual, not the mantissa",
      int(sfw.abs().max()) <= 128, f"max|s_fast|={int(sfw.abs().max())}")

# ------------------------------------------------------------------ #
print("== load_weights_anchor (LINEAR frozen anchor: deploys ~= W, NOT ~0) ==")
core = _CpuCore(8, 16)
core.load_weights_anchor(W)
dep = core.consolidated_weight().float()     # anchor deploy drops s_fast
live = core.get_weight().float()             # keeps s_fast (tighter)
sfw, ssw, vsw = core.get_state()
check("anchor: consolidated_weight ~= W to 8-bit coarse (does NOT deploy ~0)",
      0.5 < (dep.norm() / W.norm()).item() < 1.5 and rel_err(dep, W) < 0.05,
      f"||dep||/||W||={(dep.norm() / W.norm()).item():.4f} rel={rel_err(dep, W):.4f}")
check("anchor: s_slow channel is empty (coarse went to v_slow)",
      int(ssw.abs().max()) == 0, f"max|s_slow|={int(ssw.abs().max())}")
check("anchor: get_weight (keeps s_fast) is tighter than consolidated (drops it)",
      rel_err(live, W) <= rel_err(dep, W) + 1e-6,
      f"live_rel={rel_err(live, W):.4f} <= dep_rel={rel_err(dep, W):.4f}")

# ------------------------------------------------------------------ #
print("== resplit_anchor_to_even (TE-washout fix: deploy + s_fast preserved) ==")
core = _CpuCore(8, 16)
core.load_weights_anchor(W)                  # contaminated: s_slow=0, whole coarse in v_slow -> large d_sv
dep_before = core.consolidated_weight().clone()
sf_before = core.get_state()[0].clone()
gap_before = int((core.get_state()[1] - core.get_state()[2]).abs().max())
ret = core.resplit_anchor_to_even()
sf_after, ss_after, vs_after = core.get_state()
check("resplit: returns True on a contaminated anchor state (ratio > thresh)", ret is True,
      f"ret={ret}, gap_before={gap_before}")
check("resplit: deploy weight preserved EXACTLY (mass-preserving)",
      torch.equal(core.consolidated_weight(), dep_before))
check("resplit: s_fast preserved EXACTLY", torch.equal(sf_after, sf_before))
check("resplit: slow channel now even (|s_slow - v_slow| <= 1)",
      int((ss_after - vs_after).abs().max()) <= 1,
      f"gap_after={int((ss_after - vs_after).abs().max())}")

# already-even (load_weights gap-zero state) -> no-op, returns False
core2 = _CpuCore(8, 16)
core2.load_weights(W)
pw_before = core2.packed_w.clone()
ret2 = core2.resplit_anchor_to_even()
check("resplit: no-op (returns False) on an already-even load_weights state", ret2 is False)
check("resplit: packed_w unchanged on the no-op path", torch.equal(core2.packed_w, pw_before))

# ------------------------------------------------------------------ #
print("== embedding anchor init: FIXED -> deploys ~= init (was the ~0 re-split bug) ==")
# init_tokens(anchor=True) now routes through load_weights_anchor (coarse mantissa -> v_slow),
# like the Linear anchor, instead of re-reading the drained s_fast. Replay that fixed
# arithmetic on the stand-in: deploy ~= init. (The bug was ||deploy||/||init|| ~ 0 because the
# old path re-read (pw>>16) -- the <=64 residual after load_weights -- into v_slow ~ 0.)
torch.manual_seed(1)
init = torch.randn(2, 16) * 0.05
core = _CpuCore(2, 16)
core.load_weights_anchor(init)               # FIXED path: coarse mantissa -> v_slow, s_slow = 0
dep = core.consolidated_weight().float()
rel = (dep.norm() / init.norm()).item()
check("embedding anchor init deploys ~= init (coarse in v_slow; the ~0-deploy bug is fixed)",
      abs(rel - 1.0) < 0.1, f"||deploy||/||init|| = {rel:.4f}")


n_pass = sum(results)
print(f"{n_pass}/{len(results)} alt-packing CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
