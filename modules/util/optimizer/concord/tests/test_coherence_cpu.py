"""CPU unit tests for the host-side Wiener/SNR coherence gate (no GPU, no Triton).

These exercise the pure-torch HOST REPLICAS of the kernel's USE_FIXED_COH gate --
the gate that actually decides what consolidates in the live WINNER recipe -- which
until now were only run indirectly (test_autotuner_cpu.py MONKEYPATCHES
measure_coherence; gate_coherence_from_fields had no direct CPU assertion).

    coh = sig**2 / (sig**2 + noise**2),   sig = C*d_sv,  noise = d_fs - sig
    d_sv = (s_slow - v_slow) * 128,        d_fs = s_fast

prototype_packed_b.py:3138-3163. The legacy "broken units" else-branch
(mean_grad**2 / v_hat) is GPU-only (needs v_hat) and not replicated on the host,
so it is not exercisable here -- only the live USE_FIXED_COH gate is.

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_coherence_cpu.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import prototype_packed_b as ppb

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# The live drift-cancel coefficient (mass_preserve=True is the constructor default).
C = ppb.compute_drift_cancel_C(0.1, 0.001, mass_preserve=True)   # ~= 0.018036


def coh_of(s_fast, s_slow, v_slow, c=C):
    return ppb.gate_coherence_from_fields(
        torch.tensor(s_fast, dtype=torch.float32),
        torch.tensor(s_slow, dtype=torch.float32),
        torch.tensor(v_slow, dtype=torch.float32),
        c)


print("== gate_coherence_from_fields (the live Wiener gate) ==")

# 1: pure drift -- s_fast == C*d_sv exactly -> noise == 0 -> coh == 1
dsv = (10.0 - 0.0) * 128.0
sig = C * dsv
g = coh_of([sig], [10.0], [0.0])
check("pure-drift coord (s_fast = C*d_sv) -> coh ~ 1", abs(g.item() - 1.0) < 1e-5,
      f"coh={g.item():.6f}")

# 2: pure noise -- d_sv == 0 (s_slow==v_slow), s_fast != 0 -> sig == 0 -> coh == 0
g = coh_of([100.0], [5.0], [5.0])
check("pure-noise coord (s_slow==v_slow, s_fast!=0) -> coh ~ 0", g.item() < 1e-6,
      f"coh={g.item():.2e}")

# 3: sig == noise -> coh == 0.5
sig3 = C * (10.0 * 128.0)              # s_slow=10, v_slow=0
g = coh_of([2 * sig3], [10.0], [0.0])  # s_fast = 2*sig -> noise = sig
check("sig == noise -> coh ~ 0.5", abs(g.item() - 0.5) < 1e-4, f"coh={g.item():.6f}")

# 4: monotone -- raising |sig| at fixed |noise| raises coh
sigA = C * (10.0 * 128.0)
sigB = C * (20.0 * 128.0)             # 2 * sigA
noise = sigA
gA = coh_of([sigA + noise], [10.0], [0.0])
gB = coh_of([sigB + noise], [20.0], [0.0])
check("monotone: larger |sig| at fixed noise -> larger coh", gB.item() > gA.item(),
      f"cohA={gA.item():.4f} < cohB={gB.item():.4f}")
# B has sig=2N, noise=N -> 4/(4+1) = 0.8 (closed form)
check("closed form coh(sig=2N, noise=N) ~ 0.8", abs(gB.item() - 0.8) < 1e-3,
      f"coh={gB.item():.6f}")

# 5: scale invariance -- k*(s_fast, s_slow, v_slow) leaves coh unchanged
base = coh_of([50.0], [7.0], [1.0])
scaled = coh_of([150.0], [21.0], [3.0])   # x3
check("scale-invariant: k*fields leaves coh unchanged",
      abs(base.item() - scaled.item()) < 1e-5,
      f"{base.item():.6f} vs {scaled.item():.6f}")

# 6: clamp [0,1] over a random batch
torch.manual_seed(0)
sf = torch.randn(256) * 500
ss = torch.randn(256) * 40
vs = torch.randn(256) * 40
g = ppb.gate_coherence_from_fields(sf, ss, vs, C)
check("coh in [0,1] elementwise (random batch)", bool((g >= 0).all() and (g <= 1).all()),
      f"min={g.min().item():.3f} max={g.max().item():.3f}")


print("== measure_coherence (per-layer mean, unpacks packed_w) ==")

def pack(s_fast, s_slow, v_slow):
    sf = torch.tensor(s_fast, dtype=torch.int64)
    ss = torch.tensor(s_slow, dtype=torch.int64)
    vv = torch.tensor(v_slow, dtype=torch.int64)
    return (((sf & 0xFFFF) << 16) | ((ss & 0xFF) << 8) | (vv & 0xFF)).to(torch.int32)

# 7: pure-drift packed (s_fast == round(C*d_sv)) -> mean coh ~ 1
ssv, vsv = 10, 0
sfv = round(C * (ssv - vsv) * 128)        # round(23.086) = 23
layer = SimpleNamespace(packed_w=pack([[sfv]], [[ssv]], [[vsv]]), drift_cancel_C=C)
mc = float(ppb.measure_coherence(layer))
check("measure_coherence pure-drift packed -> mean coh ~ 1", mc > 0.99, f"mean_coh={mc:.4f}")

# 8: gap-zero packed (s_slow==v_slow everywhere) -> mean coh ~ 0
layer0 = SimpleNamespace(packed_w=pack([[100, 100]], [[7, 7]], [[7, 7]]), drift_cancel_C=C)
mc0 = float(ppb.measure_coherence(layer0))
check("measure_coherence gap-zero packed -> mean coh ~ 0", mc0 < 1e-6, f"mean_coh={mc0:.2e}")
check("measure_coherence returns a scalar in [0,1]", 0.0 <= mc0 <= 1.0 and 0.0 <= mc <= 1.0)


n_pass = sum(results)
print(f"{n_pass}/{len(results)} coherence-gate CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
