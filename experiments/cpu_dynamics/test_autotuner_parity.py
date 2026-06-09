"""CPU parity test for the package autotuner (concord/packed_b.py).

packed_b imports triton at module top, so on CPU we exec the exact shipped
source of `gate_coherence_from_fields`, `measure_coherence`, and
`DissipationAutoTuner` and validate:
  1. the coherence formula matches the CPU reference gate (incl. the claim
     that the row/col exponents cancel — scale invariance);
  2. packing round-trip: measure_coherence(packed) == formula(fields);
  3. the tuner's probe/commit behavior and table interpolation.
"""
import re

import torch

SRC = open("../../concord/packed_b.py").read()
ns = {"torch": torch}
for name in ("def gate_coherence_from_fields", "def measure_coherence",
             "class DissipationAutoTuner"):
    m = re.search(rf"(^@torch\.no_grad\(\)\n)?^{re.escape(name)}.*?(?=^\S)",
                  SRC, re.M | re.S)
    assert m, name
    exec(m.group(0), ns)

gate_coherence_from_fields = ns["gate_coherence_from_fields"]
measure_coherence = ns["measure_coherence"]
DissipationAutoTuner = ns["DissipationAutoTuner"]

torch.manual_seed(0)
N, K, C = 64, 96, 0.018036

# 1. formula parity vs the reference gate (concord_ref.ConcordRef.step) ----
s_fast = torch.randint(-2000, 2000, (N, K)).float()
s_slow = torch.randint(-128, 128, (N, K)).float()
v_slow = torch.randint(-128, 128, (N, K)).float()
coh_pkg = gate_coherence_from_fields(s_fast, s_slow, v_slow, C)
for trial in range(3):                       # random exponents: must cancel
    re_ = torch.randint(-8, 8, (N, 1)).float()
    ce_ = torch.randint(-8, 8, (1, K)).float()
    scale = torch.exp2(re_ + ce_ - 15)
    u = s_fast * scale                       # reference state, weight units
    S = 128.0 * s_slow * scale
    A = 128.0 * v_slow * scale
    mu = C * (S - A)
    nse = u - mu
    coh_ref = (mu * mu) / (mu * mu + nse * nse + 1e-30)
    assert torch.allclose(coh_pkg, coh_ref, atol=1e-5), \
        f"formula mismatch (trial {trial}): max diff " \
        f"{(coh_pkg - coh_ref).abs().max():.2e}"
print("1. coherence formula == reference gate, scale-invariant: OK")

# 2. packing round-trip --------------------------------------------------
packed = (((s_fast.int() & 0xFFFF) << 16)
          | ((s_slow.int() & 0xFF) << 8)
          | (v_slow.int() & 0xFF))


class FakeLayer:
    def __init__(self):
        self.packed_w = packed
        self.drift_cancel_C = C
        self.gf_consol = 0.0


assert abs(measure_coherence(FakeLayer()) - float(coh_pkg.mean())) < 1e-6
print("2. measure_coherence(packed) == formula(fields): OK")

# 3. tuner probe/commit + interpolation ----------------------------------
TABLE = [(0.387, 0.0), (0.314, 100.0), (0.288, 200.0), (0.274, 400.0),
         (0.256, 400.0)]
layers = [FakeLayer(), FakeLayer()]
tuner = DissipationAutoTuner(layers, probe_start=10, probe_end=50,
                             table=TABLE, probe_kappa=50.0,
                             measure_every=10, verbose=False)
assert all(m.gf_consol == 50.0 for m in layers), "probe kappa not set"
committed = None
for t in range(60):
    r = tuner.step(t)
    if r is not None:
        committed = (t, r)
assert committed is not None and committed[0] == 50, "commit step wrong"
assert all(m.gf_consol == committed[1] for m in layers), "kappa not applied"
expect = tuner.kappa_from_coh(float(coh_pkg.mean()))
assert abs(committed[1] - expect) < 1e-9
# interpolation endpoints + midpoints
assert tuner.kappa_from_coh(0.50) == 0.0
assert tuner.kappa_from_coh(0.20) == 400.0
mid = tuner.kappa_from_coh(0.301)
assert 100.0 < mid < 200.0, mid
print(f"3. tuner probe/commit at t=50, kappa={committed[1]:.1f}, "
      f"interp checks: OK")
print("ALL PARITY TESTS PASSED")
