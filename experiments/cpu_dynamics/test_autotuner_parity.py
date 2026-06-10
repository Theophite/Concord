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

SRC = open("../../concord/packed_b.py", encoding="utf-8").read()
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
# beta1 commit: probe runs at 0; threshold decides the committed value
probe_coh = float(coh_pkg.mean())
assert tuner.committed_beta1 == (0.1 if probe_coh >= 0.35 else 0.0)
assert all(m.beta1 == tuner.committed_beta1 for m in layers)
lo = DissipationAutoTuner([FakeLayer()], 0, 5, TABLE, verbose=False,
                          beta1_coh_threshold=0.0)     # always clears
assert all(m.beta1 == 0.0 for m in lo.layers), "probe must run at beta1=0"
for t in range(6):
    lo.step(t)
assert lo.committed_beta1 == 0.1 and lo.layers[0].beta1 == 0.1
off = DissipationAutoTuner([FakeLayer()], 0, 5, TABLE, verbose=False,
                           beta1_on=0.0)               # disabled
for t in range(6):
    off.step(t)
assert off.committed_beta1 == 0.0
# interpolation endpoints + midpoints
assert tuner.kappa_from_coh(0.50) == 0.0
assert tuner.kappa_from_coh(0.20) == 400.0
mid = tuner.kappa_from_coh(0.301)
assert 100.0 < mid < 200.0, mid
print(f"3. tuner probe/commit at t=50, kappa={committed[1]:.1f}, "
      f"interp checks: OK")
print("ALL PARITY TESTS PASSED")


# 5. live mode: one-sided re-probe watchdog (exp 11d) ----------------------
def _packed_with_coh(high, n=64, k=96, seed=0):
    """Synthetic packed state with high (~1) or low (~0) gate coherence:
    coherent -> s_fast == round(C*·d_sv·128) (noise ~ 0); incoherent ->
    s_fast random against a small telescope."""
    g = torch.Generator().manual_seed(seed)
    s_slow = torch.randint(20, 120, (n, k), generator=g)
    v_slow = torch.zeros((n, k), dtype=torch.long)
    if high:
        mu = C * (s_slow - v_slow).float() * 128.0
        s_fast = mu.round().long()
    else:
        s_fast = torch.randint(-2000, 2000, (n, k), generator=g)
    return (((s_fast.int() & 0xFFFF) << 16)
            | ((s_slow.int() & 0xFF) << 8)
            | (v_slow.int() & 0xFF))


hi, lo_pk = _packed_with_coh(True), _packed_with_coh(False)
lay = FakeLayer(); lay.packed_w = hi
tun = DissipationAutoTuner([lay], probe_start=0, probe_end=20, table=TABLE,
                           probe_kappa=50.0, measure_every=5, verbose=False,
                           reprobe_band=0.08)
k_first = None
for t in range(0, 21):
    r = tun.step(t)
    if r is not None:
        k_first = r
assert k_first is not None and k_first == TABLE[0][1], "high-coh commit should be kappa=0"
# stable hold: enough windows to prove no spurious event on a flat stream
for t in range(21, 121):
    tun.step(t)
assert tun.reprobes == 0, "spurious re-probe on a stable stream"
assert tun.committed == k_first
# regime change: swap to the low-coherence stream -> drop fires exactly once
lay.packed_w = lo_pk
k_second, t_fire = None, None
for t in range(121, 300):
    r = tun.step(t)
    if r is not None:
        k_second, t_fire = r, t
assert tun.reprobes == 1, f"expected exactly 1 re-probe, got {tun.reprobes}"
assert k_second is not None and k_second > k_first, \
    f"recommit should raise kappa ({k_first} -> {k_second})"
assert lay.gf_consol == k_second
print(f"5. live re-probe: stable hold quiet, drop fires once, "
      f"kappa {k_first:.0f} -> {k_second:.0f} at t={t_fire}: OK")
