"""CPU unit tests for the dither-accum redesign — BACKWARD-COMPAT + VELOCITY/COHERENCE
ONLY (no GPU, no Triton). Pure-torch on CPU; forces CUDA off so a live GPU run is safe.

Reference under test:
  modules/util/optimizer/concord/dither_accum_ref.py
  (design: modules/util/optimizer/concord/DITHER_ACCUM_DESIGN.md)

This module deliberately covers a NARROW slice of the reference's contract — the two
properties that protect the SDXL production path from the redesign:

  (1) BACKWARD-COMPAT (SDXL byte-safe): the disabled/degenerate config reduces to the
      CURRENT single-int16 s_fast behavior. The packed word is byte-identical to legacy
      (pack/unpack round-trips, and the two-byte view reunifies to the SAME int16 — no
      information loss from the high/low byte split, dither_accum_ref.py:191-195,758).
      Disabled mode (dither_enabled=False) matches the literal legacy s_fast tick at the
      TRAINING level: coarse/deploy word BIT-EXACT, s_fast within the SR boundary
      (<=1/coord), per dither_accum_ref.py:686-738. Disabled deploy is PURE coarse
      (s_slow+v_slow)*128 with den_frac_bits forced to 0 (dither_accum_ref.py:236-252,398).

  (2) VELOCITY / COHERENCE: d_fs as REDEFINED (d_fs = s_fast + err_s, the bounded int16
      velocity plus a sub-LSB carry, dither_accum_ref.py:84-90,495) is WELL-DEFINED and
      NON-ZERO so the coherence gate + cf-discount still function. compute_coherence
      (dither_accum_ref.py:359-376) is the byte-for-byte legacy Wiener-SNR + cf-discount;
      it must return a finite coh in [0,1], the cf-discount must actually bite
      (coh <= coh_raw when use_coh_vhat=True), and a runaway/zero d_fs must NOT drive coh
      permanently to 0 the way the rejected draft did.

We do NOT re-test the denormal channel, the carry-bound stress regimes, or recoverability
here — those are exercised by the reference self-test (8/8) and are out of this slice.

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_dither_compat_cpu.py
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only: a live GPU run must stay untouched

import torch

torch.manual_seed(0)

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

from dither_accum_ref import (  # noqa: E402
    pack_word, unpack_word, unpack_bytes, reunify_fast,
    compute_coherence, assert_disabled_matches_legacy,
    DitherAccumRef, _scale_fwd,
    INT8_MIN, INT8_MAX, INT16_MIN, INT16_MAX, CARRY,
)

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ==================================================================== #
# (1) BACKWARD-COMPAT: SDXL byte-safe — disabled/degenerate == legacy.
# ==================================================================== #
print("== (1a) byte-identical layout: pack/unpack round-trip + two-byte reunify ==")
N, K = 8, 16
s_fast = torch.randint(INT16_MIN, INT16_MAX + 1, (N, K), dtype=torch.int32)
s_slow = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
v_slow = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
packed = pack_word(s_fast, s_slow, v_slow)
sf2, ss2, vs2 = unpack_word(packed)
check("pack->unpack is bit-exact for s_fast (full int16, not int8-clamped)",
      torch.equal(s_fast, sf2))
check("pack->unpack is bit-exact for s_slow", torch.equal(s_slow, ss2))
check("pack->unpack is bit-exact for v_slow", torch.equal(v_slow, vs2))
# the crux: the high/low byte VIEW reunifies to the SAME int16 — the velocity is
# ONE bounded int16, never two clamped int8s (dither_accum_ref.py:191-195).
e_s, e_v = unpack_bytes(packed)
reuni = reunify_fast(e_s, e_v)
check("two-byte view reunifies to the SAME int16 velocity (no info loss in byte split)",
      torch.equal(reuni, s_fast),
      f"max|reunify - s_fast|={int((reuni - s_fast).abs().max())}")
# the full velocity range survives the round-trip (a draft int8 split would clamp to +-127)
extremes = torch.tensor([[INT16_MIN, INT16_MAX, 0, 1, -1]], dtype=torch.int32)
ez = torch.zeros_like(extremes)
pe = pack_word(extremes, ez, ez)
sfe, _, _ = unpack_word(pe)
check("full +-32767 int16 velocity range survives pack/unpack (not int8-clamped)",
      torch.equal(sfe, extremes),
      f"recovered={sfe.tolist()[0]}")

print("== (1b) disabled mode == legacy single-int16 s_fast at the TRAINING level ==")
# assert_disabled_matches_legacy drives a disabled step() and the literal _legacy_step
# from IDENTICAL state each step (dither_accum_ref.py:686-738): coarse/deploy word must
# be BIT-EXACT and s_fast must match to <=1/coord (the only residual is an SR-boundary
# Bernoulli float-reassociation flip; the Triton port shares one expr -> bit-exact).
ok_leg, coarse_gap, sfast_gap = assert_disabled_matches_legacy(steps=40)
check("disabled coarse/deploy word BIT-EXACT vs legacy over 40 steps",
      coarse_gap == 0, f"max coarse gap={coarse_gap}")
check("disabled s_fast matches legacy to <=1/coord (SR-boundary float reassociation)",
      sfast_gap <= 1, f"max |s_fast diff|={sfast_gap}")
check("assert_disabled_matches_legacy reports ok", ok_leg is True)

print("== (1c) disabled deploy is PURE coarse (s_slow+v_slow)*128 — no den_frac add ==")
ref_dis = DitherAccumRef(N, K, dither_enabled=False, seed=1)
# den_frac_bits is forced to 0 when disabled regardless of the ctor arg
# (dither_accum_ref.py:398) -> deploy must be byte-identical to legacy coarse.
check("disabled forces den_frac_bits == 0 (legacy: deploy is pure coarse)",
      ref_dis.den_frac_bits == 0, f"den_frac_bits={ref_dis.den_frac_bits}")
W_dis = torch.randn(N, K) * 0.05
ref_dis.load_weights(W_dis)
g_dis = -torch.sign(W_dis) * 0.02 + 0.02
for _ in range(50):
    ref_dis.step(g_dis, lr=0.05, gf_consol=0.3, drift_cancel_C=0.02, chase_floor=0.1)
sfd, ssd, vsd = unpack_word(ref_dis.packed_w)
coarse_dep = (ssd + vsd).to(torch.float32) * CARRY * _scale_fwd(ref_dis.row_exp, ref_dis.col_exp)
check("disabled deploy_weight() == (s_slow+v_slow)*128*scale EXACTLY (drops s_fast)",
      torch.equal(ref_dis.deploy_weight(), coarse_dep))
# den_frac stays identically zero in disabled mode (never fed) -> no sub-LSB injection
check("disabled den_frac is identically zero (no separate channel active)",
      float(ref_dis.den_frac.abs().max()) == 0.0,
      f"max|den_frac|={float(ref_dis.den_frac.abs().max())}")


# ==================================================================== #
# (2) VELOCITY / COHERENCE: d_fs (= s_fast + err_s) well-defined + non-zero,
#     so the coherence gate + cf-discount still function.
# ==================================================================== #
print("== (2a) compute_coherence: well-defined, finite, in [0,1]; cf-discount bites ==")
# Build a representative d_fs (= s_fast + err_s) / d_sv in W units. Use a real loaded
# state's scale so the magnitudes are physical, then feed compute_coherence directly.
ref = DitherAccumRef(N, K, dither_enabled=True, seed=2)
W = torch.randn(N, K) * 0.05
ref.load_weights(W)
scale = _scale_fwd(ref.row_exp, ref.col_exp)
sf, ss, vs = unpack_word(ref.packed_w)
d_fs = sf.to(torch.float32) + ref.err_s            # the REDEFINED velocity (ref:495)
d_sv = (ss.to(torch.float32) - vs.to(torch.float32)) * CARRY
d_fs_w = d_fs * scale
d_sv_w = d_sv * scale
# Adafactor-style positive vhat and the sum_v_inv scalar the gate expects.
vhat = torch.rand(N, K) * 1e-3 + 1e-6
sum_v_inv = 1.0 / (torch.rand(N).sum() + 1e-3)
C = 0.02
coh, coh_raw = compute_coherence(d_fs_w, d_sv_w, vhat, C, coh_kappa=1.0,
                                 sum_v_inv=sum_v_inv, N=N, K=K, use_coh_vhat=True)
check("coh is finite (no NaN/Inf) — gate is well-defined on the redefined d_fs",
      bool(torch.isfinite(coh).all()))
check("coh_raw is finite (no NaN/Inf)", bool(torch.isfinite(coh_raw).all()))
check("coh in [0,1] elementwise", bool((coh >= 0).all() and (coh <= 1).all()),
      f"range=[{float(coh.min()):.4f},{float(coh.max()):.4f}]")
check("coh_raw in [0,1] elementwise", bool((coh_raw >= 0).all() and (coh_raw <= 1).all()))
# cf-discount bites in the DOCUMENTED direction: it SHRINKS the noise term coh_n2 by
# coh_kappa/(cf+coh_kappa) < 1 (dither_accum_ref.py:374), so the chase gate coh is >=
# the un-discounted dissipation gate coh_raw (it never lowers coh). coh_raw drives
# dissipation, coh (discounted) drives the chase — memory: concord_two_pickers / the
# cf-discount protects s_fast. Assert coh >= coh_raw everywhere.
check("cf-discount functions in the documented direction: coh >= coh_raw elementwise",
      bool((coh >= coh_raw - 1e-6).all()),
      f"min(coh-coh_raw)={float((coh - coh_raw).min()):.3e}")
# and it actually BITES on at least some coords (not a silent no-op at this config).
check("cf-discount actually bites (coh strictly > coh_raw on some coord)",
      bool((coh > coh_raw + 1e-6).any()),
      f"max(coh-coh_raw)={float((coh - coh_raw).max()):.3e}")
# turning the cf-discount OFF must give coh == coh_raw (the discount is the only diff).
coh_off, coh_raw_off = compute_coherence(d_fs_w, d_sv_w, vhat, C, coh_kappa=1.0,
                                         sum_v_inv=sum_v_inv, N=N, K=K, use_coh_vhat=False)
check("use_coh_vhat=False disables the discount: coh == coh_raw",
      torch.allclose(coh_off, coh_raw_off),
      f"max|coh-coh_raw|={float((coh_off - coh_raw_off).abs().max()):.3e}")

print("== (2b) d_fs is well-defined + NON-ZERO after training; coh stays in a band ==")
# Drive the coherent-drift regime (the contract's regime) and confirm d_fs becomes a
# meaningful non-zero velocity and that coh never collapses to a permanent ~0 (the
# rejected draft went 0.0082 -> 0.0000 by step 5 from an unbounded carry; ref:777-799).
ref2 = DitherAccumRef(N, K, dither_enabled=True, seed=4)
W2 = torch.randn(N, K) * 0.05
ref2.load_weights(W2)
g2 = -torch.sign(W2) * 0.02 + 0.02
coh_trace = []
d_fs_abs_trace = []
for _ in range(300):
    info = ref2.step(g2, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                     coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
    coh_trace.append(info["coh"])
    d_fs_abs_trace.append(info["d_fs_abs_mean"])
coh_t = torch.tensor(coh_trace)
dfs_t = torch.tensor(d_fs_abs_trace)
# WELL-DEFINED: coh finite every step.
check("coh finite on all 300 steps (gate never NaN/Inf under training)",
      bool(torch.isfinite(coh_t).all()))
# NON-ZERO: d_fs (the velocity) becomes a real non-zero quantity — the gate has signal.
check("d_fs (velocity) is NON-ZERO during training (mean|d_fs| > 0 across the run)",
      float(dfs_t.max()) > 0.0, f"max mean|d_fs|={float(dfs_t.max()):.3f}")
# does NOT collapse to a permanent 0: coh is in a stable positive band after warmup.
late_coh = coh_t[50:]
check("coh does NOT collapse to permanent ~0 (late-run min strictly > 0)",
      float(late_coh.min()) > 1e-6,
      f"late coh band=[{float(late_coh.min()):.4f},{float(late_coh.max()):.4f}]")
# d_fs bounded (it is the int16 velocity + sub-LSB err_s, ref:84-90) — never a runaway.
check("d_fs stays bounded (int16 velocity, not a runaway carry): max mean|d_fs| < 32768",
      float(dfs_t.max()) < float(INT16_MAX) + 1.0, f"max mean|d_fs|={float(dfs_t.max()):.1f}")


n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} dither backward-compat + velocity/coherence CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
