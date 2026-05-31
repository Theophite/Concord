"""Correctness test for the bidirectional rebalance (tick-up + re-added
tick-down). The invariant: rebalance is VALUE-PRESERVING -- it only changes
the int representation (exponent + mantissa), not the represented weight
W = m_eff * 2^(row_exp+col_exp-bias). Tick-down (lossless left-shift) must be
EXACT; tick-up (SR right-shift + residual migration) within rounding."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
import torch
from prototype_packed_b import rebalance_packed

dev = "cuda"
BIAS = 15
MAX_M = 24000


def pack(s_fast, s_slow, v_slow):
    return (((s_fast & 0xFFFF) << 16) | ((s_slow & 0xFF) << 8)
            | (v_slow & 0xFF)).to(torch.int32)


def materialize(packed, row_exp, col_exp):
    sf = packed >> 16
    ss = (packed << 16) >> 24
    vs = (packed << 24) >> 24
    me = (ss * 128 + sf + vs * 128).float()
    exp = row_exp[:, None].float() + col_exp[None, :].float() - BIAS
    return me * torch.pow(2.0, exp)


def run(name, s_fast, s_slow, v_slow, row_exp0, col_exp0, exact):
    N, K = s_fast.shape
    packed = pack(s_fast, s_slow, v_slow)
    re = torch.full((N,), row_exp0, dtype=torch.int8, device=dev)
    ce = torch.full((K,), col_exp0, dtype=torch.int8, device=dev)
    W0 = materialize(packed, re, ce)
    m_eff = (s_slow * 128 + s_fast + v_slow * 128)
    row_max = m_eff.abs().amax(1).to(torch.int32)
    col_max = m_eff.abs().amax(0).to(torch.int32)
    pk = packed.clone()
    rebalance_packed(pk, re, ce, row_max, col_max, EXP_MIN=-8, EXP_MAX=7,
                     allow_tickdown=True)
    W1 = materialize(pk, re, ce)
    # int8 range check (no overflow corruption)
    ss1 = ((pk << 16) >> 24); vs1 = ((pk << 24) >> 24)
    ok_i8 = (ss1.abs().max() <= 127) and (vs1.abs().max() <= 127)
    relerr = (W1 - W0).abs().max().item() / (W0.abs().max().item() + 1e-30)
    dexp = (int(re[0]) + int(ce[0])) - (row_exp0 + col_exp0)
    tag = "PASS" if ((relerr < 1e-5 if exact else relerr < 1e-2) and ok_i8) else "FAIL"
    print(f"[{tag}] {name:<26} dexp={dexp:+d}  relerr={relerr:.2e}  "
          f"int8_ok={ok_i8}  (W~[{W0.abs().min():.3g},{W0.abs().max():.3g}])")
    return tag == "PASS"


torch.manual_seed(0)
N = K = 64
ok = True
# tick-DOWN: small int8s + high exponent -> reclaim precision (exact)
ok &= run("tick-down (small, high exp)",
          torch.randint(-20, 20, (N, K), dtype=torch.int32, device=dev),
          torch.randint(-14, 15, (N, K), dtype=torch.int32, device=dev),
          torch.randint(-14, 15, (N, K), dtype=torch.int32, device=dev),
          row_exp0=5, col_exp0=5, exact=True)
# tick-UP: large m_eff -> regain headroom (within SR rounding)
ok &= run("tick-up (large m_eff)",
          torch.randint(-50, 50, (N, K), dtype=torch.int32, device=dev),
          torch.full((N, K), 100, dtype=torch.int32, device=dev),
          torch.full((N, K), 100, dtype=torch.int32, device=dev),
          row_exp0=0, col_exp0=0, exact=False)
# deadband: mid-range int8 (>31) + mid m_eff -> NO tick (unchanged, exact)
ok &= run("deadband (no tick)",
          torch.randint(-50, 50, (N, K), dtype=torch.int32, device=dev),
          torch.randint(40, 60, (N, K), dtype=torch.int32, device=dev),
          torch.randint(40, 60, (N, K), dtype=torch.int32, device=dev),
          row_exp0=0, col_exp0=0, exact=True)
# floor: tick-down wanted but already at EXP_MIN -> no change
ok &= run("tick-down at EXP_MIN floor",
          torch.randint(-5, 5, (N, K), dtype=torch.int32, device=dev),
          torch.randint(-5, 5, (N, K), dtype=torch.int32, device=dev),
          torch.randint(-5, 5, (N, K), dtype=torch.int32, device=dev),
          row_exp0=-4, col_exp0=-4, exact=True)   # -4+-4 tickable until -8

print("\nALL PASS" if ok else "\nSOME FAILED")
