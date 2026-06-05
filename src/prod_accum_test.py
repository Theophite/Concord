"""End-to-end production accumulation test: drive the REAL apply_packed_adamw via
the set_consolidate() hook. N micro-step applies (tick-only for N-1, consolidate on
the last) must match a single apply on the summed gradient (true accumulation)."""
import importlib.util
import sys

import torch

_FAP = r"C:\Concord\src\fused_apply_proto.py"
_spec = importlib.util.spec_from_file_location("fap", _FAP)
fap = importlib.util.module_from_spec(_spec); sys.modules["fap"] = fap
_spec.loader.exec_module(fap)
ref = fap.ref


def apply_prod(L, gW, wb, rm, cm):
    dc = ref.compute_drift_cancel_C(0.1, 0.001)
    ref.apply_packed_adamw(
        L.packed_w, gW, wb, L.row_exp, L.col_exp, rm, cm,
        lr=1e-3, mantissa_bias=15, alpha=0.1, beta1=0.0, weight_decay=0.0, eps=1e-10,
        step_cap=10.0, v_scale=0.0, precond_p=0.5, gf_consol=50.0, drift_cancel_C=dc,
        alpha_v_fast=0.001, wd_sv=0.0, wd_sf=0.0, mass_preserve=True, apply_chase=True,
        track_rebalance=True, v_row=L.v_row, v_col=L.v_col, sum_v_inv=L._sum_v_inv,
        gf_trust_delta_sq=1.0, coh_pre=None)


def test(out_f=1280, in_f=1280, N=3):
    dev = 'cuda'
    torch.manual_seed(11)
    gmicro = [(torch.randn(out_f, in_f, device=dev) * 0.02 / N).to(torch.bfloat16) for _ in range(N)]
    gsum = torch.stack([g.float() for g in gmicro]).sum(0).to(torch.bfloat16)
    z = lambda n: torch.zeros(n, dtype=torch.int32, device=dev)

    R = fap.build_layer(out_f, in_f, seed=0)
    wbR = torch.zeros(out_f, in_f, device=dev, dtype=torch.bfloat16); rmR, cmR = z(out_f), z(in_f)
    ref.set_consolidate(dev, True)
    apply_prod(R, gsum, wbR, rmR, cmR)

    Acc = fap.build_layer(out_f, in_f, seed=0)
    wbA = torch.zeros(out_f, in_f, device=dev, dtype=torch.bfloat16); rmA, cmA = z(out_f), z(in_f)
    wb_mid = []
    for i, g in enumerate(gmicro):
        ref.set_consolidate(dev, i == N - 1)
        apply_prod(Acc, g, wbA, rmA, cmA)
        wb_mid.append(wbA.float().abs().sum().item())
    ref.set_consolidate(dev, True)   # restore default

    torch.cuda.synchronize()
    dw = wbR.float() - wbA.float(); base = wbR.float().abs()
    print(f"=== production {N}-step accumulation vs single-apply-on-sum ===")
    print(f"  weight_buf frozen during accum? sums per micro-step = {['%.3f'%v for v in wb_mid]}")
    print(f"     (first {N-1} should be IDENTICAL = frozen; last jumps = consolidated)")
    print(f"  mean|ref|={base.mean():.4e}  mean|d|={dw.abs().mean():.4e}  "
          f"rel={dw.abs().mean()/base.mean().clamp(min=1e-12):.3%}")
    print(f"  mean(d)={dw.mean():.4e} (unbiased ~0)   corr={torch.corrcoef(torch.stack([wbR.float().flatten(), wbA.float().flatten()]))[0,1]:.5f}")


if __name__ == "__main__":
    torch.cuda.init()
    test(N=2)
    test(N=3)
