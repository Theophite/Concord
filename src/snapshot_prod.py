"""Bit-exact guard for editing the production apply kernel. Run BEFORE the edit
to save a snapshot; run again AFTER (with the new consolidate flag defaulting to
1) to confirm flag=1 == today's kernel, byte for byte.
   python snapshot_prod.py save
   python snapshot_prod.py compare
"""
import importlib.util
import sys

import torch

_FAP = r"C:\Concord\src\fused_apply_proto.py"
_spec = importlib.util.spec_from_file_location("fap", _FAP)
fap = importlib.util.module_from_spec(_spec); sys.modules["fap"] = fap
_spec.loader.exec_module(fap)
ref = fap.ref

SNAP = r"C:\Concord\src\prod_snap.pt"


def run():
    torch.manual_seed(0)
    out_f, in_f = 1280, 1280
    A = fap.build_layer(out_f, in_f, seed=0)
    gW = (torch.randn(out_f, in_f, device='cuda') * 0.02).to(torch.bfloat16)
    wb = torch.zeros(out_f, in_f, device='cuda', dtype=torch.bfloat16)
    rm = torch.zeros(out_f, dtype=torch.int32, device='cuda')
    cm = torch.zeros(in_f, dtype=torch.int32, device='cuda')
    ref._get_step_counter(gW.device).fill_(99)        # next apply -> salt 100 (pinned)
    dc = ref.compute_drift_cancel_C(0.1, 0.001)
    ref.apply_packed_adamw(
        A.packed_w, gW, wb, A.row_exp, A.col_exp, rm, cm,
        lr=1e-3, mantissa_bias=15, alpha=0.1, beta1=0.0, weight_decay=0.0, eps=1e-10,
        step_cap=10.0, v_scale=0.0, precond_p=0.5, gf_consol=50.0, drift_cancel_C=dc,
        alpha_v_fast=0.001, wd_sv=0.0, wd_sf=0.0, mass_preserve=True, apply_chase=True,
        track_rebalance=True, v_row=A.v_row, v_col=A.v_col, sum_v_inv=A._sum_v_inv,
        gf_trust_delta_sq=1.0, coh_pre=None)
    torch.cuda.synchronize()
    return A.packed_w.clone(), wb.clone(), rm.clone(), cm.clone()


if __name__ == "__main__":
    torch.cuda.init()
    pk, wb, rm, cm = run()
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        s = torch.load(SNAP)
        ok = (torch.equal(pk, s['pk']) and torch.equal(wb, s['wb'])
              and torch.equal(rm, s['rm']) and torch.equal(cm, s['cm']))
        print(f"packed equal: {torch.equal(pk, s['pk'])}")
        print(f"weight equal: {torch.equal(wb, s['wb'])}")
        print(f"rowmax equal: {torch.equal(rm, s['rm'])}  colmax equal: {torch.equal(cm, s['cm'])}")
        print("=> BIT-EXACT PASS" if ok else "=> MISMATCH - FAIL")
    else:
        torch.save({'pk': pk, 'wb': wb, 'rm': rm, 'cm': cm}, SNAP)
        print(f"saved snapshot: packed.sum={pk.sum().item()} wb.sum={wb.float().sum().item():.4f}")
