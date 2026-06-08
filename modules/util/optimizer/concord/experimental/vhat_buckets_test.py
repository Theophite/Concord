"""EXPERIMENTAL: validate ConcordVHatBuckets -- per-shape v_hat snapshot/restore.

Proves, on GPU:
  A. round-trip   : train shape-A (v_hat adapts to its grad scale), switch to B, train B
                    (different scale), switch back to A -> A's v_hat is restored exactly.
  B. in-place     : v_row/v_col data_ptr is UNCHANGED across a switch (so a captured CUDA
                    graph reading those pointers needs no recapture for the swap).
  C. empty_cache  : the swap still works after torch.cuda.empty_cache() at the boundary
                    (snapshots are live tensors -> survive the arena reset).
  D. warm-start   : a never-seen shape returns False (keeps the current v_hat), a returning
                    shape returns True (restored).

Run: python experimental/vhat_buckets_test.py   (needs CUDA + the concord package).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import prototype_packed_b as ppb
from prototype_packed_b import ConcordLinearPackedB
from vhat_buckets import ConcordVHatBuckets


def train(layers, scale, steps=60):
    """Feed synthetic grads of magnitude ~scale so v_hat (EMA of g^2) settles near scale^2."""
    for _ in range(steps):
        for L in layers:
            g = torch.randn(L.out_features, L.in_features, device="cuda") * scale
            L.apply_grad_step(g)


def vhat_of(layers):
    return [(L.v_row.detach().clone(), L.v_col.detach().clone()) for L in layers]


def max_abs_diff(a, b):
    return max((x[0] - y[0]).abs().max().item() for x, y in zip(a, b)) if a else 0.0


def main():
    if not torch.cuda.is_available():
        print("NEED CUDA (Triton kernel is GPU-only). Aborting."); return
    torch.manual_seed(0)
    ppb.set_consolidate("cuda", True)
    layers = [ConcordLinearPackedB(d_in, d_out, bias=False, device="cuda", lr=1e-2)
              for d_in, d_out in [(320, 320), (640, 1280), (1280, 768)]]
    for L in layers:                       # warm-up so v_row/v_col get allocated
        L.apply_grad_step(torch.randn(L.out_features, L.in_features, device="cuda") * 1.0)

    bk = ConcordVHatBuckets(layers)
    ptr_before = [(L.v_row.data_ptr(), L.v_col.data_ptr()) for L in layers]

    # ---- shape A then shape B (10x grad scale) ----
    r_a0 = bk.switch_to("A")               # first time -> warm start (False)
    train(layers, scale=1.0)
    vhat_A = vhat_of(layers)

    r_b0 = bk.switch_to("B")               # saves A, B new -> warm start (False)
    train(layers, scale=10.0)
    vhat_B = vhat_of(layers)
    sep = max_abs_diff(vhat_A, vhat_B)     # A and B must be clearly different scales

    # ---- back to A: must restore A's v_hat exactly ----
    r_a1 = bk.switch_to("A")               # saves B, restores A -> True
    err_A = max_abs_diff(vhat_of(layers), vhat_A)

    # ---- B again, but force an arena reset first (the real boundary) ----
    torch.cuda.empty_cache()
    r_b1 = bk.switch_to("B")               # restores B -> True
    err_B = max_abs_diff(vhat_of(layers), vhat_B)

    ptr_after = [(L.v_row.data_ptr(), L.v_col.data_ptr()) for L in layers]
    ptr_stable = ptr_before == ptr_after

    print(f"[D] first-seen A/B warm-start (expect False/False): {r_a0} / {r_b0}")
    print(f"[D] returning  A/B restored   (expect  True/True ): {r_a1} / {r_b1}")
    print(f"[A] A vs B v_hat separation (expect >> 0)        : {sep:.3e}")
    print(f"[A] restore-A exact  (expect 0)                  : {err_A:.3e}")
    print(f"[C] restore-B after empty_cache (expect 0)       : {err_B:.3e}")
    print(f"[B] v_row/v_col pointers stable across switches  : {ptr_stable}")
    print(f"    cached: {len(bk.seen_shapes())} shapes, {bk.memory_bytes()/1024:.1f} KiB")
    ok = (not r_a0 and not r_b0 and r_a1 and r_b1 and sep > 1e-3
          and err_A == 0.0 and err_B == 0.0 and ptr_stable)
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
