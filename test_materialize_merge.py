"""Verify: after apply, the weight_buf matches what materialize_packed_bf16
would produce from the new packed_w state.
"""
import torch
from prototype_packed_b import (
    ConcordLinearPackedB, ConcordConv2dPackedB, materialize_packed_bf16,
)


def test_linear():
    print("=== Linear: apply keeps weight_buf in sync with packed_w ===")
    torch.manual_seed(0)
    layer = ConcordLinearPackedB(64, 32, device='cuda', lr=0.01)
    layer.set_optimizer_kind('adamw', weight_decay=0.01, eps=1.0)

    x = torch.randn(8, 64, device='cuda', dtype=torch.bfloat16,
                     requires_grad=True)
    for step in range(5):
        x.grad = None
        y = layer(x)
        y.square().mean().backward()

    # Now compare weight_buf (kept fresh by apply) with a fresh
    # materialize from the current packed_w.
    wbuf = layer._bf16_weight_buf
    fresh = torch.empty_like(wbuf)
    materialize_packed_bf16(layer.packed_w, layer.row_exp, layer.col_exp,
                             out=fresh, mantissa_bias=layer.MANTISSA_BIAS)
    diff = (wbuf.float() - fresh.float()).abs()
    print(f"  weight_buf vs fresh-materialize: "
          f"max diff = {diff.max().item():.6f}, "
          f"mean diff = {diff.mean().item():.6f}")
    if diff.max().item() < 1e-3:
        print("  PASS")
    else:
        print("  FAIL — buffers diverged")


def test_conv():
    print("=== Conv2d: apply keeps weight_buf in sync with packed_w ===")
    torch.manual_seed(0)
    layer = ConcordConv2dPackedB(3, 16, 3, padding=1, device='cuda', lr=0.01)
    x = torch.randn(4, 3, 16, 16, device='cuda', dtype=torch.bfloat16,
                     requires_grad=True)
    for step in range(5):
        x.grad = None
        y = layer(x)
        y.square().mean().backward()

    wbuf = layer._bf16_weight_buf
    fresh = torch.empty_like(wbuf)
    materialize_packed_bf16(layer.packed_w, layer.row_exp, layer.col_exp,
                             out=fresh, mantissa_bias=layer.MANTISSA_BIAS)
    diff = (wbuf.float() - fresh.float()).abs()
    print(f"  weight_buf vs fresh-materialize: "
          f"max diff = {diff.max().item():.6f}, "
          f"mean diff = {diff.mean().item():.6f}")
    if diff.max().item() < 1e-3:
        print("  PASS")
    else:
        print("  FAIL — buffers diverged")


if __name__ == "__main__":
    test_linear()
    test_conv()
