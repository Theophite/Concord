"""Smoke test for the pre-allocated backward-buffer refactor.

Validates that:
  1. ConcordLinearFused.forward/backward runs end-to-end with the new
     buffer-plumbed Function signature.
  2. ConcordConv2dFused.forward/backward runs end-to-end.
  3. State (s_slow / s_fast) actually changes after backward — the
     refactor didn't silently no-op.
  4. Buffers (_grad_W_buf, _row_max_buf, _col_max_buf) are populated
     after first call.
  5. SGD path AND three_accum path both work.

Run from C:\\foliated_onetrainer with the venv that has torch+triton:
    python test_prealloc_buffers.py
"""
import sys
import torch

torch.manual_seed(0)
DEV = 'cuda'


def _assert_state_changed(layer, label):
    s0 = layer.s_slow.clone()
    f0 = layer.s_fast.clone()
    return s0, f0


def _confirm_state_changed(layer, s0, f0, label):
    s_diff = (layer.s_slow.to(torch.int32) - s0.to(torch.int32)).abs().sum().item()
    f_diff = (layer.s_fast.to(torch.int32) - f0.to(torch.int32)).abs().sum().item()
    print(f"  [{label}] |d_s_slow|={s_diff}  |d_s_fast|={f_diff}")
    assert (s_diff + f_diff) > 0, f"{label}: state didn't change after backward!"


def _confirm_buffers_allocated(layer, label):
    # grad_W_buf was removed in the memory-optimization pass --
    # grad_W is now allocated fresh per backward (transient via the
    # CUDA graph pool when captured). Only the small row_max /
    # col_max buffers stay persistent.
    assert getattr(layer, '_row_max_buf', None) is not None, \
        f"{label}: _row_max_buf not allocated"
    assert getattr(layer, '_col_max_buf', None) is not None, \
        f"{label}: _col_max_buf not allocated"
    rm = layer._row_max_buf
    cm = layer._col_max_buf
    assert rm.dtype == torch.int32, f"{label}: row_max_buf dtype {rm.dtype}"
    assert cm.dtype == torch.int32, f"{label}: col_max_buf dtype {cm.dtype}"
    print(f"  [{label}] buffers OK: row_max{tuple(rm.shape)} "
          f"col_max{tuple(cm.shape)} (grad_W: transient)")


def test_linear_sgd():
    print("=== Test 1: ConcordLinearFused, SGD path ===")
    from concord_linear_fused import ConcordLinearFused
    layer = ConcordLinearFused(in_features=128, out_features=64,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_kind = 'sgd'
    s0, f0 = _assert_state_changed(layer, "linear-sgd")
    x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16,
                    requires_grad=False)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()
    _confirm_state_changed(layer, s0, f0, "linear-sgd")
    _confirm_buffers_allocated(layer, "linear-sgd")
    print("  PASS")


def test_linear_three_accum():
    print("=== Test 2: ConcordLinearFused, three_accum AdamW path ===")
    from concord_linear_fused import ConcordLinearFused
    layer = ConcordLinearFused(in_features=128, out_features=64,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_v_kind = 'three_accum'
    layer.set_optimizer_kind('adamw')   # auto-allocates v_slow_i8
    s0, f0 = _assert_state_changed(layer, "linear-three_accum")
    x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16,
                    requires_grad=False)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()
    _confirm_state_changed(layer, s0, f0, "linear-three_accum")
    _confirm_buffers_allocated(layer, "linear-three_accum")
    assert layer.v_slow_i8 is not None, "v_slow_i8 should be allocated"
    print("  PASS")


def test_conv2d_sgd():
    print("=== Test 3: ConcordConv2dFused, SGD path ===")
    from concord_linear_fused import ConcordConv2dFused
    layer = ConcordConv2dFused(in_channels=8, out_channels=16,
                                kernel_size=3, stride=1, padding=1,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_kind = 'sgd'
    s0, f0 = _assert_state_changed(layer, "conv-sgd")
    x = torch.randn(4, 8, 16, 16, device=DEV, dtype=torch.bfloat16,
                    requires_grad=False)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()
    _confirm_state_changed(layer, s0, f0, "conv-sgd")
    _confirm_buffers_allocated(layer, "conv-sgd")
    print("  PASS")


def test_repeat_uses_same_buffers():
    print("=== Test 4: Repeated forward+backward reuses buffers ===")
    from concord_linear_fused import ConcordLinearFused
    layer = ConcordLinearFused(in_features=64, out_features=32,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_kind = 'sgd'
    x = torch.randn(4, 64, device=DEV, dtype=torch.bfloat16)
    layer(x).square().mean().backward()
    rm_first = layer._row_max_buf
    cm_first = layer._col_max_buf
    layer(x).square().mean().backward()
    layer(x).square().mean().backward()
    assert layer._row_max_buf is rm_first, "row_max_buf was reallocated!"
    assert layer._col_max_buf is cm_first, "col_max_buf was reallocated!"
    print("  PASS (same buffer object across 3 backward passes)")


def test_loss_decreases():
    print("=== Test 5: Loss decreases over many steps (correctness) ===")
    from concord_linear_fused import ConcordLinearFused
    torch.manual_seed(1)
    layer = ConcordLinearFused(in_features=32, out_features=16,
                                bias=True, device=DEV, lr=0.05).to(DEV)
    layer.optimizer_kind = 'sgd'
    # Fixed target so the layer has something to fit to.
    W_target = torch.randn(16, 32, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(64, 32, device=DEV, dtype=torch.bfloat16)
    y_target = (x @ W_target.T).detach()
    losses = []
    for step in range(50):
        y = layer(x)
        loss = (y - y_target).square().mean()
        loss.backward()
        losses.append(loss.item())
    drop = losses[0] - losses[-1]
    print(f"  loss[0]={losses[0]:.4f}  loss[10]={losses[10]:.4f}  "
          f"loss[49]={losses[-1]:.4f}  drop={drop:.4f}")
    assert losses[-1] < losses[0], f"loss did not decrease! {losses[0]} -> {losses[-1]}"
    assert drop > 0.01, f"loss barely moved: drop={drop}"
    print("  PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        sys.exit(0)
    test_linear_sgd()
    test_linear_three_accum()
    test_conv2d_sgd()
    test_repeat_uses_same_buffers()
    test_loss_decreases()
    print("\n[OK] all pre-allocation smoke tests passed.")
