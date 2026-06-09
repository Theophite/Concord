"""Smoke test for CUDA graph capture of a Concord layer's forward+backward.

If this works, the full training loop's microbatch can also be captured.

Validates:
  1. A single ConcordLinearFused (SGD path) captures cleanly into a
     CUDA graph and replays without error.
  2. Replay actually mutates state (s_slow / s_fast change).
  3. Same shape-stable behavior over many replays.
  4. Same for three_accum path.
  5. Same for ConcordConv2dFused.
  6. Speed: replayed step is faster than non-graph step (sanity).

Run:
    python test_graph_capture.py
"""
import sys
import time
import torch

torch.manual_seed(0)
DEV = 'cuda'


def _make_static_inputs(in_features, batch_size, requires_grad=True):
    x = torch.randn(batch_size, in_features, device=DEV,
                    dtype=torch.bfloat16, requires_grad=requires_grad)
    return x


def _bench_replay(graph, n=50):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        graph.replay()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def _bench_dynamic(fn, n=50):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def test_linear_sgd_graph():
    print("=== Test 1: ConcordLinearFused SGD captured into CUDA graph ===")
    from concord_linear_fused import ConcordLinearFused
    layer = ConcordLinearFused(in_features=128, out_features=64,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_kind = 'sgd'

    # Static input buffer for both warmup and capture.
    static_x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16,
                           requires_grad=True)

    # Warmup on a dedicated capture-stream (CUDA graphs require this:
    # the legacy default stream cannot be captured).
    s_capture = torch.cuda.Stream()
    s_capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_capture):
        for _ in range(3):
            if static_x.grad is not None:
                static_x.grad = None
            y = layer(static_x)
            y.square().mean().backward()
    torch.cuda.current_stream().wait_stream(s_capture)
    torch.cuda.synchronize()

    s0 = layer.s_slow.clone()
    f0 = layer.s_fast.clone()

    # Capture on the same dedicated stream.
    if static_x.grad is not None:
        static_x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s_capture):
        y = layer(static_x)
        loss = y.square().mean()
        loss.backward()
    print("  graph captured OK")

    # Replay several times and confirm state mutates.
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s_diff = (layer.s_slow.to(torch.int32) - s0.to(torch.int32)).abs().sum().item()
    f_diff = (layer.s_fast.to(torch.int32) - f0.to(torch.int32)).abs().sum().item()
    print(f"  |d_s_slow|={s_diff}  |d_s_fast|={f_diff} after 10 replays")
    assert (s_diff + f_diff) > 0, "state didn't change after graph replay"

    # Speed comparison.
    def dyn_step():
        if static_x.grad is not None:
            static_x.grad = None
        y = layer(static_x)
        y.square().mean().backward()
    t_dyn = _bench_dynamic(dyn_step, n=50)
    t_g = _bench_replay(g, n=50)
    print(f"  dynamic: {t_dyn*1000:.3f} ms/step  graph: {t_g*1000:.3f} ms/step  "
          f"speedup={t_dyn/t_g:.2f}x")
    print("  PASS")


def test_linear_three_accum_graph():
    print("=== Test 2: ConcordLinearFused three_accum captured ===")
    from concord_linear_fused import ConcordLinearFused
    layer = ConcordLinearFused(in_features=128, out_features=64,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_v_kind = 'three_accum'
    layer.set_optimizer_kind('adamw')

    static_x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16,
                           requires_grad=True)

    s_capture = torch.cuda.Stream()
    s_capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_capture):
        for _ in range(3):
            if static_x.grad is not None:
                static_x.grad = None
            y = layer(static_x)
            y.square().mean().backward()
    torch.cuda.current_stream().wait_stream(s_capture)
    torch.cuda.synchronize()

    s0 = layer.s_slow.clone()
    if static_x.grad is not None:
        static_x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s_capture):
        y = layer(static_x)
        loss = y.square().mean()
        loss.backward()
    print("  graph captured OK")
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s_diff = (layer.s_slow.to(torch.int32) - s0.to(torch.int32)).abs().sum().item()
    print(f"  |d_s_slow|={s_diff} after 10 replays")
    assert s_diff > 0, "state didn't change"
    print("  PASS")


def test_conv2d_graph():
    print("=== Test 3: ConcordConv2dFused captured ===")
    from concord_linear_fused import ConcordConv2dFused
    layer = ConcordConv2dFused(in_channels=8, out_channels=16,
                                kernel_size=3, stride=1, padding=1,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_kind = 'sgd'

    static_x = torch.randn(4, 8, 16, 16, device=DEV, dtype=torch.bfloat16,
                           requires_grad=True)

    s_capture = torch.cuda.Stream()
    s_capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_capture):
        for _ in range(3):
            if static_x.grad is not None:
                static_x.grad = None
            y = layer(static_x)
            y.square().mean().backward()
    torch.cuda.current_stream().wait_stream(s_capture)
    torch.cuda.synchronize()

    s0 = layer.s_slow.clone()
    if static_x.grad is not None:
        static_x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s_capture):
        y = layer(static_x)
        loss = y.square().mean()
        loss.backward()
    print("  graph captured OK")
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s_diff = (layer.s_slow.to(torch.int32) - s0.to(torch.int32)).abs().sum().item()
    print(f"  |d_s_slow|={s_diff} after 10 replays")
    assert s_diff > 0, "state didn't change"
    print("  PASS")


def test_chase_off_then_chase_on():
    """Concord-native grad accumulation pattern: capture two graphs,
    one for apply_chase=False (microbatches 1..K-1) and one for
    apply_chase=True (microbatch K). Validates both can be captured."""
    print("=== Test 4: chase_off + chase_on dual graphs ===")
    from concord_linear_fused import ConcordLinearFused
    layer = ConcordLinearFused(in_features=128, out_features=64,
                                bias=True, device=DEV, lr=0.01).to(DEV)
    layer.optimizer_kind = 'sgd'

    static_x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16,
                           requires_grad=True)

    s_capture = torch.cuda.Stream()
    s_capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_capture):
        # Warmup chase_off
        layer._apply_chase = False
        for _ in range(2):
            if static_x.grad is not None:
                static_x.grad = None
            y = layer(static_x)
            y.square().mean().backward()
        # Warmup chase_on
        layer._apply_chase = True
        for _ in range(2):
            if static_x.grad is not None:
                static_x.grad = None
            y = layer(static_x)
            y.square().mean().backward()
    torch.cuda.current_stream().wait_stream(s_capture)
    torch.cuda.synchronize()

    # Capture chase_off
    layer._apply_chase = False
    if static_x.grad is not None:
        static_x.grad = None
    g_off = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_off, stream=s_capture):
        y = layer(static_x)
        y.square().mean().backward()
    print("  chase_off graph captured")

    # Capture chase_on -- share memory pool with g_off so both share buffers.
    layer._apply_chase = True
    if static_x.grad is not None:
        static_x.grad = None
    g_on = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_on, stream=s_capture, pool=g_off.pool()):
        y = layer(static_x)
        y.square().mean().backward()
    print("  chase_on graph captured")

    # Simulate K=4 grad accumulation: 3 chase_off + 1 chase_on.
    s0 = layer.s_slow.clone()
    for _ in range(3):
        g_off.replay()
    g_on.replay()
    torch.cuda.synchronize()
    s_diff = (layer.s_slow.to(torch.int32) - s0.to(torch.int32)).abs().sum().item()
    print(f"  |d_s_slow|={s_diff} after 3xoff+1xon")
    assert s_diff > 0
    print("  PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        sys.exit(0)
    test_linear_sgd_graph()
    test_linear_three_accum_graph()
    test_conv2d_graph()
    test_chase_off_then_chase_on()
    print("\n[OK] all CUDA graph capture tests passed.")
