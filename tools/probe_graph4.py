"""Custom SINGLE-graph capture of fwd+loss+bwd together (Concord). The make_graphed
path diverges because it captures fwd and bwd as SEPARATE graphs with reused static
buffers, severing Concord's forward-reads-_bf16_weight_buf <- backward-writes-it
coupling. A single graph wrapping fwd+bwd keeps that buffer in-place across the
read/write within each replay: replay does read(W_t)->...->write(W_{t+1}); next replay
read(W_{t+1}). That is exactly Concord's intended dynamic.

Blocker to solve: raw torch.cuda.graph around loss.backward() threw "operation would
make the legacy stream depend on a capturing blocking stream" (autograd engine runs on
its own threads/legacy stream). Try the documented pattern: warm up on a side stream
(incl a full fwd+bwd so autograd lazy-inits + Triton autotunes), THEN capture with
torch.cuda.graph(g) which manages the capture stream. Static input buffer; static loss.

If MATCH: this is the harness pattern. If still the legacy-stream error: print it.
Run: python tools/probe_graph4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch, torch.nn as nn, torch.nn.functional as F
from prototype_packed_b import ConcordLinearPackedB

dev = "cuda"

def build():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    m = nn.Sequential(
        ConcordLinearPackedB(64, 128, bias=False, device=dev),
        nn.GELU(),
        ConcordLinearPackedB(128, 64, bias=False, device=dev),
    )
    for layer in m:
        if hasattr(layer, "lr"): layer.lr = 0.02
    return m

def concords(m): return [l for l in m if hasattr(l, "rebalance")]

torch.manual_seed(1)
X = torch.randn(256, 64, device=dev)
Y = torch.randn(256, 64, device=dev) * 0.3
N = 60

# ---- EAGER ref (no rebalance, to match probe2's clean comparison) ----
m1 = build()
eager = []
for i in range(N):
    x = X.detach().requires_grad_(True)
    loss = F.mse_loss(m1(x), Y); loss.backward()
    eager.append(loss.item())
print(f"[eager] {eager[0]:.5f} -> {eager[-1]:.5f}")

# ---- CUSTOM single-graph capture ----
m2 = build()
static_x = torch.zeros(256, 64, device=dev, requires_grad=True)

def fwd_bwd():
    # zero the input grad in-place (static), fwd, loss, bwd. Concord layers have no
    # params so no aux grads here. loss kept as a static tensor.
    if static_x.grad is not None:
        static_x.grad = None
    out = m2(static_x)
    loss = F.mse_loss(out, Y)
    loss.backward()
    return loss

try:
    # The warmup fwd+bwd passes AND the capture-recording pass each execute a REAL
    # Concord step (the optimizer is fused in backward -> mutates packed_w). So they
    # over-step the weights before replays begin. Fix: snapshot ALL mutable Concord
    # buffers, run warmup + capture, then RESTORE so replay[0] starts from the same
    # state eager started from. (In the harness we instead just let the warmup/capture
    # passes BE real steps and account for them; here we restore for a clean A/B.)
    buf_names = ["packed_w", "row_exp", "col_exp", "v_row", "v_col", "_sum_v_inv",
                 "_bf16_weight_buf", "_row_max_buf", "_col_max_buf", "_reb_seed",
                 "_row_max_hwm", "_col_max_hwm"]
    def snapshot():
        snap = []
        for l in concords(m2):
            d = {}
            for nm in buf_names:
                b = getattr(l, nm, None)
                if isinstance(b, torch.Tensor): d[nm] = b.clone()
            snap.append(d)
        return snap
    def restore(snap):
        for l, d in zip(concords(m2), snap):
            for nm, v in d.items():
                getattr(l, nm).copy_(v)
    # also reset the global step counter so SR rng matches eager
    import prototype_packed_b as ppb
    sc = ppb._get_step_counter(torch.device(dev))

    static_x.data.copy_(X)
    snap0 = snapshot(); sc0 = sc.clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fwd_bwd()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    static_loss = None
    with torch.cuda.graph(g):
        static_loss = fwd_bwd()
    print("[capture] OK")

    restore(snap0); sc.copy_(sc0)   # rewind warmup + capture side effects
    gl = []
    for i in range(N):
        static_x.data.copy_(X)   # same fixed data each step (matches eager)
        g.replay()
        gl.append(static_loss.item())
    print(f"[graphed] {gl[0]:.5f} -> {gl[-1]:.5f}")
    md = max(abs(a-b) for a, b in zip(eager, gl))
    print(f"[compare] max |eager-graphed| over {N} = {md:.6f}")
    print("[VERDICT] " + ("MATCH -> custom single-graph capture WORKS. Port to harness."
                          if md < 1e-2 else
                          "DIVERGE -> single graph still not preserving the step. Inspect."))
except Exception as e:
    import traceback
    print("[FAILED]", type(e).__name__)
    print("\n".join(traceback.format_exc().strip().split("\n")[-12:]))
