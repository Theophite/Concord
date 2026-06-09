"""Fast, isolated test of whether the control-plane embedding is CUDA-graph
capturable -- specifically the branch-free forward AND the nested-autograd self-step
in _PackedEmbStep.backward (y.backward(G.t()) inside a custom Function.backward, never
exercised under capture before). Tiny tensors -> seconds, not a 5-min trainer run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn

from control_plane import ControlPlaneEmbedding

dev, dt = torch.device("cuda"), torch.bfloat16

base = nn.Embedding(100, 16).to(dev, dt)
cp = ControlPlaneEmbedding(base)
cp.set_zero(50)                                              # static -> zero
cp.attach_trainable([51], torch.randn(1, 16, device=dev), lr=5e-2, target_norm=1.0)

ids = torch.tensor([[51, 50, 10, 51, 7]], device=dev)       # train, static, base, train, base


def step():
    out = cp(ids)
    loss = out.float().pow(2).mean()
    loss.backward()
    return loss


# side-stream warmup, then capture
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        step()
torch.cuda.current_stream().wait_stream(s)

try:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap_loss = step()
    print("[capture] control-plane embedding captured OK (forward + nested-autograd self-step)")
    before = cp.trainable.deploy_weight().float().clone()
    g.replay(); g.replay()
    after = cp.trainable.deploy_weight().float().clone()
    moved = (after - before).norm().item()
    dn = after.norm().item()
    print(f"[replay] trainable embedding moved {moved:.4e} over 2 replays | "
          f"deploy norm {dn:.3f} (target 1.0)")
    print("[RESULT] EMBEDDING IS CAPTURABLE -> trainer graph can include the TE/token"
          if moved > 0 else "[RESULT] captured but token not moving (check self-step)")
except Exception as e:
    import traceback
    print(f"[capture] FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("[RESULT] nested autograd not capturable -> need a direct-step driver in _PackedEmbStep")
