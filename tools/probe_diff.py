"""Eager is bit-deterministic (probe_determ: 0.0). So graphed-vs-eager 0.30 is a REAL
desync. Diff every Concord buffer eager-vs-graphed after K steps to find the culprit.
Run identical setups; eager does K plain steps; graphed captures then replays K times
from the SAME restored start; compare all buffers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch, torch.nn as nn, torch.nn.functional as F
import prototype_packed_b as ppb
from prototype_packed_b import ConcordLinearPackedB
dev="cuda"
def build():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    m=nn.Sequential(ConcordLinearPackedB(64,128,bias=False,device=dev),nn.GELU(),
                    ConcordLinearPackedB(128,64,bias=False,device=dev))
    for l in m:
        if hasattr(l,"lr"): l.lr=0.02
    return m
def concords(m): return [l for l in m if hasattr(l,"rebalance")]
torch.manual_seed(1); X=torch.randn(256,64,device=dev); Y=torch.randn(256,64,device=dev)*0.3
BUFS=["packed_w","row_exp","col_exp","v_row","v_col","_sum_v_inv","_bf16_weight_buf",
      "_row_max_buf","_col_max_buf","_reb_seed"]
K=5
# eager
sc=ppb._get_step_counter(torch.device(dev)); sc.zero_()
m1=build()
for i in range(K):
    x=X.detach().requires_grad_(True); F.mse_loss(m1(x),Y).backward()
eager_state=[{nm:getattr(l,nm).clone() for nm in BUFS if isinstance(getattr(l,nm,None),torch.Tensor)} for l in concords(m1)]
# graphed
sc.zero_()
m2=build()
sx=torch.zeros(256,64,device=dev,requires_grad=True)
def fb():
    if sx.grad is not None: sx.grad=None
    F.mse_loss(m2(sx),Y).backward()
snap=[{nm:getattr(l,nm).clone() for nm in BUFS if isinstance(getattr(l,nm,None),torch.Tensor)} for l in concords(m2)]
sc0=sc.clone()
sx.data.copy_(X)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(5): fb()
torch.cuda.current_stream().wait_stream(s)
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g): fb()
# restore start state
for l,d in zip(concords(m2),snap):
    for nm,v in d.items(): getattr(l,nm).copy_(v)
sc.copy_(sc0)
for i in range(K):
    sx.data.copy_(X); g.replay()
# compare
print(f"after K={K} steps, max abs buffer diff eager-vs-graphed:")
for li,(l,ed) in enumerate(zip(concords(m2),eager_state)):
    for nm in ed:
        d=(getattr(l,nm).float()-ed[nm].float()).abs().max().item()
        flag = "  <-- DIFF" if d>1e-6 else ""
        if d>1e-6 or nm=="packed_w":
            print(f"  layer{li} {nm:18} maxdiff={d:.6g}{flag}")
