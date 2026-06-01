"""Buffers proven bit-identical eager-vs-graphed (probe_diff). Now compare the LOSS
trajectory the harness way: read static_loss.item() each replay vs eager loss each step.
If these match too, the graph is FULLY correct and probe_graph4's 0.30 was a measurement
artifact (grad/loss aliasing in that probe)."""
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
BUFS=["packed_w","row_exp","col_exp","v_row","v_col","_sum_v_inv","_bf16_weight_buf","_row_max_buf","_col_max_buf","_reb_seed"]
N=60
sc=ppb._get_step_counter(torch.device(dev)); sc.zero_()
m1=build(); eager=[]
for i in range(N):
    x=X.detach().requires_grad_(True); l=F.mse_loss(m1(x),Y); l.backward(); eager.append(l.item())
sc.zero_(); m2=build()
sx=torch.zeros(256,64,device=dev,requires_grad=True)
sl={"v":None}
def fb():
    if sx.grad is not None: sx.grad=None
    out=m2(sx); loss=F.mse_loss(out,Y); loss.backward(); sl["v"]=loss
snap=[{nm:getattr(l,nm).clone() for nm in BUFS if isinstance(getattr(l,nm,None),torch.Tensor)} for l in concords(m2)]
sc0=sc.clone(); sx.data.copy_(X)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(5): fb()
torch.cuda.current_stream().wait_stream(s)
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g): fb()
for l,d in zip(concords(m2),snap):
    for nm,v in d.items(): getattr(l,nm).copy_(v)
sc.copy_(sc0)
graphed=[]
for i in range(N):
    sx.data.copy_(X); g.replay(); graphed.append(sl["v"].item())
md=max(abs(a-b) for a,b in zip(eager,graphed))
print(f"eager  {eager[0]:.5f}->{eager[-1]:.5f}")
print(f"graphd {graphed[0]:.5f}->{graphed[-1]:.5f}")
print(f"max |eager-graphed| loss over {N} = {md:.8f}")
print("[VERDICT] " + ("MATCH -- graph is FULLY CORRECT (probe4's 0.30 was an artifact)."
                      if md<1e-3 else f"diff {md:.4f} remains."))
