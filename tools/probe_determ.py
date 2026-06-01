"""Is Concord even DETERMINISTIC run-to-run (same seed)? If two identical eager runs
differ by ~0.30, then the graphed-vs-eager 0.30 is just SR realization noise and the
CUDA graph is ALREADY CORRECT (no point chasing bit-exactness). SR is seeded by the
global step_counter (device tensor), which is a PROCESS-GLOBAL singleton -> two runs in
one process do NOT reset it, but two fresh builds share it. Test: build+run twice in the
same process, compare."""
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
torch.manual_seed(1); X=torch.randn(256,64,device=dev); Y=torch.randn(256,64,device=dev)*0.3
def run():
    # reset global step counter so SR starts identically
    sc=ppb._get_step_counter(torch.device(dev)); sc.zero_()
    m=build(); out=[]
    for i in range(60):
        x=X.detach().requires_grad_(True)
        loss=F.mse_loss(m(x),Y); loss.backward(); out.append(loss.item())
    return out
a=run(); b=run()
md=max(abs(x-y) for x,y in zip(a,b))
print(f"run A: {a[0]:.5f}->{a[-1]:.5f}   run B: {b[0]:.5f}->{b[-1]:.5f}")
print(f"max |A-B| (eager vs eager, same seed) = {md:.6f}")
print("[INTERP] " + ("eager is NON-deterministic by ~%.2f -> the graphed-vs-eager 0.30 is "
      "SR noise; GRAPH IS CORRECT." % md if md>0.05 else
      "eager IS deterministic (<0.05) -> the graphed 0.30 is a REAL desync (missing "
      "restore buffer); debug which buffer."))
