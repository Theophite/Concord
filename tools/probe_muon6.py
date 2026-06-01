"""Clean version (probe_muon5 was circular: it built m_ref from dW then correlated dW to it).
Here m_ref = EMA_0.9 of the TRUE grad_W (= grad_y^T x, exactly what the kernel forms). Then
diff three steps, all against the SAME ground-truth gradient momentum m_ref:
  - Muon step      : NS5(m_ref)              (what Muon applies)
  - Concord drift  : d_sv = s_slow - v_slow  (the direction the cascade actually commits)
  - raw momentum   : m_ref                   (plain heavy-ball SGD step)
Questions: (a) does Concord's d_sv align with the gradient momentum at all? (b) how different
is the Muon (orthogonalized) step from both? Sign-clean: compare directions of the UPDATE
(-grad), so positive alignment = same direction.
"""
import sys; sys.path.insert(0,'src')
import torch, torch.nn as nn, torch.nn.functional as F
from prototype_packed_b import ConcordLinearPackedB, S_SLOW_FACTOR, V_SLOW_FACTOR
dev='cuda'; torch.manual_seed(0)
def ns5(M,n=5):
    a,b,c=3.4445,-4.7750,2.0315
    X=(M/(M.norm()+1e-7)).bfloat16(); tr=X.shape[0]>X.shape[1]
    if tr:X=X.t()
    for _ in range(n):
        A=X@X.t(); X=a*X+(b*A+c*(A@A))@X
    if tr:X=X.t()
    return X.float()
def al(A,B):
    a=A.flatten().float(); b=B.flatten().float(); return (a@b/(a.norm()*b.norm()+1e-12)).item()
def erank(M):
    s=torch.linalg.svdvals(M.float()); return (s.sum()**2/(s**2).sum()).item()

m=ConcordLinearPackedB(384,384,bias=False,device=dev); m.lr=5e-4
Wt=(torch.randn(384,32,device=dev)@torch.randn(32,384,device=dev))*0.3
mom=torch.zeros(384,384,device=dev); beta=0.9
def state():
    pw=m.packed_w; ss=((pw<<16)>>24).float(); vs=((pw<<24)>>24).float(); sf=(pw>>16).float()
    return sf,ss,vs
rows=[]
for t in range(400):
    X=torch.randn(128,384,device=dev); Y=X@Wt.T
    xb=X.detach().requires_grad_(True); yb=m(xb); loss=F.mse_loss(yb,Y)
    # capture TRUE grad_W = d loss / d W (same product the kernel uses), via a paired nn.Linear
    gy=torch.autograd.grad(loss, yb, retain_graph=True)[0]          # grad wrt output
    gW = gy.reshape(-1,384).t() @ xb.detach().reshape(-1,384)        # grad_y^T x = grad_W
    mom = beta*mom + (1-beta)*gW                                    # true beta1=0.9 momentum
    loss.backward(); m.rebalance()
    if t>50 and t%70==0:
        sf,ss,vs=state(); d_sv=ss-vs
        upd_mom = -mom                       # heavy-ball update direction
        upd_muon= -ns5(mom)                  # Muon update direction
        rows.append((t,
            al(d_sv, upd_mom),               # does the cascade's committed drift ~ raw momentum?
            al(d_sv, upd_muon),              # ...or the orthogonalized (Muon) momentum?
            al(upd_mom, upd_muon),           # how much NS5 changes the step
            erank(mom), erank(ns5(mom))))    # eff-rank: momentum vs orthogonalized
print(" t   | al(d_sv,rawMom) | al(d_sv,Muon) | al(raw,Muon) | erank mom -> NS5")
for r in rows:
    print(f"{r[0]:4} |     {r[1]:+.3f}      |    {r[2]:+.3f}     |    {r[3]:+.3f}     |  {r[4]:.0f} -> {r[5]:.0f}")
print("\n[READ] al(d_sv,rawMom): is Concord's committed direction the gradient momentum?")
print("       al(raw,Muon)+erank jump: how much would orthogonalization CHANGE the step.")
