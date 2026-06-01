"""User's actual ask: 'calculate a beta1=0.9 momentum from the difference between two
accumulators, compare it to Muon, and diff the steps.' A DIAGNOSTIC, not a build.

Difference-of-EMAs reconstructs momentum (MACD identity). Concord has accumulators at
3 timescales: s_fast (instant velocity), s_slow (chase a=0.1 -> ~10-step EMA = beta1~0.9!),
v_slow (leak a_v=0.001 -> ~1000-step EMA). So a difference of two of them IS a momentum.
Plan:
  1. Train a real Concord layer; maintain a TRUE beta1=0.9 momentum m_ref = EMA_0.9(grad)
     as ground truth (what Muon would orthogonalize).
  2. Reconstruct momentum from accumulator differences; measure which aligns with m_ref.
  3. Muon step = NS5(m_ref). Concord's ACTUAL step = change in deployed weight this step.
  4. DIFF: is Concord's real step aligned with the RAW momentum (m_ref) or the ORTHOGONALIZED
     one (NS5 m_ref)? i.e. is Concord already ~Muon, ~momentum-SGD, or neither?
"""
import sys; sys.path.insert(0,'src')
import torch, torch.nn.functional as F
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
def al(A,B):  # cosine alignment of two matrices (flattened)
    a=A.flatten().float(); b=B.flatten().float()
    return (a@b/(a.norm()*b.norm()+1e-12)).item()

m=ConcordLinearPackedB(384,384,bias=False,device=dev); m.lr=5e-4
Wt=(torch.randn(384,32,device=dev)@torch.randn(32,384,device=dev))*0.3
def W_dep():  # deployed weight (s_slow+v_slow), in mantissa*scale units (drop the 2^exp, constant here)
    pw=m.packed_w; ss=((pw<<16)>>24).float(); vs=((pw<<24)>>24).float()
    return ss*S_SLOW_FACTOR+vs*V_SLOW_FACTOR, ss, vs
m_ref=torch.zeros(384,384,device=dev)   # true beta1=0.9 momentum of the grad
beta=0.9
prevW=None
rows=[]
for t in range(400):
    X=torch.randn(128,384,device=dev); Y=X@Wt.T
    xb=X.detach().requires_grad_(True); loss=F.mse_loss(m(xb),Y)
    # grab grad_W the same way the kernel forms it (grad wrt the linear weight)
    g=torch.autograd.grad(loss, xb, create_graph=False, retain_graph=True)[0]  # placeholder; need dW
    loss.backward(); m.rebalance()
    # true momentum EMA of the weight-grad (approx via finite-diff of deployed weight is cleaner):
    Wd,ss,vs=W_dep()
    if prevW is not None:
        dW = Wd - prevW                     # actual per-step deployed-weight delta
        m_ref = beta*m_ref + (1-beta)*(-dW) # momentum of the "gradient" (-dW direction), beta1=0.9
        if t>50 and t%50==0:
            d_sv = ss - vs                  # candidate momentum: slow - vslow (long)
            # diff the steps:
            muon = ns5(m_ref)               # what Muon would apply
            rows.append((t,
                al(d_sv, m_ref),            # does slow-vslow reconstruct the 0.9 momentum?
                al(dW, m_ref),              # is Concord's actual step aligned w/ raw momentum?
                al(dW, muon),               # ...or with the Muon (orthogonalized) step?
                al(m_ref, muon)))           # how different ARE raw vs Muon (the orthogonalization)
    prevW=Wd

print("  t   | align(d_sv,m_ref) | align(step,raw_mom) | align(step,Muon) | align(raw,Muon)")
for r in rows:
    print(f" {r[0]:4} |      {r[1]:+.3f}       |       {r[2]:+.3f}        |     {r[3]:+.3f}      |     {r[4]:+.3f}")
print("\n[READ] col2: does the accumulator-difference reconstruct the 0.9 momentum?")
print("       col3 vs col4: is Concord's ACTUAL step closer to raw-momentum-SGD or to Muon?")
print("       col5: how far apart are the raw and Muon steps (low = orthogonalization changes a lot)")
