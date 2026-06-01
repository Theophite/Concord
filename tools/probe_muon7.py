"""Rank-aware orthogonalization (user's idea): orthogonalize ONLY within the update's
actual rank, instead of full Muon (which drives all 384 singular values to 1 and pumps
~349 noise dirs -> probe_muon3 showed signal alignment collapses 0.999->0.044).

Target = U_k Vh_k (rank-k semi-orthogonal: top-k signal dirs equalized to 1, rest at 0).
Question (load-bearing, cheap): does rank-k ortho PRESERVE the true-signal subspace that
full NS5 destroys, while still equalizing (flattening) within it?

Setup mirrors probe_muon3: inject a KNOWN rank-R signal + noise, run the real chase+leak
cascade, then orthogonalize d_sv = s_slow - v_slow three ways and measure alignment with
the true signal subspace + the resulting spectrum.
"""
import sys; sys.path.insert(0,'src')
import torch
dev='cuda'; torch.manual_seed(0)

def ns5(M,n=5):
    a,b,c=3.4445,-4.7750,2.0315
    X=(M/(M.norm()+1e-7)).bfloat16(); tr=X.shape[0]>X.shape[1]
    if tr:X=X.t()
    for _ in range(n):
        A=X@X.t(); X=a*X+(b*A+c*(A@A))@X
    if tr:X=X.t()
    return X.float()

def rank_k_orth(M,k):
    """exact rank-k semi-orthogonal = U_k Vh_k (truncated-SVD orthogonalization)."""
    U,S,Vh=torch.linalg.svd(M.float(),full_matrices=False)
    return U[:,:k] @ Vh[:k,:]

def subspace_align(M,U_sig):
    M=M.float()
    if M.norm()<1e-9: return 0.0
    return ((U_sig@(U_sig.t()@M)).norm()**2/(M.norm()**2+1e-12)).item()
def erank(M):
    s=torch.linalg.svdvals(M.float()); return (s.sum()**2/(s**2).sum()).item()

N,K,R=384,384,16
Us,_=torch.linalg.qr(torch.randn(N,R,device=dev))     # TRUE signal left-subspace [N,R]
Vs,_=torch.linalg.qr(torch.randn(K,R,device=dev))
G_sig=(Us@Vs.t()); G_sig=G_sig/G_sig.abs().mean()
alpha,alpha_v,SCALE=0.1,0.001,128.0
gg=torch.Generator(device=dev); gg.manual_seed(7)
s_fast=torch.zeros(N,K,device=dev); s_slow=torch.zeros(N,K,device=dev); v_slow=torch.zeros(N,K,device=dev)
for t in range(1500):
    s_fast=s_fast+40*G_sig+40*torch.randn(N,K,device=dev,generator=gg)
    ch=alpha*s_fast/SCALE; fl=torch.floor(ch); tk=fl+(torch.rand(N,K,device=dev,generator=gg)<(ch-fl)).float()
    s_slow=s_slow+tk; s_fast=s_fast-tk*SCALE
    gap=alpha_v*(s_slow-v_slow); fl2=torch.floor(gap); lk=fl2+(torch.rand(N,K,device=dev,generator=gg)<(gap-fl2)).float()
    v_slow=v_slow+lk
d_sv=s_slow-v_slow
print(f"true signal rank R={R}.  metric: subspace_align with true signal (1=perfect) | eff_rank")
for name,M in [("raw d_sv", d_sv),
               ("full NS5 (Muon)", ns5(d_sv)),
               (f"rank-k ortho (k=R={R})", rank_k_orth(d_sv,R)),
               ("rank-k ortho (k=2R)", rank_k_orth(d_sv,2*R)),
               ("rank-k ortho (k=64)", rank_k_orth(d_sv,64))]:
    print(f"  {name:24} align={subspace_align(M,Us):.3f}  eff_rank={erank(M):.0f}")
print("\n[READ] full NS5 should have LOW align (~0.04, pumps noise to 384 dirs). rank-k should")
print("       KEEP align high (only touches the top-k signal subspace) -> idea is mechanically")
print("       sound, worth a training arm. If rank-k align is also low, the idea is dead.")
