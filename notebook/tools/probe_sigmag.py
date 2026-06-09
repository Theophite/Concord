"""Verify the cheap centered-Sigma_g noise construction. grad_W = sum_i grad_y[i] outer x[i].
Centered draw: noise = (eps*gy)^T @ x - (sum eps)*gbar, eps~N(0,1). Checks: one-matmul ==
explicit; centered (E[noise]~0); shaped (cov NOT isotropic vs white)."""
import torch
dev='cuda'; torch.manual_seed(0)
M,N,K=128,64,48
gy=torch.randn(M,N,device=dev); x=torch.randn(M,K,device=dev)
gradW=gy.transpose(0,1)@x; gbar=gradW/M
def draw_fast(eps):
    return (eps[:,None]*gy).transpose(0,1)@x - eps.sum()*gbar
def draw_explicit(eps):
    gi=torch.einsum('mn,mk->mnk', gy, x)          # [M,N,K] per-token outer (vectorized)
    return torch.einsum('m,mnk->nk', eps, gi - gbar[None])
eps=torch.randn(M,device=dev)
print(f"(a) one-matmul == explicit: max|diff|={(draw_explicit(eps)-draw_fast(eps)).abs().max().item():.2e}")
T=300; acc=torch.zeros(N,K,device=dev)
for _ in range(T): acc+=draw_fast(torch.randn(M,device=dev))
print(f"(b) centered: ||mean||/||gradW||={(acc/T).norm()/gradW.norm():.4f}  (want <<1)")
draws=torch.stack([draw_fast(torch.randn(M,device=dev)) for _ in range(200)]).reshape(200,-1)
s=torch.linalg.svdvals(torch.cov(draws.T)); s=s/s.max()
w=torch.randn(200,N*K,device=dev); sw=torch.linalg.svdvals(torch.cov(w.T)); sw=sw/sw.max()
er=lambda v:(v.sum()**2/(v**2).sum()).item()
print(f"(c/d) Sigma_g noise cov eff_rank={er(s):.0f}/{N*K} vs white={er(sw):.0f}/{N*K} (<< = shaped, good)")
