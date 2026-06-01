"""Is the ortho_k50 -0.2 result confounded by INSTANTANEOUS weight corruption (rewriting
v_slow directly perturbs the deployed weight) rather than a clean test of orthogonalized
UPDATES? Train a ConcordLinear to fit a target, measure MSE, call orthogonalize_slow ONCE
(no training), measure MSE immediately after. A big instant jump => the test measured
state-scrambling, not optimization dynamics."""
import sys; sys.path.insert(0,'src')
import torch, torch.nn.functional as F
from prototype_packed_b import ConcordLinearPackedB, S_SLOW_FACTOR, V_SLOW_FACTOR
dev='cuda'; torch.manual_seed(0)
m=ConcordLinearPackedB(256,256,bias=False,device=dev); m.lr=0.02
W=torch.randn(256,256,device=dev)*0.3
X=torch.randn(512,256,device=dev); Y=X@W.T
for _ in range(400):
    xb=X.detach().requires_grad_(True); F.mse_loss(m(xb),Y).backward(); m.rebalance()
def mse(): 
    with torch.no_grad(): return F.mse_loss(m(X),Y).item()
def dsv_align():
    pw=m.packed_w; ss=((pw<<16)>>24).float(); vs=((pw<<24)>>24).float()
    d=ss-vs
    s=torch.linalg.svdvals(d); s=s/(s.max()+1e-12)
    return d, round(s.mean().item(),3)
pre=mse(); d_pre,sp_pre=dsv_align()
# capture deployed weight before
wv_pre = m.consolidated_weight().float().clone()
m.orthogonalize_slow()
post=mse(); d_post,sp_post=dsv_align()
wv_post = m.consolidated_weight().float()
dw = (wv_post-wv_pre).norm()/(wv_pre.norm()+1e-9)
print(f"MSE   immediately before ortho: {pre:.5f}")
print(f"MSE   immediately AFTER ortho : {post:.5f}   (jump x{post/pre:.0f})")
print(f"d_sv spectrum mean_sv/max: {sp_pre} -> {sp_post}  (flattened = orthogonalized)")
print(f"deployed-weight relative change from ONE ortho call: {dw:.3f}  (||dW||/||W||)")
print(f"[READ] big MSE jump + big ||dW|| from ONE call (no training) => the -0.2 A/B was")
print(f"       measuring WEIGHT CORRUPTION, not orthogonalized-update dynamics. Test invalid.")
