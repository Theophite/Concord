"""I orthogonalized the WRONG matrix. Muon orthogonalizes the MOMENTUM and applies it as
the weight delta. In Concord momentum = s_fast (SR-ticks accumulate there, beta1 decay).
I instead orthogonalized d_sv = s_slow - v_slow = a DENOISED slow drift, NOT momentum.
probe_muon3 found d_sv low-rank -> "NS5 destructive" -- but that's because d_sv is the
denoised (concentrated) signal. The MOMENTUM s_fast carries noise -> should be FULL-RANK =
exactly what NS5 wants. Test: train a real ConcordLinear, compare the singular spectrum of
s_fast (momentum, Muon's true target) vs d_sv (what I wrongly orthogonalized)."""
import sys; sys.path.insert(0,'src')
import torch, torch.nn.functional as F
from prototype_packed_b import ConcordLinearPackedB
dev='cuda'; torch.manual_seed(0)
m=ConcordLinearPackedB(384,384,bias=False,device=dev); m.lr=5e-4
# realistic: noisy minibatch grads (LM-like: low-rank signal + lots of noise)
Wt=(torch.randn(384,16,device=dev)@torch.randn(16,384,device=dev))*0.3   # rank-16 target
def batch():
    X=torch.randn(128,384,device=dev); return X, X@Wt.T
for _ in range(300):
    X,Y=batch(); xb=X.detach().requires_grad_(True); F.mse_loss(m(xb),Y).backward(); m.rebalance()
sf,ss,vs=m.get_state()
sf=sf.float(); dsv=(ss.float()-vs.float())
def spec(M,name):
    s=torch.linalg.svdvals(M.float()); s=s/(s.max()+1e-12)
    eff_rank=(s.sum()**2/(s**2).sum()).item()      # participation ratio (stable-rank-ish)
    print(f"  {name:22} mean_sv/max={s.mean():.3f}  frac>0.1={(s>0.1).float().mean():.2f}  "
          f"eff_rank={eff_rank:.0f}/{len(s)}  |elem|={M.abs().mean():.1f}")
print("singular spectrum (full-rank = mean/max & eff_rank high = NS5-appropriate):")
spec(sf,  "s_fast (MOMENTUM)")     # Muon's true target
spec(dsv, "d_sv (what I orthog'd)")# the denoised drift I wrongly used
print("[READ] if s_fast is HIGHER-rank than d_sv, I orthogonalized the NS5-hostile matrix;")
print("       the momentum s_fast is the right Muon target and was never tested.")
