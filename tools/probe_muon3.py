"""Muon take 3: is d_sv = s_slow - v_slow the RIGHT orthogonalization target? (user's idea:
orthogonalize between slow and v_slow, not at the chase.)

Claim: d_sv is (1) DENOISED (consistent drift; s_fast is raw signal+noise), (2) SUPER-QUANTUM
by construction (int8 - int8 = integer units -> survives SR per probe_muon2), (3) the gate's
own signal sig=C*d_sv. So it's the meaningful momentum to orthogonalize.

Test: inject a KNOWN fixed low-rank signal G_sig each step + fresh noise. Run the real
cascade (chase alpha=0.1, leak alpha_v=0.001, all SR-int8). Measure, for s_fast vs d_sv:
  - subspace alignment with the TRUE signal G_sig (how denoised / on-signal is it?)
  - is it super-quantum (element magnitude in int8 units)?
  - does NS5 of it survive re-quantization?
If d_sv aligns far better with the true signal AND is super-quantum, the slow<->v_slow
boundary is the correct home for orthogonalization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
dev = "cuda"; torch.manual_seed(0)

def ns5(G, steps=5):
    a,b,c=3.4445,-4.7750,2.0315
    X=G.bfloat16(); X=X/(X.norm()+1e-7)
    tr=X.shape[0]>X.shape[1]
    if tr: X=X.T
    for _ in range(steps):
        A=X@X.T; X=a*X+(b*A+c*(A@A))@X
    if tr: X=X.T
    return X.float()

def subspace_align(M, U_sig):
    """fraction of M's energy in the true signal's column space U_sig (r x .)."""
    M=M.float()
    if M.norm()<1e-9: return 0.0
    proj = U_sig @ (U_sig.T @ M)        # project onto signal subspace
    return (proj.norm()**2 / (M.norm()**2 + 1e-12)).item()

N, K, R = 384, 384, 16
# fixed signal direction (rank R), unit-ish
Us,_ = torch.linalg.qr(torch.randn(N, R, device=dev))   # orthonormal signal basis [N,R]
Vs,_ = torch.linalg.qr(torch.randn(K, R, device=dev))
G_sig = (Us @ Vs.T)                                       # the TRUE consistent gradient dir
G_sig = G_sig / G_sig.abs().mean()                        # O(1) elements

alpha, alpha_v, SCALE = 0.1, 0.001, 128.0
torch.manual_seed(1); gg=torch.Generator(device=dev); gg.manual_seed(7)
s_fast=torch.zeros(N,K,device=dev); s_slow=torch.zeros(N,K,device=dev); v_slow=torch.zeros(N,K,device=dev)
SIG_AMP, NOISE_AMP = 40.0, 40.0
for t in range(1500):
    s_fast = s_fast + SIG_AMP*G_sig + NOISE_AMP*torch.randn(N,K,device=dev,generator=gg)
    # chase alpha*s_fast -> s_slow (SR int8)
    ch=alpha*s_fast/SCALE; fl=torch.floor(ch); tk=fl+(torch.rand(N,K,device=dev,generator=gg)<(ch-fl)).float()
    s_slow=s_slow+tk; s_fast=s_fast-tk*SCALE
    # leak alpha_v*(s_slow-v_slow) -> v_slow (SR int8)
    gap=alpha_v*(s_slow-v_slow); fl2=torch.floor(gap); lk=fl2+(torch.rand(N,K,device=dev,generator=gg)<(gap-fl2)).float()
    v_slow=v_slow+lk

d_sv = s_slow - v_slow
print("=== alignment with TRUE signal subspace (higher = more denoised / on-signal) ===")
print(f"  s_fast (raw velocity) : align={subspace_align(s_fast, Us):.3f}   |elem|_mean={s_fast.abs().mean():.1f} mantissa ({s_fast.abs().mean()/SCALE:.2f} int8u)")
print(f"  s_slow (position)     : align={subspace_align(s_slow, Us):.3f}   |elem|_mean={s_slow.abs().mean():.1f} int8u")
print(f"  d_sv  (slow - v_slow) : align={subspace_align(d_sv,  Us):.3f}   |elem|_mean={d_sv.abs().mean():.2f} int8u (super-quantum if >~1)")
print(f"  G_sig (truth)         : align={subspace_align(G_sig, Us):.3f}")

def sv_flat(M):
    s=torch.linalg.svdvals(M.float()); s=s/(s.max()+1e-12); return s.mean().item(),(s>0.1).float().mean().item()
print("\n=== orthogonalize d_sv, re-quantize at its native int8 magnitude: survives? ===")
m0,f0=sv_flat(d_sv); print(f"  d_sv raw spectrum:   mean_sv/max={m0:.3f} frac>0.1={f0:.2f}")
O=ns5(d_sv)*d_sv.abs().mean()                      # orthogonalized, scaled back to int8 magnitude
g2=torch.Generator(device=dev); g2.manual_seed(3)
fl=torch.floor(O); Q=fl+(torch.rand(N,K,device=dev,generator=g2)<(O-fl)).float()  # SR re-quantize
m1,f1=sv_flat(O); m2,f2=sv_flat(Q)
print(f"  NS5(d_sv):           mean_sv/max={m1:.3f} frac>0.1={f1:.2f}")
print(f"  NS5(d_sv) re-quant:  mean_sv/max={m2:.3f} frac>0.1={f2:.2f}  ({'survives' if m2>0.6*m1 else 'destroyed'})")
print(f"  align of NS5(d_sv) with true signal: {subspace_align(O, Us):.3f} (orthogonalization should keep the subspace, flatten within it)")
