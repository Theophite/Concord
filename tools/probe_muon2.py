"""Muon, take 2: does orthogonality survive int8-SR quantization IF the tick is large
enough to quantize? (User's fix to probe_muon's sub-quantum failure.)

probe_muon failed because the per-step chase moved alpha*src/128 ~ 0.1 int8-units/element
-> each element SR-rounds to 0/1 independently -> the relative (orthogonal) structure
becomes rounding noise. Hypothesis: orthogonality is preserved if the quantized tick is
LARGE relative to the int8 quantum (128), i.e. accumulate in s_fast (int16, fine) and chase
rarely/large so int8 rounding error is small vs signal.

Test A (isolation): take ONE orthogonalized matrix O (flat spectrum). SR-quantize at scale s
(tick magnitude ~s int8-units/elem): Q = SR(O * s) / s. Sweep s, measure spectrum flatness.
Find the threshold s* where the orthogonal spectrum survives.

Test B (cascade): chase PERIODICALLY (accumulate in s_fast for K steps, chase once) so the
per-chase tick crosses s*. Does s_slow's spectrum stay flat for large K (orth vs not)?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
dev = "cuda"
torch.manual_seed(0)

def ns5(G, steps=5):
    a,b,c = 3.4445,-4.7750,2.0315
    X = G.bfloat16(); X = X/(X.norm()+1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        A = X@X.T; X = a*X + (b*A + c*(A@A))@X
    if tr: X = X.T
    return X.float()

def sv_flat(M):
    s = torch.linalg.svdvals(M.float()); s = s/(s.max()+1e-12)
    return s.mean().item(), (s>0.1).float().mean().item()

def sr_quant(M, gen):
    """int8-style SR quantization: round to integer units (the quantum), stochastically."""
    fl = torch.floor(M); fr = M - fl
    return fl + (torch.rand(M.shape, device=dev, generator=gen) < fr).float()

N, K = 384, 384
O = ns5(torch.randn(N, K, device=dev))          # flat-spectrum orthogonal matrix
O = O / O.abs().mean()                            # elements O(1) so 'scale s' = int8-units/elem
m0, f0 = sv_flat(O)
print(f"=== Test A: SR-quantize orthogonal O at tick-scale s (s = int8 units/element) ===")
print(f"  O (continuous):  mean_sv/max={m0:.3f}  frac>0.1={f0:.2f}")
g = torch.Generator(device=dev); g.manual_seed(1)
for s in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
    Q = sr_quant(O * s, g) / s
    m, f = sv_flat(Q)
    tag = "  <- structure back" if m > 0.6*m0 else ("  (destroyed)" if m < 0.45 else "")
    print(f"  s={s:6.1f}: mean_sv/max={m:.3f}  frac>0.1={f:.2f}{tag}")
print("  (sub-quantum s<~1 destroys; s>~10 should recover O's flat spectrum -> threshold s*)")

print(f"\n=== Test B: periodic chase (accumulate K steps in s_fast, chase once) ===")
alpha = 0.1; SCALE = 128.0
def cascade(orth, K, steps=200):
    torch.manual_seed(1)
    s_fast = torch.zeros(N, K_:=K and N, device=dev) if False else torch.zeros(N, 384, device=dev)
    s_slow = torch.zeros(N, 384, device=dev)
    gg = torch.Generator(device=dev); gg.manual_seed(7)
    for t in range(steps):
        sig = (torch.randn(N,8,device=dev,generator=gg) @ torch.randn(8,384,device=dev,generator=gg))*30
        noise = torch.randn(N,384,device=dev,generator=gg)*30
        s_fast = s_fast + sig + noise
        if (t+1) % K == 0:                         # chase only every K steps (accumulate between)
            src = ns5(s_fast)*s_fast.norm()/(N**0.5) if orth else s_fast
            chase = alpha * K * src / SCALE        # K steps' worth -> super-quantum tick
            fl = torch.floor(chase); fr = chase-fl
            tick = fl + (torch.rand(N,384,device=dev,generator=gg) < fr).float()
            s_slow = s_slow + tick; s_fast = s_fast - tick*SCALE
    return s_slow*SCALE
for K in [1, 10, 50]:
    mt,ft = sv_flat(cascade(True, K)); mf,ff = sv_flat(cascade(False, K))
    print(f"  K={K:3} (chase every {K}): orth mean_sv/max={mt:.3f} | plain={mf:.3f} | "
          f"gap={mt-mf:+.3f}{'  <- orth SURVIVES' if mt-mf>0.03 else ''}")
print("  (if gap grows positive as K rises, accumulate-then-chase-large preserves orthogonality)")
