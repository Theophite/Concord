"""Is the 'rank ~35' real, or an artifact of probe_muon6's rank-32 SYNTHETIC target?
Measure the eff-rank of grad_W and of an EMA_0.9 momentum on the REAL task (real nanoGPT,
real tiny-shakespeare), for the actual Linear weight matrices. No synthetic target."""
import sys; sys.path.insert(0,'src')
import torch, torch.nn as nn, torch.nn.functional as F
from nanogpt import GPT, GPTConfig, load_char_data, get_batch
dev='cuda'; torch.manual_seed(0)
train,val,vsz,stoi = load_char_data("nanogpt_data/input.txt", dev)
cfg=GPTConfig(vocab_size=vsz,block_size=256,n_layer=6,n_head=6,n_embd=384,dropout=0.0)
m=GPT(cfg).to(dev)
opt=torch.optim.AdamW(m.parameters(), lr=3e-4)
# pick representative Linears
targets={}
for name,mod in m.named_modules():
    if isinstance(mod, nn.Linear) and name.split('.')[-1] in ('c_attn','c_fc','c_proj') and '0.' in name:
        targets[name]=mod
targets=dict(list(targets.items())[:3])
mom={n:torch.zeros_like(t.weight) for n,t in targets.items()}
def erank(M):
    s=torch.linalg.svdvals(M.float())
    pr=(s.sum()**2/(s**2).sum()).item()                 # participation ratio
    cum=torch.cumsum(s**2,0)/(s**2).sum()
    r90=int((cum<0.90).sum().item())+1                   # SVs to reach 90% energy
    return pr, r90, min(M.shape)
g=torch.Generator(device=dev); g.manual_seed(1234)
for it in range(120):
    X,Y=get_batch(train,32,256,dev,generator=g)
    opt.zero_grad(); _,loss=m(X,Y); loss.backward()
    for n,t in targets.items():
        mom[n]=0.9*mom[n]+0.1*t.weight.grad
    opt.step()
print(f"REAL tiny-shakespeare nanoGPT (vocab={vsz}), grad/momentum eff-rank @ it120:")
print(f"  {'layer':28} {'shape':12} {'grad: PR / r90':18} {'mom(0.9): PR / r90':18} of min-dim")
for n,t in targets.items():
    gp,gr,mn=erank(t.weight.grad); mp,mr,_=erank(mom[n])
    print(f"  {n:28} {str(tuple(t.weight.shape)):12} {gp:6.0f} / {gr:<8} {mp:9.0f} / {mr:<8} /{mn}")
print("\n[READ] if PR/r90 are a large FRACTION of min-dim, the gradient is HIGH-rank on the")
print("       real task -> 'rank ~35' was a synthetic artifact, and the Muon-fails-because-")
print("       low-rank story is WRONG. If still small (tens), low-rank holds on THIS task.")
