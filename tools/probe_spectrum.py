"""Is there a clean signal/noise GAP in real d_sv to threshold at? (Decides if rank-aware
ortho is even well-posed on real data.) Train a real Concord layer on tiny-shakespeare,
dump the singular-value spectrum of d_sv = s_slow - v_slow at a few checkpoints. Look for:
a knee/cliff (clean gap -> thresholdable) vs a smooth decay (no gap -> rank-aware ill-posed).
"""
import sys; sys.path.insert(0,'src')
import torch, torch.nn as nn, torch.nn.functional as F
from nanogpt import GPT, GPTConfig, load_char_data, get_batch
from prototype_packed_b import ConcordLinearPackedB, set_fixed_coh
dev='cuda'; torch.manual_seed(0)
train,val,vsz,stoi=load_char_data("nanogpt_data/input.txt",dev)
cfg=GPTConfig(vocab_size=vsz,block_size=256,n_layer=6,n_head=6,n_embd=384,dropout=0.0)
m=GPT(cfg).to(dev)
layers=[]
for parent in list(m.modules()):
    for name,ch in list(parent.named_children()):
        if isinstance(ch,nn.Linear):
            c=ConcordLinearPackedB(ch.in_features,ch.out_features,bias=ch.bias is not None,device=dev)
            c.set_optimizer_kind('adamw',weight_decay=0.0,eps=1e-10,step_cap=10.0)
            c.precond_p=0.5;c.v_scale=0.0;c.gf_trust_delta_sq=1.0
            with torch.no_grad(): c.load_weights(ch.weight)
            setattr(parent,name,c); layers.append(c)
set_fixed_coh(True)
for l in layers: l.enable_cohpre(); l.lr=5e-4
aux=[p for p in m.parameters() if p.requires_grad]; ao=torch.optim.AdamW(aux,lr=3e-4)
g=torch.Generator(device=dev); g.manual_seed(1234)
# representative layer: a square attn proj in block 0
tgt=[l for l in layers if tuple(l.packed_w.shape)==(384,384)][0]
def dsv_spec():
    pw=tgt.packed_w; ss=((pw<<16)>>24).float(); vs=((pw<<24)>>24).float()
    s=torch.linalg.svdvals((ss-vs).float()); s=s/(s.max()+1e-12)
    return s
def report(it,s):
    # cumulative-energy ranks + the drop ratio between adjacent SVs (a cliff = big ratio)
    cum=torch.cumsum(s**2,0)/(s**2).sum()
    r50=int((cum<0.5).sum())+1; r90=int((cum<0.9).sum())+1; r99=int((cum<0.99).sum())+1
    # biggest adjacent log-drop in the top-80 (where a signal/noise knee would be)
    top=s[:80]; ratios=(top[:-1]/(top[1:]+1e-9))
    knee=int(torch.argmax(ratios).item())+1; kneeval=ratios.max().item()
    # sample the curve
    idx=[0,4,9,19,39,79,159,319]; vals=[f"{s[i]:.3f}" for i in idx if i<len(s)]
    print(f" it{it:<4} r50={r50:<3} r90={r90:<3} r99={r99:<4} | sharpest knee@{knee} (x{kneeval:.1f}) | "
          f"sv[0,5,10,20,40,80,160,320]={vals}")
for it in range(801):
    X,Y=get_batch(train,32,256,dev,generator=g)
    ao.zero_grad(); _,loss=m(X,Y); loss.backward(); ao.step()
    for l in layers: l.rebalance()
    if it in (100,300,500,800):
        report(it, dsv_spec())
print("\n[READ] CLEAN GAP (thresholdable): a sharp knee (high xN) at a stable rank, r90<<384,")
print("       svs flat-ish then cliff. SMOOTH DECAY (ill-posed): knee~1.0, svs taper gradually,")
print("       no stable boundary -> rank-aware ortho has no natural k to pick.")
