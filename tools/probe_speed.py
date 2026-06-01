"""Is the graph actually displacing eager work, or are we 'trapped in eager'? Time, on the
real char-LM, 200 steps each, NO eval (pure step cost):
  (a) eager fwd+bwd+rebalance         <- baseline
  (b) graph replay only (no reb)       <- is the GRAPH itself fast?
  (c) graph replay + eager rebalance   <- does eager rebalance(25 layers/step) eat the win?
If (b) << (a): graph works; the tax is rebalance/eval. If (b) ~= (a): graph is a no-op =
trapped in eager (the user's hypothesis)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch, torch.nn as nn, torch.nn.functional as F
from nanogpt import GPT, GPTConfig, load_char_data, get_batch
from prototype_packed_b import ConcordLinearPackedB, set_fixed_coh
dev="cuda"
def build():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    train,val,vsz,stoi=load_char_data("nanogpt_data/input.txt",dev)
    cfg=GPTConfig(vocab_size=vsz,block_size=256,n_layer=6,n_head=6,n_embd=384,dropout=0.0)
    m=GPT(cfg).to(dev); layers=[]
    for parent in list(m.modules()):
        for name,ch in list(parent.named_children()):
            if isinstance(ch,nn.Linear):
                c=ConcordLinearPackedB(ch.in_features,ch.out_features,bias=ch.bias is not None,device=dev)
                c.set_optimizer_kind('adamw',weight_decay=0.0,eps=1e-10,step_cap=10.0)
                c.precond_p=0.5;c.v_scale=0.0;c.gf_trust_delta_sq=1.0
                with torch.no_grad(): c.load_weights(ch.weight)
                setattr(parent,name,c);layers.append(c)
    set_fixed_coh(True)
    for l in layers: l.enable_cohpre(); l.lr=5e-4
    return m,layers,train
def timed(fn,n=200):
    torch.cuda.synchronize(); t=time.time()
    for i in range(n): fn(i)
    torch.cuda.synchronize(); return (time.time()-t)/n*1000  # ms/iter

g=torch.Generator(device=dev); g.manual_seed(1234)
X=[get_batch(*( (None,),),) ] if False else None
# (a) eager fwd+bwd+rebalance
m,layers,train=build()
aux=[p for p in m.parameters() if p.requires_grad]; ao=torch.optim.AdamW(aux,lr=1e-3)
xb,yb=get_batch(train,64,256,dev,generator=g)
def eager(i):
    ao.zero_grad(set_to_none=True); _,l=m(xb,yb); l.backward(); ao.step()
    for L in layers: L.rebalance()
ms_a=timed(eager)
# capture for (b)/(c)
m2,layers2,_=build()
aux2=[p for p in m2.parameters() if p.requires_grad]; ao2=torch.optim.AdamW(aux2,lr=1e-3)
sx=xb.clone(); sy=yb.clone()
def fb():
    for p in aux2:
        if p.grad is not None: p.grad=None
    _,gl=m2(sx,sy); gl.backward(); return gl
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): fb()
torch.cuda.current_stream().wait_stream(s)
cap=torch.cuda.CUDAGraph()
with torch.cuda.graph(cap): sl=fb()
def replay_only(i): cap.replay()
def replay_reb(i):
    cap.replay(); ao2.step()
    for L in layers2: L.rebalance()
ms_b=timed(replay_only)
ms_c=timed(replay_reb)
print(f"(a) eager fwd+bwd+reb     : {ms_a:6.2f} ms/iter")
print(f"(b) graph replay ONLY     : {ms_b:6.2f} ms/iter   ({ms_a/ms_b:.1f}x vs eager)")
print(f"(c) graph replay + eager reb: {ms_c:6.2f} ms/iter ({ms_a/ms_c:.1f}x vs eager)")
print("[READ] if (b)<<(a): graph works, rebalance is the tax. if (b)~=(a): trapped in eager.")
