"""Pinpoint the harness graph divergence. Replicate the harness step EXACTLY (lr schedule
via m.lr=..., get_batch, eval-free) eager vs graphed, same seed, and report where they
split. Isolates: is it (a) lr-not-propagating, (b) the warmup/capture steps consuming
different batches than eager, or (c) a real replay bug. Tiny (200 iters)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch, torch.nn.functional as F, math
from nanogpt import GPT, GPTConfig, load_char_data, get_batch
from prototype_packed_b import ConcordLinearPackedB, set_fixed_coh
import torch.nn as nn
dev="cuda"

def wrap(model):
    layers=[]
    for parent in list(model.modules()):
        for name,ch in list(parent.named_children()):
            if isinstance(ch, nn.Linear):
                c=ConcordLinearPackedB(ch.in_features, ch.out_features, bias=ch.bias is not None, device=dev)
                c.set_optimizer_kind('adamw', weight_decay=0.0, eps=1e-10, step_cap=10.0)
                c.precond_p=0.5; c.v_scale=0.0; c.gf_trust_delta_sq=1.0
                with torch.no_grad():
                    c.load_weights(ch.weight)
                setattr(parent,name,c); layers.append(c)
    return layers

def build():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    train,val,enc,dec,vsz = load_char_data("nanogpt_data/input.txt", dev)
    cfg=GPTConfig(vocab_size=vsz, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0)
    m=GPT(cfg).to(dev)
    layers=wrap(m); set_fixed_coh(True)
    for l in layers:
        l.enable_cohpre()
    return m, layers, train

def lr_at(it, mx=200, peak=5e-4):
    if it<100: return peak*(it+1)/100
    p=(it-100)/max(1,mx-100); return peak*(0.1+0.45*(1+math.cos(math.pi*p)))

def run(graph):
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    import prototype_packed_b as ppb
    ppb._get_step_counter(torch.device(dev)).zero_()
    m,layers,train=build()
    aux=[p for p in m.parameters() if p.requires_grad]
    auxopt=torch.optim.AdamW(aux, lr=1e-3)
    g=torch.Generator(device='cpu'); g.manual_seed(1234)
    losses=[]; cap=None; sx=sy=None
    for it in range(200):
        lr=lr_at(it)
        for l in layers: l.lr=lr
        for grp in auxopt.param_groups: grp['lr']=1e-3*(lr/5e-4)
        x,y=get_batch(train,64,256,dev,generator=g)
        if graph and cap is None:
            sx=torch.zeros_like(x); sy=torch.zeros_like(y)
            sx.copy_(x); sy.copy_(y)
            def fb():
                for p in aux:
                    if p.grad is not None: p.grad=None
                _,gl=m(sx,sy); gl.backward(); return gl
            s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): fb()
            torch.cuda.current_stream().wait_stream(s)
            cap=torch.cuda.CUDAGraph()
            with torch.cuda.graph(cap): sl=fb()
            auxopt.step()
            for l in layers: l.rebalance()
            losses.append(sl.item())
            capsl=sl
        elif graph:
            sx.copy_(x); sy.copy_(y); cap.replay(); auxopt.step()
            for l in layers: l.rebalance()
            losses.append(capsl.item())
        else:
            auxopt.zero_grad(set_to_none=True)
            _,loss=m(x,y); loss.backward(); auxopt.step()
            for l in layers: l.rebalance()
            losses.append(loss.item())
    return losses

e=run(False); g=run(True)
print("it    eager    graph    diff")
for it in [0,1,2,5,10,50,100,150,199]:
    print(f"{it:>4} {e[it]:8.4f} {g[it]:8.4f} {abs(e[it]-g[it]):8.4f}")
md=max(abs(a-b) for a,b in zip(e,g))
print(f"max|diff|={md:.5f}  -> {'MATCH' if md<0.05 else 'DIVERGE'}")
