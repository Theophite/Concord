"""Exp 11d: the validated LIVE autotuner law — commit + one-sided event-driven
re-probe. See EXPERIMENTS.md exp 11 for the path here (11: naive continuous
laws fail on meter conditioning; 11b: continuous tracking through the
calibrated table is SELF-DEFEATING for level — the velocity-side meter
responds to the friction it controls, kappa 291->26; 11c: two-sided
change-detection fires spuriously on the benign post-commit coherence RISE).

The law: probe-then-commit exactly as shipped; then hold, tracking the
windowed meter against a slow-EMA baseline; re-probe ONLY on a drop
(m < base - BAND). Static = identical to the shipped tuner (zero events);
mid-run regime change (clean -> 30% label noise) = one re-probe, recommit
kappa ~115: deploy 92.76 vs 92.43, memorized 12.2% vs 13.4%.
Results: exp11{,b,c,d}_results.json.
"""
import torch, json
from concord_ref import ConcordRef, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
torch.set_num_threads(4)
SUBSET, EPOCHS, BATCH, LR = 4000, 25, 128, 1e-3
SEEDS=(0,1); SPB=SUBSET//BATCH
P0,P1=3*SPB,8*SPB; BAND=0.08; REPROBE=3*SPB; BETA=0.7
TABLE=[(0.3865175302951567,0.0),(0.3144563574303863,100.0),(0.28822555593265003,200.0),(0.2738887910080212,400.0),(0.2563569223688495,400.0)]
def tk(c):
    t=TABLE
    if c>=t[0][0]: return t[0][1]
    if c<=t[-1][0]: return t[-1][1]
    for (c1,k1),(c2,k2) in zip(t,t[1:]):
        if c2<=c<=c1: return k1+(c1-c)/(c1-c2)*(k2-k1)
    return t[-1][1]

class R(ConcordRef):
    def __init__(self,*a,law="commit",**k):
        super().__init__(*a,**k); self.law=law; self.kappa=0.0
        self.mode="pre"; self.acc=[]; self.base=None; self.win=[]
        self.kh=[]; self.reprobes=0
    @torch.no_grad()
    def step(self):
        super().step(); t=self.t; c=self.mean_coh()
        if self.mode=="pre" and t>=P0:
            self.mode="probe"; self.kappa=50.0; self.acc=[]; self._pend=t+(P1-P0)
        if self.mode=="probe":
            self.acc.append(c)
            if t>=self._pend:
                self.kappa=tk(sum(self.acc)/len(self.acc))
                self.mode="hold"; self.base=None; self.win=[]
        elif self.mode=="hold" and self.law=="reprobe":
            self.win.append(c)
            if len(self.win)>=SPB:
                m=sum(self.win)/len(self.win); self.win=[]
                if self.base is None:
                    self.base=m
                elif m < self.base - BAND:          # one-sided: DROP only
                    self.reprobes+=1
                    self.mode="probe"; self.kappa=50.0; self.acc=[]; self._pend=t+REPROBE
                else:
                    self.base=BETA*self.base+(1-BETA)*m   # drift-tolerant baseline
        self.kh.append(self.kappa)

def make(nf,seed,data):
    xtr,ytr,xte,yte=data
    g=torch.Generator().manual_seed(seed+100)
    sub=torch.randperm(len(xtr),generator=g)[:SUBSET]
    x,y=xtr[sub],ytr[sub].clone()
    fl=torch.rand(len(y),generator=g)<nf
    if fl.any(): y[fl]=torch.randint(0,10,(int(fl.sum()),),generator=g)
    return x,y,fl,g,xte,yte

def run(nf,seed,data,law,flip_ep=None):
    x,y,fl,g,xte,yte=make(nf,seed,data)
    net=make_net(seed); o=R(net,lr=LR,total_steps=EPOCHS*SPB,gate=True,noise=False,law=law,
                            generator=torch.Generator().manual_seed(seed+10))
    for ep in range(EPOCHS):
        if flip_ep is not None and ep==flip_ep:
            f2=torch.rand(len(y),generator=g)<0.30
            y[f2]=torch.randint(0,10,(int(f2.sum()),),generator=g); fl=f2
        perm=torch.randperm(len(x),generator=g)
        for i in range(0,len(x)-BATCH+1,BATCH):
            idx=perm[i:i+BATCH]
            torch.nn.functional.cross_entropy(net(x[idx]),y[idx]).backward()
            o.step(); o.zero_grad()
    fit=accuracy(net,x[fl],y[fl]) if fl.any() else 0.0
    with swap_to_deploy(o): dep=accuracy(net,xte,yte)
    n=len(o.kh)
    return dep,fit,sum(o.kh[-n//5:])/(n//5),o.reprobes

if __name__=="__main__":
    data=load_mnist(); mean=lambda v:sum(v)/len(v)
    out={}
    for tag,nf,fep in (("rho0",0.0,None),("rho10",0.10,None),("rho30",0.30,None),("flip",0.0,12)):
        for law in ("commit","reprobe"):
            r=[run(nf,s,data,law,fep) for s in SEEDS]
            out[f"{tag}_{law}"]=[mean([x[0] for x in r]),mean([x[1] for x in r]),mean([x[2] for x in r]),mean([x[3] for x in r])]
            print(f"  {tag:5s} {law:7s}: deploy={mean([x[0] for x in r])*100:.2f}%  mem={mean([x[1] for x in r])*100:.1f}%  "
                  f"k_late={mean([x[2] for x in r]):.0f}  reprobes={mean([x[3] for x in r]):.1f}",flush=True)
    json.dump(out,open("exp11d_results.json","w"),indent=1)
