"""MultiTimescaleOptimizer -- clean fp32 Layer-A core (spec gate 1).

Portable torch.optim.Optimizer reimplementation of the [CONFIRMED] cascade,
isolated from geometry (Layer B) and int-packing so each layer validates alone.

Gate 1 uses the VALIDATED recipe from sims/exp1_walk.py (the one shown to bound
noise + integrate signal), NOT the spec's idealized variants:
  - raw-g injection into s_f;
  - v_s is an EMA of s_s at rate alpha_v (epoch-match later = set alpha_v=1/E);
  - coh = clip((alpha_v*(s_s - s_v))^2 / vhat, 0, 1)   [alpha_v ~ 0.001];
  - coh_pre = EMA-of-coh, init 1 (start trusting, decay to earned coh) -- the
    validated bootstrap floor. (The spec's gradient-EMA coh_pre and the
    gamma-drainage are FORKS: gate-1 probes showed the naive gradient-EMA
    coh_pre leaks ~half on pure noise unless its eps = the noise scale. Revisit
    as gate-1b, don't bolt onto the confirmed core.)
  - gated acceptance: Delta = alpha*(coh + coh_pre*(1-coh))*s_f ; REQUIRE kappa<alpha.
  - closed loop: p held at w = init + s_f + s_s + s_v.
  - averaged readout: deploy w = init + G*s_s.
"""
import torch
from torch.optim.optimizer import Optimizer


class MultiTimescaleOptimizer(Optimizer):
    def __init__(self, params, lr=1.0, alpha=0.1, kappa=0.03, alpha_v=0.001,
                 lam=None, b2=0.999, eps=1e-12, G=3.0, use_coh=True,
                 whiten=False, eps_w=1e-8):
        if not (kappa < alpha):
            raise ValueError(f"require kappa<alpha (bootstrap): {kappa},{alpha}")
        # Bootstrap floor coh_pre must stay open until signal coherence latches
        # (~1/alpha_v steps), so its decay must be no faster than alpha_v.
        if lam is None:
            lam = alpha_v
        # lr = injection scale (§3 scale_inv FORK, scalar form). 1.0 = raw-g
        # (spec claims lr=1 stable via cascade gain); lower if a real net is hot.
        super().__init__(params, dict(lr=lr, alpha=alpha, kappa=kappa,
                                      alpha_v=alpha_v, lam=lam, b2=b2, eps=eps,
                                      G=G, use_coh=use_coh, whiten=whiten,
                                      eps_w=eps_w))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for grp in self.param_groups:
            a, k, av, lam = grp['alpha'], grp['kappa'], grp['alpha_v'], grp['lam']
            b2, eps, use_coh, lr = grp['b2'], grp['eps'], grp['use_coh'], grp['lr']
            whiten, eps_w = grp['whiten'], grp['eps_w']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st['init'] = p.detach().clone()
                    for nm in ('s_f', 's_s', 's_v', 'v'):
                        st[nm] = torch.zeros_like(p)
                    st['cohpre'] = torch.ones_like(p)
                    st['t'] = 0
                s_f, s_s, s_v, v = st['s_f'], st['s_s'], st['s_v'], st['v']
                st['t'] += 1
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                vh = v / (1 - b2 ** st['t']) + eps
                # injection: scalar (gated-SGD) or per-coord-whitened g/sqrt(v)
                # (gated-Adam) -- the latter normalizes heterogeneous scales,
                # required for stability on real conv/BN nets (gate-1 finding).
                ginj = (g / (vh.sqrt() + eps_w)) if whiten else g
                s_f.add_(ginj, alpha=lr)
                if use_coh:
                    coh = ((av * (s_s - s_v)) ** 2 / vh).clamp_(0, 1)
                    s_f.mul_(1 - k * (1 - coh))              # evap OFF if coherent
                    floor = st['cohpre'].clone()             # pre-update = floor
                    st['cohpre'].mul_(1 - lam).add_(coh, alpha=lam)
                    gate = coh + floor * (1 - coh)
                    Delta = (a * gate) * s_f
                else:
                    Delta = a * s_f
                s_s.add_(Delta); s_f.sub_(Delta)
                s_v.add_(av * (s_s - s_v))                   # very-slow EMA
                p.copy_(st['init'] + s_f + s_s + s_v)        # close the loop
        return loss

    @torch.no_grad()
    def deploy_(self):
        for grp in self.param_groups:
            G = grp['G']
            for p in grp['params']:
                st = self.state[p]
                if st:
                    st['_live'] = p.detach().clone()
                    p.copy_(st['init'] + G * st['s_s'])

    @torch.no_grad()
    def restore_(self):
        for grp in self.param_groups:
            for p in grp['params']:
                st = self.state[p]
                if '_live' in st:
                    p.copy_(st['_live']); del st['_live']


# ======================= gate-1 cheap 1-D probes =======================
def _probe_noise_walk(T=20000, K=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for use_coh in (False, True):
        p = torch.nn.Parameter(torch.zeros(K))
        opt = MultiTimescaleOptimizer([p], use_coh=use_coh)   # alpha_v=0.001
        var_t = []
        for t in range(T):
            p.grad = torch.randn(K, generator=g)
            opt.step()
            if (t + 1) % (T // 10) == 0:
                var_t.append(opt.state[p]['s_v'].var().item())
        out['gated' if use_coh else 'ungated'] = var_t
    return out


def _probe_selectivity(T=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    snrs = torch.tensor([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
    drift = snrs * 1.0
    p = torch.nn.Parameter(torch.zeros(len(snrs)))
    opt = MultiTimescaleOptimizer([p], use_coh=True)
    for t in range(T):
        p.grad = drift + torch.randn(len(snrs), generator=g)
        opt.step()
    st = opt.state[p]
    vh = st['v'] / (1 - 0.999 ** st['t']) + 1e-12
    coh = (((0.001 * (st['s_s'] - st['s_v'])) ** 2) / vh).clamp(0, 1)
    return snrs.tolist(), coh.tolist()


if __name__ == "__main__":
    print("=== GATE 1 PROBES (validated cascade recipe) ===\n")
    nw = _probe_noise_walk()
    print("noise-walk Var(s_v) at t=10%..100%:")
    print("  ungated:", " ".join(f"{x:.2f}" for x in nw['ungated']))
    print("  gated  :", " ".join(f"{x:.2f}" for x in nw['gated']))
    ug, gd = nw['ungated'][-1], nw['gated'][-1]
    print(f"  -> ungated {ug:.1f}; gated {gd:.3f}  "
          f"({'PASS (gated bounded)' if gd < ug / 5 else 'FAIL'})\n")
    snrs, coh = _probe_selectivity()
    print("selectivity coh vs SNR:")
    for s, c in zip(snrs, coh):
        print(f"  SNR={s:5.1f} -> coh={c:.3f}")
    spread = coh[-1] - coh[0]
    mono = all(coh[i] <= coh[i + 1] + 1e-3 for i in range(len(coh) - 1))
    print(f"  -> spans [{coh[0]:.2f},{coh[-1]:.2f}], monotone={mono}  "
          f"({'PASS' if (mono and spread > 0.3 and coh[0] < 0.3) else 'CHECK'})")
