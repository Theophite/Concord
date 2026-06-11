"""Exp 17: noise with the character of augmentation — the hierarchy test.

Claim: augmentation's gradient signature is per-example-anchored,
manifold-structured, visit-decorrelated noise, and injectable noise should
rank by how much data structure it encodes:

  L0  iso        isotropic, post-NS (exp 9b placement), sigma 0.6 rising-late
  L1  sigma_g    Sigma_g-shaped, PRE-NS (simulated batch resampling belongs in
                 gradient space): noise = sum_i eps_i*(c_i - gbar/B), built
                 per layer from per-example deltas via the one-matmul identity
                 (eps . D)^T X - mean(eps)*grad, scaled to sigma*||g|| on the
                 rising-late schedule. The repo's original _SIGMAG design,
                 re-matched under the honest gate.
  L2a vicinal    chord jitter, label unchanged: x' = x + lam*(x_j - x),
                 j random in batch, lam ~ U(0, 0.25). Input vicinity only.
  L2b mixup      x' = lam*x + (1-lam)*x_j, loss = lam*CE(y) + (1-lam)*CE(y_j),
                 lam ~ U(0,1) (= Beta(1,1)). Vicinity + label interpolation.
  ctl smallbatch batch 32 (4x natural Sigma_g noise), same epochs.

Arena: the exp-10 corner where augmentation mattered most — 4k subset, 30%
label noise, 80 epochs, NS drive, kappa=150 @ lr 1e-2 (the repaired config).
References on the books: none = 86.28 +- 0.77 (47.8% memorized);
crop-aug (L3) = 96.31 +- 0.22 (10.6%). Caveat: kappa was tuned in the
no-diversity corner; aug-character arms may prefer lower kappa — stale-high
kappa biases AGAINST the hierarchy claim, so orderings that survive it are
conservative.
"""
import json
import math

import torch
import torch.nn.functional as F

from concord_ref import swap_to_deploy
from exp3_mnist import accuracy, load_mnist
from exp6_autotune import make_run, SEEDS
from exp9_muon import MuonConcord
from exp9b_muon_noise import NoisyMuonConcord

torch.set_num_threads(4)
EPOCHS, LR, KAPPA, NF = 80, 1e-2, 150.0, 0.30
BATCH, SIGMA_PEAK = 128, 0.6


class TwoLayer(torch.nn.Module):
    """make_net's architecture with retained pre-activations, so the Sigma_g
    arm can read per-example deltas (D) and inputs (X) per Linear layer."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = torch.nn.Linear(784, 256)
        self.l2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        self.x_in = x
        self.z1 = self.l1(x)
        self.a1 = torch.relu(self.z1)
        return self.l2(self.a1)


def sigma_t(t, T, lr_min_frac=0.2):
    p = min(1.0, t / max(1, T))
    f = lr_min_frac + 0.5 * (1 - lr_min_frac) * (1 + math.cos(math.pi * p))
    return SIGMA_PEAK * (1.0 - f)


def add_sigma_g_noise(net, loss, out, sig, gen):
    """Set p.grad = grad + Sigma_g-shaped draw (pre-NS), one autograd pass.
    Per Linear: grad_W = D^T X (D includes mean-loss 1/B); centered draw
    sum_i eps_i (d_i x_i^T - grad_W/B) = (eps . D)^T X - mean(eps)*grad_W."""
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, [net.z1, out] + params)
    D = {id(net.l1.weight): (grads[0], net.x_in),
         id(net.l2.weight): (grads[1], net.a1)}
    for p_, g_ in zip(params, grads[2:]):
        g_ = g_.detach()
        if id(p_) in D and sig > 0:
            Dl, Xl = D[id(p_)]
            eps = torch.randn(Dl.shape[0], generator=gen)
            noise = (eps[:, None] * Dl).T @ Xl - eps.mean() * g_
            nn_ = noise.norm().clamp_min(1e-12)
            g_ = g_ + noise * (sig * g_.norm() / nn_)
        p_.grad = g_.clone()


def run(arm, seed, data):
    x, y, flip, gen, xte, yte = make_run(NF, seed, data)
    batch = 32 if arm == "smallbatch" else BATCH
    steps = EPOCHS * (len(x) // batch)
    ngen = torch.Generator().manual_seed(seed + 10)
    net = TwoLayer(seed)
    if arm == "iso":
        opt = NoisyMuonConcord(net, lr=LR, total_steps=steps, gate=True,
                               kappa=KAPPA, noise=True, sigma_peak=SIGMA_PEAK,
                               noise_mode="post", generator=ngen)
    else:
        opt = MuonConcord(net, lr=LR, total_steps=steps, gate=True,
                          kappa=KAPPA, noise=False, generator=ngen)
    t = 0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            if arm == "vicinal":
                j = torch.randperm(len(xb), generator=gen)
                lam = torch.rand(len(xb), 1, generator=gen) * 0.25
                loss = F.cross_entropy(net(xb + lam * (xb[j] - xb)), yb)
            elif arm == "mixup":
                j = torch.randperm(len(xb), generator=gen)
                lam = float(torch.rand((), generator=gen))
                out = net(lam * xb + (1 - lam) * xb[j])
                loss = (lam * F.cross_entropy(out, yb)
                        + (1 - lam) * F.cross_entropy(out, yb[j]))
            else:
                loss = F.cross_entropy(net(xb), yb)
            if not math.isfinite(float(loss.detach())):
                return None
            if arm == "sigma_g":
                out_full = net(xb)          # recompute with retained graph
                loss = F.cross_entropy(out_full, yb)
                add_sigma_g_noise(net, loss, out_full, sigma_t(t, steps), gen)
                opt.step()
                opt.zero_grad()
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()
            t += 1
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        return accuracy(net, xte, yte), fit


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    print("exp 17 — 4k, 30% noise, 80ep, NS kappa=150 lr 1e-2; refs: "
          "none 86.28 (47.8% mem), crop-aug 96.31 (10.6% mem)")
    out = {}
    for arm in ("iso", "sigma_g", "vicinal", "mixup", "smallbatch"):
        rs = [run(arm, s, data) for s in SEEDS]
        if any(r is None for r in rs):
            print(f"  {arm:10s} DIVERGED")
            out[arm] = None
            continue
        accs = [r[0] for r in rs]
        fits = [r[1] for r in rs]
        out[arm] = (mean(accs), spread(accs), mean(fits))
        print(f"  {arm:10s} acc={mean(accs)*100:.2f}±{spread(accs)*100:.2f}%"
              f"  memorized={mean(fits)*100:.1f}%", flush=True)
    json.dump(out, open("exp17_results.json", "w"), indent=1)
