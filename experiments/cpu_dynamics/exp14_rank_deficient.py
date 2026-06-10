"""Exp 14: the drift-referenced spectral gate on a rank-deficient task.

Exp 12+13's law: suppressors keyed on the gradient's OWN statistics lose;
working gates key on a SIGNAL REFERENCE (the telescope drift). Exp 13's
energy gate failed because energy is not SNR. This is the referenced
version, on the task family where muon actually falls down: RANK-DEFICIENT
targets, where NS5's equal-magnitude write pours energy into the task's dead
complement — directions the per-element and energy meters cannot flag, but
in which the drift D = C*(S−A) has, by construction, never consolidated
anything.

The gate, v2 (v1 — per-pair matched filter diag(U^T D V) — failed in the
smoke: D and g share the live SUBSPACE but not their singular PAIRINGS, so
the diagonal projection scatters and the gate collapsed to its floor, leak
identical to ungated muon). v2 is pairing-free — Wiener SUBSPACE projectors
built from the drift:

    W_L = DD^T (DD^T + (||D||_F^2/N) I)^{-1}     # col-space membership
    W_R = D^T D (D^T D + (||D||_F^2/K) I)^{-1}   # row-space membership
    O   = NS5(g/||g||)
    step = sqrt(max(N,K)) * (phi_c*O + (1-phi_c) * W_L @ O @ W_R)

Live directions (sigma_D^2 >> D's mean spectral energy) pass ~1; dead
directions ~0. The delta^2 scale comes from the REFERENCE's own spectrum
(exp-12/13 law: reference-keyed, not gradient-keyed), and the chase-floor
phi_c is the bootstrap (early: drift empty -> nearly ungated full NS5, no
exp-10/13 starvation; anneals 0.9 -> 0.1). The rational projectors are
NS-approximable (matmuls only) in any production port — the spectral gate
does go "inside the Newton-Schulz", but built from D, not from g.

Task: teacher y = W2 tanh(W1 x), W1 (64x64) and W2 (10x64) both RANK 4;
infinite fresh data (no memorization confound); noise arm adds 0.5*std
target noise. Metrics at the DEPLOY weights: clean-test MSE, and the
mechanism readout — complement leakage of deploy W1 (energy outside the
teacher's rank-4 column space).

Arms: vhat / muon (NS5, exp-12 class) / spec (this gate), lam in {0, 0.1},
noise in {off, on}, seeds 0/1/2.
"""
import json
import math
import time

import torch
import torch.nn as nn

from concord_ref import ConcordRef, _cos_floor, evap_term, swap_to_deploy
from exp12_muon_lambda import MuonLeak

torch.set_num_threads(4)

DIM, HID, OUT, RANK = 64, 64, 10, 4
STEPS, BATCH, LR = 3000, 128, 1e-3
SEEDS = (0, 1, 2)
NOISE_SIGMA = 0.5


def make_teacher(seed):
    g = torch.Generator().manual_seed(seed + 500)
    A1 = torch.randn(HID, RANK, generator=g) / math.sqrt(RANK)
    B1 = torch.randn(RANK, DIM, generator=g) / math.sqrt(DIM)
    A2 = torch.randn(OUT, RANK, generator=g) / math.sqrt(RANK)
    B2 = torch.randn(RANK, HID, generator=g) / math.sqrt(HID)
    W1, W2 = A1 @ B1, A2 @ B2
    def f(x):
        return torch.tanh(x @ W1.T) @ W2.T
    return f, B1


def make_student(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(DIM, HID), nn.Tanh(), nn.Linear(HID, OUT))


class SpecMuon(ConcordRef):
    """Muon with the structure-referenced spectral gate v3: Wiener subspace
    projectors from R = (S+A) - W0, the deploy position's net learned delta.
    (v2 used the recent drift D = C*(S-A) as the reference -- too dirty: the
    chase floor passes noise into the telescope window, measured effective
    rank ~13/64, leak 44.4 -> only 40.5. The drift is a VELOCITY meter; the
    live-subspace question wants the INTEGRATED structure, where noise is an
    isotropic random walk but signal accumulates coherently.)"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for st in self.swapped:
            st["W0"] = st["p"].detach().clone()

    @torch.no_grad()
    def step(self):
        self._advance_schedules()
        lr = self.lr
        for st in self.swapped:
            g = st["p"].grad
            if g is None:
                continue
            g = g.float()
            u, S, A = st["u"], st["S"], st["A"]
            D = self.Cstar * (S - A)
            nse = u - D
            coh_el = (D * D) / (D * D + nse * nse + 1e-30)
            N, K = g.shape
            R = (S + A) - st["W0"]            # net learned structure
            # v4 calibration (from the spectral diagnostic): the rank spike
            # sits ~3.4-5.4x mean sigma over a ~1.9x bulk, so the Wiener knee
            # goes at 6x mean ENERGY (between them), the projector is
            # sharpened once by the smoothstep W^2(3-2W) (a matrix polynomial
            # -- W is symmetric with eigenvalues in [0,1)), and the drive
            # floor anneals on the DISCOVERY timescale (first half) instead
            # of the chase schedule.
            RRt, RtR = R @ R.T, R.T @ R
            rl = (R * R).sum().clamp_min(1e-30)
            W_L = RRt @ torch.linalg.inv(RRt + (6.0 * rl / N) * torch.eye(N))
            W_R = RtR @ torch.linalg.inv(RtR + (6.0 * rl / K) * torch.eye(K))
            W_L = W_L @ W_L @ (3.0 * torch.eye(N) - 2.0 * W_L)
            W_R = W_R @ W_R @ (3.0 * torch.eye(K) - 2.0 * W_R)
            phid = _cos_floor(0.9, 0.05, self.t, max(1, self.T // 2))
            from exp9_muon import ns5
            O = ns5(g / g.norm().clamp_min(1e-12))
            gated = phid * O + (1.0 - phid) * (W_L @ O @ W_R)
            step_ = math.sqrt(max(g.shape)) * gated
            u += -lr * step_ - evap_term(lr, self.kappa, coh_el, u, self.min_leak)
            gc = self.phic + (1 - self.phic) * coh_el
            tr = self.alpha * gc * u
            S += tr
            u -= tr
            gl = self.phil + (1 - self.phil) * coh_el
            lk = self.alpha_v * gl * (S - A)
            A += lk
            S -= lk
            st["p"].copy_(u + S + A)
            st["last_coh"] = float(W_L.diagonal().mean())   # mean col-gate strength
        for st in self.aux:
            if st["p"].grad is None:
                continue
            st["m"].mul_(0.9).add_(st["p"].grad)
            st["p"].add_(st["m"], alpha=-lr)
        self.t += 1


DRIVES = {"vhat": ConcordRef, "muon": MuonLeak, "spec": SpecMuon}


def run(drive, noisy, kappa, seed):
    teacher, B1 = make_teacher(0)            # one fixed teacher across seeds
    net = make_student(seed)
    W1_init = net[0].weight.detach().clone()
    gen = torch.Generator().manual_seed(seed + 10)
    opt = DRIVES[drive](net, lr=LR, total_steps=STEPS, gate=True, kappa=kappa,
                        noise=False, generator=gen)
    dgen = torch.Generator().manual_seed(seed + 20)
    for _ in range(STEPS):
        x = torch.randn(BATCH, DIM, generator=dgen)
        y = teacher(x)
        if noisy:
            y = y + NOISE_SIGMA * y.std() * torch.randn(y.shape, generator=dgen)
        loss = nn.functional.mse_loss(net(x), y)
        if not math.isfinite(float(loss.detach())):
            return None
        opt.zero_grad()
        loss.backward()
        opt.step()
    with swap_to_deploy(opt):
        mses = []
        egen = torch.Generator().manual_seed(999)
        for _ in range(5):
            xt = torch.randn(1024, DIM, generator=egen)
            mses.append(float(nn.functional.mse_loss(net(xt), teacher(xt))))
        # complement leakage of the LEARNED delta: the task reads x only
        # through B1's rank-4 row space, so any learned deploy-weight mass on
        # the input complement is dead-direction accumulation (the thing the
        # whitened drive writes and a referenced gate should block). The init
        # is excluded -- it is full-rank by construction and consolidates in.
        dW = net[0].weight.detach() - W1_init
        Q, _ = torch.linalg.qr(B1.T)                    # DIM x RANK orthonormal
        resid = dW - (dW @ Q) @ Q.T
        leak = float((resid * resid).sum() / (dW * dW).sum().clamp_min(1e-30))
    return sum(mses) / len(mses), leak, opt.mean_coh()


if __name__ == "__main__":
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    results = {}
    t0 = time.time()
    for drive in DRIVES:
        for noisy in (False, True):
            for kappa in (0.0, 100.0):
                rs = [run(drive, noisy, kappa, s) for s in SEEDS]
                tag = f"{drive} noise={'on ' if noisy else 'off'} lam={kappa*LR:.1f}"
                if any(r is None for r in rs):
                    results[(drive, noisy, kappa)] = None
                    print(f"{tag}: DIVERGED   [{time.time()-t0:.0f}s]", flush=True)
                    continue
                m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
                lk, mc = mean([r[1] for r in rs]), mean([r[2] for r in rs])
                results[(drive, noisy, kappa)] = (m, sp, lk, mc)
                print(f"{tag}: deploy-MSE={m:.4f}±{sp:.4f}  leak={lk*100:.1f}%  "
                      f"coh={mc:.3f}   [{time.time()-t0:.0f}s]", flush=True)
                json.dump({f"{d}|{int(n)}|{k}": v
                           for (d, n, k), v in results.items()},
                          open("exp14_results.json", "w"), indent=1)
