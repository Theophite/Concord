"""CPU reference of the Concord winner update rule (real-valued).

Mirrors `concord/packed_b.py::_apply_packed_adamw_kernel` at the winner
configuration, minus the integer realization: stochastic rounding makes the
packed kernel equal this rule in expectation, so fp32 state stands in for the
int32 word. Used by the small CPU experiments in this directory; NOT a
training path (the real thing is the Triton kernel).

Per 2D weight, in weight units:
    u  = velocity                (s_fast · scale)
    S  = slow position half      (128 · s_slow · scale)
    A  = anchor half             (128 · v_slow · scale)
    P  = S + A                   (deploy weight)
    W  = P + u                   (live weight, what the forward uses)

Init mirrors `load_weights`: the whole pretrained/initial weight starts in
the velocity (u = W0, S = A = 0) and the chase consolidates it over the
first ~1/alpha steps — note this makes the lr *warmup* load-bearing (the
friction is lr-proportional, so init mass consolidates before friction
turns on).

1D params (biases, norms) use plain SGD(momentum=0.9), like the fork's aux
optimizer. Schedules follow `concord_winner.winner_step`.
"""
import math

import torch


def compute_drift_cancel_C(alpha=0.1, alpha_v=0.001, mass_preserve=True):
    L = (1.0 - alpha) / alpha
    if mass_preserve:
        # telescope relaxes at 2*alpha_v under the mass-preserving leak
        return L * 2 * alpha_v / (1.0 - 2 * alpha_v)
    return L * alpha_v / (1.0 - L * alpha_v)   # legacy (shipped) value


def evap_term(lr, kappa, coh, u, min_leak=0.1):
    """Dissipation with the min-leak servo floor (kernel parity: fork 4c786433,
    canonical 4eaa704). The per-step evaporated fraction is capped at
    1 - min_leak so the valve never fully shuts: at lam = lr*kappa -> 1 with
    coh ~ 0 the unclamped term wipes u's history each step -- nothing
    accumulates, the drift freezes, coh pins at 0 and the gate self-seals.
    No-op while lam*(1-coh) <= 1 - min_leak (the whole exp-5 grid)."""
    if kappa <= 0:
        return 0.0
    frac = (lr * kappa * (1.0 - coh)).clamp_(max=1.0 - min_leak)
    return frac * u


def _cos_floor(start, end, it, horizon):
    if it >= horizon:
        return end
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * it / horizon))


class ConcordRef:
    """Reference optimizer. gate=True -> ratio-coherence gates + evaporation
    (the "split" dissipation); noise=True -> rising-late isotropic gradient
    noise (the fluctuation). gate=False & noise=False & kappa=0 -> the bare
    recipe with fully-open consolidation."""

    def __init__(self, model, lr=1e-3, total_steps=1000, warmup=100,
                 alpha=0.1, alpha_v=0.001, beta1=0.0, beta2=0.999,
                 eps=1e-10, step_cap=10.0, kappa=50.0, friction_F=None,
                 sigma_peak=0.6, lr_min_frac=0.2,
                 chase_floor=(0.9, 0.1), leak_floor=(0.999, 0.1),
                 gate=True, noise=True, generator=None, min_leak=0.1):
        self.peak_lr = lr
        self.T = total_steps
        self.warmup = warmup
        self.alpha, self.alpha_v = alpha, alpha_v
        self.beta1, self.beta2 = beta1, beta2
        # friction_F: the DIMENSIONLESS friction F = lr*kappa at peak lr.
        # Preferred over kappa (which is per-unit-lr): F decouples the friction
        # sweep from the lr sweep, F < 2 is the lr-independent stability
        # ceiling, and the per-step friction F_t = F*(lr_t/lr_peak) inherits
        # the schedule's auto-fade (warmup protects init consolidation, the
        # cosine tail lets the position settle). kappa = F/lr_peak is derived.
        if friction_F is not None:
            kappa = friction_F / lr
        self.eps, self.cap, self.kappa = eps, step_cap, kappa
        self.sigma_peak, self.lr_min_frac = sigma_peak, lr_min_frac
        self.chase_floor, self.leak_floor = chase_floor, leak_floor
        self.gate, self.noise = gate, noise
        self.min_leak = float(min_leak)
        self.Cstar = compute_drift_cancel_C(alpha, alpha_v)
        self.gen = generator
        self.t = 0
        self.lr = 0.0
        self.sigma = 0.0

        self.swapped, self.aux = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 2:
                st = {
                    "p": p,
                    "u": p.detach().clone(),            # load_weights: all mass fast
                    "S": torch.zeros_like(p),
                    "A": torch.zeros_like(p),
                    "v_row": torch.zeros(p.shape[0]),
                    "v_col": torch.zeros(p.shape[1]),
                }
                self.swapped.append(st)
            else:
                self.aux.append({"p": p, "m": torch.zeros_like(p)})

    # ── schedules (winner_step) ────────────────────────────────────────
    def _advance_schedules(self):
        t, T = self.t, self.T
        p = min(1.0, t / max(1, T))
        f = self.lr_min_frac + 0.5 * (1 - self.lr_min_frac) * (1 + math.cos(math.pi * p))
        warm = min(1.0, (t + 1) / self.warmup) if self.warmup > 0 else 1.0
        self.lr = self.peak_lr * f * warm
        self.sigma = self.sigma_peak * (1.0 - f) if self.noise else 0.0
        self.phic = _cos_floor(*self.chase_floor, t, T) if self.gate else 1.0
        self.phil = _cos_floor(*self.leak_floor, t, T) if self.gate else 1.0

    # ── the step (kernel order) ────────────────────────────────────────
    @torch.no_grad()
    def step(self):
        self._advance_schedules()
        lr = self.lr
        for st in self.swapped:
            g = st["p"].grad
            if g is None:
                continue
            g = g.float()
            # fluctuation (before v-hat, as in PackedLinearFn.backward)
            if self.sigma > 0:
                xi = torch.randn(g.shape, generator=self.gen)
                g = g + xi * (self.sigma * g.norm() / xi.norm().clamp_min(1e-12))
            # AdaFactor rank-1 second moment
            g2 = g * g
            st["v_row"].mul_(self.beta2).add_(g2.sum(1), alpha=1 - self.beta2)
            st["v_col"].mul_(self.beta2).add_(g2.sum(0), alpha=1 - self.beta2)
            vhat = (st["v_row"][:, None] * st["v_col"][None, :]
                    / st["v_row"].sum().clamp_min(1e-30))
            u, S, A = st["u"], st["S"], st["A"]
            # Wiener/Kalman gain from the telescope (pre-update values)
            if self.gate:
                mu = self.Cstar * (S - A)
                nse = u - mu
                coh = (mu * mu) / (mu * mu + nse * nse + 1e-30)
            else:
                coh = torch.zeros_like(u)   # bare: no gate consumers below
            # drive + dissipation (+ optional coherence-gated momentum)
            step_ = (g / (vhat + self.eps).sqrt()).clamp_(-self.cap, self.cap)
            evap = evap_term(lr, self.kappa, coh, u, self.min_leak)
            u += self.beta1 * coh * u - lr * step_ - evap
            # chase (continuous Lookahead; W-invariant)
            gc = self.phic + (1 - self.phic) * coh if self.gate else 1.0
            tr = self.alpha * gc * u
            S += tr
            u -= tr
            # leak (P-invariant; advances the telescope)
            gl = self.phil + (1 - self.phil) * coh if self.gate else 1.0
            lk = self.alpha_v * gl * (S - A)
            A += lk
            S -= lk
            st["p"].copy_(u + S + A)
            st["last_coh"] = float(coh.mean()) if self.gate else 1.0
        for st in self.aux:   # aux SGD, momentum 0.9 (modules/util/create.py)
            if st["p"].grad is None:
                continue
            st["m"].mul_(0.9).add_(st["p"].grad)
            st["p"].add_(st["m"], alpha=-lr)
        self.t += 1

    def zero_grad(self):
        for st in self.swapped + self.aux:
            st["p"].grad = None

    # ── deploy weights (consolidated_weight: drop u) ───────────────────
    def deploy_state(self):
        return {id(st["p"]): st["S"] + st["A"] for st in self.swapped}

    def mean_coh(self):
        cs = [st.get("last_coh", 0.0) for st in self.swapped]
        return sum(cs) / max(1, len(cs))


class swap_to_deploy:
    """Context manager: evaluate the model at the deploy weights P."""

    def __init__(self, opt):
        self.opt = opt

    def __enter__(self):
        self.stash = []
        dep = self.opt.deploy_state()
        for st in self.opt.swapped:
            p = st["p"]
            self.stash.append((p, p.detach().clone()))
            with torch.no_grad():
                p.copy_(dep[id(p)])

    def __exit__(self, *a):
        for p, w in self.stash:
            with torch.no_grad():
                p.copy_(w)


def adamw_with_winner_schedule(params, lr, total_steps, warmup=100,
                               lr_min_frac=0.2):
    """AdamW baseline under the identical lr schedule."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def set_lr(t):
        p = min(1.0, t / max(1, total_steps))
        f = lr_min_frac + 0.5 * (1 - lr_min_frac) * (1 + math.cos(math.pi * p))
        warm = min(1.0, (t + 1) / warmup) if warmup > 0 else 1.0
        for grp in opt.param_groups:
            grp["lr"] = lr * f * warm

    return opt, set_lr
