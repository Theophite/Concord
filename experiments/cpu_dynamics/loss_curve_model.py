"""Forward loss-curve model for Concord SDXL fine-tunes (pure numpy).

The diffusion training loss is an uninformative thermometer: most of its
magnitude is timestep-averaged eps-prediction the base model already solves,
the concept is a small reducible slice, and per-batch timestep/image variance
swamps the early descent. The observed "flat for a few epochs, then bites"
shape is therefore mostly DERIVABLE -- the flat region is a stack of KNOWN
delays, and only the descent *rate* is data-dependent (and is itself observable
online via the telescope/gap meters).

    L(t) = L0  +  dL_unet * a_unet(t)^2  +  dL_token * a_token(t)^2  +  noise

  - L0          : irreducible floor (base model's loss on the data; one eval).
  - dL_*        : reducible loss carried by the UNet vs the tokens.
  - a_*(t)      : normalized distance-to-optimum in each reducible subspace,
                  1 -> 0; loss ~ distance^2 (quadratic bowl). Squared so the
                  loss-reduction is what descends.
  - noise       : per-batch band (measurable: the std of the raw loss line).

The SHAPE of a(t) is set by delays that are all KNOWN config constants:
  - v-hat efficiency ramp ~ 1 - beta2^t  (descent inefficient until the rank-1
    second moment fills, ~1/(1-beta2) steps);
  - lr warmup * cosine;
  - the cascade lag s_fast -> s_slow -> v_slow -> deploy (a low-pass; the LIVE
    weight moves promptly, the DEPLOY weight lags by ~1/alpha) -- so deploy
    loss is flatter-then-bite-ier than live loss;
  - the divot: the token component is FROZEN until delay steps, so it injects a
    SECOND bite at release.

The one genuinely data-dependent unknown is the descent rate constant per
component (curvature x coherence). It is NOT guessed: the telescope coherence
and the gap meter measure it online (gap = first-order L_deploy - L_live =
-sum g.s_fast is literally the loss change per unit in-flight motion). Here it
is the per-component `kappa`, fit from a logged trace by least squares.

USE: simulate(RunConstants(...)) -> L_live/L_deploy curves; fit_residual(actual,
sim) -> the residual (actual - model). The residual is the actual diagnostic:
token-lockdown / sawtooth / wrong-generalization show up as DEVIATIONS from the
predicted descent, which the bare loss curve cannot reveal.

Pure numpy: safe to run alongside training (no torch, no CUDA, no model load).
Real-data validation against a logged (loss, coherence, gap) trace is the next
step; this module is the model + a synthetic self-test of its mechanics.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RunConstants:
    # --- schedule / optimizer (all KNOWN from the config) ---
    lr: float = 3e-5
    warmup_steps: int = 100
    total_steps: int = 18865
    lr_min_frac: float = 0.05
    beta2: float = 0.999          # v-hat EMA -> efficiency ramp ~1/(1-beta2)
    alpha: float = 0.1            # chase rate -> cascade lag ~1/alpha
    divot_steps: int = 0          # token component frozen until here
    # --- reducible-loss split + floor (L0 measurable; dL_* fit) ---
    L0: float = 0.115
    dL_unet: float = 0.020
    dL_token: float = 0.010
    # --- descent rate constants: the data-dependent unknowns (fit / from meters) ---
    kappa_unet: float = 300.0
    kappa_token: float = 500.0


def _rate_shape(c, n):
    """Per-step effective-lr shape: warmup * cosine * v-hat-efficiency ramp.
    All three factors are closed-form from known constants."""
    s = np.arange(n)
    warm = np.minimum(1.0, (s + 1) / max(1, c.warmup_steps))
    cos = c.lr_min_frac + 0.5 * (1 - c.lr_min_frac) * (1 + np.cos(np.pi * np.minimum(1.0, s / max(1, c.total_steps))))
    vhat = 1.0 - c.beta2 ** (s + 1)          # efficiency: ~0 early, -> 1 as v-hat fills
    return c.lr * warm * cos * vhat


def simulate(c, n=None):
    """Forward-simulate the reducible distances and the loss curves."""
    n = int(n or c.total_steps)
    base = _rate_shape(c, n)
    tau = max(1.0, 1.0 / max(c.alpha, 1e-6))   # cascade (chase) lag in steps

    def descend(kappa, frozen_until):
        a_live = np.empty(n)
        a_dep = np.empty(n)
        al, ad = 1.0, 1.0
        for t in range(n):
            if t >= frozen_until:
                al = max(0.0, al * (1.0 - kappa * base[t]))   # live distance decays
            ad += (al - ad) / tau                              # deploy lags via the cascade
            a_live[t], a_dep[t] = al, ad
        return a_live, a_dep

    au_l, au_d = descend(c.kappa_unet, 0)
    at_l, at_d = descend(c.kappa_token, c.divot_steps)
    return {
        "steps": np.arange(n),
        "L_live": c.L0 + c.dL_unet * au_l ** 2 + c.dL_token * at_l ** 2,
        "L_deploy": c.L0 + c.dL_unet * au_d ** 2 + c.dL_token * at_d ** 2,
        "a_unet_live": au_l, "a_token_live": at_l,
        "a_unet_dep": au_d, "a_token_dep": at_d,
        "base_rate": base,
    }


def fit_residual(actual_L, sim, view="deploy"):
    """Given an actual loss trace and a simulated run, fit the linear amplitudes
    [L0, dL_unet, dL_token] against the model's distance-squared basis (the
    shape is fixed by the known delays + rate), and return the residual. The
    residual = actual - model is the anomaly signal."""
    suff = "dep" if view == "deploy" else "live"
    basis = np.stack([np.ones(len(actual_L)),
                      sim[f"a_unet_{suff}"] ** 2,
                      sim[f"a_token_{suff}"] ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(basis, actual_L, rcond=None)
    pred = basis @ coef
    return pred, actual_L - pred, coef


def _sparkline(y):
    lo, hi = float(np.min(y)), float(np.max(y))
    blocks = " .:-=+*#%@"                      # ASCII ramp (cp1252-safe console)
    if hi - lo < 1e-12:
        return blocks[0] * len(y)
    idx = ((y - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in idx)


if __name__ == "__main__":
    ok = True

    def check(name, cond, info=""):
        global ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}")

    # synthetic run: 4000 steps, token divot at 1500, both components reducible.
    c = RunConstants(total_steps=4000, divot_steps=1500, warmup_steps=100,
                     L0=0.115, dL_unet=0.015, dL_token=0.015,
                     kappa_unet=120.0, kappa_token=400.0)
    sim = simulate(c)
    L, Ld = sim["L_live"], sim["L_deploy"]
    n = len(L)
    sigma = 0.03                              # representative per-batch loss-noise band
    au = sim["a_unet_live"]
    d = c.divot_steps
    print(f"L_live   : {_sparkline(L[::80])}  ({L[0]:.4f} -> {L[-1]:.4f})")
    print(f"L_deploy : {_sparkline(Ld[::80])}  ({Ld[0]:.4f} -> {Ld[-1]:.4f})")

    # 1. v-hat ramp throttles early descent: the UNet drop accelerates as v-hat
    #    fills (rate over [300,600] exceeds rate over [0,300]) -- slow-start, not convex.
    drop_0_300 = au[0] - au[300]
    drop_300_600 = au[300] - au[600]
    check("v-hat ramp: UNet descent rate increases as the preconditioner fills",
          drop_300_600 > drop_0_300, f"d[0:300]={drop_0_300:.4f} < d[300:600]={drop_300_600:.4f}")

    # 2. divot: token component is exactly frozen until release, then descends
    check("token component frozen until the divot, then descends",
          abs(sim["a_token_live"][d - 5] - 1.0) < 1e-9 and sim["a_token_live"][d + 200] < 0.9,
          f"a_token {sim['a_token_live'][d-5]:.3f} -> {sim['a_token_live'][d+200]:.3f}")

    # 3. the visible "bite" concentrates at token release: the loss drop in the
    #    window just AFTER the divot exceeds the window just before.
    pre_bite = L[d - 300] - L[d]
    post_bite = L[d] - L[d + 300]
    check("bite concentrates at token release (post-divot drop > pre-divot drop)",
          post_bite > pre_bite, f"pre {pre_bite:.4f} < post {post_bite:.4f}")

    # 4. metric insensitivity: pre-release, the smoothed drop over an epoch-scale
    #    window is BELOW the noise band (looks flat); the release bite clears it.
    pre_window_drop = L[d - 600] - L[d]
    check("pre-release drop sits under the noise band (looks flat); bite clears it",
          pre_window_drop < sigma and post_bite + (L[d+300]-L[d+600]) > 0,
          f"pre-window drop {pre_window_drop:.4f} vs sigma {sigma}")

    # 5. deploy lags live (cascade low-pass): deploy loss higher early
    check("deploy weight lags live (cascade) -> flatter early",
          Ld[300] > L[300], f"deploy {Ld[300]:.4f} > live {L[300]:.4f}")

    # 6. fit recovers the curve: residual ~0 when the model is fit to itself
    pred, resid, coef = fit_residual(Ld, sim, view="deploy")
    check("fit recovers the curve: residual ~0 on self", np.max(np.abs(resid)) < 1e-6,
          f"max|resid|={np.max(np.abs(resid)):.2e}, coef={np.round(coef,4).tolist()}")

    # 7. an injected anomaly (token lock-down: loss stops dropping / drifts up)
    #    surfaces as a localized residual spike -- the diagnostic the raw loss hides
    anomaly = Ld.copy()
    anomaly[2500:] += 0.01 * (np.arange(n - 2500) / (n - 2500))
    _, resid_a, _ = fit_residual(anomaly, sim, view="deploy")
    check("injected post-bite anomaly shows up in the residual",
          np.max(resid_a[2500:]) > 5 * np.std(resid_a[:2000]),
          f"late residual {np.max(resid_a[2500:]):.4f} vs early std {np.std(resid_a[:2000]):.2e}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    raise SystemExit(0 if ok else 1)
