"""Single-sample factored g^2 precond  vs  the deviation D^2 fixed point.

Same closed loop as exp7, but the preconditioner estimate is built from a
FRESH per-step source, factored rank-1:
    est = rank1(base^2),  base = G (single-sample grad)  OR  D=W-Wbar (deviation)
    U   = eta * G / (est/mean + eps)^p

Key difference: the deviation D is shaped BY the precond (self-referential),
so its estimate self-degrades to sigma^(2/(1+2p)) -> can't fully whiten.
A single-sample G has E[G^2]=sigma^2 UNdegraded (minibatch noise, precond-
independent), so factoring it should fully whiten at finite p.

Predicted heterogeneity exponent k (E[U^2] ∝ sigma^k):
    deviation:      k = 2/(1+2p)     (exp7; ->0 only as p->inf)
    single-sample:  k = 2 - 4p       (->0 at p=0.5, <0 = over-whiten past it)
Full Adam = k=0. Lower |k| near 0 = correctly whitened.
"""
import numpy as np

m, n = 48, 48
beta, eta, eps = 0.999, 0.01, 0.01
WARM, T = 4000, 20000
rng = np.random.default_rng(0)


def run(spread, p, src):
    hi = spread ** 0.25
    a = np.exp(np.linspace(-np.log(hi), np.log(hi), m))
    b = np.exp(np.linspace(-np.log(hi), np.log(hi), n))
    sigma = a[:, None] * b[None, :]
    W = np.zeros((m, n)); Wbar = np.zeros((m, n))
    accU2 = np.zeros((m, n)); cnt = 0
    for t in range(T):
        G = sigma * rng.standard_normal((m, n))
        if src == "dev":
            base = W - Wbar
        else:                       # single-sample gradient
            base = G
        b2 = base * base
        R = b2.mean(1); C = b2.mean(0)
        est = R[:, None] * C[None, :] / (R.mean() + 1e-30)
        pe = p if t >= WARM else 0.0
        precond = (est / (est.mean() + 1e-30) + eps) ** pe
        U = eta * G / precond
        W = W - U; Wbar = beta * Wbar + (1 - beta) * W
        if t >= T - 8000:
            accU2 += U * U; cnt += 1
    EU2 = accU2 / cnt
    ls = np.log(sigma).ravel(); lu = np.log(EU2).ravel()
    k = np.polyfit(ls, lu, 1)[0]
    blow = (not np.isfinite(W).all()) or (np.abs(W).max() > 1e6)
    return k, blow


for src, kpred in (("dev", "2/(1+2p)"), ("grad", "2-4p")):
    print(f"\n=== src={src}   k_pred = {kpred} ===")
    print("  p     k_pred  |  k @ spread 1e2 / 1e4 / 1e6     stable?")
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        kp = (2.0 / (1.0 + 2.0 * p)) if src == "dev" else (2.0 - 4.0 * p)
        row, stab = [], []
        for spread in (1e2, 1e4, 1e6):
            k, blow = run(spread, p, src)
            row.append(f"{k:5.2f}"); stab.append("BLOW" if blow else "ok")
        print(f"  {p:4.2f}  {kp:5.2f}   |  " + " / ".join(row)
              + f"     {'/'.join(stab)}")
print("\n  k=0 = full whitening (Adam). single-sample should reach 0 at p=0.5;")
print("  deviation stays ~1 at p=0.5. If so, single-sample is the right proxy.")
