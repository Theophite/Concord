"""Closed loop: precondition by the factored slow-deviation, U = eta*G /
(est/mean + eps)^p, est = rank-1 factored D^2, D = W - W_bar (beta=0.999).
Since est is built from the SAME D the precond whitens, expect a
self-consistent PARTIAL whitening. Fixed-point algebra:
  precond ∝ (E[D^2])^p ∝ (E[U^2])^p, and E[U^2] ∝ sigma^2 / precond^2
  => precond ∝ sigma^(2p/(1+2p)),  E[U^2] ∝ sigma^(2/(1+2p)).
So measured heterogeneity exponent k (E[U^2] ∝ sigma^k) should match
k_pred = 2/(1+2p): p=0 ->2 (none), 0.5 ->1 (half), 1 ->0.667, 2 ->0.4.
Full Adam (k=0) needs p->inf -- i.e. the self-loop can't fully whiten.
Tested at several sigma-spread scales to confirm scale-invariance + stability.
"""
import numpy as np

m, n = 48, 48
beta, eta, eps = 0.999, 0.01, 0.01
WARM, T = 4000, 20000          # warm p=0 so D reaches stationary, then on
rng = np.random.default_rng(0)


def run(spread, p):
    hi = spread ** 0.25         # a_i,b_j in [1/hi, hi] -> sigma^2 spread^1
    a = np.exp(np.linspace(-np.log(hi), np.log(hi), m))
    b = np.exp(np.linspace(-np.log(hi), np.log(hi), n))
    sigma = a[:, None] * b[None, :]
    W = np.zeros((m, n)); Wbar = np.zeros((m, n))
    accU2 = np.zeros((m, n)); cnt = 0
    for t in range(T):
        G = sigma * rng.standard_normal((m, n))
        D = W - Wbar
        D2 = D * D
        R = D2.mean(1); C = D2.mean(0)
        est = R[:, None] * C[None, :] / (R.mean() + 1e-30)
        pe = p if t >= WARM else 0.0
        precond = (est / (est.mean() + 1e-30) + eps) ** pe
        U = eta * G / precond
        W = W - U; Wbar = beta * Wbar + (1 - beta) * W
        if t >= T - 8000:
            accU2 += U * U; cnt += 1
    EU2 = accU2 / cnt
    # heterogeneity exponent: slope of log E[U^2] vs log sigma
    ls = np.log(sigma).ravel(); lu = np.log(EU2).ravel()
    k = np.polyfit(ls, lu, 1)[0]               # E[U^2] ∝ sigma^k
    blow = (not np.isfinite(W).all()) or (np.abs(W).max() > 1e6)
    return k, blow


print(f"m={m} beta={beta} eta={eta} eps={eps}  (k_pred = 2/(1+2p))\n")
print("  p      k_pred   |  k measured @ spread=1e2 / 1e4 / 1e6   stable?")
for p in (0.0, 0.25, 0.5, 1.0, 2.0):
    kp = 2.0 / (1.0 + 2.0 * p)
    row = []
    stab = []
    for spread in (1e2, 1e4, 1e6):
        k, blow = run(spread, p)
        row.append(f"{k:5.2f}")
        stab.append("BLOW" if blow else "ok")
    print(f"  {p:4.2f}   {kp:5.2f}    |  " + " / ".join(row)
          + f"    {'/'.join(stab)}")
print("\n  k=2 no whitening (SGD), k=0 full Adam. Lower k = more whitened.")
print("  If measured k ~ 2/(1+2p): self-loop is a PARTIAL whitener; full")
print("  Adam (k=0) is unreachable without decoupling est from the loop.")
