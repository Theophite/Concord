"""Test the claim: for an UNWHITENED optimizer (concord, U≈ηG), the slow
deviation D = W - W_bar (W_bar = 0.999 EMA) carries the per-element gradient
2nd moment, and you can recover it for FREE by rank-1 factoring D^2
instantaneously (no accumulator).

True per-element grad std sigma_ij = a_i * b_j (rank-1, heterogeneous, ~625x
spread). Predicted: E[D_ij^2] = eta^2 * sigma_ij^2 / (2(1-beta)).
Recover via R~_i = mean_j D_ij^2, C~_j = mean_i D_ij^2,
  est_ij = R~_i * C~_j / mean(R~).
Check: does est_ij track sigma_ij^2?  How much does factoring de-noise vs the
raw single-sample D_ij^2?
"""
import numpy as np

m, n = 64, 64
beta = 0.999
eta = 0.01
T = 40000
rng = np.random.default_rng(0)

a = np.exp(np.linspace(np.log(0.2), np.log(5.0), m))   # row scales
b = np.exp(np.linspace(np.log(0.2), np.log(5.0), n))   # col scales
sigma = a[:, None] * b[None, :]                        # per-elem grad std
sig2 = sigma ** 2                                      # true E[G^2], rank-1

W = np.zeros((m, n)); Wbar = np.zeros((m, n))
for t in range(T):
    G = sigma * rng.standard_normal((m, n))            # zero-mean hetero noise
    W = W - eta * G                                    # unwhitened SGD step
    Wbar = beta * Wbar + (1 - beta) * W

D = W - Wbar
D2 = D * D

# rank-1 factored estimate of E[D^2] (zero extra state)
Rt = D2.mean(axis=1)                                   # row means
Ct = D2.mean(axis=0)                                   # col means
est = Rt[:, None] * Ct[None, :] / Rt.mean()

pred_scale = eta ** 2 / (2 * (1 - beta))               # predicted E[D2]/sig2
E_D2_pred = pred_scale * sig2

def cov(x):  # coefficient of variation (std/mean) -- 0 = perfect recovery
    return float(np.std(x) / np.mean(x))

print(f"m={m} n={n} beta={beta} eta={eta} T={T}")
print(f"sigma^2 spread: {sig2.min():.4f} .. {sig2.max():.2f}  "
      f"({sig2.max()/sig2.min():.0f}x)\n")

# 1) does factored est track the TRUE per-element scale sigma^2?
ratio_fac = est / sig2
print(f"factored est / sigma^2 :  mean={ratio_fac.mean():.4e}  "
      f"CoV={cov(ratio_fac):.3f}   (predicted scale {pred_scale:.4e})")
# 2) raw single-sample D^2 / sigma^2 (no factoring) -- how noisy?
ratio_raw = D2 / sig2
print(f"raw D^2     / sigma^2 :  mean={ratio_raw.mean():.4e}  "
      f"CoV={cov(ratio_raw):.3f}   <- single-sample noise")
# 3) sqrt: the precond proxy.  est^0.25 vs sigma^0.5 (the sqrt-v_hat target)
#    (est ~ sig2, so est^0.5 ~ sigma ; the per-elem precond proxy)
prec = np.sqrt(est)
print(f"\nsqrt(est) / sigma     :  mean={(prec/sigma).mean():.4e}  "
      f"CoV={cov(prec/sigma):.3f}   (this is the free 1/precond proxy)")
print(f"\nlog-corr(est, sigma^2) = "
      f"{np.corrcoef(np.log(est).ravel(), np.log(sig2).ravel())[0,1]:.4f}")
