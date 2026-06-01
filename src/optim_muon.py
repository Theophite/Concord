"""Faithful reference Muon (Keller Jordan's MomentUm Orthogonalized by Newton-schulz),
for the head-to-head: does REAL Muon (native fp, proper impl) beat AdamW on tiny-shakespeare?
This is the control we should have run before any Concord-cascade orthogonalization.

Muon step (per 2D weight): m = beta*m + g (momentum); O = NS5(Nesterov(m)); W -= lr * O *
sqrt(max(1, rows/cols)). NS5 = the quintic that drives singular values -> ~1. 2D hidden
weights only; embeddings/head/norms/scalars go to a standard AdamW (the standard Muon split).

Ref coeffs (3.4445, -4.7750, 2.0315), 5 NS iters, bf16 matmuls, normalize by spectral-norm
proxy (Frobenius) before iterating. Matches modded-nanogpt's Muon closely.
"""
import torch


@torch.no_grad()
def _ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.t()
    return X


class Muon(torch.optim.Optimizer):
    """Muon on 2D params. Pass ONLY >=2D hidden weights here; route everything else
    (embeddings, lm_head, norms, biases) to a separate AdamW."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']; mom = group['momentum']
            nesterov = group['nesterov']; ns = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, f"Muon is 2D-only, got {g.shape}"
                st = self.state[p]
                if 'm' not in st:
                    st['m'] = torch.zeros_like(g)
                buf = st['m']
                buf.mul_(mom).add_(g)                      # momentum
                u = g.add(buf, alpha=mom) if nesterov else buf   # Nesterov lookahead
                O = _ns5(u, ns).to(g.dtype)                # orthogonalize
                # RMS-match scale: tall matrices get sqrt(rows/cols) (Jordan's scaling)
                scale = (max(1.0, p.shape[0] / p.shape[1])) ** 0.5
                p.add_(O, alpha=-lr * scale)
