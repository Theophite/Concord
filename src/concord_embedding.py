"""Norm-preserving Concord new-token embeddings -- ALGORITHM prototype (float
accumulators; pack to int32 once the math is settled).

Textual-inversion setup: a few NEW token rows are trained while the base vocab
stays frozen. Each new row is Concord-stored (s_fast velocity / s_slow position /
v_slow anchor; deploy = s_slow + v_slow, dropping the noisy velocity) and trains
sparsely (only touched rows tick) in the backward, no optimizer.step.

NORM PRESERVATION (per the design):
  - pin the DEPLOY weight (s_slow + v_slow -- what generation ships), every step;
  - target = the MEDIAN of the frozen vocabulary's row norms. Median, not mean:
    most tokens are low-information and drag the mean around; the median is the
    "typical meaningful token" scale a new concept should sit at;
  - clamp s_fast's per-row norm so the live/forward weight can't inflate far above
    the pinned deploy norm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _EmbStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight, grad_anchor, mod):
        ctx.mod = mod
        ctx.save_for_backward(ids)
        return F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, grad_y):
        (ids,) = ctx.saved_tensors
        mod = ctx.mod
        with torch.no_grad():
            flat = ids.reshape(-1)
            g = grad_y.reshape(-1, mod.dim).float()
            uniq, inv = torch.unique(flat, return_inverse=True)   # touched rows only
            grow = torch.zeros(uniq.shape[0], mod.dim, device=g.device)
            grow.index_add_(0, inv, g)
            mod._concord_step(uniq, grow)
        return None, None, None, None


class ConcordNewTokenEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, device="cuda", lr=5e-3, alpha=0.1,
                 alpha_v=0.001, eps=1e-8, s_fast_clamp=0.5, target_norm=0.0):
        super().__init__()
        self.num_tokens, self.dim = num_tokens, dim
        self.lr, self.alpha, self.alpha_v, self.eps = lr, alpha, alpha_v, eps
        self.s_fast_clamp = s_fast_clamp
        V, D = num_tokens, dim
        z = lambda: torch.zeros(V, D, device=device)
        for n in ("s_fast", "s_slow", "v_slow", "v_hat"):
            self.register_buffer(n, z())
        self.register_buffer("target", torch.tensor(float(target_norm), device=device))
        self._grad_anchor = nn.Parameter(torch.zeros(1, device=device))

    @staticmethod
    def vocab_median_norm(vocab_weight):
        """Robust target: median row norm of the frozen vocabulary."""
        return vocab_weight.float().norm(dim=1).median().item()

    def set_target_norm(self, value):
        self.target.fill_(float(value))

    @torch.no_grad()
    def init_tokens(self, init=None, scale=0.05):
        """Init the new rows (small random, or from given vectors), then snap their
        deploy weight to the target norm."""
        if init is not None:
            self.s_slow.copy_(init.to(self.s_slow))
        else:
            self.s_slow.normal_(0, scale)
        self.s_fast.zero_(); self.v_slow.zero_()
        self._pin_deploy(torch.arange(self.num_tokens, device=self.s_slow.device))

    @property
    def weight(self):
        return self.s_fast + self.s_slow + self.v_slow

    def deploy_weight(self):
        return self.s_slow + self.v_slow

    def forward(self, ids):
        return _EmbStep.apply(ids, self.weight, self._grad_anchor, self)

    @torch.no_grad()
    def _pin_deploy(self, rows):
        dep = self.s_slow[rows] + self.v_slow[rows]
        cur = dep.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = self.target / cur
        self.s_slow[rows] *= scale
        self.v_slow[rows] *= scale

    @torch.no_grad()
    def _concord_step(self, rows, grow):
        a, av = self.alpha, self.alpha_v
        # preconditioned step into s_fast (per-element 2nd moment, Adam-style).
        self.v_hat[rows] = 0.999 * self.v_hat[rows] + 0.001 * grow * grow
        self.s_fast[rows] -= self.lr * grow / (self.v_hat[rows].sqrt() + self.eps)
        # chase s_fast -> s_slow, leak s_slow -> v_slow (mass-preserving redistribution).
        df = a * self.s_fast[rows]; self.s_slow[rows] += df; self.s_fast[rows] -= df
        dv = av * self.s_slow[rows]; self.v_slow[rows] += dv; self.s_slow[rows] -= dv
        # clamp s_fast row-norm so the live weight can't inflate far above deploy.
        cap = self.s_fast_clamp * self.target
        sfn = self.s_fast[rows].norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.s_fast[rows] *= (cap / sfn).clamp(max=1.0)
        # NORM PRESERVATION: pin the deploy weight to the (median) target.
        self._pin_deploy(rows)


class HybridCLIPEmbedding(nn.Module):
    """Drop-in replacement for a CLIP `token_embedding` (nn.Embedding): ids below
    `vocab_size` hit the FROZEN base embedding; ids at/above it route to the Concord
    new-token module (self-steps + norm-preserves in backward). "Inserts" new rows
    into the TE by swapping this one module -- the rest of the encoder is untouched.
    """
    def __init__(self, base_embedding, new_module, vocab_size):
        super().__init__()
        self.base = base_embedding              # nn.Embedding, frozen
        self.new = new_module                   # ConcordNewTokenEmbedding
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        out = self.base(input_ids.clamp(max=self.vocab_size - 1))
        is_new = input_ids >= self.vocab_size
        if is_new.any():
            new_emb = self.new(input_ids[is_new] - self.vocab_size).to(out.dtype)
            out = out.clone()
            out[is_new] = new_emb
        return out
