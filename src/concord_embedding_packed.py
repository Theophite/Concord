"""Norm-preserving Concord new-token embedding, PACKED (32 b/param) -- the clean
version that reuses packed_b's real cascade instead of re-implementing it.

Storage/optimizer = a ConcordLinearPackedB(in=dim, out=K): packed_w is [K, dim]
(one int32 per element: s_fast int16 + s_slow int8 + v_slow int8), and row_exp is
PER-TOKEN (out-row). The forward is a gather; the backward scatters the per-row
grad and drives packed_b's own fused cascade by feeding grad_W = grad_y^T @ I
through core's autograd Function (an identity matmul -- cheap for the few new
tokens of textual inversion). Then norm preservation pins each touched token's
DEPLOY norm to the target (vocab median): power-of-2 via row_exp + a mantissa
residual (col_exp=0, so this is exact per token).
"""
import torch
import torch.nn as nn

import prototype_packed_b as ppb
from prototype_packed_b import (ConcordLinearPackedB, INT16_MIN, INT16_MAX,
                                S_SLOW_FACTOR, V_SLOW_FACTOR)

MB = ConcordLinearPackedB.MANTISSA_BIAS
E_MIN, E_MAX = ConcordLinearPackedB.EXP_MIN, ConcordLinearPackedB.EXP_MAX


class _PackedEmbStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, anchor, mod):
        ctx.mod = mod
        ctx.save_for_backward(ids)
        return mod.core.get_weight()[ids]               # gather live bf16 weight

    @staticmethod
    def backward(ctx, grad_emb):
        (ids,) = ctx.saved_tensors
        mod = ctx.mod
        core = mod.core
        # scatter per-position grad into a per-token grad_W [K, dim].
        G = torch.zeros(mod.K, mod.dim, device=grad_emb.device)
        G.index_add_(0, ids.reshape(-1), grad_emb.reshape(-1, mod.dim).float())
        # drive packed_b's fused cascade: y = I @ W^T -> grad_W = grad_y^T @ I = G.
        with torch.enable_grad():
            x = mod._I.requires_grad_(True)
            y = core(x)
            y.backward(G.t().to(y.dtype))
        core._resync_weight_buf()
        # Pin ALL K rows (static shape -> CUDA-graph capturable). torch.unique would be
        # dynamic-shaped AND sync. K is tiny and untouched rows are already at target,
        # so re-pinning them is a near-no-op.
        mod._pin_norm(torch.arange(mod.K, device=ids.device))
        return None, None, None


class ConcordPackedEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, device="cuda", lr=5e-2, alpha=0.1,
                 target_norm=1.0):
        super().__init__()
        self.K, self.dim = num_tokens, dim
        self.core = ConcordLinearPackedB(dim, num_tokens, bias=False,
                                         device=device, alpha=alpha, lr=lr)
        self.register_buffer("target", torch.tensor(float(target_norm), device=device))
        self.register_buffer("_I", torch.eye(dim, device=device, dtype=torch.bfloat16))
        self._grad_anchor = nn.Parameter(torch.zeros(1, device=device))

    @staticmethod
    def vocab_median_norm(vocab_weight):
        return vocab_weight.float().norm(dim=1).median().item()

    def set_target_norm(self, v):
        self.target.fill_(float(v))

    @torch.no_grad()
    def init_tokens(self, init=None, scale=0.05):
        if init is None:
            init = torch.randn(self.K, self.dim, device=self.target.device) * scale
        self.core.load_weights(init)                     # mantissa lands in s_fast
        # move the position into s_slow so DEPLOY (s_slow+v_slow) is non-zero at init
        # (else pinning the deploy norm divides by ~0). s_slow is the x128 coarse field.
        pw = self.core.packed_w
        sf = (pw >> 16)
        ss = (sf.float() / S_SLOW_FACTOR).round().clamp(-128, 127).to(torch.int32)
        sf = (sf - ss * S_SLOW_FACTOR).clamp(INT16_MIN, INT16_MAX).to(torch.int32)
        self.core.packed_w.copy_(((sf & 0xFFFF) << 16) | ((ss & 0xFF) << 8))
        self.core._resync_weight_buf()
        self._pin_norm(torch.arange(self.K, device=self.target.device))

    def deploy_weight(self):
        return self.core.consolidated_weight()           # [K, dim], drop s_fast

    @torch.no_grad()
    def save(self, path):
        """Save the deployable embedding(s) [K, dim] -- reuse them, or feed back as
        an init vector (resolve_token_init accepts a tensor) to continue/transfer."""
        torch.save(self.deploy_weight().detach().cpu(), path)

    def forward(self, ids):
        return _PackedEmbStep.apply(ids, self._grad_anchor, self)

    @torch.no_grad()
    def _pin_norm(self, rows):
        core = self.core
        pw = core.packed_w[rows]
        s_fast = (pw >> 16)
        s_slow = ((pw << 16) >> 24)
        v_slow = ((pw << 24) >> 24)
        # deploy norm of each touched row (col_exp == 0 here).
        m_slow = s_slow.float() * S_SLOW_FACTOR + v_slow.float() * V_SLOW_FACTOR
        exp = (core.row_exp[rows, None].to(torch.float32)
               + core.col_exp[None, :].to(torch.float32) - MB)
        norm = (m_slow * torch.pow(2.0, exp)).norm(dim=1, keepdim=True).clamp_min(1e-20)
        scale = self.target / norm                       # [len(rows), 1]
        # power-of-2 via row_exp (lossless), residual r ~ [0.71, 1.41] via mantissa.
        e = torch.round(torch.log2(scale))
        new_exp = (core.row_exp[rows].float() + e.squeeze(1)).clamp(E_MIN, E_MAX)
        e = (new_exp - core.row_exp[rows].float())        # actually-applied exp delta
        core.row_exp[rows] = new_exp.to(core.row_exp.dtype)
        r = (scale / torch.pow(2.0, e.unsqueeze(1)))
        s_fast = (s_fast.float() * r).round().clamp(INT16_MIN, INT16_MAX).to(torch.int32)
        s_slow = (s_slow.float() * r).round().clamp(-128, 127).to(torch.int32)
        v_slow = (v_slow.float() * r).round().clamp(-128, 127).to(torch.int32)
        core.packed_w[rows] = (((s_fast & 0xFFFF) << 16)
                               | ((s_slow & 0xFF) << 8) | (v_slow & 0xFF))
        core._resync_weight_buf()


def resolve_token_init(specs, tokenizer, base_embedding, device="cuda"):
    """Resolve a per-new-token initializer list into a [K, dim] init tensor. Each spec:
      - str  : an INITIALIZER WORD -> mean of its frozen-vocab token embeddings
               (the new token starts pointing where that word points);
      - Tensor [dim] : an explicit init vector (e.g. torch.load'd from a saved file);
      - None : small random.
    Norm is handled afterward by init_tokens -> _pin_norm (the median target), so only
    the DIRECTION of the initializer matters here."""
    dim = base_embedding.weight.shape[1]
    rows = []
    for s in specs:
        if isinstance(s, str):
            ids = tokenizer(s, add_special_tokens=False).input_ids
            v = base_embedding.weight[ids].float().mean(0)
        elif torch.is_tensor(s):
            v = s.float().reshape(dim)
        else:
            v = torch.randn(dim) * 0.05
        rows.append(v.to(device))
    return torch.stack(rows)


def insert_new_tokens(te, tokenizer, names, init_specs=None, lr=5e-3, device="cuda"):
    """Add `names` to `tokenizer` and insert a norm-preserving Concord embedding for
    them into `te` (swap its token_embedding for a HybridCLIPEmbedding). `init_specs`
    is a per-token initializer (word / vector / None); target norm = the TE's vocab
    median. Returns the trainable ConcordPackedEmbedding. Centralizes the TI wiring."""
    from concord_embedding import HybridCLIPEmbedding
    base = te.get_input_embeddings()
    vocab, dim = base.weight.shape
    median = ConcordPackedEmbedding.vocab_median_norm(base.weight)
    for n in names:
        tokenizer.add_tokens(n)
    init = resolve_token_init(init_specs or [None] * len(names), tokenizer, base, device)
    nm = ConcordPackedEmbedding(len(names), dim, device=device, lr=lr, target_norm=median)
    nm.init_tokens(init=init)
    te.text_model.embeddings.token_embedding = HybridCLIPEmbedding(base, nm, vocab)
    return nm


if __name__ == "__main__":
    import torch.nn.functional as F
    dev = "cuda"
    torch.manual_seed(0)
    D, K = 64, 4
    base = torch.randn(4000, D, device=dev)
    base[:1200] *= 0.15
    med = ConcordPackedEmbedding.vocab_median_norm(base)

    tgt = torch.randn(K, D, device=dev)
    tgt = tgt / tgt.norm(dim=1, keepdim=True) * 30.0
    ids = torch.arange(K, device=dev)

    emb = ConcordPackedEmbedding(K, D, device=dev, lr=5e-2, target_norm=med)
    emb.init_tokens()
    p0 = emb.core.packed_w.clone()
    print(f"[init] deploy norm {emb.deploy_weight().norm(dim=1).mean():.2f} "
          f"(median target {med:.2f}) | storage int32 packed_w {tuple(emb.core.packed_w.shape)} "
          f"= {emb.core.packed_w.numel()*4} bytes ({emb.core.packed_w.numel()*32//emb.core.packed_w.numel()} b/param)")
    l0 = None
    for it in range(300):
        loss = F.mse_loss(emb(ids).float(), tgt)
        loss.backward()
        if l0 is None:
            l0 = loss.item()
    dep = emb.deploy_weight()
    cos = F.cosine_similarity(dep.float(), tgt, dim=1).mean().item()
    changed = (emb.core.packed_w != p0).float().mean().item()
    print(f"[trained] loss {l0:.3f}->{loss.item():.3f} | deploy norm "
          f"{dep.norm(dim=1).mean():.2f} (pinned {med:.2f}) | cos {cos:.3f} | "
          f"packed_w words changed {changed:.0%} (cascade ran)")
    print("-> packed (32 b/param) reuse of packed_b's real cascade; deploy norm "
          "pinned to the vocab median; learns the concept direction.")
