"""Norm-preserving Concord new-token embedding, PACKED (32 b/param) -- the clean
version that reuses packed_b's real cascade instead of re-implementing it.

Storage/optimizer = a ConcordLinearPackedB(in=dim, out=K): packed_w is [K, dim]
(one int32 per element: s_fast int16 + s_slow int8 + v_slow int8), and row_exp is
PER-TOKEN (out-row). The forward is a gather; the backward scatters the per-row
grad into grad_W [K, dim] and drives packed_b's own fused cascade by handing it
straight to core.apply_grad_step(grad_W) -- a direct kernel launch, NOT a nested
torch.autograd.backward() (the latter is illegal inside a CUDA-graph capture).
Then norm preservation pins each touched token's
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
        # Calibration accumulators read the RAW gradient (before drive), so the
        # measurement stays in data units even after a drive is applied.
        # Sightings count GRADIENT-BEARING occurrences only: the control plane's
        # branch-free forward routes EVERY position through this module (clamped
        # to row 0) and masks with torch.where afterward, so non-trainable
        # positions arrive here as row-0 ids with exact-zero grad rows --
        # measured 75.7/caption (the CLIP context minus content), which inflated
        # row 0's count ~140x before this mask. Zero-grad rows also correctly
        # exclude real tokens in dropped/masked captions from the normalizer.
        mod._accum.add_(G)
        contrib = (grad_emb.reshape(-1, mod.dim).abs().amax(dim=1) > 0).to(torch.float32)
        mod._seen.index_add_(0, ids.reshape(-1), contrib)
        # Per-token drive scaling, NOT per-row lr: evap_frac = lr*kappa*(1-coh)
        # is a FRACTION of the buffer, so scaling the drive preserves each
        # token's lambda semantics (per-row lr would push the evap fraction of
        # boosted tokens into the min_leak clamp). Device [K,1] buffer, updated
        # by .copy_() from outside the graph -> propagates into replays.
        # ORDER IS LOAD-BEARING: the v-hat EMA must see the RAW gradient and
        # the kernel the SCALED one -- rank-1 Adam is invariant to per-row
        # rescaling ((d*g)/sqrt(d^2*v_hat) = g/sqrt(v_hat)), so scaling before
        # the stats canceled the drive EXACTLY (the calibration was a no-op;
        # found 2026-06-12). Out-of-place multiply keeps G raw for the stats.
        # Drive packed_b's fused cascade DIRECTLY -- one kernel launch, no
        # re-entrant autograd. The old trick (core(x); y.backward(G.t())) ran a nested
        # torch.autograd.backward(), which is ILLEGAL inside a CUDA-graph capture: it touches
        # the legacy stream and aborts the capture (cudaErrorStreamCaptureImplicit) -- crashing
        # specifically on the post-backup graph RE-capture. apply_grad_step is the identical
        # apply path with no autograd engine, so capture (and re-capture) is safe.
        core.apply_grad_step(G * mod._drive, v_stats_from=G)
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
        # per-token gradient-drive multipliers (default 1), written by the
        # controller's divot calibration so every token moves at ONE isotropic
        # scalar rate (see ConcordController._finalize_embedding_calibration).
        self.register_buffer("_drive", torch.ones(num_tokens, 1, device=device))
        # coherent gradient accumulator: sum of RAW per-step grads. Over the
        # divot epoch this is "the change the data justifies" per token (noise
        # cancels ~sqrt(N), justified displacement adds ~N). Accumulated
        # UNCONDITIONALLY -- a python branch would bake into a captured CUDA
        # graph; after the calibration reads it nobody looks again (~[K,dim]
        # fp32, a few hundred KB).
        self.register_buffer("_accum", torch.zeros(num_tokens, dim, device=device))
        # sighting counter: GRADIENT-BEARING occurrences of each token row (the
        # control plane routes every position through row 0 with zero grad;
        # those must not count). _accum/_seen = justified distance PER
        # SIGHTING -- the calibration's normalizer.
        self.register_buffer("_seen", torch.zeros(num_tokens, device=device))
        self._grad_anchor = nn.Parameter(torch.zeros(1, device=device))

    @staticmethod
    def vocab_median_norm(vocab_weight):
        return vocab_weight.float().norm(dim=1).median().item()

    def set_target_norm(self, v):
        self.target.fill_(float(v))

    @torch.no_grad()
    def set_drive(self, mults):
        """Per-token gradient-drive multipliers, list/tensor of length K (row
        order = the attach order). Drive only -- friction kappa stays global, so
        every token keeps lambda = lr*kappa per step on its own buffer."""
        t = torch.as_tensor(mults, dtype=torch.float32, device=self._drive.device)
        if t.numel() != self.K:
            raise ValueError(f"set_drive: {t.numel()} multipliers for K={self.K} tokens")
        self._drive.copy_(t.reshape(-1, 1))

    @torch.no_grad()
    def init_tokens(self, init=None, scale=0.05, anchor=False):
        if init is None:
            init = torch.randn(self.K, self.dim, device=self.target.device) * scale
        self.core.load_weights(init)                     # mantissa lands in s_fast
        pw = self.core.packed_w
        sf = (pw >> 16)
        if anchor:
            # ANCHOR MODE: the init vector is FROZEN in v_slow (alpha_v = 0 ->
            # the leak never moves it, C* = 0 exactly), and everything learned
            # accumulates in s_slow as a friction-disciplined delta:
            #     deploy = init (immutable) + gated-learned-delta
            # The token can never drift off its founding semantics -- the
            # fried-embedding mechanism is structurally impossible. The anchor
            # also carries the norm, so the per-step norm pin (and its
            # requantization churn) is skipped; we pin ONCE here so the init
            # row lands at the vocab-median norm.
            vs = (sf.float() / V_SLOW_FACTOR).round().clamp(-128, 127).to(torch.int32)
            sf = (sf - vs * V_SLOW_FACTOR).clamp(INT16_MIN, INT16_MAX).to(torch.int32)
            self.core.packed_w.copy_(((sf & 0xFFFF) << 16) | (vs & 0xFF))
            self.core._resync_weight_buf()
            self._pin_norm(torch.arange(self.K, device=self.target.device))
            self.core.alpha_v_fast = 0.0
            self.core.drift_cancel_C = 0.0               # C*(alpha_v=0) = 0 exactly
            self._anchored = True
            return
        # legacy: position into s_slow so DEPLOY (s_slow+v_slow) is non-zero at
        # init (else pinning the deploy norm divides by ~0); the leak telescopes
        # it toward v_slow over the run. s_slow is the x128 coarse field.
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
        # Anchor mode: the frozen init carries the norm; per-step pinning would
        # only re-quantize all three fields every backward (multiplicative
        # round churn on the "frozen" anchor included). Pin only at init.
        if getattr(self, "_anchored", False):
            return
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
