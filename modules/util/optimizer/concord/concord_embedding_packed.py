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
        # Quality-tag shield: project the SUBJECT rows' step off the subspace the
        # quality-TAG rows span (mask zeroes the correction on tag rows -> they
        # train free as the sink). Q is the eagerly-refreshed tag basis; mu drops
        # out of a delta. Fixed-shape buffers, elementwise mask -> capture-safe.
        # The calibration accumulators below then measure the ALLOWED motion. No-op
        # unless setup wired a shield. See quality_orthogonal.py.
        if mod._quality_Q is not None:
            C = G @ mod._quality_Q                       # [K, n_tags]
            if mod._quality_one_sided:
                # one-sided + rho-weighted: block toward-bad motion in
                # proportion to the row's signal fraction. Unweighted, the
                # one-sided clamp RECTIFIES zero-mean noise into a
                # systematic away-from-bad drift (see update_quality_basis).
                C = torch.clamp(C, min=0.0) * mod._quality_coh
            G = G - mod._quality_subject_mask * (C @ mod._quality_Q.T)
        # Style/subject factorization: HARD two-sided projection off the
        # style tags' span (symmetric -> no rectification, no rho needed).
        # Fixed-shape buffers, elementwise mask -> capture-safe.
        if mod._style_Q is not None:
            Cs = G @ mod._style_Q                        # [K, n_style]
            G = G - mod._style_subject_mask * (Cs @ mod._style_Q.T)
        # Group-subspace shaping (exp61, set_group_shield): supervised N-group between-group
        # separation + within-group flatten, driven by the per-embedding `group` label. Placed
        # WITH the shields (before _accum) since it is supervised subspace shaping, not a
        # discovered gate -- the accumulator/meters then reflect the intended update. Eager/
        # bridge TE path only (the flatten is a per-group variable-size op). No-op unless armed.
        if mod._group_ids is not None:
            # graph_te (Option C): under capture, the eager _apply_group_shield host-syncs; route to
            # the fixed-shape capture-legal twin. _use_capture_shield is False on the bridge path,
            # so this is the ORIGINAL call, unchanged.
            G = (mod._apply_group_shield_capture(G) if mod._use_capture_shield
                 else mod._apply_group_shield(G))
        # DATA-PARALLEL SEAM for the CALIBRATION accumulators. These three
        # buffers are a variance decomposition over sightings (see
        # update_quality_basis): n = _seen, C2 = ||_accum||^2, P = _power,
        # mu2 = (C2-P)/(n(n-1)). Every term must be a GLOBAL SUM or they are
        # measured against each other at different scales. They are also the
        # one piece of per-rank state a gradient hook cannot reach: each rank
        # sees a different slice of the vocabulary, so without this every rank
        # builds its own token histogram and its own quality basis.
        # SUM, not mean: two ranks that each saw a token once saw it twice.
        # The UPDATE path is separate and stays mean-reduced -- it goes
        # through core.apply_grad_step below, which carries _GRAD_HOOK.
        ge = grad_emb.reshape(-1, mod.dim).float()
        contrib = (ge.abs().amax(dim=1) > 0).to(torch.float32)
        seen_step = torch.zeros_like(mod._seen)
        seen_step.index_add_(0, ids.reshape(-1), contrib)
        if ppb._COUNT_HOOK is not None:
            acc_step = G.clone()
            ppb._COUNT_HOOK(acc_step)
            ppb._COUNT_HOOK(seen_step)
            mod._accum.add_(acc_step)
        else:
            mod._accum.add_(G)
        mod._seen.add_(seen_step)
        if mod._track_window:
            # Incoherent power Sigma||g||^2 at SIGHTING (position) granularity:
            # the noise term of the per-token Wiener posterior (the "window
            # around init"). Position-level, NOT ||G||^2 -- a token seen twice
            # in a batch contributes ||g1||^2+||g2||^2; the COHERENT sum g1+g2
            # is what lands in _accum. Flag-gated (diagnostic, default off);
            # passive like _accum, never touches the update. Zero-grad
            # passthrough rows add 0, consistent with the _seen mask.
            power_step = torch.zeros_like(mod._power)
            power_step.index_add_(0, ids.reshape(-1), ge.pow(2).sum(-1))
            if ppb._COUNT_HOOK is not None:
                ppb._COUNT_HOOK(power_step)
            mod._power.add_(power_step)
        # Common-mode CONCENTRATION gate (see __init__): remove shrink[t,j] of row t's
        # projection on component j -- owners keep their component, co-occurrence passengers
        # are deflated, so the component MIGRATES to its owners instead of being deleted.
        # Placement is load-bearing: AFTER the calibration accumulators (_accum/_seen/_power
        # must keep seeing the raw mode, else the gate eats its own evidence next epoch and
        # oscillates) and BEFORE drive/apply. G_raw feeds v_stats so the rank-1 preconditioner
        # cannot partially re-inflate the deflated directions. Eager/bridge TE path only (a
        # graph_te capture would bake the None-branch at capture time).
        G_raw = G
        if mod._cm_Q is not None:
            G = G - ((G @ mod._cm_Q) * mod._cm_shrink) @ mod._cm_Q.T
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
        core.apply_grad_step(G * mod._drive, v_stats_from=G_raw)
        core._resync_weight_buf()
        # Pin ALL K rows (static shape -> CUDA-graph capturable). torch.unique would be
        # dynamic-shaped AND sync. K is tiny and untouched rows are already at target,
        # so re-pinning them is a near-no-op.
        mod._pin_norm(torch.arange(mod.K, device=ids.device))
        return None, None, None


def _ns5_orthogonalize(G, steps=5, eps=1e-7):
    """Newton-Schulz quintic (Muon) approximate polar factor: flattens G's singular spectrum
    toward 1 without an SVD, iterating on the smaller Gram side. Returns a matrix of G's shape;
    the caller rescales it to preserve magnitude (so the net op is a spectral RESHAPE, not a
    rescale). Coefficients are Muon's standard (a,b,c)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    n = X.norm()
    if float(n) < eps:
        return X
    X = X / n
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


def _ns5_capture(G, steps=5, eps=1e-7):
    """Branch-free Newton-Schulz (Option C, CUDA-graph capture-legal): clamp the norm instead of
    the `if float(n) < eps: return X` host-sync early-out. Bit-identical to _ns5_orthogonalize for
    n>=eps; for n<eps the caller's mn-rescale (mn/(O.norm()+eps), mn~=0) makes the block contribute
    ~0 == the eager `continue`. CPU-parity validated (option_C_group_shield/test_group_shield_parity)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    n = X.norm()
    X = X / n.clamp_min(eps)
    transpose = X.shape[0] > X.shape[1]        # static-shape compare -> capture-time constant
    if transpose:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.t()
    return X


class ConcordPackedEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, device="cuda", lr=5e-2, alpha=0.1,
                 target_norm=1.0):
        super().__init__()
        self.K, self.dim = num_tokens, dim
        self.core = ConcordLinearPackedB(dim, num_tokens, bias=False,
                                         device=device, alpha=alpha, lr=lr)
        # Deploy-norm pin target: a scalar (broadcast to all rows = legacy vocab-median
        # behavior) OR a [K] per-row tensor (preserve each token's OWN norm -- used by the
        # caption-vocab path so base-vocab tokens keep their real norm instead of being
        # homogenized to the median).
        _tn = torch.as_tensor(target_norm, dtype=torch.float32, device=device).reshape(-1)
        self.register_buffer("target", _tn)
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
        # incoherent power Sigma||g||^2 (window estimator's noise term); only
        # accumulated when _track_window is set (flag-gated diagnostic).
        self.register_buffer("_power", torch.zeros(num_tokens, device=device))
        self._track_window = False
        # Common-mode concentration gate (armed at epoch boundaries by the controller's
        # attribution meter; None = OFF = bit-exact legacy). _cm_Q [dim, m] holds the gated
        # component directions; _cm_shrink [K, m] the per-row fraction of that component's
        # projection to REMOVE (owners 0.0, passengers gamma). Plain attributes (NOT buffers):
        # rebuilt per segment from the controller's sidecar, never enters state_dict.
        self._cm_Q = None
        self._cm_shrink = None
        self._grad_anchor = nn.Parameter(torch.zeros(1, device=device))
        # Quality-tag shield (optional): protect SUBJECT rows from the subspace the
        # designated quality-TAG rows span, so a subject learns content from a bad
        # image but not its badness. _quality_Q [dim, n_tags] is rebuilt EAGERLY
        # each step from the tags' current deploy vectors (the captured backward
        # reads this fixed-shape buffer); _quality_subject_mask [K,1] is 1 on rows
        # to shield (subjects), 0 on tag rows (they train free -- they are the
        # sink). None unless setup wires it via set_quality_shield. See
        # quality_orthogonal.py.
        self._quality_Q = None
        self._quality_subject_mask = None
        self._quality_mu = None
        self._quality_tag_idx = None
        self._quality_one_sided = False
        # Style/subject factorization shield: SECOND basis, always TWO-SIDED.
        # Style is symmetric -- a subject drifting AWAY from a style absorbs
        # style information (anti-style) as surely as one drifting toward it,
        # so both directions are contamination; and a symmetric projection
        # rectifies nothing, so it needs no rho weighting. Style tags train
        # free along their own span (the sink); everyone else (including the
        # quality tags) is projected off it. None = shield off.
        self._style_Q = None
        self._style_subject_mask = None
        self._style_mu = None
        self._style_tag_idx = None
        # Group-subspace shaping (exp61, optional): _group_ids [K] long assigns each trainable
        # row to a supervised group (-1 = ungrouped); armed via set_group_shield. Between-group
        # SEPARATION projects each group's update off the others' span (N-sink generalization of
        # the style shield); within-group FLATTEN Newton-Schulz's each group's member rows so
        # members stay distinct. None = OFF = bit-exact legacy. Eager/bridge TE path only.
        self._group_ids = None
        self._group_uniq = []
        self._emb_ids = None          # [K] embedding id per row (block-aware flatten: same
                                      # multi-token embedding = same id, kept coherent)
        self._group_separate = False
        self._group_separate_gamma = 0.0
        self._group_flatten = False
        self._group_flatten_gamma = 0.0
        # graph_te (Option C): when the embedding backward is captured inside the UNet graph, the
        # group shield's per-group variable-size ops are illegal. _use_capture_shield=True routes
        # the backward through the fixed-shape capture-legal port (_apply_group_shield_capture) and
        # the cm-gate through fixed buffers; the controller sets it = should_graph_te(config) each
        # step. False (bridge, default) => the ORIGINAL eager _apply_group_shield / reassigning
        # set_common_mode_gate run unchanged (bit-identical). _grp_ops holds the fixed structure +
        # pointer-stable sep_Q buffers, built lazily by _refresh_group_operators (eager, before_step).
        self._use_capture_shield = False
        self._grp_ops = None
        self._cm_cap = 0                      # common-mode fixed-buffer column count (graph_te)

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
    def set_quality_shield(self, tag_idx, subject_mask, mu, one_sided=False):
        """Designate quality-tag rows and the subject rows to shield from them.

        tag_idx [n_tags] long, subject_mask [K] (1=shield, 0=free), mu [dim] vocab
        mean. Registers a static-shape Q buffer [dim, n_tags] (zeros until the
        controller's first eager update) so the captured backward has stable
        memory to read."""
        dev = self.target.device
        self._quality_tag_idx = tag_idx.to(dev).long()
        self._quality_subject_mask = subject_mask.to(dev).float().reshape(self.K, 1)
        self._quality_mu = mu.to(dev).float()
        self._quality_one_sided = bool(one_sided)
        self._quality_Q = torch.zeros(self.dim, int(tag_idx.numel()), device=dev)
        # rho weight for the one-sided block (ones = legacy full block;
        # refreshed eagerly alongside Q from the sighting statistics).
        self._quality_coh = torch.ones(self.K, 1, device=dev)

    def set_style_shield(self, style_idx, subject_mask, mu):
        """Designate STYLE tag rows: subjects (everyone else, quality tags
        included) are HARD-projected off the styles' learned span -- two-
        sided, enforcing the style/subject factorization symmetrically.
        Registers a static-shape basis buffer like the quality shield."""
        dev = self.target.device
        self._style_tag_idx = style_idx.to(dev).long()
        self._style_subject_mask = subject_mask.to(dev).float().reshape(self.K, 1)
        self._style_mu = mu.to(dev).float()
        self._style_Q = torch.zeros(self.dim, int(style_idx.numel()), device=dev)

    @torch.no_grad()
    def set_group_shield(self, group_ids, separate, separate_gamma, flatten, flatten_gamma,
                         emb_ids=None):
        """Supervised group-subspace shaping (exp61). group_ids [K] long = the group id of each
        trainable row (-1 = ungrouped). separate: project each group's update off the span of
        the OTHER groups' current deploy vectors (N-sink generalization of the style shield),
        scaled by separate_gamma. flatten: BLOCK-AWARE Newton-Schulz that orthogonalizes the
        per-EMBEDDING mean update (concepts distinct) while preserving each embedding's within-
        token residual (a multi-token concept's own tokens stay coherent), lerp by flatten_gamma.
        emb_ids [K] long = embedding id per row (same multi-token embedding shares an id); None
        -> each row is its own embedding (per-row flatten, the single-token behavior). EAGER/
        bridge TE path only (variable-size op). None group_ids / both flags off / <2 grouped
        rows -> disarmed (bit-exact legacy)."""
        if group_ids is None or (not separate and not flatten):
            self._group_ids = None
            return
        dev = self.target.device
        gids = group_ids.to(dev).long().reshape(-1)
        if int((gids >= 0).sum()) < 2:
            self._group_ids = None
            return
        self._group_ids = gids
        self._group_uniq = [int(g) for g in torch.unique(gids).tolist() if g >= 0]
        self._emb_ids = emb_ids.to(dev).long().reshape(-1) if emb_ids is not None else None
        self._group_separate = bool(separate)
        self._group_separate_gamma = max(0.0, float(separate_gamma))
        self._group_flatten = bool(flatten)
        self._group_flatten_gamma = max(0.0, float(flatten_gamma))

    @torch.no_grad()
    def _apply_group_shield(self, G):
        """Per-step (eager) group-subspace shaping of the gradient G [K, dim]. Between-group
        separation FIRST (carve the group subspaces apart), then within-group flatten (separate
        the members inside each subspace). Zero-gradient (unsighted) rows pass through as zeros."""
        gids, uniq = self._group_ids, self._group_uniq
        if self._group_separate and self._group_separate_gamma > 0.0 and len(uniq) >= 2:
            from quality_orthogonal import orthonormal_basis
            deploy = self.deploy_weight().to(G.device).float()
            for g in uniq:
                in_g = (gids == g)
                others = (gids >= 0) & (~in_g)
                if not bool(others.any()):
                    continue
                odirs = deploy[others]                        # [n_other, dim]
                r = min(int(odirs.shape[0]), self.dim)
                Q = orthonormal_basis(odirs, ncols=r).to(G.device)   # [dim, r]
                rows = G[in_g]
                G[in_g] = rows - self._group_separate_gamma * (rows @ Q) @ Q.t()
        if self._group_flatten and self._group_flatten_gamma > 0.0:
            # BLOCK-AWARE flatten: orthogonalize the per-EMBEDDING MEAN update (make concepts
            # distinct) and shift each embedding's tokens by the same delta, so the WITHIN-
            # embedding residual (a multi-token concept's coordinated token structure) is
            # preserved exactly. Reduces to per-row flatten when every embedding is one token
            # (mean == the token, residual == 0). eids groups rows into embeddings.
            eids = (self._emb_ids if self._emb_ids is not None
                    else torch.arange(G.shape[0], device=G.device))
            gam = self._group_flatten_gamma
            for g in uniq:
                in_g = (gids == g)
                e_here = [int(e) for e in torch.unique(eids[in_g]).tolist()]
                if len(e_here) < 2:
                    continue                                  # need >=2 concepts to make distinct
                masks = [in_g & (eids == e) for e in e_here]
                means = [G[me].mean(0) for me in masks]
                means_t = torch.stack(means)                  # [E, dim] concept-level directions
                mn = means_t.norm()
                if float(mn) < 1e-12:
                    continue
                O = _ns5_orthogonalize(means_t)
                O = O * (mn / (O.norm() + 1e-12))             # reshape not rescale
                new_means = (1.0 - gam) * means_t + gam * O
                for i, me in enumerate(masks):                # concept mean shifts; residual kept
                    G[me] = G[me] + (new_means[i] - means[i])
        return G

    # ===================== Option C: capture-legal group shield (graph_te) =====================
    # Eager STRUCTURE + Q-REFRESH (host syncs / SVD off the capture path) feed a fixed-shape,
    # branch-free in-capture apply. Numerically identical to _apply_group_shield above (CPU-parity
    # validated: option_C_group_shield/test_group_shield_parity.py). All dormant unless
    # _use_capture_shield (set = should_graph_te(config) by the controller); bridge never calls these.
    @torch.no_grad()
    def _build_group_operators(self):
        """EAGER, once (lazy, off the capture path). Precompute the FIXED group/concept structure
        (all host syncs live here) + preallocate pointer-stable [dim,dim] sep_Q buffers. Every
        tensor device-follows gids (a CPU index vs a CUDA G would fault/sync mid-capture)."""
        dev = self.target.device
        gids, uniq = self._group_ids, self._group_uniq
        K = self.K
        eids = self._emb_ids if self._emb_ids is not None else torch.arange(K, device=dev)
        sep_masks = [(gids == g).float().reshape(K, 1) for g in uniq]
        sep_others = [((gids >= 0) & (gids != g)) for g in uniq]
        concept_of_row = torch.full((K,), -1, dtype=torch.long, device=dev)
        flat_concepts, counts, ncat = [], [], 0
        for g in uniq:
            in_g = (gids == g)
            e_here = sorted(int(e) for e in torch.unique(eids[in_g]).tolist())
            if len(e_here) < 2:
                continue
            cids = []
            for e in e_here:
                me = in_g & (eids == e)
                concept_of_row[me] = ncat
                counts.append(int(me.sum()))
                cids.append(ncat); ncat += 1
            flat_concepts.append(torch.tensor(cids, dtype=torch.long, device=dev))
        concept_of_row[concept_of_row < 0] = ncat                 # sink slot (delta 0)
        self._grp_ops = dict(
            sep_masks=sep_masks, sep_others=sep_others,
            concept_of_row=concept_of_row,
            concept_count=(torch.tensor(counts, dtype=torch.float32, device=dev)
                           if ncat else torch.zeros(0, device=dev)),
            flat_concepts=flat_concepts, n_concepts=ncat,
            # POINTER-STABLE projector buffers, refreshed in place -> a captured graph's baked
            # pointers stay valid (invariant #5; mirrors _quality_Q). Zeros until 1st refresh.
            sep_Q=[torch.zeros(self.dim, self.dim, device=dev) for _ in uniq],
        )

    @torch.no_grad()
    def _refresh_group_operators(self):
        """EAGER, per step in before_step (same cadence + same pre-apply deploy as
        update_quality_basis, so the basis matches what the eager shield computes inline). Refresh
        each projector IN PLACE via .copy_() into the preallocated buffer -- NEVER rebind. No-op
        unless armed + capture-shield."""
        if self._group_ids is None or not self._use_capture_shield:
            return
        if self._grp_ops is None:
            self._build_group_operators()
        from quality_orthogonal import orthonormal_basis
        deploy = self.deploy_weight().float()
        dev = deploy.device
        tdev = self.target.device
        for gi, others in enumerate(self._grp_ops['sep_others']):
            Q = orthonormal_basis(deploy[others.to(dev)], ncols=self.dim).to(tdev)
            self._grp_ops['sep_Q'][gi].copy_(Q)

    @torch.no_grad()
    def _apply_group_shield_capture(self, G):
        """IN-CAPTURE apply. Fixed-shape, branch-free, no host sync. Numerically identical to
        _apply_group_shield (projector QQ^T is basis/pad-independent; masked full-matrix projection
        == indexed-row projection for disjoint groups; segment-mean == per-concept mean)."""
        ops = self._grp_ops
        if ops is None:                             # structure not built (eager, pre-capture)
            return self._apply_group_shield(G)      # eager-only fallback; never hit under capture
        sep_Q = ops['sep_Q']
        K, dim = G.shape
        if self._group_separate and self._group_separate_gamma > 0.0 and len(sep_Q) >= 2:
            g = self._group_separate_gamma
            for gi in range(len(sep_Q)):
                Q = sep_Q[gi]
                mask_g = ops['sep_masks'][gi]
                G = G - g * mask_g * ((G @ Q) @ Q.t())
        if self._group_flatten and self._group_flatten_gamma > 0.0 and ops['n_concepts'] > 0:
            ncat, fg = ops['n_concepts'], self._group_flatten_gamma
            csum = torch.zeros(ncat + 1, dim, dtype=G.dtype, device=G.device)
            csum.index_add_(0, ops['concept_of_row'], G)
            cmean = csum[:ncat] / ops['concept_count'][:, None].clamp_min(1.0)
            new_cmean = cmean.clone()
            for cids in ops['flat_concepts']:
                M = cmean[cids]
                mn = M.norm()
                O = _ns5_capture(M)
                O = O * (mn / (O.norm() + 1e-12))
                new_cmean[cids] = (1.0 - fg) * M + fg * O
            delta = torch.cat([new_cmean - cmean,
                               torch.zeros(1, dim, dtype=G.dtype, device=G.device)], 0)
            G = G + delta[ops['concept_of_row']]
        return G

    @torch.no_grad()
    def update_quality_basis(self):
        """Rebuild Q from the tags' CURRENT deploy vectors (mean-centered). Eager,
        called from the controller's before_step -> the captured backward replays
        against the refreshed buffer. No-op unless a shield is wired.

        Device-robust: deploy_weight() can land on CPU at save time (the saver
        materializes there) while the shield buffers live on the training device,
        so align the index/mean to deploy's device; copy_ bridges Q back into the
        (training-device) buffer."""
        if self._quality_tag_idx is None and self._style_tag_idx is None:
            return
        from quality_orthogonal import orthonormal_basis
        deploy = self.deploy_weight()
        dev = deploy.device
        if self._quality_tag_idx is not None:
            dirs = deploy[self._quality_tag_idx.to(dev)].float() - self._quality_mu.to(dev)
            self._quality_Q.copy_(orthonormal_basis(dirs, ncols=self._quality_Q.shape[1]))
        if self._style_tag_idx is not None:
            sdirs = deploy[self._style_tag_idx.to(dev)].float() - self._style_mu.to(dev)
            self._style_Q.copy_(orthonormal_basis(sdirs, ncols=self._style_Q.shape[1]))
        # rho weight for the one-sided block: the row's signal-power
        # fraction from the sighting statistics (same moments as the
        # window meter: C2 = ||sum g||^2, P = sum ||g||^2, n sightings;
        # method-of-moments mu2/(mu2+nu)). n<2 rows read 0 -> a barely-
        # seen row passes symmetrically (one small unshielded step is
        # zero-mean-harmless; directed motion raises rho with evidence
        # and re-arms the block). Requires the _power accumulator (the
        # window tracker); without it -- or with CONCORD_QSHIELD_COH=0 --
        # the weight stays at ones = the legacy full one-sided block.
        import os as _os
        if (self._quality_one_sided and self._track_window
                and _os.environ.get("CONCORD_QSHIELD_COH", "1") != "0"):
            n = self._seen.clamp_min(2.0)
            C2 = self._accum.float().pow(2).sum(dim=1)
            P = self._power.float()
            mu2 = ((C2 - P) / (n * (n - 1.0))).clamp_min(0.0)
            nu = ((P - C2 / n) / (n - 1.0)).clamp_min(0.0)
            rho = mu2 / (mu2 + nu + 1e-30)
            rho = torch.where(self._seen >= 2, rho, torch.zeros_like(rho))
            self._quality_coh.copy_(rho.reshape(self.K, 1).to(self._quality_coh.device))

    @torch.no_grad()
    def init_tokens(self, init=None, scale=0.05, anchor=False):
        if init is None:
            init = torch.randn(self.K, self.dim, device=self.target.device) * scale
        if anchor:
            # ANCHOR MODE: the init vector is FROZEN in v_slow (alpha_v = 0 -> the leak
            # never moves it, C* = 0 exactly), and everything learned accumulates in
            # s_slow as a friction-disciplined delta:
            #     deploy = init (immutable) + gated-learned-delta
            # The token can never drift off its founding semantics. The anchor carries the
            # norm, so the per-step pin (and its requant churn) is skipped; we pin ONCE here.
            #
            # load_weights_anchor puts the COARSE mantissa directly in v_slow (s_slow=0,
            # fine residual in s_fast) -> deploy = (0 + v_slow)*128 ~= init. The drift
            # d_sv = s_slow - v_slow = -coarse is large, but C* = 0 keeps the gate inert (a
            # frozen anchor has no coherence by design), so the big gap does NOT ruin the
            # gate; a creep-resume rebalances it via resplit_anchor_to_even before the leak
            # is restored. The OLD path called load_weights then re-read (pw>>16) -- only the
            # <=64 fine RESIDUAL after load_weights, NOT the mantissa -- so v_slow collapsed
            # to ~0 and the anchor deployed ~0 (nonsense samples). Mirrors the Linear
            # load_weights_anchor.
            self.core.load_weights_anchor(init)
            self._pin_norm(torch.arange(self.K, device=self.target.device))
            self.core.alpha_v_fast = 0.0
            self.core.drift_cancel_C = 0.0               # C*(alpha_v=0) = 0 exactly
            self._anchored = True
            return
        # NON-anchor: load_weights packs the mantissa into the PROTECTED slow path with the
        # EVEN split (s_slow == v_slow, gap-zero, d_sv ~= 0, alpha_v_fast>0 adaptive); deploy =
        # (s_slow+v_slow)*128 ~= W from step 0 -- exactly the adaptive state non-anchor /
        # caption-vocab tokens want. Then pin the norm. (The old in-place re-split here re-read
        # the <=64 residual and collapsed deploy to ~0; deleted -- just use load_weights' state.)
        self.core.load_weights(init)
        self._pin_norm(torch.arange(self.K, device=self.target.device))

    def deploy_weight(self):
        return self.core.consolidated_weight()           # [K, dim], drop s_fast

    @torch.no_grad()
    def save(self, path):
        """Save the deployable embedding(s) [K, dim] -- reuse them, or feed back as
        an init vector (resolve_token_init accepts a tensor) to continue/transfer."""
        torch.save(self.deploy_weight().detach().cpu(), path)

    @torch.no_grad()
    def set_common_mode_gate(self, Q, shrink):
        """Arm/refresh the common-mode concentration gate. Q [dim, m] = gated component
        directions (columns ~unit-norm); shrink [K, m] in [0, 1] = per-row fraction of that
        component's projection to REMOVE (owners 0.0, passengers gamma). Pass None, None to
        disarm. Tensors are moved to the accumulator's device; plain attribute swap -- valid
        on the eager/bridge TE path (the shipped default; graph_te captures bake the branch).

        graph_te (Option C): under capture, reassigning _cm_Q bakes a stale pointer AND m varies,
        so route to FIXED [dim, cap]/[K, cap] buffers refreshed IN PLACE (copy_ the active m modes,
        zero the rest -> the padded modes are exact no-ops; disarm = zero the buffers, NOT None, so
        the captured `if _cm_Q is not None` branch stays live). cap = _cm_cap (the controller sets
        it = concord_emb_deflate_modes). _use_capture_shield False (bridge) => original body."""
        if not self._use_capture_shield:
            if Q is None or shrink is None or Q.numel() == 0:
                self._cm_Q = None
                self._cm_shrink = None
                return
            dev = self._accum.device
            self._cm_Q = Q.detach().to(device=dev, dtype=torch.float32)
            self._cm_shrink = shrink.detach().to(device=dev, dtype=torch.float32)
            return
        # capture-shield: fixed-shape buffers, refreshed in place (never reassigned)
        dev = self._accum.device
        cap = int(self._cm_cap) or (int(Q.shape[1]) if Q is not None else 0)
        self._cm_cap = cap
        if self._cm_Q is None and cap > 0:                    # allocate ONCE, stays not-None
            self._cm_Q = torch.zeros(self.dim, cap, device=dev, dtype=torch.float32)
            self._cm_shrink = torch.zeros(self.K, cap, device=dev, dtype=torch.float32)
        if self._cm_Q is None:
            return                                            # cap 0 and no Q -> never armed
        self._cm_Q.zero_(); self._cm_shrink.zero_()           # disarm baseline (padded modes -> no-op)
        if Q is not None and shrink is not None and Q.numel():
            m = min(int(Q.shape[1]), cap)
            self._cm_Q[:, :m].copy_(Q.detach().to(dev, torch.float32)[:, :m])
            self._cm_shrink[:, :m].copy_(shrink.detach().to(dev, torch.float32)[:, :m])

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
        tgt = self.target if self.target.numel() == 1 else self.target[rows]
        scale = tgt.reshape(-1, 1) / norm                # per-row target preserves each token's norm
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
