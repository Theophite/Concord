"""OneTrainer <-> Concord glue.

Concord is NOT a torch.optim.Optimizer. It swaps the UNet's nn.Linear/nn.Conv2d for
packed self-stepping layers whose optimizer update is fused INTO the autograd backward
(there is no optimizer.step() for them). Around that it needs two per-step callbacks:
  - BEFORE the forward/backward: winner_step() advances the lr / noise-sigma / coherence
    -floor schedule (these live in device tensors the fused backward reads);
  - AFTER the update: a gated rebalance() (fires only when a packed mantissa actually
    overflows -- ~0% of steps at finetune lr, so nearly free).

So the OneTrainer-visible "optimizer" for the CONCORD choice is just a plain SGD over
the NON-swapped (aux) params -- norms, biases, embeddings -- and this controller carries
the Concord half. One controller per run, stored on the model; the trainer calls
before_step()/after_step() around its existing loop.
"""
import sys
from pathlib import Path

import torch

# the vendored Concord core lives next to this file
_CONCORD_DIR = str((Path(__file__).parent / "concord").resolve())
if _CONCORD_DIR not in sys.path:
    sys.path.insert(0, _CONCORD_DIR)


def _resolve_single_token_ids(tokenizer, words):
    """Resolve words to vocab ids, keeping only those that are a SINGLE token. A
    multi-token word (e.g. 'tok' -> 'pen','is') can't be zeroed without breaking the
    shared subwords, so it's skipped. Returns (ids, skipped_words)."""
    ids, skipped = [], []
    for w in words:
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if len(toks) == 1:
            ids.append(int(toks[0]))
        else:
            skipped.append(w)
    return ids, skipped


class SanitizePlane:
    """Independent control plane: zero the embedding rows of given single-token vocab
    words in BOTH SDXL text encoders, and keep them zeroed across steps. The saved model
    then embeds those words to ~nothing at inference (standard CLIP tokenizer + modified
    weights -> works in any SDXL tool). Works with any optimizer (not tied to Concord)."""

    def __init__(self, model, tokens_csv: str):
        words = [t.strip() for t in tokens_csv.split(",") if t.strip()]
        self.ids1, sk1 = _resolve_single_token_ids(model.tokenizer_1, words)
        self.ids2, sk2 = _resolve_single_token_ids(model.tokenizer_2, words)
        self.skipped = sorted(set(sk1) & set(sk2))   # skipped in BOTH (truly multi-token)
        self.reapply(model)
        msg = f"[concord] sanitize: zeroed {len(self.ids1)} (CLIP-L) / {len(self.ids2)} (CLIP-G) token rows"
        if self.skipped:
            msg += f" | skipped multi-token (can't zero a subword): {self.skipped}"
        print(msg)

    @torch.no_grad()
    def reapply(self, model):
        for te, ids in ((model.text_encoder_1, self.ids1), (model.text_encoder_2, self.ids2)):
            if ids:
                w = te.get_input_embeddings().weight
                w[ids] = 0.0


def make_concord_config(learning_rate: float, optimizer_config=None):
    """Map OneTrainer settings onto the validated winner config: lr comes from the main
    learning_rate field; the winner knobs (gf_consol/noise/sigmag_peak/ratio_coh/warmup/
    lr_min_frac) come from the GUI optimizer-params panel (None -> validated winner default)."""
    from concord_winner import ConcordConfig
    d = ConcordConfig()

    def pick(name, default):
        v = getattr(optimizer_config, name, None) if optimizer_config is not None else None
        return default if v is None else v

    return ConcordConfig(
        lr=float(learning_rate),
        gf_consol=float(pick("gf_consol", d.gf_consol)),
        noise=bool(pick("noise", d.noise)),
        sigmag_peak=float(pick("sigmag_peak", d.sigmag_peak)),
        ratio_coh=bool(pick("ratio_coh", d.ratio_coh)),
        warmup=int(pick("warmup", d.warmup)),
        lr_min_frac=float(pick("lr_min_frac", d.lr_min_frac)),
    )


class ConcordController:
    """Holds the swapped Concord UNet layers + the per-step schedule + the rebalance gate
    for one training run. Created in the SDXL setup (after the model is loaded, before the
    optimizer is built); driven by the trainer via before_step()/after_step()."""

    def __init__(self, unet, device, learning_rate: float, total_steps: int, optimizer_config=None,
                 module_filters=None, text_encoder=None, te_lr=None, te_wd_anchor=0.5):
        from concord_winner import swap_unet_to_winner, GatedRebalance, swap_text_encoder_to_anchor
        self.config = make_concord_config(learning_rate, optimizer_config)
        self.total_steps = max(1, int(total_steps))
        # module_filters: OneTrainer's layer_filter (ModuleFilter list). When set to a non-"full"
        # preset (e.g. attn-mlp -> ["attentions"]) only the selected layers are swapped to
        # Concord; the rest stay standard bf16 and are frozen, dropping their packed state.
        self.layers = swap_unet_to_winner(
            unet, device, self.config.lr, gf_consol=self.config.gf_consol, verbose=False,
            module_filters=module_filters)
        self.gate = GatedRebalance(self.layers)
        # Frozen-anchor TE training (CLIP-L): swapped AFTER the UNet so the shared global coh
        # flags are already set; driven with its own lr. Empty unless a text_encoder is passed.
        self.te_lr = float(te_lr) if te_lr else self.config.lr
        self.text_encoder = text_encoder          # held for the reversible TE deploy bridge
        self.te_layers = (swap_text_encoder_to_anchor(text_encoder, device, self.te_lr, te_wd_anchor)
                          if text_encoder is not None else [])
        self.te_gate = GatedRebalance(self.te_layers) if self.te_layers else None
        self.step_idx = 0
        print(f"[concord] swapped {len(self.layers)} UNet layers | lr={self.config.lr} "
              f"gf_consol={self.config.gf_consol} noise={self.config.noise} "
              f"horizon={self.total_steps} steps")

    @torch.no_grad()
    def before_step(self):
        """BEFORE forward/backward: advance the winner schedule onto the layer device
        tensors (lr / sigma / coherence floors) that the fused backward reads."""
        from concord_winner import winner_step
        winner_step(self.step_idx, self.total_steps, self.layers, config=self.config)
        if self.te_layers:
            winner_step(self.step_idx, self.total_steps, self.te_layers,
                        peak_lr=self.te_lr, config=self.config)

    @torch.no_grad()
    def after_step(self):
        """AFTER the optimizer update: gated rebalance (skips the no-op launches), tick."""
        self.gate()
        if self.te_gate is not None:
            self.te_gate()
        self.step_idx += 1

    @torch.no_grad()
    def consolidate_into_unet(self, unet):
        """DEPLOY: replace the packed Concord layers in-place with standard nn.Linear /
        nn.Conv2d holding the CONSOLIDATED weights (drops the transient s_fast), so the
        UNet saves and loads as an ordinary SDXL UNet. Destructive -- call once before the
        FINAL save; do not keep Concord-training after."""
        import torch.nn as nn
        from prototype_packed_b import ConcordConv2dPackedB, ConcordLinearPackedB
        n = 0
        for parent in unet.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, ConcordConv2dPackedB):     # subclass -> check first
                    w = child.consolidated_weight().reshape(
                        child.out_channels, child.in_channels, child.kh, child.kw)
                    new = nn.Conv2d(child.in_channels, child.out_channels, (child.kh, child.kw),
                                    stride=child.stride, padding=child.padding,
                                    bias=child.bias is not None)
                elif isinstance(child, ConcordLinearPackedB):
                    w = child.consolidated_weight()             # [out, in]
                    new = nn.Linear(child.in_features, child.out_features,
                                    bias=child.bias is not None)
                else:
                    continue
                new = new.to(device=child.packed_w.device, dtype=w.dtype)
                new.weight.data.copy_(w)
                if child.bias is not None:
                    new.bias.data.copy_(child.bias.detach().to(new.bias.dtype))
                setattr(parent, name, new)
                n += 1
        print(f"[concord] consolidated {n} layers -> standard nn.Linear/nn.Conv2d for deploy")
        return n

    @torch.no_grad()
    def materialize_te_deploy(self):
        """REVERSIBLE TE deploy: replace each text-encoder ConcordLinearPackedB with a temp
        nn.Linear holding its get_weight() (keeps s_fast -> the ~16-bit deploy, NOT the
        s_fast-dropping consolidated_weight), so the TE serializes as a standard CLIPTextModel.
        Returns a stash; pass it to restore_te_deploy() in a finally so training continues."""
        import torch.nn as nn
        from prototype_packed_b import ConcordLinearPackedB
        if not self.te_layers or self.text_encoder is None:
            return []
        # ONLY the swapped TE transformer Linears (te_layers). The control-plane embedding
        # core is also a ConcordLinearPackedB living inside the TE module tree, but it has its
        # own save bridge (materialize_packed_embeddings_to_vectors) -> must NOT be clobbered.
        te_set = {id(m) for m in self.te_layers}
        stash = []
        for parent in self.text_encoder.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, ConcordLinearPackedB) and id(child) in te_set:
                    w = child.get_weight()
                    lin = nn.Linear(child.in_features, child.out_features,
                                    bias=child.bias is not None).to(
                        device=child.packed_w.device, dtype=w.dtype)
                    lin.weight.data.copy_(w)
                    if child.bias is not None:
                        lin.bias.data.copy_(child.bias.detach().to(lin.bias.dtype))
                    setattr(parent, name, lin)
                    stash.append((parent, name, child))
        return stash

    @torch.no_grad()
    def restore_te_deploy(self, stash):
        """Undo materialize_te_deploy: put the packed ConcordLinearPackedB modules back so
        training continues with the exact pre-save state (incl. the frozen v_slow anchor)."""
        for parent, name, packed in stash:
            setattr(parent, name, packed)


# ---------------------------------------------------------------------------
# Norm-preserving packed embeddings (ControlPlaneEmbedding + ConcordPackedEmbedding).
#
# Replaces the plain-SGD AdditionalEmbeddingWrapper path for the TRAINABLE new tokens:
# each token becomes a row of a per-TE ConcordPackedEmbedding that self-steps INSIDE the
# captured backward and pins its DEPLOY norm to the vocab median -- the regularization the
# plain-SGD path lacked (embedding_learning_rate=1e-3 + no norm clamp -> overfit). The
# control plane REPLACES the TE's token_embedding (capture-safe, branch-free forward).
#
# OneTrainer's save/load is preserved: at save time the packed deploy vectors are
# materialized back into embedding.*.vector (-> standard clip_l/clip_g safetensors) and the
# ORIGINAL token_embedding is temporarily restored so the TE serializes as a plain
# CLIPTextModel. On resume _setup_embeddings restores .vector from the backup and
# setup_packed_embeddings re-packs it -> the trained tokens round-trip.
# ---------------------------------------------------------------------------

def packed_embeddings_active(config) -> bool:
    """True when the trainable new-token embeddings should route through the packed
    self-stepping core instead of plain SGD: Concord optimizer + flag on + something
    (non-output) to train."""
    from modules.util.enum.Optimizer import Optimizer
    return (config.optimizer.optimizer == Optimizer.CONCORD
            and bool(getattr(config, "concord_packed_embeddings", False))
            and config.train_any_embedding())


def _packed_trainable_uuids(config):
    """uuids of the additional embeddings that train via the packed core (train=True,
    non-output)."""
    out = set()
    for ec in config.all_embedding_configs():
        if getattr(ec, "train", False) and not getattr(ec, "is_output_embedding", False):
            out.add(ec.uuid)
    return out


def setup_packed_embeddings(model, config):
    """Replace each SDXL text encoder's token_embedding with a ControlPlaneEmbedding whose
    trainable rows are a norm-preserving ConcordPackedEmbedding. Call AFTER _setup_embeddings
    (so .vector is restored/created + the placeholder tokens are in the tokenizer) and AFTER
    the fused-matmul flag is set (the packed core reads _FUSED_MATMUL at construction).
    Stores model.concord_control_planes = [{te_idx, te, cp, base, row_map}, ...]."""
    import sys
    from pathlib import Path
    cdir = str((Path(__file__).parent / "concord").resolve())
    if cdir not in sys.path:
        sys.path.insert(0, cdir)
    from control_plane import ControlPlaneEmbedding

    train_uuids = _packed_trainable_uuids(config)
    lr = float(config.embedding_learning_rate)
    specs = [
        (1, model.text_encoder_1, model.tokenizer_1, model.all_text_encoder_1_embeddings()),
        (2, model.text_encoder_2, model.tokenizer_2, model.all_text_encoder_2_embeddings()),
    ]
    planes = []
    for te_idx, te, tokenizer, embeddings in specs:
        base = te.text_model.embeddings.token_embedding
        if isinstance(base, ControlPlaneEmbedding):          # re-setup on a persisted model
            base = base.base
            te.text_model.embeddings.token_embedding = base
        cp = ControlPlaneEmbedding(base)
        median = base.weight.float().norm(dim=1).median().item()
        tids, inits, row_map = [], [], []
        for emb in embeddings:
            if emb.uuid not in train_uuids:
                continue
            ids = tokenizer.convert_tokens_to_ids(emb.text_tokens)   # token_count ids
            for k, tid in enumerate(ids):
                tids.append(int(tid))
                inits.append(emb.vector[k].detach().float())
                row_map.append((emb, k))
        if tids:
            cp.attach_trainable(tids, torch.stack(inits).to(base.weight.device), lr, median)
        te.text_model.embeddings.token_embedding = cp
        planes.append({"te_idx": te_idx, "te": te, "cp": cp, "base": base, "row_map": row_map})
    model.concord_control_planes = planes
    # The plain-SGD wrapper path is bypassed; ensure both wrapper refs are None so
    # after_optimizer_step's preserve_embedding_norm guard short-circuits (the model's
    # __init__ leaves embedding_wrapper_2 unset).
    model.embedding_wrapper_1 = None
    model.embedding_wrapper_2 = None
    rows = len(planes[0]["row_map"]) if planes else 0
    print(f"[concord] packed embeddings ON: {rows} trainable token row(s)/TE, lr={lr}; "
          f"deploy-norm pinned to vocab median; plain-SGD embedding path bypassed")


def reenable_packed_embedding_grad(model):
    """Keep each trainable's dummy _grad_anchor requires_grad=True so its self-step autograd
    Function fires. The TE freeze in _setup_model_part_requires_grad (text_encoder_* frozen
    for embedding-only training) turns it off; this restores it. Called at setup AND every
    after_optimizer_step (both run __setup_requires_grad)."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        cp = plane["cp"]
        if cp.trainable is not None:
            cp.trainable._grad_anchor.requires_grad_(True)


def materialize_packed_embeddings_to_vectors(model):
    """Copy each trained token's DEPLOY vector (s_slow+v_slow, dropping the noisy s_fast)
    into the matching embedding.*.vector, so OneTrainer's embedding saver writes them as the
    standard clip_l/clip_g safetensors (portable). Call before any save."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        cp = plane["cp"]
        if cp.trainable is None:
            continue
        deploy = cp.trainable.deploy_weight().detach()           # [K, dim]
        with torch.no_grad():
            for row, (emb, k) in enumerate(plane["row_map"]):
                emb.vector[k].copy_(deploy[row].to(dtype=emb.vector.dtype, device=emb.vector.device))


def deactivate_packed_embeddings(model):
    """Temporarily restore each TE's ORIGINAL token_embedding so it serializes as a standard
    CLIPTextModel (the control-plane buffers must not enter its state_dict). Pair with
    reactivate_packed_embeddings in a try/finally around the save."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        plane["te"].text_model.embeddings.token_embedding = plane["base"]


def reactivate_packed_embeddings(model):
    """Re-install the control planes after a save (inverse of deactivate_packed_embeddings)."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        plane["te"].text_model.embeddings.token_embedding = plane["cp"]
