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

    def __init__(self, unet, device, learning_rate: float, total_steps: int, optimizer_config=None):
        from concord_winner import swap_unet_to_winner, GatedRebalance
        self.config = make_concord_config(learning_rate, optimizer_config)
        self.total_steps = max(1, int(total_steps))
        self.layers = swap_unet_to_winner(
            unet, device, self.config.lr, gf_consol=self.config.gf_consol, verbose=False)
        self.gate = GatedRebalance(self.layers)
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

    @torch.no_grad()
    def after_step(self):
        """AFTER the optimizer update: gated rebalance (skips the no-op launches), tick."""
        self.gate()
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
