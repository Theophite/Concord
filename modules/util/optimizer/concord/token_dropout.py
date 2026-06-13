"""Token-only caption dropout (Concord).

With probability p, replace an example's caption with ONLY its trainable
tokens -- drop every context word -- so the token must carry the concept
itself instead of leaning on the surrounding caption (the "address-sharing /
decorative token" failure). The pure tensor op lives here (torch-only,
unit-testable); BaseStableDiffusionXLSetup applies the gating: only AFTER the
embedding divot releases, and only to examples that actually contain a
trainable token.
"""
import torch


def token_only_keep(tokens, train_ids, bos_id, eos_id, drop_mask):
    """tokens [B, L] long. For each row b with drop_mask[b] True, compact the
    sequence to [bos, <trainable ids, in original order>, eos, eos-pad...];
    rows with drop_mask False are returned unchanged. train_ids [K] long is the
    set of trainable placeholder ids FOR THIS tokenizer (TE1 and TE2 differ).

    Compaction (not in-place masking) is deliberate: cross-attention then sees
    the token at a real position with EOS padding elsewhere -- a clean
    "just the token" conditioning -- and TE2's EOS-pooled vector pools a
    token-only sequence. Returns a new tensor; never mutates the input."""
    if train_ids is None or train_ids.numel() == 0:
        return tokens
    B, L = tokens.shape
    train_ids = train_ids.to(tokens.device)
    out = tokens.clone()
    is_train = torch.isin(tokens, train_ids)
    for b in range(B):
        if not bool(drop_mask[b]):
            continue
        keep = tokens[b][is_train[b]]                      # trainable ids, in order
        seq = torch.cat([tokens.new_tensor([bos_id]), keep, tokens.new_tensor([eos_id])])
        row = tokens.new_full((L,), eos_id)                # CLIP pads with EOS
        m = min(L, int(seq.numel()))
        row[:m] = seq[:m]
        out[b] = row
    return out
