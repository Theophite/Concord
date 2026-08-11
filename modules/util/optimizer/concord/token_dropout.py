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


def token_strip_keep(tokens, train_ids, eos_id, drop_mask):
    """The complement of token_only_keep: for each row b with drop_mask[b]
    True, REMOVE the trainable ids and compact what remains (bos, context
    words, eos, eos-pad...). Together the pair forms the thirds scheme
    (full / tokens-only / words-only at 1/3 each): tokens-only isolates the
    embedding-attributable gradient, words-only isolates the caption
    scaffold, so the optimizer's gate sees two coherent streams instead of
    one standing mixture (the 2026-07-12 audit's velocity-telescope
    anti-alignment; exp 68: standing mixture = the irreducible-conflict
    regime, a permanent coherence tax). Returns a new tensor."""
    if train_ids is None or train_ids.numel() == 0:
        return tokens
    B, L = tokens.shape
    train_ids = train_ids.to(tokens.device)
    out = tokens.clone()
    is_train = torch.isin(tokens, train_ids)
    for b in range(B):
        if not bool(drop_mask[b]):
            continue
        kept = tokens[b][~is_train[b]]        # bos, words, eos, pads -- in order
        row = tokens.new_full((L,), eos_id)   # CLIP pads with EOS
        row[:kept.numel()] = kept
        out[b] = row
    return out


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


def token_clause_keep(tokens, train_ids, comma_id, bos_id, eos_id, drop_mask, generator=None):
    """For each row b with drop_mask[b] True, keep the trainable "words"
    (always, wherever they occur) PLUS exactly ONE randomly chosen
    comma-delimited clause, and compact to [bos, <kept, original order>, eos,
    eos-pad...]. A clause is a maximal span bounded by commas or the caption
    ends -- [bos..comma], [comma..comma], [comma..eos] -- and the bounding
    commas are dropped.

    This is the contrastive DROPPED arm (vs a FULL-caption arm L): the cross-
    arm gap is then only the OTHER clauses, not the whole caption, so the
    coherence gate keeps the retained clause's context instead of evaporating
    all of it (the full-strip token_only_keep made the gap the entire caption,
    which the gate reads as incoherent and scales toward zero -> the label-only
    / negative-armgap regime). No commas (or comma_id None) -> one clause = the
    whole caption -> unchanged (the example opts out, gap 0). drop_mask False
    rows returned unchanged. Clause choice uses `generator` (seed it per step
    for same-seed A/B). Returns a new tensor; never mutates the input."""
    if train_ids is None or train_ids.numel() == 0:
        return tokens
    B, L = tokens.shape
    train_ids = train_ids.to(tokens.device)
    out = tokens.clone()
    is_train = torch.isin(tokens, train_ids)
    ar = torch.arange(L, device=tokens.device)
    for b in range(B):
        if not bool(drop_mask[b]):
            continue
        row = tokens[b]
        eos_hits = (row == eos_id).nonzero(as_tuple=True)[0]
        end = int(eos_hits[0]) if eos_hits.numel() > 0 else L    # first eos = caption end
        if end <= 1:
            continue                                             # empty caption (bos[,eos])
        content = ar[1:end]                                      # after bos, before eos
        if comma_id is not None:
            is_comma = (row[content] == comma_id)
        else:
            is_comma = torch.zeros(content.numel(), dtype=torch.bool, device=row.device)
        # clause id per content position: increments AFTER each comma, and the
        # comma is assigned to the clause it closes (so it can be excluded).
        cum = torch.cumsum(is_comma.to(torch.long), dim=0)
        clause_id = cum - is_comma.to(torch.long)
        n_clauses = int(clause_id.max().item()) + 1
        if generator is not None:
            sel = int(torch.randint(0, n_clauses, (1,), generator=generator,
                                    device=generator.device).item())
        else:
            sel = int(torch.randint(0, n_clauses, (1,)).item())
        keep = torch.zeros(L, dtype=torch.bool, device=row.device)
        keep[content] = (clause_id == sel) & (~is_comma)         # the chosen clause's words
        keep = keep | is_train[b]                                # + trainable words, anywhere
        keep[0] = False                                          # bos added explicitly below
        keep = keep & (ar < end)                                 # never keep eos / pads
        kept = row[keep]
        seq = torch.cat([tokens.new_tensor([bos_id]), kept, tokens.new_tensor([eos_id])])
        newrow = tokens.new_full((L,), eos_id)                   # CLIP pads with EOS
        m = min(L, int(seq.numel()))
        newrow[:m] = seq[:m]
        out[b] = newrow
    return out
