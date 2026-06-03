"""Independent token control plane for SDXL text encoders.

A separate, inspectable override layer over a FROZEN token_embedding. Specific tokens
(existing single tokens, OR whole words added as new single tokens) are routed to
controlled values; the rest of the vocab is untouched. Per-token modes:
  - sanitize -> ZERO (the token contributes nothing -> suppress the concept)
  - fix(vec) -> a fixed value (e.g. a neutral word's embedding, often a cleaner
                sanitizer than zero, since zero still leaves an attended empty slot)
  - (trainable -> the Concord hybrid composes on top, for real concept tokens)

Most explicit words are MULTI-token in CLIP (tok -> pen+is), so they can't be
zeroed at the vocab-row level (the pieces are shared with innocuous words). The
fix: add the whole word as one new token, then control THAT id. Control is
independent of the base weights -- save/load/toggle cp.state_dict() separately.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn


class ControlPlaneEmbedding(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base                                  # nn.Embedding, frozen
        self.V0, self.dim = base.weight.shape             # original vocab size
        d = base.weight.device
        self.register_buffer("route", torch.full((self.V0,), -1, dtype=torch.long, device=d))
        self.register_buffer("vals", torch.zeros(1, self.dim, dtype=base.weight.dtype, device=d))

    @torch.no_grad()
    def _grow(self, max_id):
        if max_id >= self.route.shape[0]:
            extra = torch.full((max_id + 1 - self.route.shape[0],), -1,
                               dtype=torch.long, device=self.route.device)
            self.route = torch.cat([self.route, extra])

    @torch.no_grad()
    def sanitize(self, ids):
        ids = torch.as_tensor(ids, device=self.route.device)
        self._grow(int(ids.max()))
        self.route[ids] = 0                               # -> the zero row

    @torch.no_grad()
    def fix(self, ids, vecs):
        ids = torch.as_tensor(ids, device=self.route.device)
        self._grow(int(ids.max()))
        start = self.vals.shape[0]
        self.vals = torch.cat([self.vals, vecs.to(self.vals).reshape(len(ids), self.dim)])
        self.route[ids] = torch.arange(start, start + len(ids), device=self.route.device)

    def controlled(self):
        return int((self.route >= 0).sum())

    def forward(self, input_ids):
        emb = self.base(input_ids.clamp(max=self.V0 - 1))     # new ids get overridden below
        r = self.route[input_ids.clamp(max=self.route.shape[0] - 1)]
        m = r >= 0
        if m.any():
            emb = torch.where(m.unsqueeze(-1), self.vals[r.clamp(min=0)], emb)
        return emb


def _wrap(te):
    emb = te.get_input_embeddings()
    if not isinstance(emb, ControlPlaneEmbedding):
        emb = ControlPlaneEmbedding(emb)
        te.text_model.embeddings.token_embedding = emb
    return emb


def sanitize_words(te, tokenizer, words, verbose=True):
    """Suppress each WORD: if it's a single existing token, zero it; if multi-token
    (the usual case for explicit terms), add it as one new token and zero THAT, so the
    word tokenizes to a controlled id. Returns the ControlPlaneEmbedding."""
    cp = _wrap(te)
    existing, added = [], []
    for w in words:
        ids = tokenizer(w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            cp.sanitize(ids); existing.append(w)
        else:
            tokenizer.add_tokens(w)
            cp.sanitize([tokenizer.convert_tokens_to_ids(w)]); added.append(w)
    if verbose:
        print(f"  zeroed {len(existing)} existing single tokens {existing} + "
              f"{len(added)} multi-token words added-then-zeroed {added}")
    return cp


if __name__ == "__main__":
    from diffusers import StableDiffusionXLPipeline
    dev = torch.device("cuda")
    pipe = StableDiffusionXLPipeline.from_single_file(
        r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=torch.bfloat16).to(dev)
    for m in (pipe.text_encoder, pipe.text_encoder_2):
        m.requires_grad_(False)
    ADULT = ["tok", "tok", "tok", "nude", "naked"]

    for tag, te, tok in (("L", pipe.text_encoder, pipe.tokenizer),
                         ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
        print(f"[{tag}]")
        cp = sanitize_words(te, tok, ADULT)
        nid = tok.convert_tokens_to_ids("tok")          # now a single controlled id
        ids = torch.tensor([[nid]], device=dev)
        print(f"  'tok' -> controlled id {nid}, embedding norm {cp(ids).norm():.3f} (zeroed)")
        dog = tok("dog", add_special_tokens=False).input_ids[0]
        d = torch.tensor([[dog]], device=dev)
        print(f"  unrelated 'dog' token unchanged: {torch.equal(cp(d), cp.base(d))} | "
              f"{cp.controlled()} ids controlled, rest of {cp.V0} vocab untouched")
    print("\n[done] independent control plane: explicit words tokenize to one id and "
          "zero out; base frozen + intact. fix(neutral) instead of zero is an option.")
