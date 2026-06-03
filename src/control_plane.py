"""Independent, spec-driven token control plane for SDXL text encoders.

You declare a LIST of tokens and, per token, a MODE and an explicit INIT. The
control plane (a drop-in replacement for the frozen token_embedding) sets it all
up; the embedding trainer then just runs forward/backward -- trainable tokens
self-step, static tokens stay fixed -- with NO monkeypatching, because the spec is
encoded in the module's structure (buffers for static, Concord modules for
trainable), not in the training loop.

  TokenSpec(token, mode, init)
    mode  = "sanitize" -> ZERO  (suppress; the token contributes nothing)
            "fix"      -> a fixed static value (e.g. a neutral token's embedding)
            "train"    -> a norm-preserving Concord embedding (trains, self-steps)
    init  = explicit only (NO subword-mean):
            torch.Tensor [dim] | "zero" | "random" | a SINGLE-token word to copy.
            (a multi-token init word is an error -- give a vector instead.)

Whole words that are multi-token in CLIP (tok -> pen+is) are added as one new
token so they tokenize to a single controlled id.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn

from concord_embedding_packed import ConcordPackedEmbedding


@dataclass
class TokenSpec:
    token: str
    mode: str = "train"                 # "sanitize" | "fix" | "train"
    init: object = "zero"               # tensor | "zero" | "random" | single-token word


def resolve_init(init, tokenizer, base_weight):
    """Explicit init -> a [dim] vector. No subword-mean: a multi-token init word errors."""
    dim, dev = base_weight.shape[1], base_weight.device
    if torch.is_tensor(init):
        return init.float().reshape(dim).to(dev)
    if init in (None, "zero"):
        return torch.zeros(dim, device=dev)
    if init == "random":
        return torch.randn(dim, device=dev) * 0.05
    if isinstance(init, str):
        ids = tokenizer(init, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"init word '{init}' is {len(ids)} CLIP tokens; pass a "
                             f"single-token word or an explicit vector (no subword-mean)")
        return base_weight[ids[0]].float().to(dev)
    raise ValueError(f"unrecognized init {init!r}")


class ControlPlaneEmbedding(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base                                    # nn.Embedding, frozen
        self.V0, self.dim = base.weight.shape
        d = base.weight.device
        # per-id routing: kind 0=base, 1=static, 2=trainable; idx into the kind's table.
        self.register_buffer("kind", torch.zeros(self.V0, dtype=torch.int8, device=d))
        self.register_buffer("idx", torch.zeros(self.V0, dtype=torch.long, device=d))
        self.register_buffer("static_vals", torch.zeros(1, self.dim, dtype=base.weight.dtype, device=d))
        self.trainable = None                               # ConcordPackedEmbedding (lazily)

    @torch.no_grad()
    def _grow(self, max_id):
        n = max_id + 1 - self.kind.shape[0]
        if n > 0:
            d = self.kind.device
            self.kind = torch.cat([self.kind, torch.zeros(n, dtype=torch.int8, device=d)])
            self.idx = torch.cat([self.idx, torch.zeros(n, dtype=torch.long, device=d)])

    @torch.no_grad()
    def set_zero(self, tid):
        self._grow(tid); self.kind[tid] = 1; self.idx[tid] = 0     # static row 0 = zero

    @torch.no_grad()
    def set_fixed(self, tid, vec):
        self._grow(tid)
        row = self.static_vals.shape[0]
        self.static_vals = torch.cat([self.static_vals, vec.to(self.static_vals).reshape(1, self.dim)])
        self.kind[tid] = 1; self.idx[tid] = row

    def attach_trainable(self, tids, inits, lr, target_norm):
        """tids: list of ids; inits: [K, dim]. Builds the Concord trainable embedding
        and routes those ids to its rows."""
        self.trainable = ConcordPackedEmbedding(len(tids), self.dim, device=self.kind.device,
                                                lr=lr, target_norm=target_norm)
        self.trainable.init_tokens(init=inits)
        with torch.no_grad():
            for row, tid in enumerate(tids):
                self._grow(tid); self.kind[tid] = 2; self.idx[tid] = row

    def forward(self, input_ids):
        flat = input_ids.reshape(-1)
        emb = self.base(flat.clamp(max=self.V0 - 1))
        c = self.kind[flat.clamp(max=self.kind.shape[0] - 1)]
        i = self.idx[flat.clamp(max=self.idx.shape[0] - 1)]
        sm = c == 1
        tm = c == 2
        if sm.any() or tm.any():
            emb = emb.clone()
            if sm.any():
                emb[sm] = self.static_vals[i[sm]].to(emb.dtype)
            if tm.any() and self.trainable is not None:
                emb[tm] = self.trainable(i[tm]).to(emb.dtype)
        return emb.reshape(*input_ids.shape, self.dim)


def apply_token_spec(te, tokenizer, specs, lr=5e-3):
    """Configure a control plane on `te` from a list of TokenSpec. Returns it; its
    `.trainable` (Concord) is what the trainer's loss.backward() self-steps -- the
    static tokens are buffers and never move. No monkeypatching."""
    base = te.get_input_embeddings()
    cp = base if isinstance(base, ControlPlaneEmbedding) else ControlPlaneEmbedding(base)
    if cp is not base:
        te.text_model.embeddings.token_embedding = cp
    raw = cp.base.weight
    median = raw.float().norm(dim=1).median().item()
    train_ids, train_inits = [], []
    for s in specs:
        ids = tokenizer(s.token, add_special_tokens=False).input_ids
        if len(ids) == 1 and not s.token.startswith("<"):
            tid = ids[0]                                     # existing single token
        else:
            tokenizer.add_tokens(s.token); tid = tokenizer.convert_tokens_to_ids(s.token)
        if s.mode == "sanitize":
            cp.set_zero(tid)
        elif s.mode == "fix":
            cp.set_fixed(tid, resolve_init(s.init, tokenizer, raw))
        elif s.mode == "train":
            train_ids.append(tid); train_inits.append(resolve_init(s.init, tokenizer, raw))
        else:
            raise ValueError(f"bad mode {s.mode!r}")
    if train_ids:
        cp.attach_trainable(train_ids, torch.stack(train_inits), lr, median)
    return cp


if __name__ == "__main__":
    from diffusers import StableDiffusionXLPipeline
    dev = torch.device("cuda")
    pipe = StableDiffusionXLPipeline.from_single_file(
        r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=torch.bfloat16).to(dev)
    for m in (pipe.text_encoder, pipe.text_encoder_2):
        m.requires_grad_(False)

    te, tok = pipe.text_encoder, pipe.tokenizer
    man = te.get_input_embeddings().weight[tok("man", add_special_tokens=False).input_ids[0]]
    SPEC = [
        TokenSpec("tok",   "sanitize"),                          # multi-token -> added + zeroed
        TokenSpec("tok", "fix",   init="person"),              # redirect to 'person' embedding
        TokenSpec("<tok>",  "train", init="man"),                 # trainable, copy 'man' to start
        TokenSpec("<char1>", "train", init=man.float() * 0.5),     # trainable, explicit vector
        TokenSpec("<blank>", "train", init="zero"),                # trainable, from zero
    ]
    cp = apply_token_spec(te, tok, SPEC, lr=5e-2)
    g = lambda w: cp(torch.tensor([[tok.convert_tokens_to_ids(w)]], device=dev)).norm().item()
    print(f"[modes] sanitize 'tok' norm {g('tok'):.3f} (0) | fix 'tok' norm "
          f"{g('tok'):.3f} (=person) | trainable '<tok>' norm {g('<tok>'):.3f} (median-pinned)")
    dog = torch.tensor([[tok('dog', add_special_tokens=False).input_ids[0]]], device=dev)
    print(f"[intact] 'dog' untouched: {torch.equal(cp(dog), cp.base(dog))} | "
          f"trainable module: {cp.trainable.K} tokens, {cp.trainable.core.packed_w.numel()*4} bytes")

    # the trainer just runs backward; only the 3 trainable tokens self-step.
    import torch.nn.functional as F
    ids = torch.tensor([[tok.convert_tokens_to_ids("<tok>"),
                         tok.convert_tokens_to_ids("<char1>"),
                         tok.convert_tokens_to_ids("<blank>")]], device=dev)
    before = cp.trainable.deploy_weight().float().clone()
    for _ in range(20):
        F.mse_loss(cp(ids).float(), torch.randn(1, 3, 768, device=dev) * 3).backward()
    moved = (cp.trainable.deploy_weight().float() - before).norm(dim=1)
    print(f"[train] 3 trainable tokens moved {moved.mean():.3f}; deploy norms "
          f"{cp.trainable.deploy_weight().norm(dim=1).mean():.3f} (still median-pinned). "
          f"static tokens unchanged -- no monkeypatching, the spec is the structure.")
