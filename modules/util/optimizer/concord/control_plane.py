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

    @property
    def weight(self):
        # Compat shim: code reading `token_embedding.weight` (e.g. get_input_embeddings()
        # .weight, sanitize reapply) gets the frozen base vocab. The trainable new tokens
        # live in `self.trainable` (packed), not here, so the vocab matrix is unchanged.
        return self.base.weight

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

    def attach_trainable(self, tids, inits, lr, target_norm, anchor=False):
        """tids: list of ids; inits: [K, dim]. Builds the Concord trainable embedding
        and routes those ids to its rows. anchor=True freezes the init vector in
        v_slow (deploy = init + gated-learned-delta; see init_tokens)."""
        self.trainable = ConcordPackedEmbedding(len(tids), self.dim, device=self.kind.device,
                                                lr=lr, target_norm=target_norm)
        self.trainable.init_tokens(init=inits, anchor=anchor)
        with torch.no_grad():
            for row, tid in enumerate(tids):
                self._grow(tid); self.kind[tid] = 2; self.idx[tid] = row

    def forward(self, input_ids):
        # Branch-free + static-shape so this is CUDA-graph capturable: compute every
        # route for every token and SELECT with torch.where -- no .any() host sync, no
        # dynamic masked-assignment. Masked-out tokens get no gradient path, so the
        # trainable's self-step still only sees the real train-token gradients.
        flat = input_ids.reshape(-1)
        emb = self.base(flat.clamp(max=self.V0 - 1))
        c = self.kind[flat.clamp(max=self.kind.shape[0] - 1)]
        i = self.idx[flat.clamp(max=self.idx.shape[0] - 1)]
        sm = (c == 1).unsqueeze(-1)                                  # static (zero / fixed)
        static_emb = self.static_vals[i.clamp(max=self.static_vals.shape[0] - 1)]
        emb = torch.where(sm, static_emb.to(emb.dtype), emb)
        if self.trainable is not None:          # python attr -> capture-time constant, not a sync
            tm = (c == 2).unsqueeze(-1)                              # trainable (Concord)
            train_emb = self.trainable(i.clamp(max=self.trainable.K - 1))
            emb = torch.where(tm, train_emb.to(emb.dtype), emb)
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
