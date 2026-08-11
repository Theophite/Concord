from abc import ABCMeta
from collections.abc import Callable

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.util.config.TrainConfig import TrainEmbeddingConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection

import torch
from torch import Tensor

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    Gemma2Model,
    LlamaModel,
    T5EncoderModel,
)
from transformers.tokenization_utils import PreTrainedTokenizer, Trie


class ModelSetupEmbeddingMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _remove_added_embeddings_from_tokenizer(
            self,
            tokenizer: PreTrainedTokenizer,
    ):
        if tokenizer:
            added_tokens = list(filter(lambda item: not item[1].special, tokenizer._added_tokens_decoder.items()))
            for key, added_token in added_tokens:
                tokenizer._added_tokens_decoder.pop(key)
                tokenizer._added_tokens_encoder.pop(added_token.content)
            tokenizer.tokens_trie = Trie()
            tokenizer._update_trie()

    def _create_new_embedding(
            self,
            model: BaseModel,
            embedding_config: TrainEmbeddingConfig,
            tokenizer: PreTrainedTokenizer | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection | T5EncoderModel | Gemma2Model | LlamaModel | None,
            create_output_embedding_fn: Callable[[str], Tensor] | None = None,
    ) -> Tensor | None:
        if tokenizer is None or text_encoder is None:
            return None

        with torch.no_grad():
            initial_token_ids = tokenizer(
                embedding_config.initial_embedding_text,
                padding='do_not_pad',
                truncation=embedding_config.token_count is not None,
                add_special_tokens=False,
                max_length=embedding_config.token_count,
            ).input_ids
            pad_token_id = tokenizer(
                '*',
                padding='do_not_pad',
                truncation=True,
                add_special_tokens=False,
                max_length=1,
            ).input_ids[0]

            if embedding_config.token_count is not None:
                initial_token_ids += [pad_token_id] * (embedding_config.token_count - len(initial_token_ids))

            all_embeddings = text_encoder.get_input_embeddings().weight.data
            initial_embeddings = [all_embeddings[token_id] for token_id in initial_token_ids]
            vector = torch.stack(initial_embeddings)

            if embedding_config.is_output_embedding and create_output_embedding_fn is not None:
                token_count = len(initial_token_ids)

                with model.autocast_context:
                    vector = create_output_embedding_fn(
                        embedding_config.initial_embedding_text + token_count * '*',
                    )[:token_count]

        # Concord output-space (CLIP-inversion) init (exp62): when the initializer phrase is
        # LONGER than token_count, the truncate above drops the tail. If enabled per-embedding,
        # replace the truncated init with token_count embeddings optimized so their contextualized
        # (penultimate / x-attn-layer) output reproduces the FULL phrase's effect, then renormed
        # to the vocab scale. Input embeddings only; skipped for output embeddings.
        N = embedding_config.token_count
        if (bool(getattr(embedding_config, "concord_invert_init", False))
                and not embedding_config.is_output_embedding and N is not None):
            full_ids = tokenizer(
                embedding_config.initial_embedding_text, padding='do_not_pad',
                truncation=False, add_special_tokens=False,
            ).input_ids[:75]                                # BOS + phrase + EOS must fit 77
            if len(full_ids) > N:
                vector = self._invert_init(text_encoder, tokenizer, full_ids, N, vector)

        return vector

    def _invert_init(self, text_encoder, tokenizer, full_ids, token_count, trunc_vector):
        """Output-space (CLIP-inversion) init for token_count < phrase length (exp62). Optimize
        token_count input embeddings so their penultimate (x-attn-layer) output reproduces the
        full M-token phrase's effect under a fixed random attention head, then renorm to the
        phrase-token norm (in-distribution). Drives the real text encoder on CUSTOM embeddings
        (embeddings(inputs_embeds=) + encoder + hidden_states[-2], matching encode_clip's
        default_layer=-2; assumes the default clip-skip). Falls back to the truncate init on ANY
        error (version drift, non-CLIP encoder, OOM) so it can never break setup."""
        import math
        import os as _os
        try:
            from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
            tm = text_encoder.text_model
            tbl = text_encoder.get_input_embeddings().weight
            dev, dim = tbl.device, tbl.shape[1]
            # Run the manual forward in the ENCODER BODY's dtype, NOT the input-embedding
            # table's. setup_model casts get_input_embeddings() (the token table) to
            # embedding_weight_dtype (bf16) but leaves the encoder body (LayerNorm, attention,
            # position embedding) at its load dtype (fp32) -- a MIXED encoder. The old edt=tbl.dtype
            # (bf16) fed bf16 into the fp32 body -> "expected Float but found BFloat16"; the autocast
            # fix missed because at embedding-creation time the TE is on CPU, where bf16 autocast has
            # different op coverage. We pass inputs_embeds (bypassing the bf16 token table), so
            # matching the body's dtype makes the whole path consistent -- no autocast, device-
            # agnostic. fdt = fp32 here; bf16 in a pure-bf16 TE -- either way it matches the body.
            fdt = next(tm.encoder.parameters()).dtype
            bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
            phrase_emb = tbl[torch.tensor(full_ids, device=dev)].detach().to(fdt)   # [M, D]
            _g = torch.Generator().manual_seed(0)       # fixed generic x-attn head (exp62)
            wk = (torch.randn(dim, dim, generator=_g) / math.sqrt(dim)).to(dev)
            wv = (torch.randn(dim, dim, generator=_g) / math.sqrt(dim)).to(dev)
            qq = torch.randn(64, dim, generator=_g).to(dev)

            def penult(content):                        # [BOS, content, EOS] -> penultimate [T, D]
                # Everything in the encoder body's dtype (fdt) -> a single consistent dtype through
                # embeddings + encoder, so NO autocast and no CPU/CUDA-autocast fragility. bos/eos
                # come from the bf16 token table -> cast to fdt; content (the fp32 optim var / fp32
                # phrase_emb) -> fdt too. hid.dtype (== fdt) carries into the causal mask.
                seq = torch.cat([tbl[bos].to(fdt)[None], content.to(fdt),
                                 tbl[eos].to(fdt)[None]], 0)[None]
                t = seq.shape[1]
                hid = tm.embeddings(inputs_embeds=seq,
                                    position_ids=torch.arange(t, device=dev)[None])
                cm = _create_4d_causal_attention_mask((1, t), hid.dtype, device=dev)
                out = tm.encoder(inputs_embeds=hid, attention_mask=None,
                                 causal_attention_mask=cm, output_hidden_states=True)
                return out.hidden_states[-2][0].float()

            def effect(h):                              # x-attn output over the random queries
                return (qq @ (h @ wk).T / math.sqrt(dim)).softmax(-1) @ (h @ wv)

            with torch.no_grad():
                target = effect(penult(phrase_emb)[1:]).detach()                    # drop BOS
            tgt_norm = phrase_emb.float().norm(dim=-1).mean()
            e = trunc_vector.detach().to(dev).float().clone().requires_grad_(True)
            opt = torch.optim.Adam([e], lr=float(_os.environ.get("CONCORD_INVERT_LR", "0.05")))
            steps = int(_os.environ.get("CONCORD_INVERT_STEPS", "300"))
            with torch.enable_grad():
                for _ in range(steps):
                    opt.zero_grad()
                    loss = (target - effect(penult(e)[1:])).pow(2).sum(-1).mean()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        e.mul_(tgt_norm / (e.norm(dim=-1, keepdim=True) + 1e-9))     # renorm
            print(f"[concord] invert-init: {token_count} token(s) <- {len(full_ids)}-token phrase "
                  f"(loss {float(loss):.4g}, norm {float(e.norm(dim=-1).mean()):.3f})", flush=True)
            return e.detach().to(trunc_vector.dtype)
        except Exception as _ex:
            print(f"[concord] invert-init FAILED ({type(_ex).__name__}: {_ex}) -> truncate init",
                  flush=True)
            return trunc_vector

    def _add_embeddings_to_tokenizer(
            self,
            tokenizer: PreTrainedTokenizer,
            embeddings: list[BaseModelEmbedding],
    ) -> (Tensor, list[bool]):
        for embedding in embeddings:
            tokenizer.add_tokens(embedding.text_tokens)

    def _add_embedding_param_groups(
            self,
            embeddings: list[BaseModelEmbedding],
            parameter_group_collection: NamedParameterGroupCollection,
            embedding_learning_rate: float,
            prefix: str,
    ):
        for embedding in embeddings:
            parameter = embedding.output_vector if embedding.is_output_embedding else embedding.vector
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"{prefix}/{embedding.uuid}",
                display_name=f"{prefix}/{embedding.placeholder}",
                parameters=[parameter],
                learning_rate=embedding_learning_rate,
            ))

    def _normalize_output_embeddings(self, embeddings: list[BaseModelEmbedding]):
        with torch.no_grad():
            for embedding in embeddings:
                if embedding.is_output_embedding and embedding.output_vector.requires_grad:
                    std = embedding.output_vector.std(dim=1).mean()
                    embedding.output_vector.mul_(embedding.original_output_vector_std / std)
