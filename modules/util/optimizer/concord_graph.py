"""Stage 3: CUDA-graph the UNet forward+backward in OneTrainer's SDXL step.

Only the UNet is captured -- the expensive, bsz=1 launch-overhead-bound part, and where
the Concord fused step rides (in the backward). Everything generator-derived (the
diffusion noise + timestep use a custom per-step torch.Generator that is NOT CUDA-graph-
capturable -- verified by test) stays EAGER in predict(); the captured region never
touches a custom generator. The winner's *fluctuation* noise is fine -- it's on the
default generator, which advances under replay.

We HOOK unet.forward (like OneTrainer hooks token_embedding.forward) so model.unet stays
the real module -- state_dict / parameters / EMA / saver are untouched. make_graphed_
callables graphs fwd+bwd over a positional wrapper that calls the ORIGINAL forward (so the
hook can't recurse); predict() then calls the graphed path, and loss.backward() drives the
captured backward (Concord step + aux grads). The diffusion loss stays eager.

Gated on bf16 (no GradScaler) + accum=1 (Concord steps every backward) + single-GPU +
latent caching + gradient checkpointing. EAGER FALLBACK on any failure -- the validated
non-graph path is never at risk.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

_orig_ckpt = _ckpt.checkpoint


def _capturable_checkpoint(function, *a, use_reentrant=None, preserve_rng_state=True, **kw):
    # capture-legal: drop the RNG-state save/restore (a host sync). SDXL UNet is dropout-
    # free so the recompute is bit-identical. OneTrainer already passes use_reentrant=False.
    return _orig_ckpt(function, *a, use_reentrant=False, preserve_rng_state=False, **kw)


def should_graph(config) -> bool:
    # EXPERIMENTAL, default OFF. The make_graphed_callables path captures the Concord UNet
    # but the captured graph NaNs on the first real step (a deep interaction between
    # make_graphed_callables' static-buffer backward and the layers' self-stepping +
    # checkpointing). Left in, opt-in, behind concord_cuda_graph -- the validated eager
    # Stage-1 path stays the default. A proven alternative is the standalone's MANUAL
    # capture (split predict at the UNet seam) -- the Stage-3 v2.
    from modules.util.enum.Optimizer import Optimizer
    from modules.util.enum.DataType import DataType
    return (getattr(config, "concord_cuda_graph", False)
            and config.optimizer.optimizer == Optimizer.CONCORD
            and config.train_dtype == DataType.BFLOAT_16
            and int(config.gradient_accumulation_steps) == 1
            and not config.multi_gpu
            and config.latent_caching
            and config.gradient_checkpointing.enabled())


class _UNetPositional(nn.Module):
    """make_graphed_callables wants positional tensor args + needs the UNet's params in its
    own module tree (to capture their backward). Holds unet as a submodule for the params,
    but CALLS the original (unhooked) forward so hooking unet.forward later can't recurse."""

    def __init__(self, unet, orig_forward):
        super().__init__()
        self.unet = unet
        self._orig_forward = orig_forward

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        return self._orig_forward(
            sample, timestep, encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample


def _sample_args(model, config, device, dtype):
    b = int(config.batch_size)
    res = int(str(config.resolution).split("x")[0])
    lat = res // 8
    cross = model.unet.config.cross_attention_dim                       # 2048 for SDXL
    pooled = model.unet.add_embedding.linear_1.in_features - 6 * model.unet.config.addition_time_embed_dim
    return (
        torch.randn(b, 4, lat, lat, device=device, dtype=dtype),
        torch.full((b,), 500, device=device, dtype=torch.long),
        torch.randn(b, 77, cross, device=device, dtype=dtype),
        torch.randn(b, pooled, device=device, dtype=dtype),
        torch.randn(b, 6, device=device, dtype=dtype),
    )


def install_graphed_unet(model, config, device, dtype) -> bool:
    """Hook model.unet.forward to run through a CUDA-graphed fwd+bwd. Returns True on
    success; on any failure leaves the UNet untouched (eager)."""
    unet = model.unet
    orig_forward = unet.forward
    pos = _UNetPositional(unet, orig_forward)
    _ckpt.checkpoint = _capturable_checkpoint
    # make_graphed_callables warms up with fwd+bwd on SYNTHETIC inputs, and the Concord
    # layers self-step in that backward -> it would corrupt the weights (garbage-gradient
    # updates) before real training. The per-layer lr is a DEVICE TENSOR the captured step
    # reads, so zero it during capture (warmup steps become no-ops); winner_step restores
    # the real lr at every replay.
    ctrl = getattr(model, "concord_controller", None)
    if ctrl is not None:
        for m in ctrl.layers:
            m.lr = 0.0
    try:
        with torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=False):  # bf16 like
            graphed = torch.cuda.make_graphed_callables(  # predict(); cache_enabled=False is
                pos, _sample_args(model, config, device, dtype))  # required by make_graphed_callables
    except Exception as e:
        import traceback; traceback.print_exc()
        if ctrl is not None:
            for m in ctrl.layers:
                m.lr = ctrl.config.lr
        print(f"[concord] UNet graph capture FAILED ({type(e).__name__}); eager fallback")
        return False
    if ctrl is not None:
        for m in ctrl.layers:
            m.lr = ctrl.config.lr               # winner_step re-sets per step regardless

    def graphed_forward(sample, timestep, encoder_hidden_states=None, added_cond_kwargs=None, **kw):
        pred = graphed(sample, timestep, encoder_hidden_states,
                       added_cond_kwargs["text_embeds"], added_cond_kwargs["time_ids"])
        return UNet2DConditionOutput(sample=pred)

    unet.forward = graphed_forward          # model.unet stays the real module; only fwd is hooked
    print("[concord] UNet fwd+bwd captured in a CUDA graph (Stage 3)")
    return True
