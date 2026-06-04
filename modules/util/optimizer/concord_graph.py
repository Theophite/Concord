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
import os

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


class ManualUNetGraph:
    """Stage 3 v2: manual CUDA-graph capture of UNet -> loss -> backward, fed by
    predict(return_unet_inputs=True). Unlike make_graphed_callables (v1), the captured
    region contains the REAL loss.backward(), so the warmup self-steps on REAL gradients
    -- the validated standalone pattern, which does NOT corrupt the self-stepping weights
    (the source of v1's NaN).

    Wired into GenericTrainer (gated on concord_cuda_graph): step() replaces
    predict()->calculate_loss()->backward(), and the trainer uses zero_grad(set_to_none=
    False) so the aux .grad buffers stay static for replay.

    GRADIENT BRIDGE (required for embedding / text-encoder training): the captured region
    has STATIC inputs, but ehs + text_embeds connect upstream to the text encoder + the
    trainable embeddings. They require grad, so the captured backward produces their input
    gradients; _bridge() then does ONE eager torch.autograd.backward into the live TE graph
    each step, so the text encoder + embeddings receive gradients. Without this the captured
    backward stops at the detached inputs -> embeddings never train, AND (with a live TE
    forward, i.e. text_encoder.train_embedding=True) the orphaned TE graph + the checkpointed
    UNet warmup backward double-free ("backward through the graph a second time"). Restoring
    requires_grad on the inputs (the standalone's "static, needs grad") + consuming the TE
    graph via the bridge fixes both.

    Loss scope here is plain eps-MSE; min-SNR / loss_weight weighting is a follow-up.
    """

    def __init__(self, model_setup, aux_params, dtype, warmup: int = 3):
        self.ms = model_setup
        self.aux = list(aux_params)
        self.dtype = dtype
        self.warmup = warmup
        self.static = None
        self.graph = None
        self.cap_loss = None
        self.unet = None
        _ckpt.checkpoint = _capturable_checkpoint

    def _alloc(self, prep, model):
        self.unet = model.unet
        ac = prep["added_cond_kwargs"]
        # ehs + text_embeds connect upstream to the text encoder + the trainable
        # embeddings. They REQUIRE GRAD so the captured backward produces an input
        # gradient we bridge (eager) back into the live TE graph each step -> the text
        # encoder + embeddings train under the graph. latent_input (frozen VAE, cached)
        # and timestep/time_ids/target have no upstream trainable parents -> stay detached.
        self.static = {
            "sample": prep["latent_input"].detach().clone(),
            "timestep": prep["timestep"].detach().clone(),
            "ehs": prep["encoder_hidden_states"].detach().clone().requires_grad_(True),
            "text_embeds": ac["text_embeds"].detach().clone().requires_grad_(True),
            "time_ids": ac["time_ids"].detach().clone(),
            "target": prep["target"].detach().clone(),
        }

    def _copy_in(self, prep):
        s, ac = self.static, prep["added_cond_kwargs"]
        with torch.no_grad():            # ehs/text_embeds are leaves that require grad
            s["sample"].copy_(prep["latent_input"]); s["timestep"].copy_(prep["timestep"])
            s["ehs"].copy_(prep["encoder_hidden_states"])
            s["text_embeds"].copy_(ac["text_embeds"]); s["time_ids"].copy_(ac["time_ids"])
            s["target"].copy_(prep["target"])

    def _zero_input_grads(self):
        # the captured backward ACCUMULATES into static-input .grad; zero before each
        # capture/replay so it holds exactly this step's input-gradient for the bridge.
        for k in ("ehs", "text_embeds"):
            g = self.static[k].grad
            if g is not None:
                g.zero_()

    def _bridge(self, prep):
        # Reconnect the captured (detached) UNet inputs to the eager TE graph: backward
        # the real text-encoder outputs with the captured input-grads, so the text encoder
        # + trainable embeddings receive gradients. ONE combined backward -- ehs and
        # text_embeds share the TE graph, so two separate calls would "backward twice".
        ac = prep["added_cond_kwargs"]
        tensors, grads = [], []
        for real, key in ((prep["encoder_hidden_states"], "ehs"), (ac["text_embeds"], "text_embeds")):
            g = self.static[key].grad
            if real.requires_grad and g is not None:
                tensors.append(real); grads.append(g.to(real.dtype))
        if tensors:
            torch.autograd.backward(tensors, grads)
            if os.environ.get("CONCORD_GRAPH_DEBUG"):
                gn = sum(float(g.float().norm()) for g in grads)
                print(f"[concord_graph] bridge: TE backward over {len(tensors)} inputs, "
                      f"grad-norm sum {gn:.5f}", flush=True)

    def _step_fn(self):
        s = self.static
        with torch.autocast(device_type="cuda", dtype=self.dtype, cache_enabled=False):
            pred = self.unet(s["sample"], s["timestep"], encoder_hidden_states=s["ehs"],
                             added_cond_kwargs={"text_embeds": s["text_embeds"],
                                                "time_ids": s["time_ids"]}).sample
        loss = torch.nn.functional.mse_loss(pred.float(), s["target"].float())
        loss.backward()
        return loss

    def step(self, model, batch, config, train_progress):
        prep = self.ms.predict(model, batch, config, train_progress, return_unet_inputs=True)
        if self.static is None:
            self._alloc(prep, model)
        self._copy_in(prep)
        if self.graph is None:
            if os.environ.get("CONCORD_GRAPH_DEBUG"):
                # eager warmup (no side stream) under anomaly detection -> precise traceback
                # of the op whose backward is replayed. Diagnostic only; off by default.
                with torch.autograd.detect_anomaly():
                    for i in range(self.warmup):
                        print(f"[concord_graph] anomaly warmup iter {i}", flush=True)
                        self._step_fn()
            else:
                strm = torch.cuda.Stream(); strm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(strm):
                    for _ in range(self.warmup):
                        self._step_fn()                  # real-gradient warmup (no corruption)
                torch.cuda.current_stream().wait_stream(strm)
            for p in self.aux:                           # discard warmup-accumulated aux grad
                if p.grad is not None:
                    p.grad.zero_()
            self._zero_input_grads()                     # fresh static-input grads for capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.cap_loss = self._step_fn()
        else:
            self._zero_input_grads()                     # fresh static-input grads for this replay
            self.graph.replay()
        self._bridge(prep)                               # eager: TE + embeddings receive grad
        return self.cap_loss


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
