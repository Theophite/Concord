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
    from modules.util.enum.Optimizer import Optimizer, is_concord_family
    from modules.util.enum.DataType import DataType
    # gradient_accumulation_steps > 1 is now graph-compatible: the fused backward
    # accumulates into s_fast and only consolidates on the cycle's last micro-step
    # (gated by the per-device consolidate flag the trainer toggles per replay). The
    # captured loss is scaled by 1/accum inside _step_fn so the summed micro-grads
    # equal the averaged full-batch gradient.
    return (getattr(config, "concord_cuda_graph", False)
            and is_concord_family(config.optimizer.optimizer)
            and config.train_dtype == DataType.BFLOAT_16
            and not config.multi_gpu
            and config.latent_caching
            and config.gradient_checkpointing.enabled())


def graph_te_opted_in() -> bool:
    """The config-FREE opt-in signal for graph_te: env CONCORD_GRAPH_TE=1 OR the sentinel file
    <OneTrainer-clean root>/CONCORD_GRAPH_TE.on. should_graph_te ANDs this with config-dependent
    capture-legality checks; the ConcordController uses it DIRECTLY to set _use_capture_shield,
    because the controller only holds the concord config (make_concord_config), NOT the full
    TrainConfig -- calling should_graph_te(self.config) there throws (no train_text_encoder_or_
    embedding), and a swallowed throw silently leaves the shield on the eager (capture-illegal)
    path. Create the sentinel = on; delete = off (bridge)."""
    import os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    return (_os.environ.get("CONCORD_GRAPH_TE", "") == "1"
            or _os.path.exists(_os.path.join(_root, "CONCORD_GRAPH_TE.on")))


def te_row_servo_opted_in() -> bool:
    """Off-by-default opt-in for extending the UNet per-row dynamic-lambda servo to the winner-recipe
    TE layers (converts them single-fast -> 2-fast so the ARM meter can read them). env
    CONCORD_TE_ROW_SERVO=1 OR the sentinel file <OneTrainer-clean root>/CONCORD_TE_ROW_SERVO.on.
    Config-free (the controller only holds the concord config) and read once at controller init.
    When absent the TE swap stays single-fast -> the whole extension is a structural no-op."""
    import os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    return (_os.environ.get("CONCORD_TE_ROW_SERVO", "") == "1"
            or _os.path.exists(_os.path.join(_root, "CONCORD_TE_ROW_SERVO.on")))


_TS_BANDS_CACHE = {"mtime": None, "val": None}


def concord_ts_bands():
    """Off-by-default LIVE CONTROLS for the contrast scheme (user finding, 2026-07-16:
    contrast pairs certify best at VERY LOW noise -- at low t both arms sit in the same basin,
    so the caption difference is a small CONSISTENT differential the cross-arm gate can bank;
    at mid/high noise the dropped arm hallucinates different content, the arms diverge
    everywhere, and the gate reads the window as incoherent. The ordinary full-caption
    windows WRITE best at MID noise, where conditioning has max leverage on content).
    Enable: sentinel file <root>/CONCORD_TS_BANDS.on (or env CONCORD_TS_BANDS=1 for defaults).
    The file CONTENT is live-tunable, re-read on mtime change (edit mid-run to retune):
        contrast=0.00,0.10    # contrast-pair band (fractions of T), or contrast=off
        main=0.35,0.65        # ordinary-window band, or main=off (= passthrough, for A/B)
        fire=0.5              # live override of the contrast-pair fraction (_contrast_rate)
        uncond=0.15           # live override of the uncond-pass rate (_uncond_rate)
    Any key absent -> band defaults / config rates; a malformed band falls back to its
    default. Delete the file to live-disable everything. Returns None when off, else a dict
    {"contrast": (lo,hi)|None, "main": (lo,hi)|None, "fire": float|None, "uncond":
    float|None}. The band consumer (predict) REMAPS the already-sampled timestep affinely
    into the band: no extra generator draws (the noise sampled after it stays bit-identical
    to the un-banded run at the same seed) and the configured distribution's shape is
    preserved within the band. The fire override is MEMOIZED per step in _contrast_fire --
    two sites (GenericTrainer's consolidate forcing + step()'s routing) evaluate it in the
    same iteration, and a file edit landing between them must not let them disagree."""
    import os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    _f = _os.path.join(_root, "CONCORD_TS_BANDS.on")
    try:
        _mt = _os.path.getmtime(_f)
    except OSError:
        _mt = None
    if _mt is None and _os.environ.get("CONCORD_TS_BANDS", "") != "1":
        if _TS_BANDS_CACHE["val"] is not None:
            print("[concord] ts-bands: OFF (sentinel removed)", flush=True)
        _TS_BANDS_CACHE["mtime"] = None
        _TS_BANDS_CACHE["val"] = None
        return None
    if _mt == _TS_BANDS_CACHE["mtime"] and _TS_BANDS_CACHE["val"] is not None:
        return _TS_BANDS_CACHE["val"]
    val = {"contrast": (0.00, 0.10), "main": (0.35, 0.65), "fire": None, "uncond": None}
    if _mt is not None:
        try:
            import re as _re
            with open(_f) as _fh:
                _txt = _fh.read()
            for _k in ("contrast", "main"):
                if _re.search(rf"{_k}\s*=\s*off\b", _txt):
                    val[_k] = None
                    continue
                _m = _re.search(rf"{_k}\s*=\s*([\d.]+)\s*,\s*([\d.]+)", _txt)
                if _m:
                    _b = (float(_m.group(1)), float(_m.group(2)))
                    if 0.0 <= _b[0] < _b[1] <= 1.0:
                        val[_k] = _b
                    else:
                        print(f"[concord] ts-bands: bad {_k} band {_b}; using default", flush=True)
            for _k in ("fire", "uncond"):
                _m = _re.search(rf"{_k}\s*=\s*([\d.]+)", _txt)
                if _m:
                    val[_k] = min(1.0, max(0.0, float(_m.group(1))))
        except Exception as _e:                       # torn read / bad float -> defaults
            print(f"[concord] ts-bands: parse failed ({_e}); using defaults", flush=True)
    if val != _TS_BANDS_CACHE["val"]:
        def _fmt(_b):
            return "off" if _b is None else f"[{_b[0]:.3f},{_b[1]:.3f})"
        _fr = "config" if val["fire"] is None else f"{val['fire']:g}"
        _un = "config" if val["uncond"] is None else f"{val['uncond']:g}"
        print(f"[concord] ts-bands: ON contrast={_fmt(val['contrast'])} main={_fmt(val['main'])} "
              f"fire={_fr} uncond={_un}", flush=True)
    _TS_BANDS_CACHE["mtime"] = _mt
    _TS_BANDS_CACHE["val"] = val
    return val


def should_graph_te(config) -> bool:
    # Extends should_graph: capture the text encoder too (encode_text INSIDE the graph), so
    # the embeddings train in the captured backward and the eager bridge is dropped -> recovers
    # the full UNet-graph speedup for embedding training. Requires a LIVE TE forward (otherwise
    # there's nothing to gain) and a capture-legal TE path: NO TE dropout (host RNG) and NO
    # output embeddings (_apply_output_embeddings uses a data-dependent .nonzero()).
    if not should_graph(config):
        return False
    # MASTER SWITCH -- HARD DEFAULT OFF, opt in with the env var CONCORD_GRAPH_TE=1.
    #
    # WHY env-gated and not the config field: graph_te captures encode_text INSIDE the graph,
    # so the captured backward runs the embedding shields -- and the group shield
    # (_apply_group_shield in concord_embedding_packed.py) host-syncs (`bool(others.any())`)
    # and uses data-dependent shapes, which are ILLEGAL during CUDA-graph capture
    # (cudaErrorStreamCaptureUnsupported -> the run dies at the first capture). The config
    # field `concord_graph_te` DEFAULTS TRUE (TrainConfig.py) and is not surfaced in any GUI,
    # so a saved config almost certainly serialized `true`; honoring it would silently re-enable
    # the crashing path on reload. We therefore BYPASS the config field and gate on the env var
    # alone: absent/anything-but-"1" => bridge (encoders eager, on the fused kernel, offloadable
    # for sampling; _bridge() backprops the TE gradient each step -- the proven pattern).
    # Restore config-field honoring once the group shield is ported to fixed-shape, capture-legal
    # operators (option C): precompute the per-group projectors + flatten transform eagerly in
    # before_step, apply branch-free inside the captured backward.
    if not graph_te_opted_in():                  # env CONCORD_GRAPH_TE=1 or the sentinel file
        return False
    if not (config.train_text_encoder_or_embedding() or config.train_text_encoder_2_or_embedding()):
        return False
    # TE dropout inside encode_text uses host RNG -> not capture-legal. EXCEPTION: the
    # contrastive-arms path DEFERS the CFG dropout out of encode_text (dropout_probability
    # =None) and re-applies it as the in-graph cfg_mask multiply -- a device tensor copied
    # in per replay, so capture-legal -- letting graph_te+contrast run with dropout
    # configured (the rate feeds the shared mask). Enforce no-dropout only when contrast
    # is OFF; the contrast twin (_contrast_step_graph_te) is the capture-legal dropout path.
    if not getattr(config, "concord_contrast_arms", False):
        if (config.text_encoder.dropout_probability or 0.0) > 0.0 \
                or (config.text_encoder_2.dropout_probability or 0.0) > 0.0:
            return False
    if config.embedding.is_output_embedding \
            or any(e.is_output_embedding for e in config.additional_embeddings):
        return False
    return True


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

    def __init__(self, model_setup, aux_params, dtype, warmup: int = 3, graph_te: bool = False,
                 accum: int = 1):
        self.ms = model_setup
        self.aux = list(aux_params)
        self.dtype = dtype
        self.warmup = warmup
        self.graph_te = graph_te        # True: capture encode_text->UNet (TE in the graph, no bridge)
        self._accum = max(1, int(accum))  # grad-accum: scale captured loss by 1/accum (baked at capture)
        self.static = None
        self.graph = None
        self.cap_loss = None
        self._pool = None               # persistent graph-pool handle: every capture reuses it
        self._stale_graph = None        # pool ANCHOR across a keep_pool release (never replayed)
        self._pool_bytes = 0            # measured pool footprint (boundary headroom check)
        self.unet = None
        self.model = None               # set in _alloc; needed for encode_text in TE-graph mode
        self._shape_key = None          # latent geometry the static buffers + graph are bound to
        self.vhat = None                # ConcordVHatBuckets: per-shape Adafactor v_hat (lazy)
        self.ls1 = self.ls2 = 0         # text-encoder layer skips (from config, set in step)
        # in-graph loss weighting (min-SNR-gamma + resolution cap), baked from config at first step
        self._loss_cfg_ready = False
        self._uncond_pass = False   # set in _ensure_loss_cfg; read by _step_fn at capture
        self._uncond_rate = 0.0
        self._contrast_rate = 1.0   # B: fraction of steps that run the paired contrast (1.0 = all)
        self._min_snr = False
        self._gamma = 1.0
        self._resaware = False
        self._v_pred = False
        self._alphas_cumprod = None
        _ckpt.checkpoint = _capturable_checkpoint

    def release(self, keep_pool: bool = False):
        # Drop the captured graph. step() transparently recaptures on the next
        # training step (static buffers are reused).
        #
        # keep_pool=False (sampling): also free the private memory pool -- the
        # sampler needs the VRAM. Call before any external empty_cache that
        # would otherwise invalidate the recorded graph -> replay crash.
        #
        # keep_pool=True (backup/shape-change): keep the dead graph object as a
        # POOL ANCHOR. The graph must be released because the save moves
        # weights (recorded pointers go stale), but its pool segments should
        # survive: graph-pool segments are PRIVATE (ordinary allocations can
        # never use them), so freeing them just forces the recapture to commit
        # a fresh pool into a save-churned address space -- measured +0.5G
        # committed per epoch boundary (free fell 1.16 -> 1.04 -> 0.58G and
        # the boundary recapture froze). While the anchor lives, the saver's
        # internal torch_gc()s cannot decommit the segments; the next
        # _warmup_and_capture captures into the SAME pool (self._pool) while
        # the anchor still holds it alive, THEN drops the anchor.
        import gc
        if keep_pool and self._pool_bytes:
            # Affordability guard at RELEASE time. Holding the pool through a
            # backup makes the save's transient run ON TOP of it -- measured
            # committed 25.76G / free 0.00G at the epoch boundary on the 24G
            # card, and the WDDM demotion tax lingered for the next epoch
            # (1.51 s/it vs ~1.1 baseline). The capture-time check fires too
            # late for that case: the overflow happens DURING the save. If
            # free cannot cover the held pool + margin now, demote to the full
            # release -- the save gets the headroom and the recapture pays 1x
            # from clean address space.
            try:
                free, _ = torch.cuda.mem_get_info()
            except Exception:
                free = None
            if free is not None and free < self._pool_bytes + (256 << 20):
                print(f"[concord_graph] release headroom {free / 2 ** 30:.2f}G < "
                      f"pool {self._pool_bytes / 2 ** 30:.2f}G + margin -> full "
                      f"release (pool freed before the boundary work)", flush=True)
                keep_pool = False
        if keep_pool:
            self._stale_graph = self.graph   # anchor; replay is forbidden
            self.graph = None
            self.cap_loss = None
            return
        self._stale_graph = None
        self.graph = None
        self.cap_loss = None
        # The pool dies with its last graph; the handle would dangle (capturing
        # into it trips the use_count internal assert). Mint a fresh pool on
        # the next capture.
        self._pool = None
        gc.collect()
        torch.cuda.empty_cache()

    def _ensure_vhat(self, model):
        # Lazily build the per-shape v_hat cache over every Concord packed module: the swapped
        # UNet layers (concord_controller.layers) + each trainable embedding core. An empty
        # module list makes switch_to a harmless no-op. Flat import mirrors the concord package.
        if self.vhat is None:
            import sys
            cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "concord")
            if cdir not in sys.path:
                sys.path.insert(0, cdir)
            from vhat_buckets import ConcordVHatBuckets
            mods = []
            ctrl = getattr(model, "concord_controller", None)
            if ctrl is not None:
                mods.extend(ctrl.layers)
            for plane in getattr(model, "concord_control_planes", None) or []:
                cp = plane.get("cp") if isinstance(plane, dict) else None
                core = getattr(getattr(cp, "trainable", None), "core", None)
                if core is not None:
                    mods.append(core)
            self.vhat = ConcordVHatBuckets(mods)
        return self.vhat

    def _alloc(self, prep, model):
        self.unet = model.unet
        self.model = model
        if self.graph_te:
            # TE captured from raw tokens: token IDs are detached int buffers (the trainable
            # embeddings they index require grad INSIDE the capture, so no bridge is needed).
            self.static = {
                "tokens_1": prep["tokens_1"].clone(),
                "tokens_2": prep["tokens_2"].clone(),
                "sample": prep["latent_input"].detach().clone(),
                "timestep": prep["timestep"].detach().clone(),
                "time_ids": prep["time_ids"].detach().clone(),
                "target": prep["target"].detach().clone(),
                # UNCOND-PASS gate: multiplied onto ehs/text_embeds INSIDE the
                # captured region (1 = cond, 0 = uncond replay). A device scalar
                # so its value crosses the graph boundary per replay.
                "cond_flag": torch.ones((), device=prep["latent_input"].device,
                                        dtype=self.dtype),
                # Per-example CFG mask for the graph_te CONTRAST path: [B], multiplied onto
                # ehs/text_embeds inside _step_fn (gated by self._contrast_cfg_mask, so it is a
                # no-op -- bit-identical -- for ordinary graph_te). The SAME mask is copied into
                # both arm replays so a CFG-dropped example is uncond in BOTH (armgap 0 = opts out),
                # matching the eager contrast's shared mask. 1.0 by default = identity.
                "cfg_mask": torch.ones((prep["latent_input"].shape[0],),
                                       device=prep["latent_input"].device, dtype=self.dtype),
            }
            return
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
            # UNCOND-PASS gate (see the TE-graph branch above).
            "cond_flag": torch.ones((), device=prep["latent_input"].device,
                                    dtype=self.dtype),
        }

    def _copy_in(self, prep):
        s = self.static
        with torch.no_grad():            # ehs/text_embeds (bridge mode) are leaves that require grad
            if self.graph_te:
                s["tokens_1"].copy_(prep["tokens_1"]); s["tokens_2"].copy_(prep["tokens_2"])
                s["sample"].copy_(prep["latent_input"]); s["timestep"].copy_(prep["timestep"])
                s["time_ids"].copy_(prep["time_ids"]); s["target"].copy_(prep["target"])
                # contrast path supplies a per-example CFG mask; ordinary graph_te has none -> 1.0.
                s["cfg_mask"].copy_(prep["cfg_mask"]) if "cfg_mask" in prep else s["cfg_mask"].fill_(1.0)
                return
            ac = prep["added_cond_kwargs"]
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

    def _ensure_loss_cfg(self, model, config):
        # Bake the loss-weighting config once (constant across the run). The captured graph
        # computes the loss from copied-in timestep/time_ids, so the per-sample weight recomputes
        # each replay -- but the gamma/flags/schedule read here are constants.
        if self._loss_cfg_ready:
            return
        self._min_snr = (config.loss_weight_fn.name == "MIN_SNR_GAMMA")
        self._gamma = float(config.loss_weight_strength)
        self._resaware = bool(getattr(config, "resolution_aware_loss_weight", False))
        self._v_pred = (getattr(getattr(model.noise_scheduler, "config", None),
                                "prediction_type", "epsilon") == "v_prediction")
        ac = getattr(model.noise_scheduler, "alphas_cumprod", None)
        self._alphas_cumprod = ac.to(self.ms.train_device).float() if ac is not None else None
        # UNCOND-PASS partition (opt-in): see the step() block. Rate = fraction of micro-steps that
        # get an extra zeroed-conditioning replay. Default off = the block never runs (bit-identical).
        self._uncond_pass = bool(getattr(config, "concord_uncond_pass", False))
        self._uncond_rate = min(1.0, max(0.0, float(getattr(config, "concord_uncond_pass_rate", 0.15) or 0.15)))
        self._contrast_rate = min(1.0, max(0.0, float(
            getattr(config, "concord_contrast_fraction", 1.0) or 1.0)))
        self._accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1) or 1))
        # graph_te contrast: bake the per-example CFG-mask multiply into the captured
        # _step_fn. A Python bool read at capture time -> MUST be set before the first
        # _warmup_and_capture and stay stable for the graph's life (invariant: a scalar
        # baked at capture cannot change per replay). True only for graph_te+contrast;
        # then _step_fn multiplies ehs/text_embeds by the copied-in cfg_mask (identity
        # x1.0 for ordinary graph_te steps, a real shared mask for contrast replays).
        # False (ordinary graph_te / bridge) => _step_fn skips the multiply => the
        # validated graph is bit-identical. See _contrast_step_graph_te.
        self._contrast_cfg_mask = bool(getattr(config, "concord_contrast_arms", False)) and self.graph_te
        if bool(getattr(config, "concord_contrast_arms", False)) and self._contrast_rate < 1.0:
            print(f"[concord_graph] CONTRAST FRACTION = {self._contrast_rate:g}: only this fraction "
                  f"of steps run the paired same-image contrast (deterministic per-step hash); the "
                  f"rest are ordinary held-out-router steps.", flush=True)
        if self._uncond_pass:
            print(f"[concord_graph] UNCOND-PASS partition ON (rate={self._uncond_rate:g}): an extra "
                  f"tick-only replay with conditioning gated to zero IN-GRAPH (cond_flag) accumulates "
                  f"the CFG uncond gradient into s_fast; the TE/embedding gradient scales by the same "
                  f"zero, so embeddings keep 100% of their gradient events (works in TE-bridge AND "
                  f"TE-in-graph modes). Fire pattern = deterministic hash of the global step. Set "
                  f"caption dropout to 0 (redundant). ⚠ still not GPU-validated -- watch the first "
                  f"fire steps' loss/health lines.", flush=True)
        self._loss_cfg_ready = True

    def _weighted_loss(self, pred, target, timestep, time_ids):
        # In-graph equivalent of ModelSetupDiffusionLossMixin's MIN_SNR_GAMMA path: per-sample
        # eps-MSE weighted by min(snr, gamma_eff)/snr, with gamma_eff = gamma * min(1, orig/crop
        # area) the spectrum-principled resolution cap (orig/crop read from the SDXL time_ids:
        # [orig_h, orig_w, crop_top, crop_left, target_h, target_w]). Pure tensor ops on copied-in
        # buffers -> recomputed each graph replay, no host sync. weight==1 recovers plain MSE.
        mse = torch.nn.functional.mse_loss(
            pred.float(), target.float(), reduction="none").mean(dim=list(range(1, pred.dim())))
        if not self._min_snr or self._alphas_cumprod is None:
            return mse.mean()
        ac = self._alphas_cumprod[timestep.long()]
        snr = ac / (1.0 - ac)
        if self._resaware:
            ti = time_ids.float()
            ratio = (ti[:, 0] * ti[:, 1] / (ti[:, 4] * ti[:, 5]).clamp(min=1.0)).clamp(max=1.0)
            gamma_t = self._gamma * ratio
        else:
            gamma_t = torch.full_like(snr, self._gamma)
        denom = snr + 1.0 if self._v_pred else snr
        weight = torch.minimum(snr, gamma_t) / denom
        return (mse * weight).mean()

    def _log_resaware(self):
        # Eager (pre-capture) diagnostic of the per-shape resolution cap -- safe to host-sync
        # here, unlike inside the captured _step_fn. Prints once per (re)capture.
        ti = self.static["time_ids"].float()
        oa = ti[:, 0] * ti[:, 1]
        ca = (ti[:, 4] * ti[:, 5]).clamp(min=1.0)
        ratio = (oa / ca).clamp(max=1.0) if self._resaware else torch.ones_like(oa)
        seen = {}
        for i in range(ti.shape[0]):
            seen[(int(ti[i, 0]), int(ti[i, 1]), int(ti[i, 4]), int(ti[i, 5]))] = float(self._gamma * ratio[i])
        msg = " | ".join(f"{a}x{b}->{c}x{d} g_eff={g:.2f}" for (a, b, c, d), g in seen.items())
        print(f"[resaware-graph] gamma={self._gamma} resaware={self._resaware} v_pred={self._v_pred} {msg}",
              flush=True)

    def _step_fn(self):
        s = self.static
        with torch.autocast(device_type="cuda", dtype=self.dtype, cache_enabled=False):
            if self.graph_te:
                # text encoder INSIDE the graph: backward reaches the embeddings here.
                # dropout off (None) + tokens provided -> capture-legal (gated by should_graph_te).
                te1, te2, pooled = self.model.encode_text(
                    train_device=self.ms.train_device, batch_size=s["tokens_1"].shape[0], rand=None,
                    tokens_1=s["tokens_1"], tokens_2=s["tokens_2"],
                    text_encoder_1_layer_skip=self.ls1, text_encoder_2_layer_skip=self.ls2,
                    text_encoder_1_output=None, text_encoder_2_output=None,
                    pooled_text_encoder_2_output=None,
                    text_encoder_1_dropout_probability=None, text_encoder_2_dropout_probability=None,
                )
                ehs, text_embeds = self.model.combine_text_encoder_output(te1, te2, pooled)
                ehs = ehs.to(self.dtype); text_embeds = text_embeds.to(self.dtype)
            else:
                ehs, text_embeds = s["ehs"], s["text_embeds"]
            if self._uncond_pass:
                # UNCOND-PASS conditioning gate: x1.0 is an exact identity, so
                # cond replays are numerically unchanged; filled to 0.0 for the
                # uncond replay, which also scales the backward into the TE /
                # embeddings to exactly zero (they never see the uncond pass,
                # in EITHER mode). Gated on the flag so the validated graph is
                # untouched bit-for-bit when the feature is off.
                ehs = ehs * s["cond_flag"]
                text_embeds = text_embeds * s["cond_flag"]
            if getattr(self, "_contrast_cfg_mask", False):
                # Shared per-example CFG mask for the graph_te contrast (see _alloc). Baked at
                # capture ONLY when graph_te+contrast is active, so ordinary graph_te (flag False)
                # stays bit-identical. Same mask in both arm replays -> CFG-dropped rows are uncond
                # in both -> armgap 0 (opts out), matching the eager contrast's shared-mask fix.
                _cm = s["cfg_mask"]
                ehs = ehs * _cm[:, None, None]
                text_embeds = text_embeds * _cm[:, None]
            pred = self.unet(s["sample"], s["timestep"], encoder_hidden_states=ehs,
                             added_cond_kwargs={"text_embeds": text_embeds,
                                                "time_ids": s["time_ids"]}).sample
        loss = self._weighted_loss(pred, s["target"], s["timestep"], s["time_ids"])
        # grad-accumulation: 1/accum scaling so the summed micro-step gradients equal
        # the averaged full-batch gradient (mirrors the eager path's loss/=accum). Baked
        # at capture; accum==1 -> no op -> the validated single-step graph is unchanged.
        if self._accum > 1:
            loss = loss / self._accum
        loss.backward()
        return loss

    def _uncond_fire(self, train_progress):
        # Deterministic per-micro-step fire decision for the uncond pass: a
        # murmur3-style integer hash of the RESUMED global step, thresholded at
        # the configured rate. A pure function of training progress, so same-seed
        # runs share the fire pattern (trajectory-level A/B stays valid) and a
        # restart-wrapper relaunch resumes it instead of re-rolling.
        gs = int(getattr(train_progress, "global_step", 0) or 0)
        h = (gs * 0x9E3779B1 + 0x517CC1B7) & 0xFFFFFFFF
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        # LIVE OVERRIDE (uncond= in CONCORD_TS_BANDS.on; single call site, no memo
        # needed). Sentinel absent -> None -> the configured rate, bit-identical.
        _ctl = concord_ts_bands()
        _rate = self._uncond_rate if (_ctl is None or _ctl.get("uncond") is None) else _ctl["uncond"]
        return (h / 4294967296.0) < _rate

    def _contrast_fire(self, train_progress):
        # WHICH steps run the paired same-image contrast vs an ordinary held-out-
        # router step. Two regimes:
        #   accum>1: ONE pair per accumulation cycle, pinned to micro 0
        #     (gs % accum == 0). The other accum-1 micros are ordinary (dithered
        #     by arm parity), the cycle's last consolidates e_L+e_H -> a cycle is
        #     {pair, ordinary..., update}. contrast_fraction<1 gates the pair over
        #     CYCLES.
        #   accum==1: every step is micro 0 -> the contrast_fraction hash gate over
        #     steps (B). rate>=1.0 -> always paired (no hash, bit-identical).
        # Deterministic murmur hash of the resumed step/cycle (same-seed A/B +
        # restart-resume valid); DISTINCT constants from _uncond_fire.
        gs = int(getattr(train_progress, "global_step", 0) or 0)
        # MEMO (one decision per gs). TWO sites evaluate this in the same iteration --
        # GenericTrainer's whole-step consolidate forcing and step()'s pair routing --
        # and the fire= live override below re-reads a user-editable file: an edit
        # landing between the two calls must not let them disagree, or the consolidate
        # flag desyncs from the actual routing (mis-consolidated window). With no
        # override active the memo just caches a pure function -> bit-identical.
        _memo = getattr(self, "_fire_memo", None)
        if _memo is not None and _memo[0] == gs:
            return _memo[1]
        accum = max(1, int(getattr(self, "_accum", 1)))
        if accum > 1 and (gs % accum) != 0:
            self._fire_memo = (gs, False)
            return False                                # ordinary micro of the cycle
        # LIVE OVERRIDE (fire= in CONCORD_TS_BANDS.on). Sentinel absent -> None ->
        # the configured _contrast_rate, bit-identical.
        _ctl = concord_ts_bands()
        _rate = self._contrast_rate if (_ctl is None or _ctl.get("fire") is None) else _ctl["fire"]
        if _rate >= 1.0:
            self._fire_memo = (gs, True)
            return True
        key = (gs // accum) if accum > 1 else gs         # per-cycle at accum>1, per-step at accum==1
        h = (key * 0x27D4EB2F + 0x165667B1) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 0xD3A2646C) & 0xFFFFFFFF
        h ^= h >> 14
        _res = (h / 4294967296.0) < _rate
        self._fire_memo = (gs, _res)
        return _res

    def step(self, model, batch, config, train_progress):
        # Contrastive arms (graph-native): pair the SAME image full-vs-dropped
        # across the two held-out arms as TWO replays of the one captured graph
        # -- batch processed ONCE (no loader yield-twice, no pool disturbance,
        # unlike the reverted loader-level version that segfaulted at capture).
        # Contrastive arms: route to the bridge twin (_contrast_step, eager TE encode
        # + gradient bridge) or the graph_te twin (_contrast_step_graph_te, TE captured
        # in the graph, no bridge). Off => the validated single-replay path below,
        # bit-identical.
        if getattr(config, "concord_contrast_arms", False) \
                and self._contrast_fire(train_progress):
            # B: fire -> paired same-image contrast; else fall through to the
            # ordinary single-replay held-out-router step below.
            if self.graph_te:
                return self._contrast_step_graph_te(model, batch, config, train_progress)
            return self._contrast_step(model, batch, config, train_progress)
        if self.graph_te:
            self.ls1 = config.text_encoder_layer_skip
            self.ls2 = config.text_encoder_2_layer_skip
            prep = self.ms.predict(model, batch, config, train_progress, return_raw_inputs=True)
        else:
            prep = self.ms.predict(model, batch, config, train_progress, return_unet_inputs=True)
        self._ensure_loss_cfg(model, config)
        # Shape-aware: the static buffers + captured graph are bound to ONE latent geometry.
        # When aspect-ratio bucketing changes the shape, release the graph (frees its pool +
        # empty_cache), rebuild the buffers, and restore this shape's cached Concord v_hat
        # before the recapture warms up. No-op while the shape is constant (single bucket).
        shape_key = tuple(prep["latent_input"].shape)
        if self.static is not None and shape_key != self._shape_key:
            # keep_pool: bucket alternation recaptures constantly; reusing the pool (it
            # grows once to the max-shape footprint) beats committing a fresh one per flip.
            # NOTE (2026-06-15, measured via the overfit census): this reuse FRAGMENTS the
            # graph pool under multi-shape recapture -- torch_reserved ratchets ~0.2G/flip
            # while live tensors stay flat. The fragmented-but-committed reserved is NOT
            # reclaimable in-process: a full release(keep_pool=False) at a flip both fails to
            # reclaim it AND reintroduces the dangling-pool segfault the anchor was added to
            # prevent. This is the WDDM/allocator limit the checkpoint-restart wrapper exists
            # for (fresh process = clean allocator). Bound the per-segment fragmentation by
            # restarting often enough (concord_train_restart) + maximizing bucket contiguity.
            self.release(keep_pool=True)
            self.static = None                   # force _alloc to rebuild for the new geometry
        if self.static is None:
            self._alloc(prep, model)
            self._shape_key = shape_key
            self._ensure_vhat(model).switch_to(shape_key)   # save outgoing, restore incoming v_hat
        self._copy_in(prep)
        # gamma-SNR dissipation modulation: the batch's timesteps are known here
        # (eager prep) and the kappa device buffers are read inside the replay --
        # write them now (no-op unless optimizer.autotune_gamma_snr is set).
        _ctrl = getattr(model, "concord_controller", None)
        if _ctrl is not None and self._alphas_cumprod is not None:
            _ctrl.on_timesteps(prep["timestep"], self._alphas_cumprod)
        if self.graph is None:
            try:
                self._warmup_and_capture(model)
            except RuntimeError as e:
                # OOM, or the allocator's pool-state internal assert (a stale/
                # dangling shared pool). Both degrade to the same recovery:
                # full release (fresh pool handle) + one retry from headroom.
                if (not isinstance(e, torch.cuda.OutOfMemoryError)
                        and "use_count" not in str(e)):
                    raise
                # Recapture on a heavily-exercised allocator (post-sample,
                # post-cache) can OOM on fragmentation where the train-start
                # capture succeeded. Retry ONCE from maximum headroom: drop the
                # static buffers + any partial pool, gc, rebuild, recapture.
                # If the retry also OOMs, the allocator is truly wedged --
                # run via scripts/concord_train_restart.py (fresh-process
                # relaunch per sample) instead.
                print("[concord_graph] capture OOM (fragmented allocator) -> "
                      "dropping static buffers and retrying once; if this also "
                      "fails, run via scripts/concord_train_restart.py",
                      flush=True)
                self.release()
                self.static = None
                self._alloc(prep, model)
                self._shape_key = shape_key
                self._ensure_vhat(model).switch_to(shape_key)
                self._copy_in(prep)
                self._warmup_and_capture(model)
        else:
            # UNCOND-PASS partition (opt-in; ⚠ built + reasoned, still not GPU-
            # validated). An EXTRA replay with the conditioning gated to ZERO INSIDE
            # the captured region: _step_fn multiplies ehs/text_embeds by the
            # cond_flag device scalar (1 = cond, 0 = uncond), so the same captured
            # graph serves BOTH modes -- TE bridge and TE-in-graph -- and the
            # backward scales the TE/embedding gradient by the same zero: the uncond
            # pass reaches them as exactly 0 and they never lose a gradient event
            # (the point: uncond training without the caption-dropout tax). Zero
            # conditioning IS the SDXL inference uncond (force_zeros_for_empty_
            # prompt), with real time_ids. Forced TICK-ONLY (set_consolidate False):
            # the replay only ACCUMULATES the uncond gradient into s_fast; the cond
            # replay below consolidates cond+uncond together. Order is load-bearing:
            # uncond BEFORE cond so the consolidation sweeps it up. The fire decision
            # is a deterministic hash of the resumed global step -- host RNG here
            # decorrelates same-seed A/B runs and re-rolls at every restart-wrapper
            # relaunch. Known accounting: the pass is ADDITIVE (~(1+rate) gradient
            # mass vs cond-only), and the sigmag noise injection rides the extra
            # backward too -- zero-mean and admitted, so harmless in expectation
            # (the arms mean-center it).
            if self._uncond_pass and self.graph is not None \
                    and self._uncond_fire(train_progress):
                from modules.util.optimizer.concord.prototype_packed_b import (
                    set_consolidate, _get_consolidate_flag)
                _dev = self.static["sample"].device
                _saved = int(_get_consolidate_flag(_dev).item())  # the trainer's intended gate
                set_consolidate(_dev, False)            # tick-only: accumulate, never consolidate
                self.static["cond_flag"].fill_(0.0)     # gate conditioning to zero in-graph
                self.graph.replay()                     # UNet(0, latent)->loss->bwd: s_fast += uncond
                self.static["cond_flag"].fill_(1.0)     # cond replays always see gate = 1
                set_consolidate(_dev, bool(_saved))     # restore the trainer's gate
            if not self.graph_te:
                self._zero_input_grads()                 # fresh static-input grads for this replay
            self.graph.replay()
        if not self.graph_te:
            self._bridge(prep)                           # eager: TE + embeddings receive grad (bridge mode)
        elif os.environ.get("CONCORD_GRAPH_DEBUG"):
            opt = getattr(model, "optimizer", None)
            if opt is not None:
                aux_ids = {id(p) for p in self.aux}
                gn = sum(float(p.grad.float().norm()) for g in opt.param_groups for p in g["params"]
                         if id(p) not in aux_ids and p.grad is not None)
                print(f"[concord_graph] TE-graph: embedding/TE grad-norm sum {gn:.5f}", flush=True)
        return self.cap_loss

    def _contrast_step(self, model, batch, config, train_progress):
        """Graph-native contrastive arms (concord_contrast_arms). TWO replays
        of the ONE captured graph on the SAME image at the SAME timestep/noise
        (shared seed): FULL caption -> arm L (tick), token-DROPPED -> arm H
        (consolidate). The 2fast gap floor reads e_L - e_H = the token's
        CONTEXT and evaporates it, banking the token's context-invariant core
        (exp 72). The batch is prepped TWICE eagerly (two TE encodes; VAE is
        cached, timestep/noise shared) but flows through the graph as two
        ordinary replays with different ehs copied in -- the graph pool is
        never disturbed. (The reverted loader-level pairing re-ran the batch
        through the whole eager-prep->graph pipeline twice and segfaulted the
        pool: 0xC0000005 at capture.) This call IS the whole update, so it
        needs accum==1 (the setup guard permits heldout_router+accum==1 under
        contrast) and the TE bridge (dropout>0 -> should_graph_te False).
        NOT GPU-VALIDATED. Validation protocol in the PR notes: (1) contrast
        OFF is bit-identical (different code path); (2) first armed run --
        watch capture (~step 50) survives, then per-step loss finite and the
        [concord-health] gap (e_L-e_H proxy) is nonzero on token-bearing
        batches; (3) confirm the deploy weight moves (consolidation fired on
        arm H). Known rough edge: the warmup self-steps at capture tick e_L
        only (arm L) with no paired e_H -- a one-time capture-time imbalance
        that should wash out; watch the first post-capture health line."""
        from modules.util.optimizer.concord.prototype_packed_b import (
            set_consolidate, set_arm_sel, _get_consolidate_flag)
        _accum = max(1, int(config.gradient_accumulation_steps))
        if _accum > 1 and not getattr(self, "_contrast_accum_warned", False):
            if _accum == 2:
                print(f"[concord_graph] CONTRASTIVE ARMS @ accum=2 (WHOLE-STEP): a paired cycle IS "
                      f"the whole 2-tick step -- arm L (tick) + arm H (tick + CONSOLIDATE e_L+e_H) "
                      f"in one call, forced-update, counted as 2 micros (global_step += 2). Ordinary "
                      f"cycles are 2 held-out-router micros (arm parity). Every cycle fills the arms "
                      f"1:1 -- no odd-accum 2:1 asymmetry, no double-consolidation.", flush=True)
            else:
                print(f"[concord_graph] CONTRASTIVE ARMS @ accum={_accum}: one paired "
                      f"micro per cycle (micro 0, tick-only into BOTH arms) + "
                      f"{_accum - 1} ordinary held-out-router micros (dithered by arm "
                      f"parity); the cycle's LAST micro consolidates e_L+e_H. The pair "
                      f"now respects __is_update_step -- no double-consolidation. NOTE: at "
                      f"odd accum the ordinary micros fill the arms unevenly (2:1); prefer accum=2.",
                      flush=True)
            self._contrast_accum_warned = True
        seed = int(getattr(train_progress, "global_step", 0))
        # Two preps, SAME seed -> identical sample/timestep/noise/target; ehs
        # differs only by caption. The token-dropout hook reads
        # _concord_contrast_mode; the shared seed also makes the coupled CFG
        # dropout mask identical, so CFG-dropped examples are uncond in BOTH
        # arms (gap 0 -> they opt out of the contrast, as intended).
        model._concord_contrast_mode = "full"
        model._concord_contrast_seed = seed
        prep_L = self.ms.predict(model, batch, config, train_progress,
                                 return_unet_inputs=True)
        model._concord_contrast_mode = None
        model._concord_contrast_seed = None
        # prep_H: REUSE prep_L's diffusion inputs VERBATIM (same noisy latent,
        # timestep, target tensors) and recompute ONLY the ehs from the
        # token-dropped caption. Two separate predict() calls do NOT share the
        # timestep/noise -- a persistent sampling state or any RNG-consumption
        # difference desyncs them, and the gap becomes (full@t1)-(dropped@t2),
        # swamped by the timestep mismatch, not the caption (the "gap goes the
        # other way" symptom). Reusing the tensors makes the two arms differ by
        # caption ALONE. The CFG mask is re-seeded to prep_L's batch_seed so
        # CFG-dropped examples stay uncond in both arms (exact single-GPU;
        # multi-GPU would need the world_size/rank factor).
        from random import Random as _Random
        _bd = dict(batch)
        model._concord_contrast_mode = "dropped"
        # Deterministic per-step generator: the dropped arm now picks ONE random
        # comma-delimited clause (token_clause_keep), so it needs a seeded RNG
        # for same-seed A/B + restart-resume (was an unseeded Generator when the
        # dropped branch was deterministic token-only). Distinct salt from the
        # CFG-mask / uncond streams.
        _cgen = torch.Generator(device=self.ms.train_device)
        _cgen.manual_seed((seed ^ 0x5EEDC1A5) & 0x7FFFFFFF)
        _bd = self.ms._concord_token_only_dropout(model, config, _bd, _cgen)
        model._concord_contrast_mode = None
        _bs = prep_L["latent_input"].shape[0]
        _tr1 = config.train_text_encoder_or_embedding()
        _tr2 = config.train_text_encoder_2_or_embedding()
        # dropped ehs with CFG dropout DEFERRED (None/False); prep_L's ehs is
        # likewise nodrop (predict defers CFG while _concord_contrast_mode is
        # set). One shared mask is applied to BOTH below.
        # The TE forward MUST run under the model's autocast (predict wraps its
        # own encode_text the same way); a bare call feeds bf16 activations to
        # fp32 CLIP layer-norm -> "expected Float but found BFloat16".
        with model.autocast_context:
            ehs_d, pooled_d = model.combine_text_encoder_output(*model.encode_text(
                train_device=self.ms.train_device, batch_size=_bs, rand=_Random(seed),
                tokens_1=_bd['tokens_1'], tokens_2=_bd['tokens_2'],
                text_encoder_1_layer_skip=config.text_encoder_layer_skip,
                text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
                text_encoder_1_output=(_bd['text_encoder_1_hidden_state'] if not _tr1 else None),
                text_encoder_2_output=(_bd['text_encoder_2_hidden_state'] if not _tr2 else None),
                pooled_text_encoder_2_output=(_bd['text_encoder_2_pooled_state'] if not _tr2 else None),
                text_encoder_1_dropout_probability=None,
                text_encoder_2_dropout_probability=None,
                couple_dropout=False,
            ))
        # ONE shared coupled CFG mask (rate max of the two configured) applied
        # to BOTH arms -> a CFG-dropped example is uncond in each, so its gap is
        # 0 (opts out) instead of a spurious cond-vs-uncond gap. This is why the
        # per-arm masks mattered: without sharing, the UNet is conditioned
        # differently even with the SAME caption (nonzero gap in the divot).
        # Mask matches the ehs dtype (bf16) so the copy_in into the static
        # graph buffers stays clean.
        _ehs_L = prep_L["encoder_hidden_states"]
        _te_L = prep_L["added_cond_kwargs"]["text_embeds"]
        _p = max(float(config.text_encoder.dropout_probability or 0.0),
                 float(config.text_encoder_2.dropout_probability or 0.0))
        if _p > 0.0:
            _r = _Random(seed)
            _m = (torch.tensor([_r.random() for _ in range(_bs)],
                               device=_ehs_L.device) > _p).to(_ehs_L.dtype)
            _ehs_L = _ehs_L * _m[:, None, None]
            _te_L = _te_L * _m[:, None]
            ehs_d = ehs_d * _m[:, None, None]
            pooled_d = pooled_d * _m[:, None]
        prep_L["encoder_hidden_states"] = _ehs_L
        prep_L["added_cond_kwargs"] = {**prep_L["added_cond_kwargs"],
                                       "text_embeds": _te_L}
        prep_H = dict(prep_L)                          # same latent/timestep/target
        prep_H["encoder_hidden_states"] = ehs_d
        prep_H["added_cond_kwargs"] = {**prep_L["added_cond_kwargs"],
                                       "text_embeds": pooled_d}
        self._ensure_loss_cfg(model, config)
        # geometry (both preps share it): recapture on aspect-bucket flips
        shape_key = tuple(prep_L["latent_input"].shape)
        if self.static is not None and shape_key != self._shape_key:
            self.release(keep_pool=True)
            self.static = None
        if self.static is None:
            self._alloc(prep_L, model)
            self._shape_key = shape_key
            self._ensure_vhat(model).switch_to(shape_key)
        dev = self.static["sample"].device
        # Update-micro flag: the trainer set the consolidate device flag to
        # __is_update_step BEFORE step(); read it BEFORE we override, so the pair
        # consolidates ONLY on the cycle's update micro. accum==1 -> the pair IS
        # the update (flag=1, banks e_L+e_H now). accum>1 -> the pair sits on
        # micro 0 (flag=0, tick-only); the ordinary last micro banks e_L+e_H.
        _is_update = int(_get_consolidate_flag(dev).item())
        _ctrl = getattr(model, "concord_controller", None)
        if _ctrl is not None and self._alphas_cumprod is not None:
            _ctrl.on_timesteps(prep_L["timestep"], self._alphas_cumprod)
        # --- arm L: full caption, TICK ONLY (consf=0); arm H's apply banks both ---
        # ONE consolidation per paired step, on arm H. The 2-fast apply chases
        # s_slow toward the SUM e_L+e_H (prototype_packed_2fast: fine_post =
        # e_L+e_H, gated by consf, NOT arm_sel) and the momentum/evap are consf-
        # gated per arm -- so a SINGLE consf=1 apply banks BOTH arms and compares
        # them through the gap e_L-e_H in one coherence gate. set_consolidate(False)
        # here does NOT drop arm L: arm L ticks e_L now, and it is consolidated in
        # arm H's apply below. Consolidating on BOTH replays fires the chase +
        # evaporation TWICE per step -> the direction whips and the two arms are
        # never weighed against each other in a single gate. (Both arms are still
        # ticked, so the step is 2 ticks for the clock; see _contrast_step_ticks.)
        # ANTITHETIC arm assignment: alternate WHICH caption lands in which accumulator (e_L vs
        # e_H) per cycle, so the arm asymmetry (bracket friction lam*(1±d), held-out role) de-
        # correlates from the caption contrast instead of stamping a CONSTANT directional lean on
        # the consolidated e_L+e_H. The apply drains the SUM (fine_post = e_L+e_H, consf-gated), so
        # which caption sits in the consolidating arm (H) is content-symmetric. Deterministic on the
        # CYCLE index parity -> same-seed A/B + restart-resume valid; the token-invariant part still
        # agrees in BOTH arms (banks) and context-only words still disagree (evaporate) either way.
        _swap = ((seed // max(1, self._accum)) & 1) == 1
        _prep1, _prep2 = (prep_H, prep_L) if _swap else (prep_L, prep_H)   # replay1->e_L, replay2->e_H
        # --- replay 1 -> arm L (e_L), TICK ONLY (consf=0); arm H's apply banks BOTH via e_L+e_H ---
        self._copy_in(_prep1)
        set_arm_sel(dev, True)
        set_consolidate(dev, False)
        if self.graph is None:
            self._warmup_and_capture(model)        # captures on the replay-1 pass (shape-only; both preps share geometry)
        else:
            if not self.graph_te:
                self._zero_input_grads()
            self.graph.replay()
        if not self.graph_te:
            self._bridge(_prep1)                   # TE/embedding grad from replay 1
        _loss1 = self.cap_loss
        if torch.is_tensor(_loss1):
            _loss1 = _loss1.detach().clone()
        # --- replay 2 -> arm H (e_H), TICK (+ CONSOLIDATE e_L+e_H only on the cycle's update micro) ---
        self._copy_in(_prep2)
        set_arm_sel(dev, False)
        set_consolidate(dev, bool(_is_update))
        if not self.graph_te:
            self._zero_input_grads()
        self.graph.replay()
        if not self.graph_te:
            self._bridge(_prep2)
        _loss2 = self.cap_loss
        if torch.is_tensor(_loss2):
            _loss2 = _loss2.detach().clone()
        # Contrast telemetry (meter only): the two replays ran on the SAME latent/timestep/noise,
        # differing by caption ALONE, so L_dropped - L_full is a DIRECT read of how much the caption
        # context lowers the loss. Label by CAPTION, NOT by arm -- else the swap flips the armgap sign
        # every other pair and scrambles armgap x SNR. cap_loss is the static graph loss buffer
        # (overwritten every replay) so the clones above decouple it; .item() is deferred to the
        # health line. Detached clones live outside the graph pool (no pinned segment).
        _loss_full, _loss_drop = (_loss2, _loss1) if _swap else (_loss1, _loss2)
        if _ctrl is not None:
            _ctrl._last_contrast_full = _loss_full
            _ctrl._last_contrast_drop = _loss_drop
            # Per-pair meter (armgap x SNR): buffer this pair's timestep vector with its arm
            # losses so the trainer can drain it to telemetry. cap_loss is the batch-MEAN over a
            # timestep-spanning micro, so the offline harness deconvolves per-SNR-bin armgap
            # (armgap_s = sum_b w_bs armgap_b) -- the same trick the CSNR meter uses. .clone() so
            # the copy is decoupled from the graph's static input buffer (overwritten next copy_in);
            # bounded so a drain stall can't grow it unbounded. Meter-only, never touches training.
            _ts = prep_L.get("timestep")
            if _ts is not None:
                _pend = getattr(_ctrl, "_contrast_pending", None)
                if _pend is None:
                    _pend = _ctrl._contrast_pending = []
                _pend.append((_ts.detach().clone(), _loss_full, _loss_drop))
                if len(_pend) > 4096:
                    del _pend[:len(_pend) - 4096]
            if _is_update and self._accum != 2:
                # Clock counts the paired UPDATE as 2 ticks (both arms) -- only when the pair is
                # EXTRA on top of the cycle (accum==1: the pair IS the update and injects 2 passes
                # where an ordinary step injects 1). At accum>1 (old design) the pair is micro 0,
                # not the update, so this never fires and the ordinary update micro ticks the clock
                # by 1. At accum==2 (whole-step design) the pair REPLACES the cycle -- it is one
                # optimizer step like any other, 2 passes across one image just as an ordinary
                # accum==2 cycle is 2 passes across two images -- so it counts as 1, NOT 2 (the
                # default in after_step applies). Bumping to 2 here would run step_idx ~10% fast.
                _ctrl._contrast_step_ticks = 2
        return _loss_full

    def _contrast_step_graph_te(self, model, batch, config, train_progress):
        """graph_te contrastive arms -- the TE-in-graph twin of _contrast_step. The two
        arms differ by TOKEN SET (full caption vs clause-dropped) instead of by an eagerly
        encoded ehs: encode_text runs INSIDE the captured graph (_step_fn), so each arm is
        a replay of the ONE captured encode_text->UNet->loss->backward with a DIFFERENT
        token buffer copied in (a device int tensor crossing the graph boundary, exactly as
        ordinary graph_te copies its tokens). No eager TE encode and NO bridge -- the
        captured backward reaches the trainable embeddings directly (that is the whole point
        of graph_te). The shared per-example CFG mask rides the in-graph cfg_mask multiply
        (baked at capture via _contrast_cfg_mask, set in _ensure_loss_cfg), so a CFG-dropped
        row is uncond in BOTH arms -> armgap 0 (opts out), the graph twin of the eager
        shared-mask fix. Everything downstream of the two preps -- antithetic caption<->arm
        swap, arm_sel/consolidate routing, one-consolidation-per-pair, armgap x SNR
        telemetry, the clock-ticks gate -- is byte-for-byte the same reasoning as
        _contrast_step; see that method's comments for the arm/consolidation derivation.

        Requires should_graph_te True (concord_graph_te + the contrast dropout carve-out).
        Designed for contrast_fraction=1.0: an ORDINARY graph_te step (contrast_fraction<1
        non-pair micro) carries NO CFG dropout (its cfg_mask is filled 1.0), so sub-1.0
        fractions drop the CFG-dropout on the ordinary micros -- prefer fraction=1.0 until
        that path also draws a mask. NOT GPU-VALIDATED: the open question is whether
        encode_text survives capture with this run's embedding machinery (shields /
        group-subspace / packed embeddings). Validate ordinary graph_te (contrast OFF)
        captures cleanly FIRST, then this."""
        from modules.util.optimizer.concord.prototype_packed_b import (
            set_consolidate, set_arm_sel, _get_consolidate_flag)
        from random import Random as _Random
        # layer skips for the in-graph encode_text (mirrors the ordinary graph_te step()).
        self.ls1 = config.text_encoder_layer_skip
        self.ls2 = config.text_encoder_2_layer_skip
        seed = int(getattr(train_progress, "global_step", 0))
        # prep_L: FULL-caption RAW inputs (tokens + diffusion inputs, TE NOT run).
        # contrast_mode='full' => _concord_token_only_dropout is a no-op (full tokens) and
        # return_raw_inputs skips encode_text entirely (CFG dropout deferred). The shared
        # seed pins the timestep/noise so prep_H reuses them verbatim (caption-only gap).
        model._concord_contrast_mode = "full"
        model._concord_contrast_seed = seed
        prep_L = self.ms.predict(model, batch, config, train_progress, return_raw_inputs=True)
        model._concord_contrast_mode = None
        model._concord_contrast_seed = None
        # Dropped-arm tokens: token_clause_keep on a SHALLOW batch copy (original batch's
        # token tensors untouched -- _concord_token_only_dropout reassigns the dict keys to
        # new tensors, never mutates in place). Same seeded stream + distinct salt as the
        # bridge twin, so same-seed A/B + restart-resume stay valid.
        _bd = dict(batch)
        model._concord_contrast_mode = "dropped"
        _cgen = torch.Generator(device=self.ms.train_device)
        _cgen.manual_seed((seed ^ 0x5EEDC1A5) & 0x7FFFFFFF)
        _bd = self.ms._concord_token_only_dropout(model, config, _bd, _cgen)
        model._concord_contrast_mode = None
        _bs = prep_L["latent_input"].shape[0]
        dev = prep_L["latent_input"].device
        # ONE shared coupled CFG mask (max of the two configured rates), 1.0 = keep /
        # 0.0 = uncond, applied to BOTH arms inside _step_fn -> a CFG-dropped row is uncond
        # in each arm (armgap 0). SAME _Random(seed) draw as the bridge twin's shared mask,
        # so the CFG-dropped rows match bridge<->graph_te for A/B parity.
        _p = max(float(config.text_encoder.dropout_probability or 0.0),
                 float(config.text_encoder_2.dropout_probability or 0.0))
        if _p > 0.0:
            _r = _Random(seed)
            _cfg = (torch.tensor([_r.random() for _ in range(_bs)], device=dev) > _p).to(self.dtype)
        else:
            _cfg = torch.ones((_bs,), device=dev, dtype=self.dtype)
        prep_L["cfg_mask"] = _cfg
        # prep_H: SAME diffusion inputs + SAME cfg_mask (shared via the shallow copy so both
        # arms multiply by the identical mask tensor), only the token set differs.
        prep_H = dict(prep_L)
        prep_H["tokens_1"] = _bd["tokens_1"]
        prep_H["tokens_2"] = _bd["tokens_2"]
        self._ensure_loss_cfg(model, config)
        # geometry (both preps share it): recapture on aspect-bucket flips.
        shape_key = tuple(prep_L["latent_input"].shape)
        if self.static is not None and shape_key != self._shape_key:
            self.release(keep_pool=True)
            self.static = None
        if self.static is None:
            self._alloc(prep_L, model)
            self._shape_key = shape_key
            self._ensure_vhat(model).switch_to(shape_key)
        # Update-micro flag: read the trainer's intended consolidate gate BEFORE overriding,
        # so the pair consolidates only on the cycle's update micro (accum==1/whole-step: the
        # pair IS the update; accum>1: tick-only, the ordinary last micro banks e_L+e_H).
        _is_update = int(_get_consolidate_flag(dev).item())
        _ctrl = getattr(model, "concord_controller", None)
        if _ctrl is not None and self._alphas_cumprod is not None:
            _ctrl.on_timesteps(prep_L["timestep"], self._alphas_cumprod)
        # ANTITHETIC caption<->arm swap (see _contrast_step): decorrelate the arm friction
        # asymmetry from the caption contrast; the apply drains the SUM e_L+e_H so which
        # caption sits in the consolidating arm H is content-symmetric.
        _swap = ((seed // max(1, self._accum)) & 1) == 1
        _prep1, _prep2 = (prep_H, prep_L) if _swap else (prep_L, prep_H)   # replay1->e_L, replay2->e_H
        # --- replay 1 -> arm L (e_L), TICK ONLY (consf=0); arm H's apply banks BOTH ---
        self._copy_in(_prep1)
        set_arm_sel(dev, True)
        set_consolidate(dev, False)
        if self.graph is None:
            self._warmup_and_capture(model)        # captures on the replay-1 pass (shape-only)
        else:
            self.graph.replay()                    # no _zero_input_grads / _bridge: graph_te has no static-input grads
        _loss1 = self.cap_loss
        if torch.is_tensor(_loss1):
            _loss1 = _loss1.detach().clone()
        # --- replay 2 -> arm H (e_H), TICK (+ CONSOLIDATE e_L+e_H only on the update micro) ---
        self._copy_in(_prep2)
        set_arm_sel(dev, False)
        set_consolidate(dev, bool(_is_update))
        self.graph.replay()
        _loss2 = self.cap_loss
        if torch.is_tensor(_loss2):
            _loss2 = _loss2.detach().clone()
        # Contrast telemetry (meter only; label by CAPTION, not arm, so the swap doesn't flip
        # the armgap sign): armgap x SNR pending buffer, mirroring _contrast_step exactly.
        _loss_full, _loss_drop = (_loss2, _loss1) if _swap else (_loss1, _loss2)
        if _ctrl is not None:
            _ctrl._last_contrast_full = _loss_full
            _ctrl._last_contrast_drop = _loss_drop
            _ts = prep_L.get("timestep")
            if _ts is not None:
                _pend = getattr(_ctrl, "_contrast_pending", None)
                if _pend is None:
                    _pend = _ctrl._contrast_pending = []
                _pend.append((_ts.detach().clone(), _loss_full, _loss_drop))
                if len(_pend) > 4096:
                    del _pend[:len(_pend) - 4096]
            if _is_update and self._accum != 2:
                _ctrl._contrast_step_ticks = 2
        return _loss_full

    def _warmup_and_capture(self, model):
        # Real-gradient warmup on a side stream, then capture. Factored out so
        # the fragmentation-OOM retry in step() can re-run it after a rebuild.
        if self.graph_te:
            # The captured encode_text runs the FULL forward for BOTH text encoders (no cached
            # hidden states -- backward has to reach the embeddings). A non-trained encoder (e.g.
            # CLIP-G when only CLIP-L trains) lives on the temp/CPU device, and a pre-capture
            # sample offloads even the trained ones, so force every TE onto the train device
            # before warmup/capture. Otherwise the embedding lookup gets CPU weights vs CUDA
            # token ids ("Expected all tensors to be on the same device"). No-op if already there.
            for _to in ("text_encoder_1_to", "text_encoder_2_to"):
                _fn = getattr(self.model, _to, None)
                if _fn is not None:
                    _fn(self.ms.train_device)
        if os.environ.get("CONCORD_RESAWARE_DEBUG") and self._min_snr:
            self._log_resaware()
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
        self._zero_warmup_grads(model)               # discard warmup-accumulated grads
        if not self.graph_te:
            self._zero_input_grads()                 # fresh static-input grads for capture
        import gc
        had_anchor = self._stale_graph is not None
        if had_anchor and self._pool_bytes:
            # Headroom check. Capturing while the anchor holds the old pool
            # peaks at ~2x the pool (old + new blocks coexist until the anchor
            # drops). On a card without that slack, WDDM does not OOM -- it
            # silently demotes to shared memory (observed: slight overflow at
            # the first epoch boundary), so the OOM-retry never fires. If free
            # memory cannot cover another pool plus margin, degrade up front
            # to the full-release path: fresh pool at 1x peak; the post-capture
            # cleanup below prevents the old ratchet on this path too.
            try:
                free, _ = torch.cuda.mem_get_info()
            except Exception:
                free = None
            need = self._pool_bytes + (256 << 20)
            if free is not None and free < need:
                print(f"[concord_graph] boundary headroom {free / 2 ** 30:.2f}G < "
                      f"pool {self._pool_bytes / 2 ** 30:.2f}G + margin -> releasing "
                      f"pool before recapture (1x peak)", flush=True)
                self._stale_graph = None
                self._pool = None
                gc.collect()
                torch.cuda.empty_cache()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        self.graph = torch.cuda.CUDAGraph()
        # ORDER IS LOAD-BEARING: the anchor (old graph) must stay alive THROUGH
        # capture_begin -- a pool handle alone does not keep the pool alive;
        # dropping the last graph zombifies it and capturing into the dangling
        # handle trips the allocator's use_count>0 internal assert (crashed
        # 2026-06-12 at the first aspect-bucket flip). Capturing while the old
        # graph holds the pool is the make_graphed_callables pattern; the old
        # graph is never replayed again, so capture-order replay constraints
        # don't bind. Peak = old+new blocks (~2x one step's transient) during
        # this capture only; on a too-tight card that OOMs into step()'s
        # full-release retry, which degrades gracefully to a fresh pool.
        with torch.cuda.graph(self.graph, pool=self._pool):
            self.cap_loss = self._step_fn()
        if had_anchor:
            # Anchor served its purpose (kept the pool committed through the
            # boundary; kept it ALIVE through capture_begin). Drop it, then the
            # one cleanup the boundary was missing: every torch_gc in the
            # backup path runs BEFORE this recapture, so the boundary
            # transient's ordinary segments were never decommitted -- the
            # +0.5G/epoch ratchet. The new graph's pool blocks are live and
            # untouched by empty_cache; the old graph's blocks return to the
            # pool for the next recapture.
            self._stale_graph = None
            gc.collect()
            torch.cuda.empty_cache()
            try:
                free, _ = torch.cuda.mem_get_info()
                a = torch.cuda.memory_allocated() / 2 ** 30
                r = torch.cuda.memory_reserved() / 2 ** 30
                print(f"[concord_graph] post-recapture: torch_alloc={a:.2f}G "
                      f"torch_reserved={r:.2f}G device_free={free / 2 ** 30:.2f}G",
                      flush=True)
            except Exception:
                pass
        # Record the pool footprint for the boundary headroom check: right
        # after a capture (+cleanup), reserved-minus-allocated is dominated by
        # the live pool blocks. Keep the high-water mark.
        try:
            self._pool_bytes = max(
                self._pool_bytes,
                int(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()))
        except Exception:
            pass

    def _zero_warmup_grads(self, model):
        # Discard grads accumulated during warmup so the captured backward writes exactly this
        # step's grad. graph_te: the captured backward writes UNet-aux AND embedding grads, so
        # zero the full optimizer param set; bridge mode: only UNet aux (the embeddings get grad
        # from the eager bridge, which runs after).
        if self.graph_te and getattr(model, "optimizer", None) is not None:
            for group in model.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.zero_()
        else:
            for p in self.aux:
                if p.grad is not None:
                    p.grad.zero_()


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
