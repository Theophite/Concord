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


def make_concord_config(learning_rate: float):
    """Map OneTrainer settings onto the validated winner config. Stage 1: take lr from
    OneTrainer's learning_rate; the rest are the validated sf_060 winner defaults."""
    from concord_winner import ConcordConfig
    return ConcordConfig(lr=float(learning_rate))


class ConcordController:
    """Holds the swapped Concord UNet layers + the per-step schedule + the rebalance gate
    for one training run. Created in the SDXL setup (after the model is loaded, before the
    optimizer is built); driven by the trainer via before_step()/after_step()."""

    def __init__(self, unet, device, learning_rate: float, total_steps: int):
        from concord_winner import swap_unet_to_winner, GatedRebalance
        self.config = make_concord_config(learning_rate)
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
