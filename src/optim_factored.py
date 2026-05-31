"""FactoredAdam -- Adam with a RANK-1 (Adafactor) v-hat on 2D weights.

Isolates ONE variable vs torch AdamW: the RANK of the second-moment estimate.
Everything else (momentum, bias correction, decoupled wd, betas, lr schedule)
is identical to AdamW. So on the enwik8 comparability bench (same init + batch
order), where it lands between SGD-chase (1.43) and full-vhat Adam (1.07) tells
us how much of the gap a rank-1 v-hat captures -- i.e. how many "ranks with the
next-lowest noise" you actually need.

  2D weight g (out x in):
    R = b2 R + (1-b2) sum_j g^2_ij        # row accumulator  [out]
    C = b2 C + (1-b2) sum_i g^2_ij        # col accumulator  [in]
    v_hat_ij = (Rhat_i * Chat_j) / sum(Rhat)         # rank-1 reconstruction
  1D params (LayerNorm, biases): full diagonal v-hat (they're tiny; the rank
  question is about the big matrices).
"""
import torch
from torch.optim.optimizer import Optimizer


class FactoredAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for grp in self.param_groups:
            b1, b2 = grp['betas']
            lr, eps, wd = grp['lr'], grp['eps'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st['t'] = 0
                    st['m'] = torch.zeros_like(p)
                    if p.dim() == 2:
                        st['R'] = torch.zeros(p.shape[0], device=p.device)
                        st['C'] = torch.zeros(p.shape[1], device=p.device)
                    else:
                        st['v'] = torch.zeros_like(p)
                st['t'] += 1
                t = st['t']
                m = st['m']
                m.mul_(b1).add_(g, alpha=1 - b1)
                mh = m / (1 - b1 ** t)
                g2 = g * g
                if p.dim() == 2:
                    R, C = st['R'], st['C']
                    R.mul_(b2).add_(g2.sum(dim=1), alpha=1 - b2)
                    C.mul_(b2).add_(g2.sum(dim=0), alpha=1 - b2)
                    bc = 1 - b2 ** t
                    Rh = R / bc
                    Ch = C / bc
                    # rank-1 reconstruction of E[g^2]; sum(Rh)=sum(Ch)=total
                    v_hat = (Rh.unsqueeze(1) * Ch.unsqueeze(0)) / (Rh.sum() + 1e-30)
                else:
                    v = st['v']
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    v_hat = v / (1 - b2 ** t)
                upd = mh / (v_hat.sqrt() + eps)
                if wd != 0:
                    upd = upd + wd * p          # decoupled (AdamW-style)
                p.add_(upd, alpha=-lr)
        return loss
