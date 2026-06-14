"""Quality-tag shielding for Concord packed embeddings.

The user marks some of their trainable embeddings as "low-quality tags" and
captions the quality-problem images with them. Those tag embeddings train
FREELY -- they co-occur with the defect across many subjects, so the shared
low-quality signal accumulates in them (subjects cancel): they become the sink,
and a droppable / negative-promptable knob at inference. What we protect is
every OTHER (subject) embedding: its gradient is projected off the subspace the
tags currently span, so it learns subject content from a bad image but not the
badness. Overlap is fine -- a subject can sit anywhere; it just can't be PUSHED
along the tag directions.

The tag directions are LEARNED and moving, so the basis is rebuilt each step
(eagerly, outside the CUDA graph) from the tags' current deploy vectors and
handed to the captured backward as a fixed buffer.

    dir_t   = tag_deploy_t - vocab_mean                  # one per quality-tag row
    Q       = orthonormal basis of { dir_t }             # [dim, r], via SVD
    shield a SUBJECT step:  G - (G Q) Q^T                 # hard: no motion along Q
                            G - relu(G Q) Q^T             # one-sided: only block
                                                          # motion TOWARD bad
    (mu drops out of a gradient; it matters only for projecting POSITIONS at save.)

MEAN-CENTERING IS LOAD-BEARING: CLIP embeddings are anisotropic (a dominant axis
every token shares, the tags included), so projecting against the RAW tag vectors
would strip that shared component out of every subject. Centering by the vocab
mean removes only the quality-specific contrast the tag actually carries.

Pure-torch; the math is mirrored by tests/test_quality_orthogonal_np.py (numpy)
so it can be validated without importing torch.
"""
import torch


def orthonormal_basis(dirs, ncols=None, tol=1e-6):
    """dirs [n, dim] -> Q [dim, ncols] with orthonormal columns spanning the row
    space of `dirs`. Dependent / near-zero directions are dropped (SVD); if
    `ncols` is given the result is zero-padded (or truncated) to exactly that many
    columns so the buffer handed to the captured graph keeps a STATIC shape (a
    zero column projects to nothing, so padding is a clean no-op)."""
    if dirs.numel() == 0:
        dim = dirs.shape[-1] if dirs.dim() == 2 else 0
        return torch.zeros(dim, ncols or 0, dtype=torch.float32, device=dirs.device)
    _U, S, Vh = torch.linalg.svd(dirs.float(), full_matrices=False)
    keep = S > (tol * S[0].clamp_min(tol))
    Q = Vh[keep].T.contiguous()                          # [dim, r]
    if ncols is not None:
        r = Q.shape[1]
        if r < ncols:
            Q = torch.cat([Q, torch.zeros(Q.shape[0], ncols - r, device=Q.device)], dim=1)
        elif r > ncols:
            Q = Q[:, :ncols].contiguous()
    return Q.float()


class QualityProjector:
    """Eager-side helper around a frozen (Q, mu) for projecting POSITIONS at save
    and for diagnostics. The per-step gradient shield in the captured backward
    reads the Q buffer directly (see concord_embedding_packed) -- it does not go
    through this object."""

    def __init__(self, Q, mu):
        self.Q = Q          # [dim, r] orthonormal columns (zero columns allowed)
        self.mu = mu        # [dim]

    @classmethod
    def from_directions(cls, dirs, mu, ncols=None):
        return cls(orthonormal_basis(dirs, ncols), mu.float())

    def project_positions(self, X, one_sided=False):
        """X [K, dim] -> X with the tag-subspace component of (X - mu) removed."""
        if self.Q.numel() == 0:
            return X
        Xf = X.float()
        C = (Xf - self.mu) @ self.Q
        if one_sided:
            C = torch.clamp(C, min=0.0)
        return (Xf - C @ self.Q.T).to(X.dtype)

    def project_gradient(self, G, one_sided=False):
        """G [K, dim] -> the step with its tag-subspace component removed (mu drops
        out). one_sided removes only the part moving TOWARD a tag direction."""
        if self.Q.numel() == 0:
            return G
        Gf = G.float()
        C = Gf @ self.Q
        if one_sided:
            C = torch.clamp(C, min=0.0)
        return (Gf - C @ self.Q.T).to(G.dtype)

    def overlap(self, X):
        """Max |coefficient| of (X - mu) on the tag subspace; 0.0 == clean."""
        if self.Q.numel() == 0:
            return 0.0
        return ((X.float() - self.mu) @ self.Q).abs().max().item()
