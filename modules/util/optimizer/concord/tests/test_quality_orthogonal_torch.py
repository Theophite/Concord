"""Torch-level test of quality_orthogonal (CPU-only; no CUDA, no model load).

Covers what the numpy mirror cannot: the torch SVD basis with zero-padding to a
static column count (so the captured-graph buffer keeps a fixed shape), dtype
handling, and the masked shield exactly as _PackedEmbStep.backward applies it.
DEFERRED: do not run while a training job holds the GPU -- it imports torch. Run
on the next idle window:  python test_quality_orthogonal_torch.py
"""
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the concord/ dir

from quality_orthogonal import orthonormal_basis, QualityProjector


def main():
    torch.manual_seed(0)
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}")

    dim = 48
    shared = torch.randn(dim)
    shared = shared / shared.norm() * 8.0
    vocab = shared + 0.5 * torch.randn(100, dim)             # anisotropic
    mu = vocab.float().mean(0)

    # 3 tag directions, but ask for 4 columns -> 1 zero-padded (rank-deficient case)
    tag_deploy = shared + 0.5 * torch.randn(3, dim)
    Q = orthonormal_basis(tag_deploy - mu, ncols=4)
    check("static-shape Q: padded to requested ncols", Q.shape == (dim, 4), f"{tuple(Q.shape)}")
    nz = Q.abs().sum(0) > 0
    check("rank-deficient column is zero-padded", int(nz.sum()) == 3, f"{int(nz.sum())} nonzero cols")
    QtQ = Q[:, nz].T @ Q[:, nz]
    check("nonzero columns orthonormal",
          torch.allclose(QtQ, torch.eye(int(nz.sum())), atol=1e-5),
          f"max off-I {(QtQ - torch.eye(int(nz.sum()))).abs().max():.2e}")

    # masked shield exactly as the backward applies it (K=5: rows 0,1 tags)
    K = 5
    G = torch.randn(K, dim)
    subject_mask = torch.tensor([0., 0., 1., 1., 1.]).reshape(K, 1)
    C = G @ Q
    Gs = G - subject_mask * (C @ Q.T)
    subj = subject_mask.bool().reshape(-1)
    check("shield zeroes subject tag-overlap", (Gs[subj] @ Q).abs().max() < 1e-4,
          f"{(G[subj] @ Q).abs().max():.3e} -> {(Gs[subj] @ Q).abs().max():.3e}")
    check("shield leaves tag rows untouched", torch.allclose(Gs[~subj], G[~subj], atol=1e-6))

    # one-sided shield: a subject step moving away from every tag dir is untouched
    away = -(Q[:, nz].sum(dim=1)).unsqueeze(0)
    C2 = torch.clamp(away @ Q, min=0.0)
    away_s = away - (C2 @ Q.T)
    check("one-sided leaves away-from-tag step untouched",
          torch.allclose(away, away_s, atol=1e-5), f"max delta {(away - away_s).abs().max():.2e}")

    # QualityProjector.project_positions (save path): subject cleaned, dtype kept
    proj = QualityProjector(Q, mu)
    x = (shared + 0.5 * torch.randn(4, dim)).to(torch.bfloat16)
    xc = proj.project_positions(x)
    check("save-path output dtype preserved", xc.dtype == torch.bfloat16)
    check("save-path zeroes overlap", proj.overlap(xc) < 1e-2,
          f"overlap {proj.overlap(x):.3e} -> {proj.overlap(xc):.3e}")   # bf16 tol
    kept = proj.project_positions(shared.unsqueeze(0)).norm() / shared.norm()
    check("mean-centering preserves the shared component", kept > 0.95, f"kept {kept:.3f}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
