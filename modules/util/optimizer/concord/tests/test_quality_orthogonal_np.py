"""Numpy mirror of quality_orthogonal + the backward shield's masked projection
-- validates the math (and why mean-centering is load-bearing) without importing
torch, so it is safe to run alongside a live training job.

The numpy functions below mirror orthonormal_basis / QualityProjector (eager) and
the masked gradient shield in concord_embedding_packed._PackedEmbStep.backward;
if you change one, change the other.
"""
import numpy as np


# --- numpy mirror of orthonormal_basis / QualityProjector ----------------------------
def basis(dirs, tol=1e-6):
    """dirs [n, dim] (already mean-centered) -> Q [dim, r] orthonormal."""
    U, S, Vh = np.linalg.svd(dirs, full_matrices=False)
    keep = S > (tol * max(S[0], tol))
    return Vh[keep].T            # [dim, r]


def project_positions(X, Q, mu, one_sided=False):
    C = (X - mu) @ Q
    if one_sided:
        C = np.clip(C, 0.0, None)
    return X - C @ Q.T


def project_gradient(G, Q, one_sided=False):
    C = G @ Q
    if one_sided:
        C = np.clip(C, 0.0, None)
    return G - C @ Q.T


def shield(G, Q, subject_mask, one_sided=False):
    """The backward shield: project only the subject rows (mask=1); tag rows free."""
    C = G @ Q
    if one_sided:
        C = np.clip(C, 0.0, None)
    return G - subject_mask[:, None] * (C @ Q.T)


def overlap(X, Q, mu):
    return np.abs((X - mu) @ Q).max() if Q.size else 0.0


def basis_raw(label_embs, tol=1e-6):
    U, S, Vh = np.linalg.svd(label_embs, full_matrices=False)
    keep = S > (tol * max(S[0], tol))
    return Vh[keep].T


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, info=""):
        global ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}")

    dim = 64
    # Anisotropic vocab: a large SHARED mean direction every token has, plus a
    # small per-token part. This is the CLIP regime that makes centering matter.
    shared = rng.normal(size=dim)
    shared = shared / np.linalg.norm(shared) * 10.0          # dominant shared axis
    vocab = shared[None, :] + 0.5 * rng.normal(size=(5000, dim))
    mu = vocab.mean(0)

    # Quality-TAG deploy vectors = a handful of learned tokens; one direction each
    # (centered by the vocab mean).
    tag_rows = shared[None, :] + 0.5 * rng.normal(size=(4, dim))
    dirs = tag_rows - mu
    Q = basis(dirs)
    print(f"tag subspace rank = {Q.shape[1]} from {dirs.shape[0]} tags (dim={dim})")

    # A subject embedding in the same anisotropic regime.
    x = shared + 0.5 * rng.normal(size=dim)

    # 1. hard projection -> subject has zero overlap with every tag direction
    xc = project_positions(x[None], Q, mu)[0]
    check("hard projection zeroes the overlap", overlap(xc[None], Q, mu) < 1e-10,
          f"overlap {overlap(x[None], Q, mu):.3e} -> {overlap(xc[None], Q, mu):.3e}")

    # 2. mean-centering is load-bearing: the shared anisotropic mass survives.
    kept_centered = np.linalg.norm(xc) / np.linalg.norm(x)
    Qraw = basis_raw(tag_rows)                               # basis of RAW tag embs
    xraw = x - (x @ Qraw) @ Qraw.T
    kept_raw = np.linalg.norm(xraw) / np.linalg.norm(x)
    check("mean-centering preserves the shared component (centered keeps ~norm)",
          kept_centered > 0.97, f"kept {kept_centered:.3f}")
    check("raw (uncentered) projection would gut the subject (contrast)",
          kept_raw < 0.5, f"raw kept {kept_raw:.3f} vs centered {kept_centered:.3f}")

    # 3. THE SHIELD: subjects (mask=1) projected, tags (mask=0) left free.
    K = 6
    G = rng.normal(size=(K, dim))
    subject_mask = np.array([0, 0, 1, 1, 1, 1], dtype=float)   # rows 0,1 are tags
    Gs = shield(G, Q, subject_mask)
    subj = subject_mask.astype(bool)
    check("shield zeroes the tag-overlap of SUBJECT steps",
          np.abs(Gs[subj] @ Q).max() < 1e-10, f"|Gsub.Q| {np.abs(Gs[subj] @ Q).max():.2e}")
    check("shield leaves TAG steps completely untouched (they are the sink)",
          np.allclose(Gs[~subj], G[~subj], atol=1e-12),
          f"max tag delta {np.abs(Gs[~subj] - G[~subj]).max():.2e}")

    # 4. one-sided shield: a subject step moving AWAY from every tag dir is untouched
    away = -(Q.sum(axis=1))                                   # negative on every dir
    Gw = np.stack([away, rng.normal(size=dim)])
    mask2 = np.array([1.0, 1.0])
    Gw_s = shield(Gw, Q, mask2, one_sided=True)
    check("one-sided leaves an away-from-tag subject step untouched",
          np.allclose(Gw[0], Gw_s[0], atol=1e-10), f"max delta {np.abs(Gw[0]-Gw_s[0]).max():.2e}")

    # 5. idempotence of the position projection
    xcc = project_positions(xc[None], Q, mu)[0]
    check("position projection is idempotent", np.allclose(xc, xcc, atol=1e-10),
          f"max delta {np.abs(xc - xcc).max():.2e}")

    # 6. a clean step survives the gradient projection (only the tag part is removed)
    g = rng.normal(size=(3, dim))
    g_clean = g - (g @ Q) @ Q.T
    check("gradient projection preserves the orthogonal (clean) part",
          np.allclose(g_clean, project_gradient(g_clean, Q), atol=1e-10))

    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    raise SystemExit(0 if ok else 1)
