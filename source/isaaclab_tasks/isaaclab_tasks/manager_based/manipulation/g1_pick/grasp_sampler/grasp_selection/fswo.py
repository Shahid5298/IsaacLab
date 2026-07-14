"""FSWO (Frictionless Self-Balancing Wrench Optimizer) stability scoring.

Reference: Lightning Grasp (Yin & Abbeel, arXiv:2511.07418, 2025), FSWO stage only
-- see Misc./optimizer.md S1-S2 for the full derivation. Replaces the previous
nearest-palm heuristic used to pick a single grasp from the 32-grasp UltraDexGrasp
library with a force-closure-quality ranking.

Given k contact points {(p_i, n_i)} (p_i = contact position, n_i = INWARD surface
normal), frictionless force closure asks for alpha_i >= 0, max_i(alpha_i) = 1, with

    sum_i alpha_i n_i = 0                  (force balance)
    sum_i alpha_i (p_i x n_i) = 0          (torque balance about the origin)

Relaxed to a minimum-residual QP with Gram matrix Q_ij = n_i.n_j + lam*(tau_i.tau_j),
tau_i = p_i x n_i (this Q is PSD by construction: it's the Gram matrix of the
stacked vectors [n_i; sqrt(lam)*tau_i]):

    alpha* = argmin_{alpha_{-j} >= 0, alpha_j=1} alpha^T Q alpha

solved once per choice of the fixed index j, keeping the global minimum over j.
Score S = -alpha*^T Q alpha* in (-inf, 0]; S = 0 is perfect force closure.

IMPORTANT: this is NOT the same problem as NNLS directly on Q's columns. Recasting
argmin alpha^T Q alpha as argmin ||Q_free @ alpha_free + Q_j||^2 (using raw Q as the
least-squares design matrix) changes the KKT/complementary-slackness conditions
whenever the true optimum has some alpha_i pinned to 0 -- verified numerically
against a brute-force constrained solver (SLSQP) on random contact sets: the raw-Q
version is measurably wrong (reports a worse score than the true optimum) whenever
an active non-negativity constraint exists. The correct NNLS reduction uses a
matrix square root L of Q (Q = L^T L, via eigendecomposition since Q is PSD but not
necessarily full rank) as the design matrix instead:

    alpha* = argmin_{alpha_{-j} >= 0} || L[:, -j] @ alpha_{-j} + L[:, j] ||^2

which is a genuine || . ||^2 <-> alpha^T Q alpha equivalence (alpha^T Q alpha =
alpha^T L^T L alpha = ||L alpha||^2), so NNLS's complementary slackness on the
residual L@alpha matches the QP's complementary slackness on Q@alpha exactly.
Depends only on cube-frame contact geometry, so it is invariant to the cube's pose
in the world -- computed once offline, never per episode.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def _psd_sqrt(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetric matrix square root L of a PSD matrix Q, s.t. Q = L @ L (= L.T @ L).
    Negative eigenvalues (numerical noise; Q is PSD by construction) are clipped to 0.
    """
    w, V = np.linalg.eigh(Q)
    w = np.clip(w, eps, None) - eps  # clip, keeping the floor at exactly 0 for near-zero modes
    return V @ np.diag(np.sqrt(w)) @ V.T


def fswo_score(contacts: np.ndarray, lam: float = 1.0) -> float:
    """FSWO stability score for one grasp.

    Args:
        contacts: (k, 6) array, each row [px, py, pz, nx, ny, nz] -- contact
            position and inward surface normal, both in the same frame (cube
            frame, per Misc./optimizer.md S3.1).
        lam: torque-balance weight relative to force balance. Default 1.0
            matches Lightning Grasp's normalized-coordinate default; increase
            (e.g. 10.0) to penalize rocking instability more on a small object
            where torque arms are numerically small relative to force terms.

    Returns:
        Score in (-inf, 0]; 0 = perfect frictionless force closure.
    """
    if contacts.shape[0] < 2:
        raise ValueError(f"FSWO needs >= 2 contacts, got {contacts.shape[0]}.")

    p, n = contacts[:, :3], contacts[:, 3:]
    tau = np.cross(p, n)  # (k, 3) torque contribution of unit force at each contact

    Q = n @ n.T + lam * (tau @ tau.T)  # (k, k) Gram matrix, PSD
    L = _psd_sqrt(Q)  # Q = L @ L; NNLS design matrix must be L, not Q itself (see module docstring)

    k = contacts.shape[0]
    best_val = np.inf
    for j in range(k):
        mask = np.arange(k) != j
        A = L[:, mask]  # (k, k-1)
        b = -L[:, j]  # (k,)
        alpha_free, _ = nnls(A, b)
        alpha = np.zeros(k)
        alpha[mask] = alpha_free
        alpha[j] = 1.0
        val = float(alpha @ Q @ alpha)
        if val < best_val:
            best_val = val

    return -best_val


def batch_fswo_scores(contacts_batch: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """FSWO scores for a batch of grasps.

    Args:
        contacts_batch: (N, k, 6) contact arrays, one per grasp.
        lam: see ``fswo_score``.

    Returns:
        (N,) scores; argmax gives the mechanically best grasp index.
    """
    return np.array([fswo_score(contacts_batch[i], lam) for i in range(len(contacts_batch))])
