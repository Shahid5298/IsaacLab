# Grasp Optimizer: FSWO-Based Selection from UltraDex Candidates

The grasp sampler (UltraDexGrasp + BODex, documented in `Misc./UltraDex.md`) produces 32
pre-synthesized Inspire Hand grasps for the 5 cm cube, stored in
`grasp_sampler/grasp_dataset/cube_5cm_grasps_valid.npz`. The current selection heuristic
uses a fixed index (`fixed_grasp_idx: 15`) chosen by nearest-palm frequency from an early
training run. This document defines a principled replacement: a Frictionless Self-Balancing
Wrench Optimizer (FSWO) that scores each of the 32 candidates by force-closure quality and
selects the mechanically optimal one.

---

## 1. Lightning Grasp — Relevant Background

Lightning Grasp (Yin and Abbeel, arXiv:2511.07418, 2025) is a procedural grasp synthesis
system achieving 300–1000 effective samples per second on a single A100 GPU. The key
innovation is a **Contact Field** data structure: a BVH-organized collection of contact
vectors in $\mathbb{R}^3 \times S^2$ (position, inward-normal pairs) indexed by hand link,
joint configuration, and object surface point. This decouples geometric computation (which
part of the hand contacts which part of the object) from the stability optimization (whether
those contacts produce force closure).

The full Lightning Grasp pipeline is: preprocess → object placement → contact domain
detection → **FSWO contact optimization** → kinematics optimization → postprocess. We adopt
only the FSWO stage. The UltraDexGrasp + BODex pipeline has already solved contact domain
detection and kinematics optimization for our setting: each of the 32 library grasps
specifies a palm pose and joint configuration that places fingertips on the cube surface
with force-closure constraints applied during BODex synthesis. FSWO is applied as a
post-hoc ranking criterion to identify the grasp with the best stability margin among the 32.

### 1.1 Contact Field (Reference)

For a dexterous hand $H$ with kinematic model $\mathcal{M}$:
$$\mathcal{CF}(H) = \bigcup_{i \in \text{links}} \bigcup_{\substack{p \in \partial H_i \\ n \in \hat{n}(p,\mathcal{M}_i)}} \left\{ T_i(q) \cdot (p, n) \;\middle|\; q \in \mathcal{C} \right\}$$
where $T_i(q)$ maps link $i$'s local surface point $(p, n)$ to world frame via FK at
configuration $q$. The contact domain is the intersection of the hand's contact field with
the object surface (positions where hand and object surfaces coincide with antipodal normals):
$$\mathcal{D}(H, O) = \mathcal{CF}(H) \cap \{(p, -n) \mid p \in \partial O,\; n \in \hat{n}(p, O)\}$$

The 32 UltraDex library entries are effectively samples drawn from $\mathcal{D}(H, O)$ for
the Inspire right hand and the 5 cm cube; we do not need to recompute the contact domain.

---

## 2. FSWO Formulation

### 2.1 Stability Criterion

Given $k$ fingertip contact points $\{(p_i, n_i)\}_{i=1}^{k}$ where $p_i \in \mathbb{R}^3$
is the contact position on the object surface and $n_i \in S^2$ is the **inward** surface
normal (pointing into the object, i.e., the direction the finger pushes), a grasp achieves
**frictionless force closure** if there exist $\alpha_i \geq 0$ such that:
$$\sum_{i=1}^k \alpha_i n_i = 0 \qquad \text{(force balance)}$$
$$\sum_{i=1}^k \alpha_i (p_i \times n_i) = 0 \qquad \text{(torque balance about origin)}$$
with $\max_i \alpha_i = 1$ (normalization; avoids the trivial $\alpha = 0$ solution).

Perfect force closure requires both equalities simultaneously. FSWO relaxes this to a
minimum-residual problem by combining both conditions into a quadratic objective:
$$\min_{\alpha \geq 0,\; \max_i \alpha_i = 1} \left\| \sum_i \alpha_i n_i \right\|^2 + \lambda \left\| \sum_i \alpha_i (p_i \times n_i) \right\|^2$$

### 2.2 Matrix Reduction to NNLS

Let $f_i = n_i \in \mathbb{R}^3$ and $\tau_i = p_i \times n_i \in \mathbb{R}^3$ (torque
contribution of unit force at contact $i$ about the origin). Define the Gram matrix:
$$Q_{ij} = f_i \cdot f_j + \lambda\, (\tau_i \cdot \tau_j) \in \mathbb{R}^{k \times k}$$

The objective becomes $\alpha^T Q \alpha$, which is a convex QP in $\alpha$. To enforce
$\max_i \alpha_i = 1$ with $\alpha \geq 0$, fix $\alpha_j = 1$ for each candidate $j \in
\{1,\ldots,k\}$ and solve for the remaining components. **Correction (2026-07-13):**
an earlier version of this document reduced this directly to
$\min_{\alpha_{-j}\ge0}\|Q_{:,-j}\alpha_{-j}+Q_{:,j}\|^2$ using $Q$ itself as the NNLS
design matrix — this is **not** equivalent to minimizing $\alpha^TQ\alpha$ whenever the
true constrained optimum has any $\alpha_i$ pinned to $0$ (verified numerically against
a brute-force SLSQP solve on random contact sets: the raw-$Q$ version reports a
strictly worse score than the true optimum in that case). The correct reduction uses a
matrix square root $L$ of $Q$ (any $L$ with $Q = L^\top L$; since $Q$ is PSD but not
necessarily full rank, computed via eigendecomposition with negative/near-zero
eigenvalues clipped to $0$), so that $\alpha^TQ\alpha = \|L\alpha\|^2$ genuinely:
$$\min_{\alpha_{-j} \geq 0} \| L_{:,-j}\, \alpha_{-j} + L_{:,j} \|^2$$
which **is** a faithful Non-Negative Least Squares (NNLS) reduction of the QP (NNLS's
complementary slackness on the residual $L\alpha$ matches the QP's complementary
slackness on $Q\alpha$ exactly, because $Q=L^\top L$). Iterating over all $k$
candidates for the fixed maximum takes the global minimum:
$$\alpha^*, j^* = \arg\min_{j} \min_{\alpha_{-j} \geq 0} \alpha^T Q \alpha \bigg|_{\alpha_j = 1}$$

### 2.3 Score

$$S_{\text{FSWO}} = -(\alpha^*)^T Q\, \alpha^* \;\in\; (-\infty,\; 0]$$

$S = 0$ corresponds to perfect force closure (residual forces and torques vanish exactly).
More negative values indicate less stable grasps. The optimal grasp from the library is:
$$g^* = \arg\max_{g \in \{0,\ldots,31\}} S_{\text{FSWO}}(g)$$

### 2.4 Hyperparameter $\lambda$

Controls torque balance weight relative to force balance. Lightning Grasp uses $\lambda = 1$
in normalized contact coordinates. For the 5 cm cube, the contact point spread is
$\sim 0.025$ m (half-edge), so torques are numerically smaller than forces; increasing
$\lambda$ (e.g., $\lambda = 10$) emphasizes rotational stability. Default: $\lambda = 1$
(match Lightning Grasp; tune if the selected grasp tends to have a rocking instability).

**Critical note:** FSWO scores depend only on contact geometry in the cube frame, not on
the cube's pose in the world. All 32 scores can be computed once offline; no per-episode
recomputation is needed until the clutter-aware extension (§5.2) is added.

---

## 3. Applying FSWO to the 32-Grasp Library

### 3.1 Data Layout

`grasp_dataset/cube_5cm_grasps_valid.npz` — relevant fields:
- `grasp_pose`: `float32 (32, 1, 3, 13)` — 32 grasps × 1 hand × 3 stages × 13 values
  - Stage 0: pregrasp (approach pose, ~5 cm above cube)
  - Stage 1: grasp (fingertips on cube surface) ← used for FSWO
  - Stage 2: squeeze (tightened grip)
  - Values: $[x,\; y,\; z,\; q_w,\; q_x,\; q_y,\; q_z,\; \theta_{\text{ty}},\; \theta_{\text{tp}},\; \theta_{\text{idx}},\; \theta_{\text{mid}},\; \theta_{\text{rng}},\; \theta_{\text{pnk}}]$
  - All poses expressed in the cube frame (cube centered at origin, axis-aligned)
- `T_usdbase_urdfbase`: `float64 (4, 4)` — Kabsch-calibrated transform correcting the 90° axis
  permutation between the URDF base frame and USD `R_hand_base_link` (0.3 mm residual)

### 3.2 Inspire Hand FK for Contact Extraction

The Inspire Hand has 6 proximal DOF commanding 12 DOF via mimic ratios:

| Proximal (master) | Slave joints |
|---|---|
| `thumb_yaw` | — |
| `thumb_pitch` | `thumb_inter = thumb_pitch × 0.8024`, `thumb_distal = thumb_inter × 0.9487` |
| `index_proximal` | `index_inter = index_proximal × 1.0843` |
| `middle_proximal` | `middle_inter = middle_proximal × 1.0843` |
| `ring_proximal` | `ring_inter = ring_proximal × 1.0843` |
| `pinky_proximal` | `pinky_inter = pinky_proximal × 1.0843` |

As documented in UltraDex.md (Discovery 1), USD slave joints have no position drives (the
`PhysxMimicJointAPI` gearing is active but the drives are absent). During BODex synthesis,
the **synthesis URDF** was used with slave joints manually frozen at measured values:
$$\theta_{\text{thumb\_inter}} = -0.16\;\text{rad}, \quad \theta_{\text{thumb\_distal}} = -0.24\;\text{rad}, \quad \theta_{\text{finger\_inter}} = 1.15\;\text{rad (all 4 fingers)}$$
The optimizer uses the same synthesis URDF with the same frozen slave joints; FK produces
the actual contact configuration used during BODex synthesis.

The friend's `check_grasps_offline.py` already implements `fk_tips()` (pytorch_kinematics
chain, same URDF). Import directly rather than re-implementing.

Fingertip world position for grasp $g$, finger $\ell$:
$$p_\ell^{\text{cube}} = T^*_{g} \cdot T_{\text{usdbase}}^{\text{urdfbase}} \cdot \text{FK}_\ell(q^*_g, q_{\text{fixed}})$$
where $T^*_g = (R^*_g, t^*_g)$ is the Stage-1 palm transform from the library (in cube
frame) and $\text{FK}_\ell$ evaluates the URDF kinematic chain to link $\ell$'s origin.

### 3.3 Contact Point Projection

For a point-contact model on the cube (half-edge $r = 0.025$ m, axis-aligned, centered at
origin), the nearest surface point and inward normal for fingertip position $p \in
\mathbb{R}^3$:
$$k^* = \arg\max_{k \in \{x,y,z\}} \left| p_k / r \right|$$
$$p^{\text{contact}} = p \;\text{with}\; p_{k^*} \leftarrow \text{sign}(p_{k^*}) \cdot r$$
$$n = -\text{sign}(p_{k^*}) \cdot e_{k^*}$$
where $e_{k^*}$ is the standard basis vector along axis $k^*$. Well-synthesized BODex
grasps place fingertips within $\lesssim 2$ mm of the cube surface, so the projection
correction is negligible. For a grasp $g$ with 5 contact points ($k = 5$, one per finger),
the FSWO input is $\{(p^\text{contact}_\ell, n_\ell)\}_{\ell=1}^{5}$.

---

## 4. Implementation

### 4.1 File Structure

```
grasp_sampler/
├── grasp_dataset/
│   └── cube_5cm_grasps_valid.npz        ← friend's output (32 grasps)
├── check_grasps_offline.py              ← friend's validation script; provides fk_tips()
└── grasp_selection/
    ├── __init__.py
    ├── fswo.py                           ← FSWO scoring (numpy + scipy, no Isaac Lab)
    └── select_optimal_grasp.py          ← top-level script; outputs ranked indices + best
```

Run `select_optimal_grasp.py` once from the `env_isaaclab` conda env on any machine (CPU
is sufficient). The output JSON contains the best index and the full ranking, to be
substituted into the env config.

### 4.2 `grasp_selection/fswo.py`

**Note (2026-07-13):** the snippet below reflects the corrected NNLS reduction
(§2.2) — it uses the PSD square root $L$ of $Q$ as the NNLS design matrix, not
$Q$ itself. See `grasp_selection/fswo.py` for the actual, current file (this
block is illustrative and may drift from it; the real file is authoritative).

```python
"""FSWO stability scoring for grasps on the 5 cm cube."""
import numpy as np
from scipy.optimize import nnls


def _psd_sqrt(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetric square root L of a PSD matrix Q, s.t. Q = L @ L."""
    w, V = np.linalg.eigh(Q)
    w = np.clip(w, eps, None) - eps
    return V @ np.diag(np.sqrt(w)) @ V.T


def fswo_score(contacts: np.ndarray, lam: float = 1.0) -> float:
    """
    Compute FSWO stability score for one grasp.

    contacts : (k, 6) — each row [px, py, pz, nx, ny, nz];
               positions and inward surface normals at k contact points
    lam      : torque weight; default 1.0 (Lightning Grasp paper default)
    Returns  : float in (-inf, 0]; 0 = perfect force closure
    """
    p = contacts[:, :3]           # (k, 3) contact positions
    n = contacts[:, 3:]           # (k, 3) inward normals
    tau = np.cross(p, n)          # (k, 3) torque contributions

    # Q_ij = n_i·n_j + lam*(tau_i·tau_j); PSD Gram matrix
    Q = n @ n.T + lam * (tau @ tau.T)   # (k, k)
    L = _psd_sqrt(Q)                   # Q = L @ L -- NNLS needs L, not Q (see S2.2)

    k = contacts.shape[0]
    best_val = np.inf

    for j in range(k):
        mask = np.arange(k) != j
        A = L[:, mask]             # (k, k-1): columns for the free alphas
        b = -L[:, j]                # (k,): shift from fixing alpha_j = 1
        alpha_free, _ = nnls(A, b)
        alpha = np.zeros(k)
        alpha[mask] = alpha_free
        alpha[j] = 1.0
        val = float(alpha @ Q @ alpha)   # objective value uses Q (not L) -- alpha^T Q alpha
        if val < best_val:
            best_val = val

    return -best_val   # higher is better


def batch_fswo_scores(contacts_batch: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    FSWO scores for a batch of grasps.

    contacts_batch : (N, k, 6)
    Returns        : (N,) scores; argmax gives the best grasp index
    """
    return np.array([fswo_score(contacts_batch[i], lam) for i in range(len(contacts_batch))])
```

### 4.3 `grasp_selection/select_optimal_grasp.py`

```python
"""
Select the mechanically optimal grasp from the 32-sample UltraDex library.

Run once offline:
    cd grasp_sampler/
    python grasp_selection/select_optimal_grasp.py \\
        --library grasp_dataset/cube_5cm_grasps_valid.npz \\
        --lam 1.0 \\
        --out grasp_selection/scores.json

Then update fixed_grasp_idx in g1_pick_env_cfg.py to result["best_idx"].
"""
import sys, argparse, json
import numpy as np

# Reuse the friend's FK implementation
sys.path.insert(0, ".")
from check_grasps_offline import fk_tips   # returns (5, 3) fingertip positions in cube frame

from grasp_selection.fswo import batch_fswo_scores

CUBE_HALF = 0.025   # m; half-edge of the 5 cm cube


def cube_contact(p: np.ndarray) -> tuple:
    """Return (contact_point, inward_normal) for fingertip position p on cube surface."""
    k = int(np.argmax(np.abs(p) / CUBE_HALF))
    contact = p.copy()
    contact[k] = np.sign(p[k]) * CUBE_HALF
    normal = np.zeros(3)
    normal[k] = -np.sign(p[k])
    return contact, normal


def build_contacts(
    grasps: np.ndarray,           # (N, 1, 3, 13)
    T_cal: np.ndarray,            # (4, 4) calibration transform
) -> np.ndarray:
    """Compute (N, 5, 6) contact array for all N grasps at the grasp stage."""
    N = grasps.shape[0]
    contacts = np.zeros((N, 5, 6))

    for g in range(N):
        stage1 = grasps[g, 0, 1, :]         # (13,); Stage 1 = grasp
        palm_pos  = stage1[:3]
        palm_quat = stage1[3:7]              # [w, x, y, z]
        joint_q   = stage1[7:]               # 6 proximal DOF

        tips = fk_tips(palm_pos, palm_quat, joint_q, T_cal)  # (5, 3) in cube frame

        for i, tip in enumerate(tips):
            c, n = cube_contact(tip)
            contacts[g, i, :3] = c
            contacts[g, i, 3:] = n

    return contacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="grasp_dataset/cube_5cm_grasps_valid.npz")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="grasp_selection/scores.json")
    args = ap.parse_args()

    data   = np.load(args.library, allow_pickle=True)
    grasps = data["grasp_pose"]           # (32, 1, 3, 13)
    T_cal  = data["T_usdbase_urdfbase"]   # (4, 4)

    contacts = build_contacts(grasps, T_cal)           # (32, 5, 6)
    scores   = batch_fswo_scores(contacts, lam=args.lam)   # (32,)

    best_idx = int(np.argmax(scores))
    ranking  = np.argsort(scores)[::-1].tolist()

    result = {
        "best_idx":      best_idx,
        "best_score":    float(scores[best_idx]),
        "ranked_indices": ranking,
        "scores":        scores.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Best grasp: index {best_idx}  (FSWO = {scores[best_idx]:.5f})")
    print(f"Ranking (best → worst): {ranking}")
    print(f"Previous fixed index 15: FSWO = {scores[15]:.5f}")


if __name__ == "__main__":
    main()
```

**Dependency on `fk_tips`.** The signature expected above is:
```python
def fk_tips(
    palm_pos:  np.ndarray,   # (3,)   palm position in cube frame
    palm_quat: np.ndarray,   # (4,)   [w, x, y, z]
    joint_q:   np.ndarray,   # (6,)   [thumb_yaw, thumb_pitch, idx, mid, rng, pnk]
    T_cal:     np.ndarray,   # (4, 4) calibration transform from .npz
) -> np.ndarray:             # (5, 3) fingertip positions in cube frame
```
If the friend's `check_grasps_offline.py` uses a different signature, adapt the wrapper in
`build_contacts` accordingly. The internal FK chain must use the **synthesis URDF** with
slave joints frozen at the BODex values ($-0.16$, $-0.24$, $1.15$ rad; see §3.2).

---

## 5. Integration with the RL Policy

### 5.1 Offline Mode (Immediate)

Run the selector once, note the best index, and substitute it into the env config:

```python
# g1_pick_env_cfg.py — inside G1RightArmLiftEnvCfg_V2
grasp_goal_event = EventTerm(
    func=mdp.sample_grasp_goal,
    mode="reset",
    params={
        "fixed_grasp_idx": 7,   # ← replace 15 with FSWO best_idx from scores.json
        "library_path": "grasp_sampler/grasp_dataset/cube_5cm_grasps_valid.npz",
    },
)
```

The existing reward terms in `mdp/grasp_goal.py` are unchanged. For completeness, the
target $(p^*, q^*)$ for the selected grasp is transformed to world frame at each episode
reset using the cube's sampled initial pose, and updated every step via `update_live_goals`:

$$r_{\text{palm}} = 1 - \tanh\!\left(\frac{\|p_{\text{palm}} - p^*\|}{0.15}\right), \quad w_{\text{palm}} = 1.0$$

$$r_{\text{hand}} = \underbrace{\left(1 - \tanh\!\left(\frac{\|p_{\text{palm}} - p^*\|}{0.20}\right)\right)}_{\text{proximity gate}} \cdot \left(1 - \tanh\!\left(\frac{\|q - q^*\|}{0.5}\right)\right), \quad w_{\text{hand}} = 0.3$$

The FSWO selector improves on the heuristic because BODex grasps with lower force-closure
quality have wider acceptable contact regions (lower wrench resistance → policy can "cheat"
by approaching at a shallower angle). The FSWO-optimal grasp has the narrowest acceptable
contact region and the highest wrench resistance, making it the most demanding but also the
most transferable to real hardware.

### 5.2 Goal-Conditioned Mode (Next Step After Baseline Converges)

The current single-fixed-grasp constraint (Problem 17 in UltraDex.md) prevents the policy
from adapting its approach to the cube orientation, which varies across episodes. To remove
this constraint without the hidden-goal ceiling, extend the observation by 13 dims:

$$o_t \in \mathbb{R}^{109}: \quad o_t = \left[\underbrace{o_t^{\text{base}}}_{96},\; \underbrace{p^*_{\text{palm}}}_{3},\; \underbrace{q^*_{\text{palm}}}_{4},\; \underbrace{q^*_{\text{joints}}}_{6}\right]$$

where all goal quantities are expressed in the robot base frame (same convention as the rest
of $o_t$). The policy can then condition its approach on which grasp is currently active.

At episode reset, the optimizer selects per-episode:
1. Transform all 32 library grasps from cube frame to world frame using the sampled cube
   initial pose (already done inside `sample_grasp_goal`).
2. Look up pre-computed offline FSWO scores (no re-computation needed; FSWO is invariant to
   rigid-body transformations of the cube).
3. Optionally weight by a clutter corridor score $\rho_g \in [0,1]$: probability that a
   straight-line pregrasp trajectory is unobstructed by distractor positions (computable
   from the 10 distractor positions in $o_t$ using a capsule-cylinder intersection test).
4. Select $g^* = \arg\max_g S_{\text{FSWO}}(g) \cdot \rho_g$.

Combined score: $S_g = S_{\text{FSWO}}(g) \cdot \rho_g$. The FSWO term is fixed (lookup);
only the clutter term varies per episode and is cheap to evaluate ($O(32 \times 10)$ capsule
tests ≈ $10^{-4}$ s per env).

**Observation extension in code:**

```python
# mdp/observations.py — add new term
@torch.jit.export
def grasp_goal_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current grasp goal (palm pose + joint config) in robot base frame. Shape: (N, 13)."""
    goal = env.goal_buffer  # (N, 13); maintained by sample_grasp_goal + update_live_goals
    return goal

# g1_pick_env_cfg.py — ObservationsCfg
@configclass
class ObservationsCfg:
    ...
    @configclass
    class PolicyCfg(ObsGroup):
        ...
        grasp_goal: ObsTerm = ObsTerm(func=mdp.grasp_goal_obs)

# Also update:
observation_space = 109   # was 96
```

---

## 6. Verification Checklist

Before deploying the new `fixed_grasp_idx`:

1. **Sanity-check FSWO scores.** Run `select_optimal_grasp.py` and confirm all 32 scores
   are finite and $\leq 0$. A `NaN` score indicates a degenerate FK result (fingertip far
   from cube surface, so the contact projection is invalid — check the synthesis URDF path
   and `T_cal` loading).

2. **Compare against grasp 15 visually.** Use `check_grasps_offline.py --idx <best_idx>` 
   and `--idx 15` side by side. The FSWO-optimal grasp should show more symmetric finger
   placement around the cube (no single finger dominating the contact, balanced wrenches).

3. **Check fingertip-surface distances.** Print `np.linalg.norm(tips - p_contacts, axis=1)`
   for the optimal grasp; should be $< 5$ mm. Larger distances indicate that FK does not
   reproduce the BODex contact (likely a URDF mismatch or wrong slave-joint values).

4. **Update config and retrain.** Either retrain from scratch or fine-tune from the current
   checkpoint (fine-tuning is faster but may need a temporary learning rate warmup if the
   new target is far from the old one in joint-config space).

5. **Monitor TensorBoard.** `grasp_goal_palm_reward` and `grasp_goal_hand_config_reward`
   should climb faster with the FSWO-optimal target than with grasp 15 (if not, the
   force-closure quality did not translate to a more learnable reward signal — revisit the
   clutter corridor term).
