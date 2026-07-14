# Quick Reference: UltraDex + FSWO Optimizer

Read time: ~10 min. Covers everything needed to start implementing immediately.

---

## Pipeline at a Glance

```
OFFLINE (done, friend's code)
  cube mesh → BODex → 100 raw grasps → calibrate + filter → 32-grasp library (.npz)

OFFLINE (your task, ~1 hr)
  load .npz → FK each grasp → project fingertips to cube surface → FSWO score → pick best index

ONLINE (RL training, running now)
  each episode reset → load library → select grasp[fixed_grasp_idx] → transform to world → reward policy
```

---

## The Data: `grasp_sampler/grasp_dataset/cube_5cm_grasps_valid.npz`

```python
data = np.load("cube_5cm_grasps_valid.npz", allow_pickle=True)
grasps = data["grasp_pose"]         # float32 (32, 1, 3, 13)
T_cal  = data["T_usdbase_urdfbase"] # float64 (4, 4)  — frame correction
```

Shape meaning: `(32 grasps, 1 hand, 3 stages, 13 values)`

| Stage | Index | What it is |
|---|---|---|
| pregrasp | 0 | palm ~5 cm above surface, fingers open |
| **grasp** | **1** | fingertips on cube surface ← **use this for FSWO** |
| squeeze | 2 | fingers pressed in |

13 values per row: `[x, y, z, qw, qx, qy, qz, thumb_yaw, thumb_pitch, index, middle, ring, pinky]`

All poses in **cube frame** (cube centered at origin, axis-aligned). The calibration matrix
`T_cal` corrects a 90° axis permutation between the synthesis URDF base and the USD
`R_hand_base_link` (Kabsch fit over 10 link pairs, 0.3 mm residual).

---

## What Already Exists (Friend's Code)

| File | What it does |
|---|---|
| `grasp_sampler/grasp_dataset/cube_5cm_grasps_valid.npz` | 32-grasp goal library |
| `grasp_sampler/check_grasps_offline.py` | has `fk_tips(palm_pos, palm_quat, joint_q, T_cal) → (5,3)` — fingertip positions in cube frame |
| `mdp/grasp_goal.py` | `sample_grasp_goal` (reset event), `update_live_goals` (per-step), `grasp_goal_palm_reward`, `grasp_goal_hand_config_reward` |
| `g1_pick_env_cfg.py` | `fixed_grasp_idx: 15` in the event params — **this is what you change** |

**`fk_tips` uses the synthesis URDF** (not the USD), with slave joints frozen at measured
postures because the USD slave joints have no drives (Discovery 1):
- `thumb_inter = −0.16 rad`, `thumb_distal = −0.24 rad`
- `index/middle/ring/pinky_inter = 1.15 rad`

Slave joints in the USD are stochastic — the `InspireMimicAction` software ratios in
`g1_pick_env_cfg.py` write nothing to them. The synthesis URDF freezes them to match.

---

## Reward Structure (Already in Policy, Unchanged)

$$r_{\text{palm}} = 1 - \tanh\!\left(\frac{\|p_{\text{palm}} - p^*\|}{0.15}\right) \qquad w = 1.0$$

$$r_{\text{hand}} = \underbrace{\left(1 - \tanh\!\left(\frac{\|p_{\text{palm}} - p^*\|}{0.20}\right)\right)}_{\text{gate: activates when palm within 20 cm}} \cdot \left(1 - \tanh\!\left(\frac{\|q - q^*\|}{0.5}\right)\right) \qquad w = 0.3$$

$(p^*, q^*)$ = palm position + 6 proximal joint angles from the chosen library entry,
transformed to world frame and updated every step. You only change **which entry** is chosen.

---

## Why Fixed Index (Not Per-Episode Selection)

With per-episode selection the goal varies but is not in the observation → policy can only
optimize the average → hovers at the centroid of candidate poses, never commits. This is
Problem 17 in UltraDex.md and killed two training runs. `fixed_grasp_idx: 15` makes the
goal deterministic from the observed cube position, which is learnable. FSWO replaces the
heuristic choice of 15 with the mechanically best grasp.

---

## FSWO Math

Given 5 contact points `{(pᵢ, nᵢ)}` where `pᵢ` = contact position on cube surface,
`nᵢ` = inward normal (pointing into cube):

**Build Q matrix** (k=5, so 5×5):
$$Q_{ij} = n_i \cdot n_j + \lambda\,(p_i \times n_i)\cdot(p_j \times n_j), \qquad \lambda = 1.0$$

**Solve for each candidate fixed-maximum finger** j ∈ {0..4}, using L (a PSD square
root of Q, Q = LᵀL, via eigendecomposition) as the NNLS design matrix — **not Q
itself** (using Q directly gives a different, wrong answer whenever the true
optimum has an active α=0 constraint; verified 2026-07-13 against brute-force
SLSQP on random contact sets — see `Misc./optimizer.md` §2.2 for the derivation):
$$\min_{\alpha_{-j} \geq 0} \|L_{:,-j}\,\alpha_{-j} + L_{:,j}\|^2 \qquad \text{(NNLS)}$$

**Score:**
$$S = -\alpha^{*T} Q\,\alpha^* \in (-\infty, 0]; \quad S = 0 \Rightarrow \text{perfect force closure}$$

Best grasp: `argmax S` over all 32 candidates. FSWO scores depend only on cube-frame
contact geometry → compute once offline, never per-episode.

**Contact projection** (cube half-edge r = 0.025 m, centered at origin, axis-aligned):
```python
k = argmax(|p| / r)               # dominant axis → nearest face
contact = p; contact[k] = sign(p[k]) * r
normal = zeros(3); normal[k] = -sign(p[k])   # inward
```

---

## What You Need to Implement

Two new files inside `grasp_sampler/grasp_selection/` (create this folder):

### File 1: `fswo.py` (see the real file for the current, authoritative version)

```python
import numpy as np
from scipy.optimize import nnls

def _psd_sqrt(Q, eps=1e-12):
    w, V = np.linalg.eigh(Q)
    w = np.clip(w, eps, None) - eps
    return V @ np.diag(np.sqrt(w)) @ V.T

def fswo_score(contacts: np.ndarray, lam: float = 1.0) -> float:
    """contacts: (k, 6) = [px py pz nx ny nz] per contact. Returns score in (-inf, 0]."""
    p, n = contacts[:, :3], contacts[:, 3:]
    tau = np.cross(p, n)
    Q = n @ n.T + lam * (tau @ tau.T)   # (k, k), PSD
    L = _psd_sqrt(Q)                    # Q = L @ L -- NNLS design matrix must be L, not Q
    k = len(contacts)
    best = np.inf
    for j in range(k):
        mask = np.arange(k) != j
        alpha_free, _ = nnls(L[:, mask], -L[:, j])
        alpha = np.zeros(k); alpha[mask] = alpha_free; alpha[j] = 1.0
        val = float(alpha @ Q @ alpha)
        if val < best:
            best = val
    return -best
```

### File 2: `select_optimal_grasp.py`

```python
import sys, json
import numpy as np
sys.path.insert(0, "..")          # so it can import check_grasps_offline
from check_grasps_offline import fk_tips
from grasp_selection.fswo import fswo_score

CUBE_HALF = 0.025

def cube_contact(p):
    k = int(np.argmax(np.abs(p) / CUBE_HALF))
    c = p.copy(); c[k] = np.sign(p[k]) * CUBE_HALF
    n = np.zeros(3); n[k] = -np.sign(p[k])
    return np.concatenate([c, n])   # (6,)

data   = np.load("grasp_dataset/cube_5cm_grasps_valid.npz", allow_pickle=True)
grasps = data["grasp_pose"]           # (32, 1, 3, 13)
T_cal  = data["T_usdbase_urdfbase"]   # (4, 4)

scores = []
for g in range(32):
    s1 = grasps[g, 0, 1, :]                # stage 1 (grasp)
    tips = fk_tips(s1[:3], s1[3:7], s1[7:], T_cal)   # (5, 3) in cube frame
    contacts = np.array([cube_contact(t) for t in tips])  # (5, 6)
    scores.append(fswo_score(contacts))

scores = np.array(scores)
best   = int(np.argmax(scores))
print(f"Best index: {best}  score: {scores[best]:.5f}")
print(f"Ranking: {np.argsort(scores)[::-1].tolist()}")
print(f"Current index 15 score: {scores[15]:.5f}")

with open("grasp_selection/scores.json", "w") as f:
    json.dump({"best_idx": best, "scores": scores.tolist(),
               "ranking": np.argsort(scores)[::-1].tolist()}, f, indent=2)
```

Run from `grasp_sampler/`:
```bash
conda activate env_isaaclab
cd grasp_sampler/
mkdir -p grasp_selection
touch grasp_selection/__init__.py
python grasp_selection/select_optimal_grasp.py
```

---

## Wiring the Result

Open `g1_pick_env_cfg.py`, find the `sample_grasp_goal` EventTerm params, change:
```python
"fixed_grasp_idx": 15,   # old heuristic
```
to whatever index the script printed as best. That's the only change needed to start training.

---

## Check Before Running

1. `fk_tips` signature in `check_grasps_offline.py` — confirm it takes `(palm_pos, palm_quat, joint_q, T_cal)` and returns `(5, 3)`. Adjust the call in `select_optimal_grasp.py` if the signature differs.
2. All 32 scores should be finite and `≤ 0`. A `NaN` means FK placed a fingertip far from the cube (wrong URDF or wrong T_cal loading).
3. Visually confirm the top-ranked grasp looks balanced using the existing visualization in `check_grasps_offline.py`.
