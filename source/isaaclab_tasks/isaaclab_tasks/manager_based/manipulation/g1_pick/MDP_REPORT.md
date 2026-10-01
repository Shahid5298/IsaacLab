# G1 Pick — MDP Technical Report

A full specification of the Markov Decision Process defined in this environment, derived from the source code.

---

## 1. Problem Statement

Train a Unitree G1 humanoid robot (with Inspire dexterous right hand) to reach into a cluttered tray and pick a specific red target cube, lifting it ~29 cm clear of the tray surface, while leaving all distractor cubes on the tray.

The robot's lower body (legs, waist) and left arm are frozen. Only the **right arm** (7 DOF) and **right Inspire hand** (6 controllable proximal joints) are active.

---

## 2. Scene / Physical Setup

| Entity | Description | Position (x, y, z) |
|--------|-------------|---------------------|
| Robot (G1 + Inspire) | Fixed standing pose, root at (−0.1, 0, 0.74) | frozen in place |
| Table | Static cuboid 0.6 × 1.2 × 0.80 m | (0.4, 0, 0.40) |
| Tray | Kinematic slab 0.4 × 0.6 × 0.02 m | (0.4, 0, 0.81) |
| **Target cube** (red) | 5 cm × 5 cm × 5 cm, mass 0.2 kg, friction 1.0 | (0.35, 0, **0.845**) ± random jitter |
| Distractors 1–10 (colored) | Identical 5 cm cubes, mass 0.2 kg | Ring layout around target |

**Tray surface height**: 0.820 m. **Target cube center** init at 0.845 m.

**Distractor anchor positions** — this is the fixed layout `Isaac-G1-Pick-v0` (the default/"stage 3" task) actually spawns; distances are to the target's nominal center [0.35, 0.0], recomputed and verified directly from `SceneCfg`'s per-distractor `init_state.pos` (2026-08-13; the previous version of this table had 6, 7, and 8 in the wrong ring):

| Ring | Distractors | Position (x, y) | Distance from cube |
|------|-------------|------------------|---------------------|
| Inner (~7-8.5 cm) | 3, 8 | (0.42, 0.0), (0.28, 0.0) | 7.00 cm |
| Inner (~7-8.5 cm) | 1, 2 | (0.38, ±0.08) | 8.54 cm |
| Mid (~12 cm) | 4, 5 | (0.32, ±0.12) | 12.37 cm |
| Outer (~17-18 cm) | 9 | (0.52, 0.0) | 17.00 cm |
| Outer (~17-18 cm) | 6, 7 | (0.48, ±0.12) | 17.69 cm |
| Outer (~17-18 cm) | 10 | (0.35, 0.18) | 18.00 cm |

So the true split is 4 inner / 2 mid / 4 outer, not the 3/4/3 the ring boundaries previously implied.

Each distractor resets with ±3 cm uniform jitter per episode (`reset_distractor_1`…`reset_distractor_10`, §8).

**This is one of three registered distractor layouts** — see §9 for the other two (`Isaac-G1-Pick-Stage1-v0`, `Isaac-G1-Pick-Stage2-v0`), which use different, less adversarial anchor positions for curriculum training.

---

## 3. Action Space

**Dimension: 13** (continuous, joint position targets)

Actions are relative offsets from the default pose (`use_default_offset=True`).

### 3a. Right Arm — 7 DOF (scale 0.3)

| # | Joint |
|---|-------|
| 0 | `right_shoulder_pitch_joint` |
| 1 | `right_shoulder_roll_joint` |
| 2 | `right_shoulder_yaw_joint` |
| 3 | `right_elbow_joint` |
| 4 | `right_wrist_roll_joint` |
| 5 | `right_wrist_pitch_joint` |
| 6 | `right_wrist_yaw_joint` |

### 3b. Right Hand — 6 DOF proximal joints (scale 0.5)

| # | Joint | Controls |
|---|-------|----------|
| 7 | `R_thumb_proximal_yaw_joint` | Thumb abduction |
| 8 | `R_thumb_proximal_pitch_joint` | Thumb flex (drives intermediate × 0.8024, distal × 0.7622) |
| 9 | `R_index_proximal_joint` | Index flex (drives intermediate × 1.0843) |
| 10 | `R_middle_proximal_joint` | Middle flex (drives intermediate × 1.0843) |
| 11 | `R_ring_proximal_joint` | Ring flex (drives intermediate × 1.0843) |
| 12 | `R_pinky_proximal_joint` | Pinky flex (drives intermediate × 1.0843) |

**Mimic coupling** (`InspireMimicAction`): The policy only controls the 6 proximal joints. The 6 downstream intermediate/distal joints are driven automatically using hardware transmission ratios from the Inspire URDF, reducing the effective action space while preserving physical realism.

**Arm actuator**: stiffness 300, damping 30 — stiff position control.  
**Hand actuator**: stiffness 100, damping 0.5 — compliant for grasping.

---

## 4. Observation Space

**Total dimension: ~96** (all concatenated into a flat 1D tensor for an MLP policy)

| # | Observation Term | Dim | Description |
|---|-----------------|-----|-------------|
| 1 | `joint_pos` | 13 | Relative joint positions (arm 7 + hand 6) |
| 2 | `joint_vel` | 13 | Relative joint velocities, clipped ±50 |
| 3 | `object_pos_b` | 3 | Target cube XYZ in robot root frame |
| 4 | `object_vel` | 6 | Target cube linear + angular velocity (world frame) |
| 5–14 | `distractor_{1–10}_pos_b` | 30 | All 10 distractor positions in robot root frame |
| 15 | `right_fingertip_pos` | 18 | 6 tracked hand bodies (palm + 5 tips) × 3, in robot root frame |
| 16 | `actions` | 13 | Previous action (action history, 1 step) |

**Total: 13 + 13 + 3 + 6 + 30 + 18 + 13 = 96 dims**

**Tracked hand bodies** (for rewards and observations):

| Index | Body | Role |
|-------|------|------|
| 0 | `right_wrist_yaw_link` | Palm anchor |
| 1 | `R_thumb_distal` | Thumb tip |
| 2 | `R_index_intermediate` | Index tip |
| 3 | `R_middle_intermediate` | Middle tip |
| 4 | `R_ring_intermediate` | Ring tip |
| 5 | `R_pinky_intermediate` | Pinky tip |

No observation noise is added (`enable_corruption = False`). All positions are expressed in the robot's local frame via `quat_apply_inverse` to be pose-invariant.

---

## 5. Reward Function

> **Rewritten from source on 2026-07-30, updated 2026-08-13, updated 2026-08-19.** The
> 2026-07-30 pass added `grasp_goal_palm`/`grasp_goal_hand`/`grasp_reach` (previously
> undocumented) and the `use_posture=False, pose_gated_success=True` parameterization.
> The 2026-08-13 pass added `target_object_accel` (weight $-5.0$), a direct penalty on
> the cube's own sudden velocity change, added as a checkpoint-5500+ regression fix
> (§5.3). The 2026-08-19 pass adds two more terms — **`joint_speed`** (§5.3) and
> **`sustained_reach_bonus`** (§5.5, new) — for **12 reward terms total** as of this
> update. Everything below is verified against the current `g1_pick_env_cfg.py`.
>
> **This section documents the full/destination weights** — i.e. what each term's
> mechanism is and what it's weighted once fully phased in. As of 2026-08-19 this
> environment is mid-way through a **from-scratch 5-phase curriculum** (see §9) where
> most terms are deliberately weight-zeroed until their phase arrives — the table below
> shows destination weights, not necessarily what's active in any specific run right now.
> `RewardsCfg`'s own inline comments in `g1_pick_env_cfg.py` are the source of truth for
> current-phase weights; each zeroed param carries a `# target: X -- Phase N` comment.

All reward terms are computed every control step (30 Hz) and summed with their configured
weight (when active for the current curriculum phase — see the note above).

### 5.0 Term overview

| Term | Destination weight | Role |
|---|---:|---|
| `task_reward` | $1.0$ | reach + grasp + lift + pose-gated success — §5.1 |
| `grasp_goal_palm` | $2.0$ | pulls the palm toward the UltraDexGrasp goal pose — §5.2.1 |
| `grasp_goal_hand` | $1.0$ | pulls the 6 finger joints toward the goal hand shape, gated on palm proximity — §5.2.2 |
| `grasp_reach` | $1.0$ | pulls each fingertip toward its own contact point on the cube — §5.2.4 |
| `action_smoothness` | $-3.0$ | penalizes jerky actions / high joint velocity (rate + L2, not speed — see `joint_speed`) |
| `joint_speed` | $-1.0$ | tanh-bounded penalty on raw joint speed — §5.3 |
| `fingertip_impact` | $-2.0$ | penalizes sudden hand-body acceleration (slams) |
| `distractor_accel` | $-3.0$ | penalizes sudden distractor acceleration (bumps) |
| `target_object_accel` | $-5.0$ | penalizes sudden TARGET CUBE velocity change — regression fix, §5.3 |
| `sustained_reach_bonus` | $10.0$ | sparse bonus for holding a close reach — §5.5 |
| `distractor_off_tray` | $-10.0$ | per-step penalty while any distractor sits below tray height |
| `distractor_drop` | $-100.0$ | one-time penalty (+ termination) if any distractor falls off the table |

**Two different bodies are both informally "the palm" in this codebase — keep them
distinct:**

- `right_wrist_yaw_link` — read by `compute_task_reward`'s (currently disabled) posture
  term.
- `R_hand_base_link` — read by both grasp-goal terms (§5.2) and by the pose-gated
  success bonus (§5.1.6). The two links sit a few centimeters apart.

### 5.1 Task reward — `compute_task_reward` (weight $1.0$)

Let $\mathbf{p}_c \in \mathbb{R}^3$ be the cube position, $\mathbf{p}_{\text{palm}}$ the
`right_wrist_yaw_link` position, and $\{\mathbf{p}_{\text{tip},i}\}_{i=0}^{4}$ the 5
tracked fingertip bodies ($i=0$: thumb, $i=1\ldots4$: index/middle/ring/pinky).

**Deadband distances** (subtract the cube's 2.5 cm half-edge, so touching the surface
reads as zero):

$$
d_{\text{thumb}} = \max\!\Big(\lVert \mathbf{p}_{\text{tip},0} - \mathbf{p}_c \rVert - 0.025,\ 0\Big)
\qquad
d_{\text{finger}} = \max\!\Big(\tfrac{1}{4}\textstyle\sum_{i=1}^{4}\lVert \mathbf{p}_{\text{tip},i} - \mathbf{p}_c \rVert - 0.025,\ 0\Big)
$$

#### 5.1.1 Posture reward — **disabled** on this branch (`use_posture=False`)

$$
r_{\text{posture}} =
\begin{cases}
\displaystyle 1 - \tanh\!\frac{\max\big(\lVert \mathbf{p}_{\text{palm}} - (\mathbf{p}_c + [0,0,0.08])\rVert - 0.03,\ 0\big)}{0.3} & \texttt{use\_posture=True} \\[6pt]
0 & \texttt{use\_posture=False\ (current)}
\end{cases}
$$

Why it's off: this term hard-codes the palm target 8 cm **straight above** the cube — a
top-down assumption. Grasp #12 is a ~65° side approach, so this term would fight the
grasp-goal palm reward (§5.2.1) instead of agreeing with it. Palm placement is now owned
entirely by `grasp_goal_palm`.

#### 5.1.2 Reach reward

Centroid of all 5 fingertips toward the cube:

$$
\bar{\mathbf{p}}_{\text{tip}} = \frac{1}{5}\sum_{i=0}^{4} \mathbf{p}_{\text{tip},i}
\qquad
r_{\text{reach}} = 1 - \tanh\!\left(\frac{\lVert \bar{\mathbf{p}}_{\text{tip}} - \mathbf{p}_c \rVert}{0.25}\right)
$$

#### 5.1.3 Grasp reward

$$
r_{\text{thumb}} = 1 - \tanh\!\left(\frac{d_{\text{thumb}}}{0.055}\right)
\qquad
r_{\text{finger}} = 1 - \tanh\!\left(\frac{d_{\text{finger}}}{0.055}\right)
\qquad
r_{\text{grasp}} = \frac{r_{\text{thumb}} + r_{\text{finger}}}{2}
$$

Counted with weight $2$ inside the task-reward sum (§5.1.7).

#### 5.1.4 Grasp gate

Soft AND of "thumb close" and "fingers close", used to gate lift + success so the policy
can't earn them by shoving the cube with an open palm:

$$
g = \Big(1 - \tanh\tfrac{d_{\text{thumb}}}{0.06}\Big)\Big(1 - \tanh\tfrac{d_{\text{finger}}}{0.06}\Big) \ \in [0,1]
$$

#### 5.1.5 Continuous lift reward

$$
\Delta h = \operatorname{clamp}(p_{c,z} - 0.845,\ 0,\ 0.30)
\qquad
r_{\text{lift}} = 2.0\,\Delta h \cdot g \ \in [0,\ 0.6]
$$

#### 5.1.6 Success bonus — now **pose-gated** (`pose_gated_success=True`)

This is where the grasp goal reaches into the terminal bonus. Every step the live goal
is re-attached to the cube's current pose (mechanism shared with §5.2), then compared
against the current `R_hand_base_link` pose $(\mathbf{p}_{\text{hand}}, \mathbf{q}_{\text{hand}})$
against the goal $(\mathbf{p}^\star, \mathbf{q}^\star)$:

$$
e_{\text{pos}} = \lVert \mathbf{p}_{\text{hand}} - \mathbf{p}^\star \rVert
\qquad
e_{\text{ang}} = \operatorname{quat\_error\_magnitude}(\mathbf{q}_{\text{hand}}, \mathbf{q}^\star) \ \in [0,\pi]
$$

$$
\text{pose\_match} = \Big(1 - \tanh\tfrac{e_{\text{pos}}}{0.10}\Big)\Big(1 - \tanh\tfrac{e_{\text{ang}}}{0.8}\Big) \ \in [0,1]
$$

$$
\text{success\_scale} = \underbrace{0.2}_{\text{success\_floor}} + (1-0.2)\cdot\text{pose\_match} \ \in [0.2,\ 1.0]
$$

$$
r_{\text{success}} = \mathbb{1}[p_{c,z} > 1.134]\cdot g \cdot \text{success\_scale}\cdot 1000
$$

**Interpretation**: lifting the cube always pays **at least 200** (the `success_floor`
$=0.2$), so the pick signal never fully vanishes even if the exact UltraDex pose proves
unreachable — but lifting it **in grasp #12's pose** pays the full **1000**. This is a
structural fix, not just reward shaping: instead of *adding* a small pose term next to a
1000-point bonus (which the policy learns to ignore entirely — see `CONTEXT.md` §7–8),
the size of the bonus itself now *depends on* the pose match.

#### 5.1.7 Assembly + drop override

$$
r_{\text{task}} =
\begin{cases}
-0.5 & p_{c,z} < 0.600 \quad \text{(cube fell off the table)} \\[4pt]
r_{\text{posture}} + r_{\text{reach}} + 2\,r_{\text{grasp}} + r_{\text{lift}} + r_{\text{success}} & \text{otherwise}
\end{cases}
$$

---

### 5.2 Grasp-goal pose-mimicking rewards — `grasp_goal_palm` (weight $2.0$) + `grasp_goal_hand` (weight $1.0$)

These are the two terms this doc previously omitted, and the ones most worth
understanding in detail: they pull the robot toward a **specific grasp pulled from the
UltraDexGrasp/BODex library** (`grasp_sampler/grasp_dataset/cube_5cm_grasps_valid.npz`),
rather than just rewarding "get close to the cube center" the way §5.1 does.

**The goal itself.** At every episode reset, the `sample_grasp_goal` event term
(`mdp/grasp_goal.py`) assigns the **same fixed library entry to every environment** —
currently **grasp #12**, the sphere-contact-gated FSWO optimizer's pick (a 5-finger
envelope wrap; confirmed reachable off-center by the arm, unlike the far-tray positions
where no grasp in the library works because the arm physically can't reach). This
produces three per-env goal buffers, all in **world frame**:

- $\mathbf{p}^\star \in \mathbb{R}^3$ — target `R_hand_base_link` position (`goal_pos_w`)
- $\mathbf{q}^\star \in \mathbb{R}^4$ — target `R_hand_base_link` orientation (`goal_quat_w`)
- $\mathbf{q}^\star_{\text{hand}} \in \mathbb{R}^6$ — target 6 proximal joint angles, in
  policy joint order (`goal_hand_q`)

Because grasp #12 is stored **relative to the cube**, $\mathbf{p}^\star$ and
$\mathbf{q}^\star$ are **re-attached to the cube's live pose every step**
(`update_live_goals`, cached per `env.common_step_counter` so it only recomputes once
even though both reward terms call it):

$$
\mathbf{p}^\star_t = R(\mathbf{q}_{\text{cube},t})\,\mathbf{p}^\star_{\text{obj}} + \mathbf{p}_{\text{cube},t}
\qquad
\mathbf{q}^\star_t = \mathbf{q}_{\text{cube},t} \otimes \mathbf{q}^\star_{\text{obj}}
$$

where $\mathbf{p}^\star_{\text{obj}}, \mathbf{q}^\star_{\text{obj}}$ are grasp #12's pose
in the cube's own frame (constant, loaded once from the `.npz` at env construction).
Without this live re-attachment, the goal would freeze at the cube's spawn pose — the
moment anything nudges the cube, the reward would point at empty air (this was an
actual bug early in the project; see `grasp_sampler/README.md` problem #14).

#### 5.2.1 Palm-pose reward — `grasp_goal_palm_reward` (weight $2.0$)

Pulls `R_hand_base_link` toward $(\mathbf{p}^\star, \mathbf{q}^\star)$ — position **and**
orientation, in the current config (`pos_mode="full"`):

$$
d_{\text{pos}} = \lVert \mathbf{p}_{\text{hand}} - \mathbf{p}^\star \rVert
\qquad
d_{\text{ang}} = \operatorname{quat\_error\_magnitude}(\mathbf{q}_{\text{hand}}, \mathbf{q}^\star)
$$

$$
r_{\text{pos}} = 1 - \tanh\!\left(\frac{d_{\text{pos}}}{0.15}\right)
\qquad
r_{\text{orient}} = 1 - \tanh\!\left(\frac{d_{\text{ang}}}{0.6}\right)
$$

$$
r_{\text{palm}} = (1-w_o)\, r_{\text{pos}} + w_o\, r_{\text{orient}}, \qquad w_o = 0.5
$$

i.e. $r_{\text{palm}} = 0.5\, r_{\text{pos}} + 0.5\, r_{\text{orient}} \in [0,1]$. This is
a **dense, ungated** reward — it pays from anywhere in the workspace, purely as a
function of how close the hand's pose is to the goal, every single step.

(`pos_mode` also supports `"height"`, which would reward only the vertical offset
$|\,p_{\text{hand},z} - p^\star_z\,|$ — meant for when a separate posture term already
centers the palm horizontally. Not used here since `use_posture=False`.)

#### 5.2.2 Hand-configuration reward — `grasp_goal_hand_config_reward` (weight $1.0$)

Pulls the 6 controllable proximal joints $\mathbf{q} \in \mathbb{R}^6$ toward the goal's
joint angles $\mathbf{q}^\star_{\text{hand}}$ — but **gated** on palm proximity, so the
policy can't get finger-shape credit while the hand is still across the room:

$$
\text{gate} = \operatorname{clamp}\!\left(1 - \tanh\frac{d_{\text{pos}}}{0.20},\ 0,\ \infty\right)
$$

$$
r_{\text{hand}} = \text{gate}\cdot\left(1 - \tanh\frac{\lVert \mathbf{q} - \mathbf{q}^\star_{\text{hand}}\rVert}{0.5}\right)
$$

The gate reuses the **same** $d_{\text{pos}}$ from §5.2.1, but with its own width ($0.20$
m vs. $0.15$ m) — wide enough that the finger-shaping signal starts to switch on well
before the palm has fully arrived, giving a smooth handoff instead of a cliff.

#### 5.2.3 Four independent pose-matching signals — don't conflate them

It's tempting to think of this as one "match the grasp" reward. There are actually
**four**, all reading the same live goal but with different tolerances and different
jobs:

| # | Where | Std devs | When it pays | Job |
|---|---|---|---|---|
| 1 | `grasp_goal_palm` (§5.2.1) | $0.15$ m pos / $0.6$ rad ang | dense, every step | pulls the palm toward the goal pose continuously |
| 2 | `grasp_goal_hand` (§5.2.2) | gate $0.20$ m; joint std $0.5$ rad | dense, once palm-gated | pulls fingers toward the goal *joint angles* once the palm is close |
| 3 | pose-gated success (§5.1.6) | $0.10$ m pos / $0.8$ rad ang | **only at the instant of a successful lift** | scales the terminal 1000-point bonus by pose quality |
| 4 | `grasp_reach` (§5.2.4) | $\sigma_{\text{reach}}=0.05$ m per finger | dense, every step | pulls each fingertip toward its own *Cartesian contact point* on the cube |

Per `CONTEXT.md` §7–8, signal (1) has sat essentially flat at $\approx 0.028$–$0.030$
across 6000+ training iterations despite being dense and weighted $2.0$ — the policy
isn't moving toward grasp #12's pose at all, even though it picks the cube reliably
(94–96%) using whatever grasp it discovered on its own. That's the open problem this
whole reward structure exists to diagnose. Now that reachability of #12 is confirmed
(your IK check), signal (4) is a complementary shaping term that attacks the same
problem from task space instead of configuration space — see the rationale in §5.2.4.

#### 5.2.4 Fingertip contact-point reward — `grasp_reach` (weight $1.0$, implemented 2026-07-30)

**Motivation.** Signal (2), `grasp_goal_hand`, matches the policy's 6 proximal **joint
angles** to grasp #12's joint angles. But joint-space matching is once removed from what
actually matters: whether each fingertip lands on the *specific patch of cube surface*
the optimizer identified as a good contact (per `grasp_selection`'s sphere-contact
scoring — §"Grasp gate" in `OPTIMIZER_IMPLEMENTATION_SPEC.md`). Two hands with slightly
different joint angles can produce nearly the same fingertip placement, and — per
`grasp_sampler/README.md` Discovery 2 — the synthesis URDF and the simulated USD hand are
known to disagree geometrically by 1–3 cm at the fingertips, so joint-angle matching
doesn't guarantee contact-point matching anyway. A reward defined directly in **task
space** (Cartesian distance from real fingertip to intended contact point) is more
directly tied to what a good grasp physically requires, and is robust to that model
mismatch in a way joint matching isn't.

**Fingertips vs. collision spheres.** The offline optimizer scores wrap quality using
~41 collision spheres across the whole hand (§"Version 2" of the optimizer spec), not
just the 5 fingertips. In principle the online reward could do the same. My
recommendation, matching your instinct: **use the 5 fingertips**, not the spheres, for
this online term:

- The 5 fingertip bodies are *already* tracked (`_RIGHT_HAND_BODIES`) and already read by
  `compute_task_reward`'s reach/grasp terms — no new body tracking, no new per-step FK.
- Tracking ~41 spheres online would mean live-FK'ing ~13 hand links every step for every
  parallel env — real engineering cost for what's likely marginal benefit here: the
  spheres already did their job *offline* (they're why grasp #12 was selected as the
  best wrap in the first place). The online reward doesn't need to re-derive wrap
  quality; it just needs to nudge the policy toward the finger placement that a
  pre-vetted grasp already specifies.
- If fingertip-only guidance turns out to be too coarse (e.g. the policy matches the 5
  tip points but still doesn't wrap correctly), sphere-based shaping is the natural
  escalation — but it's not the right place to start.

**Target contact points (precomputed offline, cached, loaded lazily at runtime).**
`ultradex_repo/` (the URDF + collision-sphere YAML `grasp_selection` needs for FK) is
gitignored and not present in every checkout — the same reason `get_optimal_grasp_idx()`
falls back to the cached `scores.json` instead of recomputing from the optimizer every
time. This term follows the identical idiom: `grasp_selection/cache_fingertip_contacts.py`
(run once against grasp #12's stored root pose $(\mathbf p_g, \mathbf q_g)$ and 6 joint
angles $\boldsymbol\theta_g$) forward-kinematics the 5 fingertip links
(`thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip`) in the cube frame, and caches
the result to `grasp_selection/fingertip_contacts.json` (committed to the repo). The
reward function (`mdp.grasp_reach_reward`) just loads that cache on first call — no
URDF/FK dependency at training time. If the optimizer's pick ever changes from #12, rerun
the caching script.

$$
\mathbf t_i = \mathrm{FK}_i(\mathbf p_g, \mathbf q_g, \boldsymbol\theta_g), \qquad i \in \{\text{thumb, index, middle, ring, pinky}\}
$$

then project each onto the nearest cube face — exactly the `cube_contact()` step
`grasp_selection/hand_model.py` already implements and the optimizer already uses (here
reused purely as a geometry helper, not for re-selecting a grasp):

$$
\mathbf c^\star_i = \Pi_{\text{cube}}(\mathbf t_i), \qquad
\Pi_{\text{cube}}(\mathbf p) = \mathbf p + \big(h\,\mathrm{sign}(p_k) - p_k\big)\hat{\mathbf e}_k,
\quad k = \arg\max_{a\in\{x,y,z\}} \frac{|p_a|}{h},\ \ h=0.025\text{ m}
$$

This yields 5 fixed points $\mathbf c^\star_i$ in the **cube's own frame** — "where this
fingertip should touch the cube surface for grasp #12."

**Live tracking (goal follows the cube, identical mechanism to §5.2's palm/hand goals):**

$$
\mathbf c^\star_{i,t} = R(\mathbf q_{\text{cube},t})\,\mathbf c^\star_i + \mathbf p_{\text{cube},t}
$$

**Reward** — mean, over the 5 tracked fingertips, of a bounded per-finger term (the same
mean-of-tanh style already used by `fingertip_impact`/`distractor_accel`, chosen over
tanh-of-mean so one badly-placed finger can't be washed out by four good ones):

$$
d_{\text{reach},i} = \big\lVert \mathbf p_{\text{tip},i} - \mathbf c^\star_{i,t} \big\rVert
$$

$$
r_{\text{grasp\_reach}} = \frac{1}{5}\sum_{i=0}^{4}\left(1 - \tanh\frac{d_{\text{reach},i}}{\sigma_{\text{reach}}}\right), \qquad \sigma_{\text{reach}} = 0.05\text{ m}
$$

Dense and **ungated** — like `grasp_goal_palm`, it should pay from anywhere in the
workspace, since "move each fingertip toward its own target point" is meaningful at any
distance (it's a per-finger refinement of the existing §5.1.2 reach reward, which only
uses one shared centroid target for all 5 tips).

**Plain uniform mean over the 5 fingertips — no thumb/finger split.** `compute_task_reward`'s
`grasp_rew` (§5.1.3) splits 0.5-thumb/0.5-fingers because it compares distance to *one
shared target* (the cube center), where a badly-placed thumb could get diluted by four
decent fingers averaged together before the `tanh`. That risk doesn't apply here: each
finger already has its **own unique target point**, and each is passed through `tanh`
*individually* before averaging (mean-of-tanh, not tanh-of-mean) — so a badly-placed
thumb still shows up as its own near-zero term regardless of the other four. The thumb's
special importance is already encoded in *where* its target point sits (the optimizer
chose it as the opposing contact for force closure); weighting it again here would
double-count that.

**Weight**: $w_{\text{grasp\_reach}} = 1.0$ — same order as `grasp_goal_hand`, since it
plays a similar complementary role.

$$
r_{\text{grasp\_reach}} \cdot w_{\text{grasp\_reach}} \ \text{ (proposed addition to §5.4's total)}
$$

---

### 5.3 Penalty terms

| Term | Weight | Formula | Purpose |
|---|---:|---|---|
| `action_smoothness` | $-3.0$ | $0.005\sum_i(a_i-a_i^{\text{prev}})^2$ | penalizes jerky actions (rate + L2 only — joint velocity moved to `joint_speed` below, 2026-08-19) |
| `joint_speed` | $-1.0$ | $\tanh\!\big(\textstyle\sum_j \dot q_j^2 \,/\, 3000\big)$ | tanh-bounded "never move fast" prior — see rationale below |
| `fingertip_impact` | $-2.0$ | $\dfrac{1}{6}\sum_{k=1}^{6}\tanh\!\big(\lVert\Delta \mathbf{v}_{\text{tip},k}\rVert/3.0\big)$ | penalizes sudden hand-body acceleration (slam/jab) |
| `distractor_accel` | $-3.0$ | $\dfrac{1}{10}\sum_{d=1}^{10}\tanh\!\big(\lVert\Delta \mathbf{v}_d\rVert/2.0\big)$ | penalizes bumping/knocking distractors |
| `target_object_accel` | $-5.0$ | $\tanh\!\big(\lVert\Delta \mathbf{v}_{\text{cube}}\rVert/2.0\big)$ | penalizes sudden TARGET CUBE velocity change (a violent "throw" can otherwise satisfy the soft `is_grasped` gate for a few post-impact frames and pay out lift/success reward without a real grasp — this closes that loophole directly, independent of what `is_grasped` reads at that instant; see the docstring on `target_object_acceleration_penalty` in `g1_pick_env_cfg.py` for the full root-cause writeup of the checkpoint-5450→5500 collapse this was added to fix) |
| `distractor_off_tray` | $-10.0$ | $\sum_{d=1}^{10}\mathbb{1}[z_d < 0.835]$ | per-step, accumulates while any distractor sits off the tray |
| `distractor_drop` | $-100.0$ | $\mathbb{1}[\exists\, d: z_d < 0.600]$ | one-time; episode also terminates |

**Weight scale, for context**: `target_object_accel` ($-5.0$) is deliberately the largest of the impact-style penalties (vs. $-2.0$/$-3.0$ for `fingertip_impact`/`distractor_accel`) so that a "hit it hard" exploit strategy can't out-earn the penalty for triggering it.

**`joint_speed`'s tanh bound, and why it matters beyond "encourage/discourage" (2026-08-19).**
This term used to be a raw, unbounded quadratic (`sum(joint_vel²) * 0.001`, weight $-1.0$).
An inference-time reward-breakdown plot on checkpoint 150 of the first from-scratch
curriculum attempt showed it spiking to **$-35$ per step** during aggressive flailing —
$\sum \dot q_j^2 \approx 35{,}000$ — roughly 35-70x larger than every other active term
(`task_reward` capped near $0.6$). The fix isn't a bigger weight; an unbounded outlier this
large inflates return *variance* enough to corrupt the critic's value predictions (and
therefore GAE advantage estimates), and triggers PPO's adaptive-KL learning-rate schedule
to throttle down (since large noisy advantages risk destabilizing updates) — actively
slowing learning of *everything*, not just discouraging the fast motion it targets.
Bounding it with the same `tanh` pattern every other impact-style penalty here already
uses puts it on the same footing as `task_reward` instead of drowning it out.
`threshold=3000` is a starting value (saturates near raw$\approx 9000$, well below the
observed $35{,}000$ worst case) — tune after seeing how training responds, not a measured
optimum.

### 5.4 Full per-step reward

$$
R = 1.0\, r_{\text{task}} \;+\; 2.0\, r_{\text{palm}} \;+\; 1.0\, r_{\text{hand}} \;+\; 1.0\, r_{\text{grasp\_reach}}
\;+\; 10.0\, r_{\text{sustained\_reach}}
\;-\; 3.0\, p_{\text{smooth}} \;-\; 1.0\, p_{\text{joint\_speed}} \;-\; 2.0\, p_{\text{impact}} \;-\; 3.0\, p_{\text{accel}} \;-\; 5.0\, p_{\text{target\_accel}}
\;-\; 10.0\, p_{\text{off\_tray}} \;-\; 100.0\, p_{\text{drop}}
$$

### 5.5 Sustained-reach bonus — `sustained_reach_bonus` (weight $10.0$, added 2026-08-19)

Paired with the `sustained_reach` termination (§6): the fingertip centroid (same signal as
`reach_rew`, §5.1.2) must stay within $0.10$m of the cube for $30$ *consecutive* control
steps ($\approx 1$s at 30Hz) for either to fire. `sustained_reach_termination` tracks the
consecutive-steps counter (it always runs — `TerminationManager.compute()` isn't
weight-gated the way `RewardManager.compute()` is) and caches the resulting boolean on the
env; `sustained_reach_bonus_reward` reads that same-step cache
(`ManagerBasedRLEnv.step()` calls `termination_manager.compute()` **before**
`reward_manager.compute()`, so the cache is always current when the reward term reads it).

$$
r_{\text{sustained\_reach}} = \mathbb{1}\big[\text{counter} \ge 30\big], \qquad
\text{counter} \leftarrow \begin{cases} \text{counter} + 1 & \lVert \bar{\mathbf p}_{\text{tip}} - \mathbf p_c \rVert < 0.10 \\ 0 & \text{otherwise} \end{cases}
$$

**Motivation**: Phase 1 (reach-only) previously had no positive-outcome termination at
all — every episode ended via `time_out` or `target_dropped`, so a policy that reached
the cube and immediately drifted away again looked identical, reward-wise, to one that
never reached at all. This gives "reach and *hold*" its own genuine success signal and
episode boundary. `distance_threshold=0.10`m and `hold_steps=30` and `weight=10.0` are
starting defaults (the user's own suggested range was "10 or 20") — tune after seeing how
often it fires in practice.

---

## 6. Termination Conditions

Episodes end when any of the following occur:

| Condition | Type | Trigger |
|-----------|------|---------|
| `time_out` | Truncation | 8 seconds elapsed (960 sim steps at 120 Hz × 4 decimation = 240 control steps) |
| `target_lifted` | **Success** | Target cube `z > 1.134 m` (~29 cm above tray) |
| `target_dropped` | Failure | Target cube `z < 0.600 m` (fell off table) |
| `distractor_dropped` | Failure | ANY of the 10 distractors `z < 0.600 m` |
| `sustained_reach` | **Success** (added 2026-08-19) | Fingertip centroid within `0.10`m of the cube for `30` consecutive control steps (~1s) — see §5.5 |

**Episode length**: 8 seconds → 960 physics steps (120 Hz) → **240 control steps** (4× decimation).

---

## 7. Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Physics timestep | 1/120 s (120 Hz) |
| Control decimation | 4 (policy runs at 30 Hz) |
| Episode length | 8 s (240 control steps) |
| Parallel environments (train, class default) | 4096 — `SceneCfg(num_envs=4096, ...)` in `G1RightArmLiftEnvCfg_V2` |
| Parallel environments (eval, `*_PLAY` variants) | 64 |

**Note:** `train.py`'s `--num_envs` CLI flag overrides the class default above — every training run so far has been launched with `--num_envs 2048`, not the class's 4096. If you invoke `train.py` without `--num_envs`, you'll get 4096 instead, which changes throughput/VRAM footprint and effective batch size.
| Solver position iterations (robot) | 32 |
| Solver position iterations (objects) | 16 |
| Cube contact offset | 0.005 m |
| Physics: `bounce_threshold_velocity` | 0.2 m/s |

---

## 8. Domain Randomization (Reset Events)

Per episode reset, the following randomizations are applied:

| Event | Joints/Objects | Randomization |
|-------|---------------|---------------|
| `reset_right_arm` | 7 arm joints | ±0.05 rad offset from default pose |
| `reset_right_hand` | 6 hand joints | ±0.05 rad offset from default pose |
| `freeze_left_arm` | 13 left arm+hand joints | Exact zero offset (hard frozen) |
| `freeze_lower_body` | All leg + waist joints | Exact zero offset (hard frozen) |
| `reset_target_object` | Target cube | x: ±10 cm, y: ±5 cm from anchor |
| `reset_distractor_{1–10}` | Each distractor | x: ±3 cm, y: ±3 cm from anchor |

The ±10 cm / ±5 cm target jitter forces the policy to generalize across a workspace region rather than memorizing a single cube location.

---

## 9. Curriculum

> **Rewritten 2026-08-13 — the previous version of this section described a mechanism
> that does not actually run.** `PickingCurriculumScheduler` (`mdp/curriculum.py`) and
> `reset_clutter_based_on_difficulty` (`mdp/events.py`) both exist as library code, with
> logic matching what used to be documented here almost verbatim — but **no `CurriculumCfg`
> is ever assigned on any registered env cfg class**, so the scheduler is never instantiated
> and the difficulty score never leaves 0. Even if it were wired up, the reward-term names
> it looks for (`reaching_target`, `lifting_target`, `declutter`, `pick_success`) don't
> exist in the current `RewardsCfg` (§5) — those names predate the terms that actually
> exist today (`task_reward`, `grasp_goal_palm`, `grasp_goal_hand`, `grasp_reach`, etc.).
> Treat both files as designed-but-orphaned code, not active behavior, unless someone
> wires a `CurriculumCfg` back in and renames the reward terms to match.
>
> **What actually runs today is a manual, 3-stage curriculum over separate registered gym
> tasks** — you pick the difficulty by choosing `--task`, not by anything the environment
> adjusts on its own mid-run.

### 9a. The three registered distractor-difficulty tasks

| gym task ID | Env cfg class | Distractor layout | Distance to cube (nominal) |
|---|---|---|---|
| `Isaac-G1-Pick-Stage1-v0` | `G1RightArmLiftEnvCfg_V2_Stage1` | 5 distractors in a tight ring around the cube, other 5 parked on the table margins (out of the way, still in the observation space) | ring: 5.0 cm to cube, 6.8 cm to each other |
| `Isaac-G1-Pick-Stage2-v0` | `G1RightArmLiftEnvCfg_V2_Stage2` | all 10 distractors in a tight 3×4 grid centered on the cube (2 grid cells nearest the cube left empty) | every distractor's nearest neighbor: 6.0 cm; cube's nearest distractor: 7.3 cm |
| `Isaac-G1-Pick-v0` / `Isaac-G1-Pick-Play-v0` | `G1RightArmLiftEnvCfg_V2` / `_PLAY` | all 10 distractors at the original fixed adversarial-clutter anchors (§2) | 7.0 – 18.0 cm, clustered close to the approach path |

Each `*-Play-v0` variant is identical to its training counterpart except `num_envs=64`, for lower-footprint eval/inference. Every `RewardsCfg`/`TerminationsCfg`/`ObservationsCfg` term (§5, §6, §4) is shared across all three — only the distractor `init_state.pos` values (and, for Stage 1 only, which distractors `distractor_off_tray` applies to — see the class docstring) differ between them.

**Usage pattern**: train (or resume) against one task, then launch a fresh run resuming from the previous stage's checkpoint but pointed at the next `--task`. There's no automatic promotion — advancing stages is a manual decision made by watching TensorBoard (episode reward / success-rate curves flattening), not a fixed iteration count or a script.

### 9b. Empty / no-obstruction baseline

`Isaac-G1-Pick-Empty-v0` / `-Play-v0` (added 2026-08-13, `G1RightArmLiftEnvCfg_V2_Empty`):
all 10 distractors relocated to the table margins — present in the scene and the
96-dim observation space (so nothing about the observation shape changes when a later
phase moves them back onto the tray), just physically out of the reach/grasp workspace.
This is the base environment for the from-scratch reward curriculum's Phases 1-4 (§9c).

### 9c. From-scratch 5-phase reward curriculum (started 2026-08-13)

Separate from — and orthogonal to — the distractor-difficulty tasks in §9a/9b: rather
than warm-starting from a pre-trained checkpoint (as every run before this one did, see
`working_models/`), this trains fresh from a random initialization, introducing reward
terms one phase at a time so a training stall can be attributed to a specific,
just-added term instead of an opaque multi-term reward from step one. Implemented
entirely via `RewardsCfg` weights/params in `g1_pick_env_cfg.py` (each currently-zeroed
one carries a `# target: X -- Phase N` comment) — no RewTerms are added or removed
between phases, only their weights change, and advancing is a manual decision (watch
TensorBoard, not a fixed iteration count).

| Phase | Task | Adds |
|---|---|---|
| 1 | `Isaac-G1-Pick-Empty-v0` | `task_reward` (reach_weight=1.0 only) + `action_smoothness` + `joint_speed` + `distractor_accel` (inert — no distractors nearby) + `target_object_accel` + `sustained_reach_bonus`/`sustained_reach` termination |
| 2 | `Isaac-G1-Pick-Empty-v0` | `task_reward.grasp_weight` (thumb/finger proximity) |
| 3 | `Isaac-G1-Pick-Empty-v0` | `grasp_goal_palm`/`grasp_goal_hand`/`grasp_reach` (UltraDexGrasp pose mimicking) + `task_reward.pose_gated_success=True` |
| 4 | `Isaac-G1-Pick-Empty-v0` | `task_reward.lift_weight`/`success_weight` |
| 5 | `Stage1-v0` → `Stage2-v0` → `Isaac-G1-Pick-v0` | `fingertip_impact`, `distractor_off_tray`, `distractor_drop`, and the distractor layout itself ramps through §9a's three stages |

**`target_object_accel` and `sustained_reach_bonus` are Phase-1 additions, not
originally planned there** (2026-08-19 revision): `target_object_accel` was moved up
from Phase 4 after a reward-breakdown plot (§5.5, §5.3) showed `reach_rew` can be
exploited by disturbing the cube even with `lift_weight`/`success_weight` still at 0 —
the exploit doesn't need lift/success to be reachable. `sustained_reach_bonus` was added
because Phase 1-4 previously had no positive-outcome termination at all.

---

## 10. Policy Architecture

**Algorithm**: PPO (Proximal Policy Optimization) via RSL-RL

| Hyperparameter | Value |
|---------------|-------|
| Actor/Critic network | MLP [512, 256, 128] with ELU activation |
| Input normalization | Enabled for both actor and critic |
| Rollout steps per env | 32 |
| Mini-batches | 4 |
| Learning epochs per rollout | 5 |
| Learning rate | 3×10⁻⁴ (adaptive schedule) |
| PPO clip ε | 0.2 |
| Entropy coefficient | 0.005 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Desired KL | 0.01 |
| Max grad norm | 1.0 |
| Initial noise std | 0.8 |
| Max training iterations | 5000 |
| Checkpoint interval | 50 iterations |

---

## 11. Height Thresholds Summary

| Constant | Value (m) | Role |
|----------|-----------|------|
| `_OBJ_INIT_Z` | 0.845 | Cube center at rest on tray |
| `_OFF_TRAY_Z` | 0.835 | Cube below this → distractor_off_tray penalty fires |
| `_SUCCESS_Z` | 1.134 | Cube above this → `target_lifted` success termination |
| `_DROP_Z` | 0.600 | Below this → early termination (target OR distractor) |
| Lift threshold (curriculum) | 0.150 | Used internally by curriculum to count a pick as successful |

---

## 12. Key Design Choices

**Mimic joints**: The Inspire hand has 12 joints but the policy only controls 6 proximal joints. Downstream joints are driven by fixed gear ratios (`×0.8024`, `×0.9487`, `×1.0843`) matching hardware transmission specs. This halves the hand action space without losing physical fidelity.

**Deadband in grasp reward**: Fingertip distances are clamped: `clamp(raw_dist − 0.025, min=0)` — a 2.5 cm contact radius. Tips physically touching the cube surface (within 2.5 cm of center) receive maximum grasp reward, preventing reward from peaking at impossible zero-distance configurations.

**Drop override in reward**: When the target cube falls off the table (`z < 0.600`), the task reward is hard-overridden to −0.5, providing a clear negative gradient before the termination fires.

**Left arm/lower body frozen via stiffness override**: Instead of removing joints, the left hand and leg actuators are given extreme stiffness (10,000) and damping (1,000) at environment init. This keeps the articulation physically consistent while making those DOF immovable for the policy.

**Distractor penalty hierarchy**: Three layers create a smooth gradient away from knocking distractors: (1) acceleration penalty on any contact, (2) per-step off-tray penalty while cube is displaced, (3) large sparse penalty + termination if cube falls off the table.
