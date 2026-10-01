# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-grasp reward terms driven by BODex/UltraDexGrasp-synthesized Inspire Hand grasps.

At every reset, :class:`sample_grasp_goal` transforms the validated grasp library
(object frame) to the current cube pose and selects, per environment, the grasp
whose palm position is closest to the robot's current palm. Reward terms then pull
the palm pose and the six proximal hand joints toward that goal grasp.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_matrix, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

_DEFAULT_GRASP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grasp_sampler", "grasp_dataset", "cube_5cm_grasps_valid.npz",
)

# EDIT ME: rotate every goal grasp about the cube's vertical (robot-frame z) axis.
# Rotates position AND orientation together about the cube center, so the grasp
# stays valid (cube is symmetric under 90 deg z-rotation). -90 = clockwise seen
# from above (robot looking down). Set 0.0 to disable.
_GOAL_YAW_DEG = 0.0

# BODex joint order == policy hand-action order (thumb yaw/pitch, index, middle, ring, pinky)
_HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]


class sample_grasp_goal(ManagerTermBase):
    """Reset event: assigns each env a goal grasp (palm pose in world + 6 hand joints).

    Stores goals on the class instance; the reward terms below look this term up
    through the event manager.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        grasp_file = cfg.params.get("grasp_file", _DEFAULT_GRASP_FILE)
        data = np.load(grasp_file, allow_pickle=True)
        grasps = data["grasp_pose"]  # (N, 1, 3, 13) stages: pregrasp/grasp/squeeze
        if grasps.shape[0] == 0:
            raise RuntimeError(f"No grasps in {grasp_file}")
        # use the 'grasp' stage (index 1) as the goal
        g = grasps[:, 0, 1, :]  # (N, 13)
        # convert hand root pose from the synthesis URDF base frame to the USD
        # R_hand_base_link frame using the calibration saved by the validator
        if "T_usdbase_urdfbase" in data:
            T = data["T_usdbase_urdfbase"]  # p_usd = R p_urdf + t
            R_c = torch.tensor(T[:3, :3], dtype=torch.float32)
            # T_obj<-usdbase = T_obj<-urdfbase @ inv(T_usdbase<-urdfbase)
            from scipy.spatial.transform import Rotation as Rot
            R_g = Rot.from_quat(g[:, 3:7], scalar_first=True).as_matrix()  # obj<-urdfbase
            R_inv = T[:3, :3].T
            t_inv = -R_inv @ T[:3, 3]
            R_new = R_g @ R_inv
            p_new = g[:, :3] + np.einsum("nij,j->ni", R_g, t_inv)
            quat_new = Rot.from_matrix(R_new).as_quat(scalar_first=True)
            g = np.concatenate([p_new, quat_new, g[:, 7:]], axis=1)

        device = env.device
        self.grasp_pos_obj = torch.tensor(g[:, :3], dtype=torch.float32, device=device)
        self.grasp_quat_obj = torch.tensor(g[:, 3:7], dtype=torch.float32, device=device)
        self.grasp_q = torch.tensor(g[:, 7:], dtype=torch.float32, device=device)

        # rotate only the goal ORIENTATION about the cube's vertical z (see _GOAL_YAW_DEG);
        # position is left unchanged
        if _GOAL_YAW_DEG != 0.0:
            a = torch.deg2rad(torch.tensor(_GOAL_YAW_DEG, device=device))
            qz = torch.tensor([torch.cos(a / 2), 0.0, 0.0, torch.sin(a / 2)], device=device)  # wxyz, about +z
            qz_b = qz.unsqueeze(0).expand(self.grasp_quat_obj.shape[0], 4)
            self.grasp_quat_obj = quat_mul(qz_b, self.grasp_quat_obj)
        # pregrasp/squeeze finger stages (used by the demonstration oracle)
        self.grasp_q_pre = torch.tensor(grasps[:, 0, 0, 7:], dtype=torch.float32, device=device)
        self.grasp_q_squeeze = torch.tensor(grasps[:, 0, 2, 7:], dtype=torch.float32, device=device)
        self.num_grasps = g.shape[0]

        # per-env goal buffers
        n = env.num_envs
        self.goal_pos_w = torch.zeros(n, 3, device=device)
        self.goal_quat_w = torch.zeros(n, 4, device=device)
        self.goal_hand_q = torch.zeros(n, 6, device=device)
        self.goal_hand_q_pre = torch.zeros(n, 6, device=device)
        self.goal_hand_q_squeeze = torch.zeros(n, 6, device=device)
        self.goal_grasp_idx = torch.zeros(n, dtype=torch.long, device=device)
        self._object_name = "target_object"
        self._last_live_update = -1

        robot: Articulation = env.scene["robot"]
        self._palm_body_idx = robot.body_names.index("R_hand_base_link")
        self._hand_joint_ids = [robot.joint_names.index(nm) for nm in _HAND_JOINT_NAMES]

        # distractors present in the scene (for clutter-aware goal selection)
        self._distractor_names = [n for n in (f"distractor_{i}" for i in range(1, 11))
                                  if n in env.scene.keys()]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
        grasp_file: str = "",
        fixed_grasp_idx: int = -1,
        random_selection: bool = False,
    ):
        obj: RigidObject = env.scene[object_cfg.name]
        robot: Articulation = env.scene["robot"]

        obj_pos = obj.data.root_pos_w[env_ids]      # (E,3)
        obj_quat = obj.data.root_quat_w[env_ids]    # (E,4) wxyz

        # single-grasp mode: the same library grasp for every env and episode.
        # This makes the goal a deterministic function of the (observed) cube
        # pose, eliminating the hidden-goal noise of per-episode selection.
        # random_selection: draw a uniform random grasp per episode instead —
        # viable with the max(pose, grasp-gate) palm reward, where the grasp
        # endpoint pays fully regardless of which goal was drawn.
        if fixed_grasp_idx >= 0 or random_selection:
            E = len(env_ids)
            if random_selection:
                best = torch.randint(self.num_grasps, (E,), device=env.device)
            else:
                best = torch.full((E,), fixed_grasp_idx, dtype=torch.long, device=env.device)
            gq = self.grasp_quat_obj[best]
            self.goal_grasp_idx[env_ids] = best
            self._object_name = object_cfg.name
            self.goal_pos_w[env_ids] = quat_apply(obj_quat, self.grasp_pos_obj[best]) + obj_pos
            self.goal_quat_w[env_ids] = quat_mul(obj_quat, gq)
            self.goal_hand_q[env_ids] = self.grasp_q[best]
            self.goal_hand_q_pre[env_ids] = self.grasp_q_pre[best]
            self.goal_hand_q_squeeze[env_ids] = self.grasp_q_squeeze[best]
            return

        palm_pos = robot.data.body_pos_w[env_ids, self._palm_body_idx]  # (E,3)

        E, G = len(env_ids), self.num_grasps
        # grasp palm positions in world: obj pose * grasp pos
        oq = obj_quat.unsqueeze(1).expand(E, G, 4).reshape(-1, 4)
        gp = self.grasp_pos_obj.unsqueeze(0).expand(E, G, 3).reshape(-1, 3)
        cand_pos_w = (quat_apply(oq, gp) + obj_pos.unsqueeze(1).expand(E, G, 3).reshape(-1, 3)).view(E, G, 3)

        # selection score 1: reach cost (distance from current palm)
        d = torch.norm(cand_pos_w - palm_pos.unsqueeze(1), dim=2)  # (E,G)

        # selection score 2: clutter risk — count distractors inside each
        # candidate's approach corridor (segment from 12 cm behind the palm
        # goal, along the palm normal, to the object center)
        score = d.clone()
        if self._distractor_names:
            gq_all = self.grasp_quat_obj.unsqueeze(0).expand(E, G, 4).reshape(-1, 4)
            cand_quat_w = quat_mul(oq, gq_all)
            palm_normal = quat_apply(cand_quat_w, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(E * G, 3))
            p1 = cand_pos_w.view(-1, 3)                     # palm goal
            p0 = p1 - 0.12 * palm_normal                    # pregrasp end of corridor
            p2 = obj_pos.unsqueeze(1).expand(E, G, 3).reshape(-1, 3)  # object center
            dist_pos = torch.stack(
                [env.scene[n].data.root_pos_w[env_ids] for n in self._distractor_names], dim=1
            )  # (E,D,3)
            on_tray = dist_pos[:, :, 2] > 0.7               # ignore hidden distractors
            dist_pos = dist_pos.unsqueeze(1).expand(E, G, -1, 3).reshape(E * G, -1, 3)
            n_block = torch.zeros(E * G, device=env.device)
            for a, b in ((p0, p1), (p1, p2)):               # two corridor segments
                ab = (b - a).unsqueeze(1)                   # (EG,1,3)
                t = ((dist_pos - a.unsqueeze(1)) * ab).sum(-1) / (ab.pow(2).sum(-1) + 1e-9)
                closest = a.unsqueeze(1) + t.clamp(0, 1).unsqueeze(-1) * ab
                seg_d = torch.norm(dist_pos - closest, dim=-1)  # (EG,D)
                blocked = (seg_d < 0.07) & on_tray.unsqueeze(1).expand(E, G, -1).reshape(E * G, -1)
                n_block += blocked.float().sum(-1)
            # each blocking distractor costs as much as 0.5 m of extra reach
            score += 0.5 * n_block.view(E, G)

        best = score.argmin(dim=1)  # (E,)

        gq = self.grasp_quat_obj[best]  # (E,4)
        self.goal_grasp_idx[env_ids] = best
        self._object_name = object_cfg.name
        self.goal_pos_w[env_ids] = cand_pos_w[torch.arange(E, device=env.device), best]
        self.goal_quat_w[env_ids] = quat_mul(obj_quat, gq)
        self.goal_hand_q[env_ids] = self.grasp_q[best]
        self.goal_hand_q_pre[env_ids] = self.grasp_q_pre[best]
        self.goal_hand_q_squeeze[env_ids] = self.grasp_q_squeeze[best]

    def update_live_goals(self, env: ManagerBasedRLEnv):
        """Re-attach each env's chosen grasp to the object's CURRENT pose.

        The grasps are stored relative to the object, so they remain valid when
        the object is pushed around; without this, the goal would stay frozen at
        the object's spawn pose and the policy would chase empty space after any
        contact moves the object.
        """
        if self._last_live_update == env.common_step_counter:
            return  # already refreshed this step (both reward terms call this)
        self._last_live_update = env.common_step_counter
        obj: RigidObject = env.scene[self._object_name]
        gp = self.grasp_pos_obj[self.goal_grasp_idx]    # (N,3)
        gq = self.grasp_quat_obj[self.goal_grasp_idx]   # (N,4)
        self.goal_pos_w = quat_apply(obj.data.root_quat_w, gp) + obj.data.root_pos_w
        self.goal_quat_w = quat_mul(obj.data.root_quat_w, gq)


def _get_goal_term(env: ManagerBasedRLEnv) -> sample_grasp_goal:
    term = getattr(env, "_grasp_goal_term", None)
    if term is None:
        idx = env.event_manager.active_terms["reset"].index("sample_grasp_goal")
        term = env.event_manager._mode_term_cfgs["reset"][idx].func
        env._grasp_goal_term = term
    return term


def grasp_goal_palm_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_std: float = 0.15,
    orient_std: float = 0.6,
    orient_weight: float = 0.5,
    pos_mode: str = "full",
    blend: str = "gated",
    pos_shape: str = "tanh",
    pos_ramp_dist: float = 0.5,
) -> torch.Tensor:
    """POSE guidance toward the UltraDexGrasp goal: palm position + wrist orientation.

    Deliberately does NOT reward grasping — that is owned entirely by
    compute_task_reward's grasp term. This term only pulls the palm toward the
    goal grasp's pose; the task reward then handles closing and lifting.

    pos_mode:
      "full"   - distance to the goal position in all 3 axes.
      "height" - only the VERTICAL (z) offset to the goal. Use this when the
                 task reward already centres the palm horizontally (posture +
                 reach) and the grasp library's unique positional contribution
                 is the correct grasp HEIGHT (~hand-length above the object),
                 which task_reward's posture target undershoots.
    """
    term = _get_goal_term(env)
    term.update_live_goals(env)  # goal follows the object if it gets pushed
    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
    palm_quat = robot.data.body_quat_w[:, term._palm_body_idx]
    if pos_mode == "height":
        d_pos = torch.abs(palm_pos[:, 2] - term.goal_pos_w[:, 2])
    else:
        d_pos = torch.norm(palm_pos - term.goal_pos_w, dim=1)
    if pos_shape == "linear":
        # "[Option B]" 2026-08-29 -- CONSTANT gradient regardless of distance, unlike tanh
        # whose gradient shrinks the farther the palm is from the goal. Measured on this
        # project's own runs: with pos_std=0.15, gradient (1/std)*sech^2(d/std) falls from
        # 6.67 at d=0 to 3.73 at d=12cm to 1.63 at d=20cm to 0.47 at d=30cm -- already down
        # to a quarter of its peak in exactly the 12-20cm band where the policy kept
        # converging and stalling. A plain linear ramp keeps the gradient at a constant
        # 1/pos_ramp_dist everywhere from d=0 out to pos_ramp_dist, so there is no distance
        # at which the position term goes quiet.
        #
        # Ported from OpenAI's Learning Dexterous In-Hand Manipulation (Andrychowicz et al.
        # 2020, IJRR 39(1)) and IsaacLab's own bundled reference implementation of that task
        # (isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py), which use
        # dist_rew = dist_reward_scale * goal_dist -- a raw, unbounded linear penalty
        # (dist_reward_scale=-10.0 in their tuned config). Kept bounded to [0,1] here
        # (clamped ramp, not raw unbounded) so it composes safely with the multiplicative
        # gate below rather than flipping sign under it.
        pos_rew = (1.0 - d_pos / pos_ramp_dist).clamp(min=0.0, max=1.0)
    else:
        pos_rew = 1.0 - torch.tanh(d_pos / pos_std)
    ang = quat_error_magnitude(palm_quat, term.goal_quat_w)  # radians, [0, pi]
    orient_rew = 1.0 - torch.tanh(ang / orient_std)

    if blend == "additive":
        # ORIGINAL. Position and orientation are INDEPENDENT, so orientation pays on its
        # own -- and a wrist can be aligned anywhere in space. Measured consequence at
        # orient_weight=0.5: the policy sat 12.6cm from the goal with wrist_err of only
        # 11-17 deg, banking 68% of this term from orientation while never fixing position.
        # Kept only for reproducing old runs; do not use for new ones.
        return (1.0 - orient_weight) * pos_rew + orient_weight * orient_rew

    if blend == "multiplicative":
        # Closes the loophole but overcorrects: the position gradient gets scaled by
        # orient_rew, which is ~0.05 right after a position-only stage, collapsing the
        # position drive ~22x (measured -5.05 -> -0.23) and throwing away exactly the
        # progress that stage just bought.
        return pos_rew * orient_rew

    # "gated" (DEFAULT for stage 2, 2026-08-27): orientation is a BONUS ON TOP of position,
    # never a substitute for it.
    #   pos_rew * ((1 - w) + w * orient_rew)
    # - orientation cannot pay anything on its own: if pos_rew -> 0 the whole term -> 0
    # - the position gradient is preserved (measured -2.64, vs -2.52 for additive and only
    #   -0.23 for multiplicative)
    # - the orientation gradient is live and GROWS as position improves, so the two are
    #   learned in the right order rather than traded against each other
    return pos_rew * ((1.0 - orient_weight) + orient_weight * orient_rew)


def grasp_goal_hand_config_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    q_std: float = 0.5,
    gate_dist: float = 0.10,
    use_squeeze: bool = False,
) -> torch.Tensor:
    """Reward for matching the goal hand configuration, gated on palm proximity.

    The gate prevents the policy from curling fingers to the goal pose while the
    hand is still far from the object.

    use_squeeze (2026-08-26): target the grasp library's SQUEEZE stage
    (`goal_hand_q_squeeze`) instead of its grasp stage (`goal_hand_q`).

    Why this matters, from a decisive measurement (`lift_capability_probe.py` on
    `phase4v7` ckpt 2099, the best-contact policy of the project -- thumb 0.50cm with
    64/64 envs inside the 1cm buffer): forcing the arm upward raised the palm +2.50cm and
    moved the cube -0.01cm. Follow ratio -0.005, i.e. **no grip whatsoever**. Proximity was
    excellent and grip force was zero.

    The reason is structural, and it invalidates every reward this project has tried so far:
    all of them are GEOMETRIC (fingertip position / joint angle). The hand joints are
    position-controlled, so commanding a finger to exactly where it already rests produces
    no force. A hand hovering at zero distance scores identically to one squeezing hard --
    in fact slightly better, since squeezing jostles the cube and pays `target_object_accel`.
    Nothing anywhere rewarded grip FORCE.

    The grasp stage is a *touching* pose; the squeeze stage closes each finger a further
    0.10-0.23 rad (L2 0.35 rad). Commanding it drives the fingers PAST the cube surface,
    PhysX stops them at the surface, and the PD controller then generates real normal force
    -- which is what friction, and therefore lifting, requires.
    """
    term = _get_goal_term(env)
    term.update_live_goals(env)  # goal follows the object if it gets pushed
    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
    palm_d = torch.norm(palm_pos - term.goal_pos_w, dim=1)
    gate = (1.0 - torch.tanh(palm_d / gate_dist)).clamp(min=0.0)

    target_q = term.goal_hand_q_squeeze if use_squeeze else term.goal_hand_q
    q = robot.data.joint_pos[:, term._hand_joint_ids]
    q_err = torch.norm(q - target_q, dim=1)
    return gate * (1.0 - torch.tanh(q_err / q_std))


# Cached fingertip target contact points for the active grasp (see MDP_REPORT.md §5.2.4).
# Computed offline by scratchpad/compute_fingertip_contacts.py from the grasp_selection
# optimizer's own cube_contact() projection -- reused here purely as geometry, not for
# re-selecting a grasp. Cached (not recomputed from the gitignored ultradex_repo/URDF at
# runtime) so training doesn't depend on those assets being present, same idiom as
# grasp_selection/scores.json.
_FINGERTIP_CONTACTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grasp_selection", "fingertip_contacts.json",
)
# order must match the tracked fingertip bodies [R_thumb_distal, R_index_intermediate,
# R_middle_intermediate, R_ring_intermediate, R_pinky_intermediate] i.e. _RIGHT_HAND_BODIES[1:]
_FINGERTIP_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def grasp_reach_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    std: float = 0.05,
    contacts_file: str = _FINGERTIP_CONTACTS_FILE,
    shape: str = "tanh",
    ramp_dist: float = 0.5,
) -> torch.Tensor:
    """Pulls each of the 5 tracked fingertips toward its OWN target contact point on
    the cube surface for the active grasp, instead of one shared centroid target
    (c.f. compute_task_reward's reach_rew). Dense, ungated. Uses mean-of-tanh (not
    tanh-of-mean) so one badly-placed finger isn't washed out by four good ones --
    see MDP_REPORT.md §5.2.4 for the full derivation and design rationale.

    robot_cfg.body_ids must resolve to the 5 fingertip bodies in the order
    [thumb, index, middle, ring, pinky] (_RIGHT_HAND_BODIES[1:]), matching the
    cached contact points' fingertip_order.

    shape="linear" (2026-08-29, per the user): same motivation as the palm-position and
    multi-link fixes -- tanh's gradient shrinks with distance, so once a fingertip gets
    within a few std of its target the incentive to close the LAST centimetre (let alone
    reach genuine contact) largely disappears. Linear keeps a constant gradient of
    1/ramp_dist all the way to the target, so there is no distance at which "close enough"
    sets in and the pull to actually reach the surface never goes quiet.
    """
    if not hasattr(env, "_grasp_reach_contacts_obj"):
        with open(contacts_file) as f:
            data = json.load(f)
        assert data["fingertip_order"] == _FINGERTIP_ORDER, (
            f"cached fingertip order {data['fingertip_order']} != expected {_FINGERTIP_ORDER}"
        )
        env._grasp_reach_contacts_obj = torch.tensor(
            data["contact_pos_cube_frame"], dtype=torch.float32, device=env.device
        )  # (5,3), constant, in the cube's own object frame

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (N,5,3)
    cube_quat = obj.data.root_quat_w                        # (N,4)
    cube_pos = obj.data.root_pos_w                          # (N,3)

    n = tips_w.shape[0]
    contacts_obj = env._grasp_reach_contacts_obj.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
    cube_quat_exp = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
    cube_pos_exp = cube_pos.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 3)
    targets_w = (quat_apply(cube_quat_exp, contacts_obj) + cube_pos_exp).view(n, 5, 3)

    d = torch.norm(tips_w - targets_w, dim=-1)  # (N,5)
    if shape == "linear":
        return (1.0 - d / ramp_dist).clamp(min=0.0, max=1.0).mean(dim=-1)
    return (1.0 - torch.tanh(d / std)).mean(dim=-1)


def multi_link_position_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    pos_ramp_dist: float = 0.5,
    contacts_file: str = _FINGERTIP_CONTACTS_FILE,
) -> torch.Tensor:
    """"[Option C]" 2026-08-29 -- POSITION component of grasp_pose, redefined per the user.

    Replaces the single palm-centroid distance ("one simple point... I don't think it's
    doing much" -- the user's words, and the data backed it: v6 converged in ~90 iterations
    to a palm 11-12cm from goal and sat there for 700+ iterations with no further progress)
    with the mean LINEAR distance of each of the 5 tracked fingertips to ITS OWN target
    point on the cube surface -- the same points and body order `grasp_reach_reward` above
    already uses (reused deliberately: those targets are already validated and cached,
    no new data needed).

    Two structural differences from grasp_reach_reward, both per the user's explicit
    request this round:
      1. LINEAR shape, not tanh -- same reasoning as the palm-position fix in "[Option B]":
         tanh's gradient shrinks with distance, linear keeps it constant out to
         pos_ramp_dist so there is no distance at which this term goes quiet.
      2. This is now the sole POSITION signal -- grasp_goal_palm_reward's old single-point
         palm-centroid distance is removed, not kept alongside this.

    Also serves the user's SECOND request implicitly: matching 5 fingertip positions
    simultaneously constrains orientation too (a correctly-POSITIONED but wrongly-ORIENTED
    hand will have its fingertips systematically off-target even though the palm centroid
    might be close), which is exactly the richer signal multi-point matching provides over
    a single centroid point.
    """
    if not hasattr(env, "_grasp_reach_contacts_obj"):
        with open(contacts_file) as f:
            data = json.load(f)
        assert data["fingertip_order"] == _FINGERTIP_ORDER, (
            f"cached fingertip order {data['fingertip_order']} != expected {_FINGERTIP_ORDER}"
        )
        env._grasp_reach_contacts_obj = torch.tensor(
            data["contact_pos_cube_frame"], dtype=torch.float32, device=env.device
        )

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (N,5,3)
    cube_quat = obj.data.root_quat_w
    cube_pos = obj.data.root_pos_w

    n = tips_w.shape[0]
    contacts_obj = env._grasp_reach_contacts_obj.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
    cube_quat_exp = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
    cube_pos_exp = cube_pos.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 3)
    targets_w = (quat_apply(cube_quat_exp, contacts_obj) + cube_pos_exp).view(n, 5, 3)

    d = torch.norm(tips_w - targets_w, dim=-1)  # (N,5)
    per_finger = (1.0 - d / pos_ramp_dist).clamp(min=0.0, max=1.0)
    return per_finger.mean(dim=-1)


# Body name for each entry of _FINGERTIP_ORDER. Note the non-thumb fingers' LAST link is
# the "intermediate" one (those fingers have 2 segments, the thumb has 3), so these are
# the tip links in every case.
_FINGER_BODY = {
    "thumb": "R_thumb_distal",
    "index": "R_index_intermediate",
    "middle": "R_middle_intermediate",
    "ring": "R_ring_intermediate",
    "pinky": "R_pinky_intermediate",
}


def finger_contact_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    fingers: tuple = ("middle", "ring", "pinky"),
    std: float = 0.02,
    gate_dist: float = 0.08,
    contacts_file: str = _FINGERTIP_CONTACTS_FILE,
) -> torch.Tensor:
    """Proximity-gated pull of SPECIFIC fingertips toward their own cube-surface contact
    points -- the "curl these fingers in, but only once you're actually in grasp position"
    term (2026-08-25).

    Why this exists rather than another `grasp_goal_hand` retune: two attempts at tuning
    that term failed in opposite directions (q_std=0.3 saturated the gradient to nothing;
    q_std=1.0 made the term dominant and it got REWARD-HACKED -- its own reward rose 45x
    while measured fingertip-to-cube distance got ~2x WORSE, because `grasp_goal_hand`
    scores JOINT ANGLES, which are satisfiable anywhere in space, behind a loose 20cm palm
    gate). This term is structurally immune to that failure: the only way to score is for
    the fingertip to actually approach a point ON the cube, so "right shape, wrong place"
    pays nothing.

    Targets middle/ring/pinky by default because those are the three the policy was
    measurably refusing to close -- `mimic_track_probe.py` showed their proximal joints
    pinned at ~0% of travel (commanded BELOW their lower limit) while the thumb used
    ~80-88% of its range. `measure_coupling.py` separately confirmed those joints track
    position commands fine up to 1.0 rad, so this is a reward problem, not a mechanical one.

    NOTE on the target points: the FSWO/UltraDexGrasp optimizer picks contact points by
    PROJECTING each fingertip onto the cube surface and solving force closure, so a valid
    grasp does NOT require every finger to physically touch -- some target points sit at a
    small standoff by construction. So this term is deliberately a dense pull toward the
    per-finger target, not a hard contact requirement, and should not be weighted as if
    zero distance were mandatory for all three.

    Bodies are resolved by explicit name lookup rather than through `robot_cfg.body_ids`
    to guarantee the fingertip<->contact-point pairing, since SceneEntityCfg does not
    preserve the caller's body_names order by default.
    """
    if not hasattr(env, "_grasp_reach_contacts_obj"):
        with open(contacts_file) as f:
            data = json.load(f)
        assert data["fingertip_order"] == _FINGERTIP_ORDER, (
            f"cached fingertip order {data['fingertip_order']} != expected {_FINGERTIP_ORDER}"
        )
        env._grasp_reach_contacts_obj = torch.tensor(
            data["contact_pos_cube_frame"], dtype=torch.float32, device=env.device
        )  # (5,3), constant, in the cube's own object frame

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "_finger_contact_ids"):
        sel = [_FINGERTIP_ORDER.index(f) for f in fingers]
        body_ids = [robot.body_names.index(_FINGER_BODY[f]) for f in fingers]
        env._finger_contact_ids = (
            torch.tensor(sel, device=env.device),
            torch.tensor(body_ids, device=env.device),
        )
    sel_ids, body_ids = env._finger_contact_ids
    k = len(body_ids)

    # gate on the palm actually being at the grasp goal (which tracks the cube), so the
    # fingers don't get paid for curling on the way in
    term = _get_goal_term(env)
    term.update_live_goals(env)
    palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
    palm_d = torch.norm(palm_pos - term.goal_pos_w, dim=1)
    gate = (1.0 - torch.tanh(palm_d / gate_dist)).clamp(min=0.0)

    tips_w = robot.data.body_pos_w[:, body_ids]  # (N,k,3)
    n = tips_w.shape[0]
    contacts_obj = env._grasp_reach_contacts_obj[sel_ids]  # (k,3)
    contacts_obj = contacts_obj.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
    cube_quat_exp = obj.data.root_quat_w.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    cube_pos_exp = obj.data.root_pos_w.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)
    targets_w = (quat_apply(cube_quat_exp, contacts_obj) + cube_pos_exp).view(n, k, 3)

    d = torch.norm(tips_w - targets_w, dim=-1)  # (N,k)
    return gate * (1.0 - torch.tanh(d / std)).mean(dim=-1)


# ---------------------------------------------------------------------------------------
# 2026-08-27, "[Sufian Plan]" clean-slate reward set. Only three rewards exist now:
# grasp pose, cube lift, and REAL contact. This is the third.
# ---------------------------------------------------------------------------------------

_CONTACT_FINGER_BODIES = [
    "R_thumb_distal", "R_index_intermediate", "R_middle_intermediate",
    "R_ring_intermediate", "R_pinky_intermediate",
]


_CONTACT_SENSOR_NAMES = ["contact_thumb", "contact_index", "contact_middle",
                         "contact_ring", "contact_pinky"]


def fingertip_contact_force_reward(
    env: ManagerBasedRLEnv,
    sensor_names: tuple = tuple(_CONTACT_SENSOR_NAMES),
    force_thresh: float = 0.1,
    sat_force: float = 5.0,
) -> torch.Tensor:
    """Reward REAL fingertip<->cube contact, read from PhysX, not from geometry.

    Every reward this project used before was geometric (fingertip position, joint angle),
    and that is precisely why it failed: the hand joints are position-controlled, so a
    finger commanded to where it already rests exerts no force, and a hand hovering at zero
    distance scored identically to one gripping. Measured on the best policy to date
    (`lift_capability_probe.py`): raising the arm moved the palm +2.50cm and the cube
    -0.01cm -- perfect proximity, zero usable grip.

    This term instead reads the contact sensor's force matrix filtered to the target cube,
    so it can only be earned by actually pressing on it. Per finger it saturates at
    `sat_force` newtons, then averages over the five fingers -- so the maximum is reached by
    all five making solid contact, not by one finger mashing hard.

    Counting fingers-in-contact (rather than raw summed force) is deliberate: summed force
    is maximised by shoving, which is the exact failure mode observed -- slamming the hand
    shut moved the cube 7.3cm mean / 19.7cm max while lifting it 0.09cm. A saturating
    per-finger term pays for spreading contact around the cube instead.
    """
    # One sensor per finger -- see the SceneCfg comment for why a single 5-body sensor
    # silently reports all zeros.
    total = torch.zeros(env.num_envs, device=env.device)
    for name in sensor_names:
        fm = env.scene.sensors[name].data.force_matrix_w      # (N, 1, 1, 3)
        if fm is None:
            continue
        mag = torch.linalg.norm(fm, dim=-1).sum(dim=(1, 2))   # (N,) newtons on this finger
        contrib = (mag / sat_force).clamp(max=1.0)
        total += torch.where(mag > force_thresh, contrib, torch.zeros_like(contrib))
    return total / max(len(sensor_names), 1)


def cube_lift_height_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    init_z: float = 0.845,
    max_h: float = 0.20,
) -> torch.Tensor:
    """Dense reward on how high the cube is above its resting height. Linear (not tanh) so
    the gradient is constant all the way up and never saturates -- the mistake made twice
    with tanh stds in this project. Clamped at `max_h` so the policy cannot farm reward by
    flinging the cube arbitrarily high."""
    obj: RigidObject = env.scene[object_cfg.name]
    return (obj.data.root_pos_w[:, 2] - init_z).clamp(min=0.0, max=max_h)


def excessive_joint_speed_termination(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    max_speed: float = 3.0,
) -> torch.Tensor:
    """Terminate if any policy-controlled joint exceeds `max_speed` rad/s.

    Replaces the deleted joint_speed_penalty ("[Sufian Plan]", 2026-08-27). A penalty is a
    gradient the policy can trade against reward -- and it demonstrably did, costing ~-1.8
    of return while never actually preventing the thrashing it was meant to stop. A
    termination is a hard constraint instead. `max_speed` sits well above the actuator's own
    velocity_limit_sim (0.5) so this fires only on genuine solver-level violence, not on
    ordinary fast motion.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vel = robot.data.joint_vel[:, robot_cfg.joint_ids]
    return (vel.abs() > max_speed).any(dim=1)


def cube_disturbed_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    max_lateral: float = 0.10,
) -> torch.Tensor:
    """Terminate if the cube is shoved more than `max_lateral` HORIZONTALLY from where it
    started the episode.

    Replaces the deleted target_object_acceleration_penalty. Deliberately measures lateral
    displacement, not acceleration or total displacement: vertical motion is the goal, so
    penalising it would fight the lift reward. Sized from a real measurement -- slamming the
    hand shut moved the cube 7.3cm mean / 19.7cm max sideways while lifting it 0.09cm, which
    is the shove-instead-of-pinch failure this is meant to cut off.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    xy = obj.data.root_pos_w[:, :2]
    if not hasattr(env, "_cube_reset_xy"):
        env._cube_reset_xy = xy.clone()
    # Re-latch the reference for any env at the start of a fresh episode. Without this the
    # reference is captured once, ever, and every later episode measures displacement from a
    # stale origin -- which would fire this termination almost immediately and permanently.
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        env._cube_reset_xy[fresh] = xy[fresh]
    moved = torch.linalg.norm(xy - env._cube_reset_xy, dim=1)
    return (moved > max_lateral) & (~fresh)
