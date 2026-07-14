# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp-goal integration: load the FSWO-selected grasp from the offline
UltraDexGrasp+BODex library (grasp_sampler/), re-express it in the cube's
current world pose every step ("live", not frozen at reset -- see
Misc./UltraDex.md Problem 14), and shape the policy toward it with two reward
terms whose weight is ramped by mdp.GraspGoalCurriculum (grasp_goal_curriculum.py)
instead of being active at full strength from iteration 0.

Library schema (produced by grasp_sampler/build_goal_library.py):
    grasp_pose         : float32 (N, 1, 3, 13)
        N grasps x 1 hand x 3 stages x
        [x, y, z, qw, qx, qy, qz, thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        Stage 0 = pregrasp, 1 = grasp (fingertips on cube surface, used here),
        2 = squeeze. All poses in the CUBE frame (cube centered at origin,
        axis-aligned, as synthesized).
    T_usdbase_urdfbase : float64 (4, 4)
        Kabsch-calibrated rigid transform from the synthesis URDF's base frame
        to the USD's R_hand_base_link frame (Misc./UltraDex.md S3.3, S9b).

Note: grasp_selection/select_optimal_grasp.py separately writes a `scores.json`
with the full FSWO ranking (Misc./optimizer.md) -- this module does not read it
back from the npz. Only the chosen `fixed_grasp_idx` (set in g1_pick_env_cfg.py
from that ranking) is consumed here; per-grasp scores would only matter again
for the multi-grasp mode this module deliberately does not implement (see below).

Selection: single fixed grasp only (``fixed_grasp_idx >= 0``), chosen offline
by FSWO force-closure score instead of the nearest-palm heuristic it replaces.
Per-episode multi-grasp selection is intentionally NOT implemented here: doing
so before the observation is goal-conditioned reintroduces the hidden-goal
ceiling (Problem 17 in UltraDex.md, S5.2 in optimizer.md) -- it is future work,
gated on the 96->109 dim observation extension.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul

from .grasp_goal_curriculum import GraspGoalCurriculum

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Must match ActionsCfg.right_hand_action.joint_names in g1_pick_env_cfg.py exactly --
# this is also the order of columns [7:13] in every grasp_pose row.
HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]

_DEFAULT_LIBRARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grasp_sampler",
    "grasp_dataset",
    "cube_5cm_grasps_valid.npz",
)

_STAGE_GRASP = 1  # index into the 3-stage axis: 0=pregrasp, 1=grasp, 2=squeeze


class SampleGraspGoal(ManagerTermBase):
    """Reset-time event: assign each resetting env its (fixed) grasp goal.

    Loads the goal library once at construction. Nothing here depends on the
    cube's pose -- that transform happens every step in ``live_goal_pose``,
    called from both reward functions below, so the goal can never go stale
    relative to a cube that got bumped mid-episode.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        library_path = cfg.params.get("library_path", _DEFAULT_LIBRARY_PATH)
        if not os.path.isfile(library_path):
            raise FileNotFoundError(
                f"Grasp goal library not found at '{library_path}'.\n"
                "Generate it on a machine with GPU + the 'ultradex' conda env first:\n"
                "  1. grasp_sampler/synthesize_inspire_grasps.py   (BODex synthesis)\n"
                "  2. grasp_sampler/validate_grasps_isaaclab.py    (calibration + re-centering)\n"
                "  3. grasp_sampler/build_goal_library.py          (filter -> 32-grasp library)\n"
                "  4. grasp_sampler/grasp_selection/select_optimal_grasp.py (FSWO best index)\n"
                "See grasp_sampler/README.md for exact commands. Until then, set "
                "_ENABLE_GRASP_GOALS = False at the top of g1_pick_env_cfg.py to train "
                "without goal shaping."
            )
        data = np.load(library_path, allow_pickle=True)
        grasp_pose = data["grasp_pose"]  # (N, 1, 3, 13)
        T_cal = data["T_usdbase_urdfbase"]  # (4, 4)
        num_grasps = grasp_pose.shape[0]
        device = env.device

        R_cal = torch.tensor(np.asarray(T_cal[:3, :3]), dtype=torch.float32, device=device)
        t_cal = torch.tensor(np.asarray(T_cal[:3, 3]), dtype=torch.float32, device=device)
        q_cal = quat_from_matrix(R_cal).unsqueeze(0).expand(num_grasps, -1)  # (N, 4)

        stage = torch.tensor(grasp_pose[:, 0, _STAGE_GRASP, :], dtype=torch.float32, device=device)  # (N, 13)
        pos_urdf = stage[:, 0:3]
        quat_urdf = stage[:, 3:7]

        # Apply the Kabsch calibration once here, at load time, rather than every
        # step: synthesis-URDF base frame -> USD R_hand_base_link frame.
        self.grasp_pos_cube = quat_apply(q_cal, pos_urdf) + t_cal  # (N, 3), cube frame
        self.grasp_quat_cube = quat_mul(q_cal, quat_urdf)  # (N, 4), cube frame
        self.grasp_joint_q = stage[:, 7:13]  # (N, 6), policy joint order

        self.num_grasps = num_grasps
        self.chosen_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_grasp_idx: int = -1,
        library_path: str | None = None,  # noqa: ARG002 -- consumed in __init__ only;
        # declared here so EventManager's **term_cfg.params unpacking doesn't raise.
    ) -> None:
        if len(env_ids) == 0:
            return
        if fixed_grasp_idx < 0:
            raise ValueError(
                "SampleGraspGoal requires fixed_grasp_idx >= 0 (single-grasp mode). "
                "Run grasp_selection/select_optimal_grasp.py and set the result here; "
                "see the module docstring for why per-episode selection is not "
                "implemented yet."
            )
        self.chosen_idx[env_ids] = fixed_grasp_idx


def _get_grasp_goal_term(env: ManagerBasedRLEnv) -> SampleGraspGoal:
    """Resolve the registered SampleGraspGoal instance, caching on env."""
    term = getattr(env, "_grasp_goal_term", None)
    if term is None:
        for name, cfg in zip(env.event_manager._mode_term_names["reset"], env.event_manager._mode_term_cfgs["reset"]):
            if isinstance(cfg.func, SampleGraspGoal):
                term = cfg.func
                break
        if term is None:
            raise RuntimeError(
                "No SampleGraspGoal event term found in EventCfg (mode='reset'). "
                "Add one before using grasp_goal_palm_reward / grasp_goal_hand_config_reward."
            )
        env._grasp_goal_term = term
    return term


def live_goal_pose(
    env: ManagerBasedRLEnv,
    term: SampleGraspGoal,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-attach each env's chosen grasp to the cube's CURRENT pose.

    Recomputed from scratch every call (cheap: one quat_apply + one quat_mul
    over num_envs) rather than cached at reset -- this is what makes the goal
    "live" and structurally rules out the staleness in Problem 14, since there
    is no cache to go stale.

    Returns (goal_pos_w, goal_quat_w, goal_joint_q), shapes (N,3), (N,4), (N,6).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    idx = term.chosen_idx
    pos_cube = term.grasp_pos_cube[idx]
    quat_cube = term.grasp_quat_cube[idx]
    goal_pos_w = obj.data.root_pos_w + quat_apply(obj.data.root_quat_w, pos_cube)
    goal_quat_w = quat_mul(obj.data.root_quat_w, quat_cube)
    goal_joint_q = term.grasp_joint_q[idx]
    return goal_pos_w, goal_quat_w, goal_joint_q


def grasp_goal_palm_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    std: float = 0.15,
) -> torch.Tensor:
    r"""Pull the palm toward the live goal pose, ramped by the training-time
    curriculum: :math:`w_g \cdot (1 - \tanh(\lVert p_{palm} - p^* \rVert / std))`.

    ``robot_cfg`` must resolve a single body -- the palm (``right_wrist_yaw_link``).
    Also drives GraspGoalCurriculum's reach-quality accumulation for this step
    (see grasp_goal_curriculum.py) -- this is the one place that side effect is
    triggered, so this term must be declared in RewardsCfg for the ramp to move.
    """
    GraspGoalCurriculum.accumulate(env, robot_cfg=robot_cfg, object_cfg=object_cfg)

    term = _get_grasp_goal_term(env)
    goal_pos_w, _, _ = live_goal_pose(env, term, object_cfg)

    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    dist = torch.linalg.norm(palm_pos - goal_pos_w, dim=-1)
    r_palm = 1.0 - torch.tanh(dist / std)

    ramp = getattr(env, "_ggc_ramp", 0.0)
    return r_palm * ramp


def grasp_goal_hand_config_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    gate_std: float = 0.20,
    q_std: float = 0.5,
) -> torch.Tensor:
    r"""Pull the 6 proximal hand joints toward the live goal configuration,
    gated by palm proximity (so the policy doesn't curl its fingers from
    across the room) and ramped by the same training-time curriculum as
    ``grasp_goal_palm_reward``:

    .. math::
        w_g \cdot \underbrace{(1 - \tanh(\lVert p_{palm}-p^*\rVert / \text{gate\_std}))}_{\text{proximity gate}}
        \cdot (1 - \tanh(\lVert q - q^* \rVert / q\_std))

    ``robot_cfg`` must resolve both the palm body (index 0 of ``body_ids``,
    ``right_wrist_yaw_link``) and the 6 hand joints in ``HAND_JOINT_NAMES`` order
    (``joint_ids``).
    """
    term = _get_grasp_goal_term(env)
    goal_pos_w, _, goal_joint_q = live_goal_pose(env, term, object_cfg)

    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    gate_dist = torch.linalg.norm(palm_pos - goal_pos_w, dim=-1)
    gate = 1.0 - torch.tanh(gate_dist / gate_std)

    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    q_dist = torch.linalg.norm(joint_pos - goal_joint_q, dim=-1)
    r_hand = 1.0 - torch.tanh(q_dist / q_std)

    ramp = getattr(env, "_ggc_ramp", 0.0)
    return gate * r_hand * ramp
