# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for G1 picking environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_target_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
):
    """Reset target object position on the tray.

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        pose_range: Dictionary with position ranges for x, y, z.
        object_cfg: Scene entity for the target object.
    """
    target_object: RigidObject = env.scene[object_cfg.name]
    
    # Sample random positions within range
    num_resets = len(env_ids)
    pos_x = torch.rand(num_resets, device=env.device) * (pose_range["x"][1] - pose_range["x"][0]) + pose_range["x"][0]
    pos_y = torch.rand(num_resets, device=env.device) * (pose_range["y"][1] - pose_range["y"][0]) + pose_range["y"][0]
    pos_z = torch.rand(num_resets, device=env.device) * (pose_range["z"][1] - pose_range["z"][0]) + pose_range["z"][0]
    
    # Set positions
    target_object.data.root_pos_w[env_ids, 0] = env.scene.env_origins[env_ids, 0] + pos_x
    target_object.data.root_pos_w[env_ids, 1] = env.scene.env_origins[env_ids, 1] + pos_y
    target_object.data.root_pos_w[env_ids, 2] = env.scene.env_origins[env_ids, 2] + pos_z
    
    # Reset velocities
    target_object.data.root_lin_vel_w[env_ids] = 0.0
    target_object.data.root_ang_vel_w[env_ids] = 0.0
    
    # Write to simulation
    target_object.write_root_state_to_sim(target_object.data.root_state_w[env_ids], env_ids)


def reset_clutter_based_on_difficulty(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distractor_names: list[str],
    tray_surface_height: float = 0.845,
    hidden_height: float = -5.0,
    tray_x_half: float = 0.14,
    tray_y_half: float = 0.17,
    table_center_x: float = 0.5,
) -> None:
    """Place distractor objects on the tray or hide them based on curriculum difficulty.

    Clutter activation schedule (difficulty 0–60):
    - distractor 0: appears at difficulty ≥ 30
    - distractor 1: appears at difficulty ≥ 40
    - distractor 2: appears at difficulty ≥ 50

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        distractor_names: Scene entity names of the distractor rigid objects.
        tray_surface_height: Z of tray surface (from env origin).
        hidden_height: Z to hide inactive distractors (below table).
        tray_x_half: Half-size of tray in x for random placement.
        tray_y_half: Half-size of tray in y for random placement.
        table_center_x: X offset from env origin to table/tray centre.
    """
    # Get per-environment difficulty from the curriculum manager.
    # After init, term_cfg.func holds the PickingCurriculumScheduler instance.
    difficulties = torch.zeros(env.num_envs, device=env.device)
    try:
        cm = env.curriculum_manager
        idx = cm._term_names.index("picking_curriculum")
        difficulties = cm._term_cfgs[idx].func.current_difficulties
    except Exception:
        pass  # No curriculum yet or name mismatch → default to no clutter

    diff_for_ids = difficulties[env_ids]  # (num_resets,)
    num_resets = len(env_ids)

    for dist_idx, name in enumerate(distractor_names):
        obj: RigidObject = env.scene[name]
        default_states = obj.data.default_root_state[env_ids].clone()
        orientations = default_states[:, 3:7]  # keep default orientation (identity)

        # Each distractor activates at a progressively higher difficulty threshold
        activation_threshold = 30.0 + dist_idx * 10.0
        is_active = diff_for_ids >= activation_threshold  # (num_resets,) bool

        # Random on-tray positions
        rand_x = (torch.rand(num_resets, device=env.device) - 0.5) * 2.0 * tray_x_half
        rand_y = (torch.rand(num_resets, device=env.device) - 0.5) * 2.0 * tray_y_half

        pos = torch.zeros(num_resets, 3, device=env.device)
        pos[:, 0] = env.scene.env_origins[env_ids, 0] + table_center_x + rand_x
        pos[:, 1] = env.scene.env_origins[env_ids, 1] + rand_y
        active_z = env.scene.env_origins[env_ids, 2] + tray_surface_height
        hidden_z = env.scene.env_origins[env_ids, 2] + hidden_height
        pos[:, 2] = torch.where(is_active, active_z, hidden_z)

        velocities = torch.zeros(num_resets, 6, device=env.device)

        obj.write_root_pose_to_sim(torch.cat([pos, orientations], dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(velocities, env_ids=env_ids)


_RSI_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
_RSI_HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
    "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
]


def reset_arm_hand_object_with_rsi(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    rsi_probability: float,
    arm_ref_pos: list[float],
    hand_ref_pos: list[float],
    shoulder_pitch_cm_per_rad: float,
    height_range_cm: tuple[float, float],
    obj_xy_range: dict[str, tuple[float, float]],
    obj_init_z: float,
    obj_yaw: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
):
    """Reference-state initialization (RSI) -- 2026-08-21, REWARD_CURRICULUM_LOG.md
    "[Phase 4 v2 Round 5]". For a random fraction of resetting envs, instead of the
    normal cold-start reset (arm/hand near default, cube resting on the table), starts
    the arm/hand DIRECTLY in the policy's own converged grasp joint configuration
    (measured empirically via capture_grasp_joints.py, median over 64 real rollouts, NOT
    hand-derived IK) with an added shoulder_pitch offset that raises the hand by a
    randomly sampled height (shoulder_pitch confirmed empirically, via arm_swing_xyz.py,
    as the dominant z-control joint -- ~10x more z-sensitive than shoulder_yaw), and
    spawns the cube to match that elevated hand position. Gives the value function/policy
    direct experience of high-lift-reward states instead of relying on exploration noise
    alone to ever reach them from a cold start -- the last resort after four straight
    rounds of reward-shaping and exploration-noise changes alone failed to produce a
    single measurable lift.

    MUST run (per its position in EventsCfg) after reset_right_arm/reset_right_hand/
    reset_target_object (this function overrides their result for the RSI-selected
    subset of envs) and BEFORE sample_grasp_goal (which reads the object's post-reset
    pose to compute the goal -- an object pose set by THIS function must be visible to
    it, or the goal would be computed against the wrong, pre-RSI cube position).

    The remaining (1 - rsi_probability) fraction of envs are left untouched -- this is a
    MIX of RSI and cold-start resets, not 100% RSI, so the policy keeps practicing the
    full reach+grasp sequence from scratch too, not just completing an already-started
    grasp."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    num_resets = len(env_ids)
    is_rsi = torch.rand(num_resets, device=env.device) < rsi_probability
    rsi_ids = env_ids[is_rsi]
    n = len(rsi_ids)
    if n == 0:
        return

    arm_ids = [robot.joint_names.index(name) for name in _RSI_ARM_JOINT_NAMES]
    hand_ids = [robot.joint_names.index(name) for name in _RSI_HAND_JOINT_NAMES]

    height_cm = torch.rand(n, device=env.device) * (height_range_cm[1] - height_range_cm[0]) + height_range_cm[0]
    # shoulder_pitch's measured sensitivity: delta=-1.0 (action units) -> palm +3.21cm over
    # a 40-step sustained swing (see REWARD_CURRICULUM_LOG.md). Action-to-joint-radian scale
    # is folded into shoulder_pitch_cm_per_rad at the call site (params), so this stays a
    # plain linear scaling here -- negative offset raises the hand, per that measurement.
    shoulder_pitch_offset = -(height_cm / shoulder_pitch_cm_per_rad)

    arm_pos = torch.tensor(arm_ref_pos, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    arm_pos[:, 0] += shoulder_pitch_offset
    hand_pos = torch.tensor(hand_ref_pos, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    arm_vel = torch.zeros_like(arm_pos)
    hand_vel = torch.zeros_like(hand_pos)

    # clamp to the articulation's own soft joint limits -- safety net in case a sampled
    # height pushes shoulder_pitch's offset past what the joint can actually reach.
    arm_limits = robot.data.soft_joint_pos_limits[rsi_ids][:, arm_ids]
    arm_pos = arm_pos.clamp(arm_limits[..., 0], arm_limits[..., 1])
    hand_limits = robot.data.soft_joint_pos_limits[rsi_ids][:, hand_ids]
    hand_pos = hand_pos.clamp(hand_limits[..., 0], hand_limits[..., 1])

    robot.write_joint_state_to_sim(arm_pos, arm_vel, joint_ids=arm_ids, env_ids=rsi_ids)
    robot.write_joint_state_to_sim(hand_pos, hand_vel, joint_ids=hand_ids, env_ids=rsi_ids)

    # cube: same xy jitter range as the normal reset_target_object, but z matches the
    # elevated hand height and orientation matches the same fixed +45deg yaw used
    # everywhere else in this curriculum (see reset_target_object's own comment).
    root_states = obj.data.default_root_state[rsi_ids].clone()
    x_range = obj_xy_range.get("x", (0.0, 0.0))
    y_range = obj_xy_range.get("y", (0.0, 0.0))
    rand_xy = torch.rand(n, 2, device=env.device)
    rand_xy[:, 0] = rand_xy[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    rand_xy[:, 1] = rand_xy[:, 1] * (y_range[1] - y_range[0]) + y_range[0]

    positions = root_states[:, 0:3] + env.scene.env_origins[rsi_ids]
    positions[:, 0:2] += rand_xy
    positions[:, 2] = obj_init_z + height_cm / 100.0

    zeros_n = torch.zeros(n, device=env.device)
    yaw_delta = quat_from_euler_xyz(zeros_n, zeros_n, torch.full((n,), obj_yaw, device=env.device))
    orientations = quat_mul(root_states[:, 3:7], yaw_delta)

    obj.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=rsi_ids)
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=rsi_ids)
