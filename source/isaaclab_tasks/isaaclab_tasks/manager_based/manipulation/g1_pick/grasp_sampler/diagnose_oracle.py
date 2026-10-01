"""Instrumented variant of collect_demos.py's oracle: runs a handful of episodes and
prints, per phase transition, exactly WHERE the scripted grasp-execution fails --
IK/reachability (does the palm reach the goal pose?), finger closure (do the coupled
joints reach their commanded closed configuration?), or the lift itself (does a
genuinely-closed hand still fail to lift?). Built after collect_demos.py returned 0/512
successes, to find out which of those three it is.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, compute_pose_error
from isaaclab_tasks.manager_based.manipulation.g1_pick.mdp.grasp_goal import _get_goal_term

ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINTS = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
    "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
]
MIM_JOINTS = [
    "R_thumb_intermediate_joint", "R_thumb_distal_joint", "R_index_intermediate_joint",
    "R_middle_intermediate_joint", "R_ring_intermediate_joint", "R_pinky_intermediate_joint",
]
ARM_SCALE, HAND_SCALE = 0.3, 0.5
PREGRASP_OFFSET = 0.08
LIFT_HEIGHT = 0.35
APPROACH, DESCEND, CLOSE, LIFT = 0, 1, 2, 3
PHASE_BUDGET = {APPROACH: 90, DESCEND: 60, CLOSE: 45, LIFT: 999}
CLOSE_STEPS = 45
_OBJ_INIT_Z = 0.845


def main():
    env_cfg = parse_env_cfg("Isaac-G1-Pick-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-G1-Pick-v0", cfg=env_cfg)
    obs_dict, _ = env.reset()
    uenv = env.unwrapped
    device = uenv.device
    N = args_cli.num_envs

    robot = uenv.scene["robot"]
    obj = uenv.scene["target_object"]
    goal_term = _get_goal_term(uenv)
    palm_idx = robot.body_names.index("R_hand_base_link")
    fingertip_ids = [robot.body_names.index(n) for n in
                     ["R_thumb_distal", "R_index_intermediate", "R_middle_intermediate",
                      "R_ring_intermediate", "R_pinky_intermediate"]]
    arm_ids = torch.tensor([robot.joint_names.index(n) for n in ARM_JOINTS], device=device)
    hand_ids = torch.tensor([robot.joint_names.index(n) for n in HAND_JOINTS], device=device)
    mim_ids = torch.tensor([robot.joint_names.index(n) for n in MIM_JOINTS], device=device)
    q_def = robot.data.default_joint_pos.clone()
    fixed_base = robot.is_fixed_base
    jac_body_idx = palm_idx - 1 if fixed_base else palm_idx

    phase = torch.zeros(N, dtype=torch.long, device=device)
    phase_step = torch.zeros(N, dtype=torch.long, device=device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).expand(N, 3)
    obs = obs_dict["policy"]
    n_done = 0
    prev_phase = phase.clone()

    def ik_action(goal_pos, goal_quat):
        palm_pos = robot.data.body_pos_w[:, palm_idx]
        palm_quat = robot.data.body_quat_w[:, palm_idx]
        pos_err, ax_err = compute_pose_error(palm_pos, palm_quat, goal_pos, goal_quat,
                                             rot_error_type="axis_angle")
        err = torch.cat([pos_err, 0.5 * ax_err], dim=1).unsqueeze(-1)
        jac = robot.root_physx_view.get_jacobians()[:, jac_body_idx, :, :]
        J = jac[:, :, arm_ids]
        JT = J.transpose(1, 2)
        lam = 0.05
        dq = (JT @ torch.linalg.solve(J @ JT + lam**2 * torch.eye(6, device=device), err)).squeeze(-1)
        q_des = robot.data.joint_pos[:, arm_ids] + torch.clamp(dq, -0.15, 0.15)
        return torch.clamp((q_des - q_def[:, arm_ids]) / ARM_SCALE, -1.0, 1.0), pos_err.norm(dim=1)

    step_count = 0
    while n_done < args_cli.episodes and step_count < 3000:
        goal_pos = goal_term.goal_pos_w
        goal_quat = goal_term.goal_quat_w
        approach = quat_apply(goal_quat, y_axis)
        pregrasp_pos = goal_pos - PREGRASP_OFFSET * approach
        lift_pos = goal_pos + torch.tensor([0.0, 0.0, LIFT_HEIGHT], device=device)

        tgt_pos = torch.where(phase.unsqueeze(1) == APPROACH, pregrasp_pos,
                  torch.where(phase.unsqueeze(1) >= LIFT, lift_pos, goal_pos))
        arm_act, ik_err = ik_action(tgt_pos, goal_quat)

        alpha = (phase_step.float() / CLOSE_STEPS).clamp(0, 1).unsqueeze(1)
        q_pre, q_grasp, q_sq = goal_term.goal_hand_q_pre, goal_term.goal_hand_q, goal_term.goal_hand_q_squeeze
        close_q = torch.where(alpha < 0.6, q_pre + (alpha / 0.6) * (q_grasp - q_pre),
                              q_grasp + ((alpha - 0.6) / 0.4) * (q_sq - q_grasp))
        fing_q = torch.where(phase.unsqueeze(1) < CLOSE, q_pre,
                 torch.where(phase.unsqueeze(1) == CLOSE, close_q, q_sq))
        hand_act = torch.clamp((fing_q - q_def[:, hand_ids]) / HAND_SCALE, -1.0, 1.0)
        action = torch.cat([arm_act, hand_act], dim=1)

        obs_dict, _, terminated, truncated, _ = env.step(action)
        obs = obs_dict["policy"]
        dones = (terminated | truncated)

        # print on phase transitions for env 0
        if phase[0].item() != prev_phase[0].item() or step_count % 30 == 0:
            names = {0: "APPROACH", 1: "DESCEND", 2: "CLOSE", 3: "LIFT"}
            hand_q_actual = robot.data.joint_pos[0, hand_ids].cpu().numpy()
            hand_q_target = fing_q[0].cpu().numpy()
            mim_actual = robot.data.joint_pos[0, mim_ids].cpu().numpy()
            tip_pos = robot.data.body_pos_w[0, fingertip_ids]
            tip_dist = (tip_pos - obj.data.root_pos_w[0]).norm(dim=1).cpu().numpy() * 100 - 2.5
            cube_z = obj.data.root_pos_w[0, 2].item()
            print(f"[diag] step={step_count} env0 phase={names[phase[0].item()]} "
                  f"ik_err={ik_err[0].item()*100:.2f}cm cube_z={cube_z*100:.2f}cm "
                  f"(init={_OBJ_INIT_Z*100:.1f})", flush=True)
            print(f"       hand_q_actual={np.round(hand_q_actual,2)}", flush=True)
            print(f"       hand_q_target={np.round(hand_q_target,2)}", flush=True)
            print(f"       mim_actual   ={np.round(mim_actual,2)}", flush=True)
            print(f"       fingertip_surface_dist_cm={np.round(tip_dist,2)}", flush=True)
        prev_phase = phase.clone()

        palm_pos = robot.data.body_pos_w[:, palm_idx]
        err = torch.norm(palm_pos - tgt_pos, dim=1)
        phase_step += 1
        budget = torch.tensor([PHASE_BUDGET[APPROACH], PHASE_BUDGET[DESCEND],
                               PHASE_BUDGET[CLOSE], PHASE_BUDGET[LIFT]], device=device)[phase]
        advance = ((phase == APPROACH) & ((err < 0.025) | (phase_step > budget))) | \
                  ((phase == DESCEND) & ((err < 0.015) | (phase_step > budget))) | \
                  ((phase == CLOSE) & (phase_step > budget))
        phase = torch.where(advance, phase + 1, phase)
        phase_step = torch.where(advance, torch.zeros_like(phase_step), phase_step)

        done_ids = torch.nonzero(dones).squeeze(-1).tolist()
        if 0 in done_ids:
            lifted = uenv.termination_manager.get_term("target_lifted")[0].item()
            print(f"[diag] === env0 episode ended, lifted={lifted} ===", flush=True)
            n_done += 1
        for i in done_ids:
            phase[i] = APPROACH
            phase_step[i] = 0

        step_count += 1

    os._exit(0)


if __name__ == "__main__":
    main()
