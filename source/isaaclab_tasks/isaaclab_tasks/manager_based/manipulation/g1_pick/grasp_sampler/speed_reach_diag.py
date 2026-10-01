"""Standalone diagnostic (not a standing feature): measures, for a trained g1_pick
checkpoint over many parallel envs, (1) per-joint velocity statistics for the 13
policy-controlled joints (to size a velocity_limit_sim clamp), (2) the closest the
fingertip-centroid-to-cube distance ever gets per episode, and (3) DWELL TIME -- the
longest run of CONSECUTIVE steps each episode spends under each distance threshold,
mirroring sustained_reach_termination's own counter logic exactly so this directly
answers "would the termination have fired, and by how much did it miss."

Usage:
  python speed_reach_diag.py --task Isaac-G1-Pick-Empty-Play-v0 --num_envs 64 \
    --steps 500 --checkpoint <path/model_499.pt> --headless --device cuda:0
"""
import argparse
import os
import sys

_RSL_RL_DIR = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
                           "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL_RL_DIR)

from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Empty-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
_RIGHT_HAND_ACTION_JOINTS = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
    "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
]
_RIGHT_HAND_BODIES = [
    "right_wrist_yaw_link", "R_thumb_distal", "R_index_intermediate",
    "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate",
]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["target_object"]
    dev = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    arm_ids = [robot.joint_names.index(n) for n in _RIGHT_ARM_JOINTS]
    hand_ids = [robot.joint_names.index(n) for n in _RIGHT_HAND_ACTION_JOINTS]
    controlled_ids = arm_ids + hand_ids
    other_ids = [i for i in range(len(robot.joint_names)) if i not in controlled_ids]
    fingertip_ids = [robot.body_names.index(n) for n in _RIGHT_HAND_BODIES[1:]]

    arm_vel_hist = []
    hand_vel_hist = []
    other_vel_hist = []
    min_dist = torch.full((num_envs,), float("inf"), device=dev)
    episode_min_dists = []  # one entry per completed episode (any env), in meters

    dwell_threshes_cm = [5.0, 7.0, 10.0]
    dwell_counters = {t: torch.zeros(num_envs, dtype=torch.long, device=dev) for t in dwell_threshes_cm}
    dwell_max_run = {t: torch.zeros(num_envs, dtype=torch.long, device=dev) for t in dwell_threshes_cm}
    episode_max_dwell = {t: [] for t in dwell_threshes_cm}  # one entry per completed episode, in steps

    obs = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        jv = robot.data.joint_vel  # (num_envs, num_joints)
        arm_vel_hist.append(jv[:, arm_ids].abs().cpu().numpy())
        hand_vel_hist.append(jv[:, hand_ids].abs().cpu().numpy())
        other_vel_hist.append(jv[:, other_ids].abs().cpu().numpy())

        tips = robot.data.body_pos_w[:, fingertip_ids]
        centroid = tips.mean(dim=1)
        dist = torch.linalg.norm(centroid - obj.data.root_pos_w, dim=-1)
        min_dist = torch.minimum(min_dist, dist)

        # Dwell-time tracking: same consecutive-run logic as sustained_reach_termination
        # itself (reset to 0 the instant the env steps back outside the threshold).
        for t in dwell_threshes_cm:
            within = dist < (t / 100.0)
            dwell_counters[t] = torch.where(within, dwell_counters[t] + 1, torch.zeros_like(dwell_counters[t]))
            dwell_max_run[t] = torch.maximum(dwell_max_run[t], dwell_counters[t])

        done_mask = dones.bool()
        if done_mask.any():
            # record each just-completed episode's closest approach BEFORE resetting the
            # running min, then reset it so the next episode starts counting fresh.
            episode_min_dists.extend(min_dist[done_mask].cpu().tolist())
            min_dist[done_mask] = float("inf")
            for t in dwell_threshes_cm:
                episode_max_dwell[t].extend(dwell_max_run[t][done_mask].cpu().tolist())
                dwell_max_run[t][done_mask] = 0
                dwell_counters[t][done_mask] = 0

        if step % 50 == 0:
            print(f"[diag] step {step}/{args_cli.steps}", flush=True)

    arm_vel = np.concatenate([a.reshape(-1) for a in arm_vel_hist])
    hand_vel = np.concatenate([a.reshape(-1) for a in hand_vel_hist])
    other_vel = np.concatenate([a.reshape(-1) for a in other_vel_hist])
    # episodes still in-flight at the end of the run (never hit `dones`) have a valid
    # running min too -- include them so short runs still yield a usable distribution.
    still_running = min_dist.cpu().numpy()
    still_running = still_running[np.isfinite(still_running)]
    d = np.concatenate([np.array(episode_min_dists, dtype=np.float64), still_running])

    print("\n===== JOINT VELOCITY (rad/s, |value|) =====", flush=True)
    for name, arr in [("right_arm (7 joints)", arm_vel), ("right_hand (6 action joints)", hand_vel),
                       ("all other joints (legs/waist/left side)", other_vel)]:
        print(f"{name}: mean={arr.mean():.3f}  p50={np.percentile(arr,50):.3f}  "
              f"p95={np.percentile(arr,95):.3f}  p99={np.percentile(arr,99):.3f}  max={arr.max():.3f}", flush=True)

    print(f"\n===== FINGERTIP-CENTROID -> CUBE-CENTER MIN DISTANCE (cm) "
          f"[{len(episode_min_dists)} completed episodes + {len(still_running)} in-flight] =====", flush=True)
    if len(d) > 0:
        dc = d * 100.0
        print(f"n_samples={len(dc)}  mean={dc.mean():.2f}  min={dc.min():.2f}  "
              f"p10={np.percentile(dc,10):.2f}  p50={np.percentile(dc,50):.2f}  p90={np.percentile(dc,90):.2f}", flush=True)
        for thresh_cm in [5.0, 7.0, 10.0, 15.0]:
            frac = (dc < thresh_cm).mean()
            print(f"  fraction that ever got within {thresh_cm:.0f}cm: {frac*100:.1f}%", flush=True)
    else:
        print("no distance samples collected", flush=True)

    print(f"\n===== DWELL TIME: longest CONSECUTIVE steps spent under each threshold, per episode "
          f"(30Hz control -> steps/30 = seconds) =====", flush=True)
    for t in dwell_threshes_cm:
        # episodes still in-flight at the end of the run haven't had their dwell run
        # closed out by a reset -- include their max-so-far too, same as the distance stat.
        ep_vals = np.array(episode_max_dwell[t] + dwell_max_run[t].cpu().tolist(), dtype=np.float64)
        if len(ep_vals) == 0:
            print(f"  {t:.0f}cm: no samples", flush=True)
            continue
        print(f"  {t:.0f}cm: n={len(ep_vals)}  mean={ep_vals.mean():.1f} steps  "
              f"p50={np.percentile(ep_vals,50):.1f}  p90={np.percentile(ep_vals,90):.1f}  "
              f"max={ep_vals.max():.0f} steps ({ep_vals.max()/30.0:.2f}s)", flush=True)
        for hold in [10, 20, 30]:
            frac = (ep_vals >= hold).mean()
            print(f"    fraction of episodes reaching >= {hold} consecutive steps "
                  f"(would fire sustained_reach at hold_steps={hold}): {frac*100:.1f}%", flush=True)

    # simulation_app.close() hangs indefinitely in headless mode after the run finishes --
    # same workaround already used by play_with_goal_markers.py in this directory. os._exit
    # skips stdio flushing though, so flush explicitly first or the stats above never reach
    # a redirected/piped log (discovered the hard way: first run produced an empty log).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
