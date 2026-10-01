# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep every (or every Nth) saved checkpoint in a training run and report
success rate + hand/cube penetration for each, without relaunching Isaac Sim
per checkpoint (only the policy weights are reloaded each iteration).

Answers "where did this run actually plateau/start degrading" directly from
saved checkpoints, instead of reading the noisy per-iteration reward curve.
"""

import argparse
import os
import re
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Sweep checkpoints of a training run: success rate + penetration.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--load_run", type=str, required=True, help="Run folder under logs/rsl_rl/<experiment> to sweep.")
parser.add_argument("--every", type=int, default=4, help="Evaluate every Nth checkpoint file (by iteration order).")
parser.add_argument("--num_steps", type=int, default=1000, help="Control steps to roll out per checkpoint.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

_RIGHT_HAND_BODIES = [
    "right_wrist_yaw_link",
    "R_thumb_distal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_ring_intermediate",
    "R_pinky_intermediate",
]
_CUBE_HALF_EDGE = 0.025  # 5cm cube


def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    B = q.shape[:-1]
    R = torch.zeros(*B, 3, 3, device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def box_sdf_batch(points_w, box_pos_w, box_quat_w, half_edge):
    """points_w: (E,K,3); box_pos_w: (E,3); box_quat_w: (E,4) wxyz. Returns (E,K) signed distance."""
    Rm = quat_to_mat(box_quat_w)  # (E,3,3)
    p_local = torch.einsum("ekc,ecd->ekd", points_w - box_pos_w.unsqueeze(1), Rm)
    d = torch.abs(p_local) - half_edge
    outside = torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1)
    inside = torch.clamp(torch.amax(d, dim=-1), max=0.0)
    return outside + inside  # (E,K)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_dir = os.path.join(log_root_path, args_cli.load_run)
    env_cfg.log_dir = run_dir

    ckpts = [f for f in os.listdir(run_dir) if re.match(r"model_\d+\.pt$", f)]
    ckpts.sort(key=lambda f: int(re.match(r"model_(\d+)\.pt$", f).group(1)))
    ckpts = ckpts[:: args_cli.every]
    if ckpts[-1] != sorted(
        [f for f in os.listdir(run_dir) if re.match(r"model_\d+\.pt$", f)],
        key=lambda f: int(re.match(r"model_(\d+)\.pt$", f).group(1)),
    )[-1]:
        # always include the final checkpoint even if stride skips past it
        all_ckpts = [f for f in os.listdir(run_dir) if re.match(r"model_\d+\.pt$", f)]
        ckpts.append(sorted(all_ckpts, key=lambda f: int(re.match(r"model_(\d+)\.pt$", f).group(1)))[-1])
    print(f"[sweep] {len(ckpts)} checkpoints to evaluate (every {args_cli.every}): {ckpts}")

    # create the env ONCE -- only weights get reloaded per checkpoint, not Isaac Sim itself
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    cube = base_env.scene["target_object"]
    body_ids, _ = robot.find_bodies(_RIGHT_HAND_BODIES)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    results = []  # (iter, success_rate, n_episodes, pen_frac, worst_gap_cm)
    for ckpt_name in ckpts:
        it = int(re.match(r"model_(\d+)\.pt$", ckpt_name).group(1))
        ckpt_path = os.path.join(run_dir, ckpt_name)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=base_env.device)

        term_names = list(base_env.termination_manager.active_terms)
        counts = {name: 0 for name in term_names}
        num_episodes = 0
        gap_min_hist = []

        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args_cli.num_steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                done_mask = dones.bool()
                if int(done_mask.sum().item()) > 0:
                    num_episodes += int(done_mask.sum().item())
                    for name in term_names:
                        just_now = base_env.termination_manager.get_term(name) & done_mask
                        counts[name] += int(just_now.sum().item())

                hand_pos_w = robot.data.body_pos_w[:, body_ids]  # (E,6,3)
                cube_pos_w = cube.data.root_pos_w  # (E,3)
                cube_quat_w = cube.data.root_quat_w  # (E,4)
                gaps = box_sdf_batch(hand_pos_w, cube_pos_w, cube_quat_w, _CUBE_HALF_EDGE)  # (E,6)
                gap_min_hist.append(gaps.min(dim=-1).values.cpu().numpy() * 100.0)  # cm, per env per step

        gap_arr = np.concatenate(gap_min_hist) if gap_min_hist else np.array([0.0])
        success = counts.get("target_lifted", 0)
        succ_rate = 100.0 * success / num_episodes if num_episodes > 0 else float("nan")
        pen_frac = 100.0 * float((gap_arr < 0).sum()) / len(gap_arr)
        worst_gap = float(gap_arr.min())
        results.append((it, succ_rate, num_episodes, pen_frac, worst_gap))
        print(
            f"[sweep] iter {it:>6d}  success={succ_rate:5.1f}% ({success}/{num_episodes})  "
            f"penetrating_steps={pen_frac:5.1f}%  worst_gap={worst_gap:+.2f}cm"
        )

    env.close()

    its = [r[0] for r in results]
    succ = [r[1] for r in results]
    pen = [r[3] for r in results]
    worst = [r[4] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.plot(its, succ, "o-", color="seagreen")
    ax1.set_ylabel("success rate (%)")
    ax1.set_title(f"Checkpoint sweep -- {args_cli.load_run}")
    ax1.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2.plot(its, pen, "o-", color="crimson", label="% steps penetrating")
    ax2b.plot(its, worst, "s--", color="darkorange", label="worst gap (cm)")
    ax2.set_xlabel("training iteration")
    ax2.set_ylabel("% steps penetrating", color="crimson")
    ax2b.set_ylabel("worst gap (cm)", color="darkorange")
    ax2.grid(alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(run_dir, "checkpoint_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n[sweep] graph written -> {out_path}")

    csv_path = os.path.join(run_dir, "checkpoint_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("iteration,success_rate_pct,num_episodes,penetrating_steps_pct,worst_gap_cm\n")
        for r in results:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]},{r[3]:.2f},{r[4]:.3f}\n")
    print(f"[sweep] csv written -> {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
