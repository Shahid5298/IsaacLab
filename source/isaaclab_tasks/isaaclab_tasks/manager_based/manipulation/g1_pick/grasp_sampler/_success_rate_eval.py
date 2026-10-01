"""Headless, no-video success-rate sweep across multiple g1_pick checkpoints.

For each checkpoint: runs --num_envs in parallel for --steps control steps and counts,
per step, which envs terminated (any termination term) vs. which of those terminations
were specifically the `target_lifted` term (a genuine successful pick). Success rate =
successes / total completed episodes over that window.

Usage:
  python _success_rate_eval.py --task Isaac-G1-Pick-Play-v0 --num_envs 1024 --steps 800 \
    --device cuda:1 --run_dir /path/to/run_dir \
    --checkpoints 5450 5500 5550 ... 6200
"""
import argparse
import os
import sys

_RSL_RL_DIR = os.path.join(os.environ.get("ISAACLAB_PATH", "/home/umar/IsaacLab"),
                           "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL_RL_DIR)

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Success-rate sweep across g1_pick checkpoints.")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel envs for the sweep.")
parser.add_argument("--steps", type=int, default=800, help="Control steps to run per checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Play-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--run_dir", type=str, default=None, help="Dir containing model_<N>.pt files.")
parser.add_argument("--checkpoints", type=int, nargs="*", default=[],
                    help="Checkpoint iteration numbers within --run_dir (model_<N>.pt).")
parser.add_argument("--checkpoint_paths", type=str, nargs="*", default=[],
                    help="Arbitrary absolute .pt checkpoint paths, evaluated in addition to "
                         "--checkpoints (e.g. a working_models/ reference checkpoint outside "
                         "any run_dir). Labeled by filename in the output.")
parser.add_argument("--tray_distractors", action="store_true", default=False,
                    help="Restore the ORIGINAL tray-clutter distractor layout (positions right "
                         "around the cube's own spawn, physically in the hand's approach path) "
                         "instead of the table-margin layout the live env_cfg's __post_init__ "
                         "currently applies for no-interference training.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.tray_distractors:
        # Original per-distractor spawn positions from SceneCfg (before the V2 __post_init__
        # override relocates them to the table margins) -- ring 1 sits directly in the
        # approach path to the cube at [0.35, 0.0]. Same tray height all ten originally shared.
        _ORIGINAL_TRAY_POS = {
            1: (0.38, 0.08), 2: (0.38, -0.08), 3: (0.42, 0.0), 4: (0.32, 0.12), 5: (0.32, -0.12),
            6: (0.48, 0.12), 7: (0.48, -0.12), 8: (0.28, 0.0), 9: (0.52, 0.0), 10: (0.35, 0.18),
        }
        _ORIGINAL_TRAY_Z = 0.845
        for i in range(1, 11):
            d_cfg = getattr(env_cfg.scene, f"distractor_{i}")
            x, y = _ORIGINAL_TRAY_POS[i]
            d_cfg.init_state.pos = (x, y, _ORIGINAL_TRAY_Z)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    term_mgr = env.unwrapped.termination_manager

    # unified (label, path) list -- numbered run_dir checkpoints first, then arbitrary paths
    to_eval = []
    for ckpt in args_cli.checkpoints:
        to_eval.append((str(ckpt), os.path.join(args_cli.run_dir, f"model_{ckpt}.pt")))
    for p in args_cli.checkpoint_paths:
        to_eval.append((os.path.splitext(os.path.basename(p))[0], p))

    results = []
    for ckpt, ckpt_path in to_eval:
        if not os.path.isfile(ckpt_path):
            print(f"[eval] SKIP ckpt {ckpt}: {ckpt_path} not found", flush=True)
            continue

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations()
        successes = 0
        completed = 0
        for step in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            done_mask = dones.bool()
            success_mask = term_mgr.get_term("target_lifted") & done_mask
            completed += int(done_mask.sum().item())
            successes += int(success_mask.sum().item())
            if (step + 1) % 200 == 0:
                rate = (successes / completed * 100.0) if completed > 0 else 0.0
                print(f"[eval] ckpt {ckpt}  step {step + 1}/{args_cli.steps}  "
                      f"episodes so far={completed}  successes so far={successes}  "
                      f"rate={rate:.1f}%", flush=True)

        rate = (successes / completed * 100.0) if completed > 0 else float("nan")
        print(f"[RESULT] ckpt {ckpt}: episodes={completed} successes={successes} "
              f"success_rate={rate:.2f}%", flush=True)
        results.append((ckpt, completed, successes, rate))

    print("\n=== FINAL SUMMARY ===", flush=True)
    for ckpt, completed, successes, rate in results:
        print(f"ckpt {ckpt:>5}: {successes:6d}/{completed:6d} episodes = {rate:6.2f}% success", flush=True)

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
