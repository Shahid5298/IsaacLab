"""Verifies the cube spawns resting stably on the new riser asset at the expected height,
and stays there (doesn't sink through / bounce off) over a handful of zero-action steps."""
import argparse
import faulthandler
import os
import sys

_RSL_RL_DIR = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
                           "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL_RL_DIR)

from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Empty-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs

    dump_log = open("/tmp/riser_faulthandler_dump.log", "w")
    faulthandler.enable(file=dump_log)
    faulthandler.dump_traceback_later(45, exit=False, file=dump_log)

    print("[verify_riser] creating env...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    faulthandler.cancel_dump_traceback_later()
    print("[verify_riser] env created, calling reset()...", flush=True)
    obs, _ = env.reset()
    print("[verify_riser] reset() returned", flush=True)

    obj = env.unwrapped.scene["target_object"]
    cube_z = obj.data.root_pos_w[:, 2].clone()
    print("\n===== IMMEDIATELY POST-RESET =====")
    print("cube_z (cm):", (cube_z * 100).tolist())
    print("expected (cm): ~86.0 (0.845 + 0.015)")

    zero_actions = torch.zeros(env.unwrapped.num_envs, 13, device=env.unwrapped.device)
    for i in range(30):
        obs, rew, terminated, truncated, extras = env.step(zero_actions)

    cube_z2 = obj.data.root_pos_w[:, 2].clone()
    print("\n===== AFTER 30 ZERO-ACTION STEPS =====")
    print("cube_z (cm):", (cube_z2 * 100).tolist())
    print("max |cube_z change| (cm):", ((cube_z2 - cube_z).abs() * 100).max().item())
    print("any NaN:", torch.isnan(cube_z2).any().item())

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
