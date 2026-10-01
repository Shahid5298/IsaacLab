"""Directly inspects post-reset state to verify RSI produces physically sane, elevated
states -- no policy involved, just env.reset() + a few physics steps to let anything
unstable settle or reveal itself (explosion, NaN, cube falling straight through, etc.)."""
import argparse
import faulthandler
import os
import sys
import threading

_RSL_RL_DIR = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
                           "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL_RL_DIR)

from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Empty-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
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

_PALM_BODY = "right_wrist_yaw_link"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    # force RSI to fire on every env for this check
    env_cfg.events.rsi_arm_hand_object.params["rsi_probability"] = 1.0

    # Self-dump all thread stacks after 45s if still running -- works from inside the
    # process itself via signal handlers, no ptrace/SYS_PTRACE needed (unlike py-spy/gdb),
    # so it isn't blocked by this container's missing capability.
    dump_log = open("/tmp/rsi_faulthandler_dump.log", "w")
    faulthandler.enable(file=dump_log)
    faulthandler.dump_traceback_later(45, exit=False, file=dump_log)
    print(f"[verify_rsi] faulthandler armed (45s), main thread ident={threading.get_ident()}", flush=True)

    print("[verify_rsi] creating env...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    faulthandler.cancel_dump_traceback_later()
    print("[verify_rsi] env created, calling reset()...", flush=True)
    obs, _ = env.reset()
    print("[verify_rsi] reset() returned", flush=True)

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["target_object"]
    palm_id = robot.body_names.index(_PALM_BODY)

    palm_z = robot.data.body_pos_w[:, palm_id, 2]
    cube_z = obj.data.root_pos_w[:, 2]
    print("\n===== IMMEDIATELY POST-RESET (RSI forced to 100%) =====")
    print("palm_z (cm):", (palm_z * 100).tolist())
    print("cube_z (cm):", (cube_z * 100).tolist())
    print("cube_z - 84.5cm (height above resting, cm):", ((cube_z - 0.845) * 100).tolist())
    print("any NaN in palm_z:", torch.isnan(palm_z).any().item())
    print("any NaN in cube_z:", torch.isnan(cube_z).any().item())

    # step a few times with zero action to let PhysX resolve/settle, check for blowup
    zero_actions = torch.zeros(env.unwrapped.num_envs, 13, device=env.unwrapped.device)
    for i in range(20):
        obs, rew, terminated, truncated, extras = env.step(zero_actions)
        print(f"[verify_rsi] step {i} done", flush=True)

    palm_z2 = robot.data.body_pos_w[:, palm_id, 2]
    cube_z2 = obj.data.root_pos_w[:, 2]
    print("\n===== AFTER 20 ZERO-ACTION STEPS =====")
    print("palm_z (cm):", (palm_z2 * 100).tolist())
    print("cube_z (cm):", (cube_z2 * 100).tolist())
    print("max |palm_z change| (cm):", ((palm_z2 - palm_z).abs() * 100).max().item())
    print("max |cube_z change| (cm):", ((cube_z2 - cube_z).abs() * 100).max().item())
    print("reward this step - min/mean/max:", rew.min().item(), rew.mean().item(), rew.max().item())
    print("any NaN in reward:", torch.isnan(rew).any().item())

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
