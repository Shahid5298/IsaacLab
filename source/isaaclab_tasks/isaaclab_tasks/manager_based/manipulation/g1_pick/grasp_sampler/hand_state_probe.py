"""Standalone diagnostic: reports the ACTUAL internal state of the grasp_goal_hand reward
at a converged checkpoint -- the palm gate, the hand joint-space error q_err, the per-joint
breakdown, and where that error sits on the tanh curve (value AND gradient). Answers "is
the hand-closing reward saturated?" with numbers instead of inference from episode sums.
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
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.manipulation.g1_pick.mdp.grasp_goal import _get_goal_term

_OBJ_INIT_Z = 0.845


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
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
    term = _get_goal_term(env.unwrapped)

    q_errs, gates, cube_dz, per_joint = [], [], [], []
    obs = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        term.update_live_goals(env.unwrapped)
        palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
        palm_d = torch.norm(palm_pos - term.goal_pos_w, dim=1)
        gate = (1.0 - torch.tanh(palm_d / 0.20)).clamp(min=0.0)
        q = robot.data.joint_pos[:, term._hand_joint_ids]
        dq = q - term.goal_hand_q
        q_err = torch.norm(dq, dim=1)

        gates.append(gate.cpu().numpy())
        q_errs.append(q_err.cpu().numpy())
        per_joint.append(dq.abs().cpu().numpy())
        cube_dz.append((obj.data.root_pos_w[:, 2] - _OBJ_INIT_Z).cpu().numpy())

    gates = np.concatenate(gates); q_errs = np.concatenate(q_errs)
    per_joint = np.concatenate(per_joint); cube_dz = np.concatenate(cube_dz)

    # steady state = last 2/3 of the episode (after the reach transient)
    n = args_cli.steps * args_cli.num_envs
    ss = slice(n // 3, n)
    g, qe = gates[ss].mean(), q_errs[ss].mean()

    print("\n" + "=" * 68)
    print("===== grasp_goal_hand INTERNAL STATE (steady state) =====")
    print("=" * 68)
    print(f"palm gate            : {g:.4f}   (1.0 = palm exactly at goal)")
    print(f"q_err (hand joints)  : {qe:.4f} rad   L2 over {per_joint.shape[1]} hand joints")
    print(f"  per-joint |dq| mean: {np.array2string(per_joint[ss].mean(axis=0), precision=3)}")
    _ids = term._hand_joint_ids
    _ids = _ids.tolist() if hasattr(_ids, "tolist") else list(_ids)
    print(f"  hand joint names   : {[robot.joint_names[i] for i in _ids]}")

    print("\n----- where that error sits on the tanh curve -----")
    print(f"{'q_std':>7} {'u=err/std':>10} {'value':>9} {'x weight':>10} {'d(rew)/d(q_err)':>17}")
    for q_std, w in [(0.3, 4.0), (0.5, 1.0), (0.5, 4.0), (0.8, 4.0), (1.0, 4.0), (1.2, 4.0)]:
        u = qe / q_std
        val = g * (1.0 - np.tanh(u))
        grad = w * g * (1.0 / q_std) * (1.0 - np.tanh(u) ** 2)
        tag = "  <-- CURRENT" if (q_std, w) == (0.3, 4.0) else ("  <-- before p23v2" if (q_std, w) == (0.5, 1.0) else "")
        print(f"{q_std:>7.1f} {u:>10.2f} {val:>9.4f} {w*val:>10.4f} {grad:>17.4f}{tag}")

    print("\n----- cube height above init (m) -----")
    print(f"mean={cube_dz.mean():.6f}  max={cube_dz.max():.6f}  "
          f"p99={np.percentile(cube_dz, 99):.6f}")
    print(f"steps with cube >1cm up: {(cube_dz > 0.01).sum()}/{len(cube_dz)}")
    print(f"SUCCESS needs +0.289 m.  Best ever seen here: {cube_dz.max():.4f} m")

    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
