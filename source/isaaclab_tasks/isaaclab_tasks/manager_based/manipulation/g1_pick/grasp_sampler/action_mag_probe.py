"""What magnitude of ARM actions does the trained policy actually output, and how far do
the resulting joint offsets get from the default posture? Distinguishes two very different
root causes for the palm stalling ~9.4cm short of the grasp goal:
  (a) ACTION-SCALE-LIMITED -- policy is already pushing |action| near/above 1 and still
      cannot express the offsets the goal needs (an unclamped IK oracle needs up to
      1.377 rad = |action| 4.59 at scale=0.3). Fix: raise the arm action scale.
  (b) PENALTY-LIMITED -- policy outputs small actions because action_smoothness/joint_speed
      make big ones unprofitable. Fix: rebalance those penalties.
"""
import argparse, os, sys
_RSL = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
                    "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL)
from isaaclab.app import AppLauncher
import cli_args
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Empty-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import numpy as np, torch, gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.manipulation.g1_pick.mdp.grasp_goal import _get_goal_term

ARM = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
       "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
ARM_SCALE = 0.3

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None: env_cfg.sim.device = args_cli.device
    rp = retrieve_file_path(args_cli.checkpoint); env_cfg.log_dir = os.path.dirname(rp)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(rp); policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    term = _get_goal_term(env.unwrapped)
    arm_ids = [robot.joint_names.index(n) for n in ARM]
    q_def = robot.data.default_joint_pos.clone()

    acts, offs, palmd = [], [], []
    obs = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            a = policy(obs)
            obs, _, dones, extras = env.step(a)
        acts.append(a[:, :7].cpu().numpy())
        offs.append((robot.data.joint_pos[:, arm_ids] - q_def[:, arm_ids]).cpu().numpy())
        term.update_live_goals(env.unwrapped)
        palm = robot.data.body_pos_w[:, term._palm_body_idx]
        palmd.append(torch.norm(palm - term.goal_pos_w, dim=1).cpu().numpy())

    acts = np.concatenate(acts); offs = np.concatenate(offs); palmd = np.concatenate(palmd)
    n = acts.shape[0]; ss = slice(n // 3, n)
    print("\n" + "="*74)
    print("ARM ACTION MAGNITUDE (steady state)")
    print("="*74)
    print(f"per-joint mean |action| : {np.round(np.abs(acts[ss]).mean(axis=0),3)}")
    print(f"per-joint  max |action| : {np.round(np.abs(acts[ss]).max(axis=0),3)}")
    print(f"overall mean |action|   : {np.abs(acts[ss]).mean():.3f}")
    print(f"overall  max |action|   : {np.abs(acts[ss]).max():.3f}")
    print(f"fraction of |action|>1.0: {(np.abs(acts[ss])>1.0).mean():.2%}")
    print(f"fraction of |action|>2.0: {(np.abs(acts[ss])>2.0).mean():.2%}")
    print()
    print("RESULTING ARM JOINT OFFSET FROM DEFAULT (rad)")
    print(f"per-joint mean |offset| : {np.round(np.abs(offs[ss]).mean(axis=0),3)}")
    print(f"per-joint  max |offset| : {np.round(np.abs(offs[ss]).max(axis=0),3)}")
    print(f"overall  max |offset|   : {np.abs(offs[ss]).max():.3f} rad")
    print(f"  (IK oracle needs up to 1.377 rad to reach the grasp pose)")
    print()
    print(f"palm->goal distance     : mean={palmd[ss].mean()*100:.2f}cm  min={palmd[ss].min()*100:.2f}cm")
    print(f"  (unclamped IK oracle reaches ~3.0cm)")
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
