"""GROUND TRUTH: how many NEWTONS is the hand actually applying to the cube?

Every diagnostic so far has been geometric and they disagree with each other:
  - contact_check.py tracks 5 fingertip LINK ORIGINS -> reports positive gaps (not touching)
  - play_with_goal_markers' penetration check uses ~40 approximate collision SPHERES from
    inspire_right.yml -> reports the hand 1.7cm INSIDE the cube
Both cannot be right, and neither measures force. A sphere model inflated relative to the
real USD colliders would produce exactly this contradiction.

This probe sidesteps geometry entirely by asking PhysX for contact forces, filtered to the
cube. Force is the thing that actually determines whether a lift is possible: friction
needs normal force, and no amount of proximity substitutes for it.

Requires activate_contact_sensors=True on the robot spawn, which the training config
deliberately leaves False -- set here on THIS SCRIPT'S OWN cfg instance only.
"""
import argparse, os, sys
_RSL = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
                    "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL)
from isaaclab.app import AppLauncher
import cli_args
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Empty-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None: env_cfg.sim.device = args_cli.device

    # contact reporting must be on at SPAWN time for the sensor to see anything
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/R_.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetObject"],
        history_length=1, update_period=0.0,
    )

    rp = retrieve_file_path(args_cli.checkpoint); env_cfg.log_dir = os.path.dirname(rp)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(rp); policy = runner.get_inference_policy(device=env.unwrapped.device)

    sensor = env.unwrapped.scene["hand_contact"]
    print("\ncontact-reporting bodies:", len(sensor.body_names))
    print(sensor.body_names)

    per_step_total, per_body_max, touching_steps = [], None, 0
    obs = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            a = policy(obs); obs, _, _, _ = env.step(a)
        fm = sensor.data.force_matrix_w        # (N, B, F, 3) bodies x filtered targets
        if fm is None:
            print("force_matrix_w is None -- filtering not active"); break
        mag = torch.linalg.norm(fm, dim=-1).squeeze(-1)   # (N, B)
        per_step_total.append(mag.sum(dim=1).cpu().numpy())
        bmax = mag.max(dim=0).values.cpu().numpy()
        per_body_max = bmax if per_body_max is None else np.maximum(per_body_max, bmax)
        touching_steps += int((mag.sum(dim=1) > 1e-3).any())

    tot = np.concatenate([p[None] for p in per_step_total]) if per_step_total else np.zeros((1,1))
    ss = slice(len(tot)//3, len(tot))
    print("\n" + "=" * 70)
    print("HAND -> CUBE CONTACT FORCE (PhysX ground truth)")
    print("=" * 70)
    print(f"steps with ANY contact : {touching_steps}/{args_cli.steps}")
    print(f"total |F| steady-state : mean={tot[ss].mean():.4f} N   max={tot[ss].max():.4f} N")
    print(f"per-env mean |F|       : {np.round(tot[ss].mean(axis=0)[:8],4)} N  (first 8 envs)")
    if per_body_max is not None:
        order = np.argsort(-per_body_max)[:10]
        print("\ntop bodies by peak |F|:")
        for i in order:
            if per_body_max[i] > 1e-4:
                print(f"  {sensor.body_names[i]:28s} {per_body_max[i]:8.3f} N")
        if per_body_max.max() <= 1e-4:
            print("  (none -- every body reports zero force against the cube)")
    print("\nInterpretation: a 5cm cube of ~0.1kg needs roughly 1 N of grip force per")
    print("contact to lift against gravity with mu~0.5. Near-zero total force means the")
    print("hand is NOT pressing the cube, whatever the geometric diagnostics claim.")
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
