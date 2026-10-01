"""Decisive test: does the trained grasp actually HOLD the cube?

Runs the policy normally so it settles into its grasp, then OVERRIDES the arm action to
drive the shoulder upward while leaving the policy's hand action untouched, and tracks the
cube. This separates two very different failure modes that look identical in training:

  (a) EXPLORATION problem -- the grip works, the policy just never tries lifting. Then the
      cube rises with the hand here, and the fix is restoring arm exploration.
  (b) GRIP problem -- the fingers are near the cube but exert no usable normal force, so
      there is no friction to lift with. Then the hand rises and the cube stays put, and no
      amount of arm exploration or lift reward can ever work.

Phase 4 v7 ended with the best contact of the project (thumb 0.50cm, 64/64 envs inside the
1cm buffer) and still target_lifted=0.0022, which is what motivated this.
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
parser.add_argument("--settle_steps", type=int, default=140)
parser.add_argument("--lift_steps", type=int, default=140)
parser.add_argument("--mode", type=str, default="lift", choices=["lift", "squeeze"],
                    help="lift = force the arm up; squeeze = slam the hand shut")
parser.add_argument("--squeeze_cmd", type=float, default=3.0,
                    help="value written to all 6 hand actions in squeeze mode")
parser.add_argument("--lift_delta", type=float, default=1.5,
                    help="added to the shoulder_pitch ACTION (scale 0.3 => 1.5 ~ 0.45 rad)")
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

_OBJ_INIT_Z = 0.845
_TIPS = ["R_thumb_distal", "R_index_intermediate", "R_middle_intermediate",
         "R_ring_intermediate", "R_pinky_intermediate"]

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

    robot = env.unwrapped.scene["robot"]; obj = env.unwrapped.scene["target_object"]
    palm = robot.body_names.index("R_hand_base_link")
    tips = [robot.body_names.index(n) for n in _TIPS]

    obs = env.get_observations()
    for _ in range(args_cli.settle_steps):
        with torch.inference_mode():
            a = policy(obs); obs, _, _, _ = env.step(a)

    cz0 = obj.data.root_pos_w[:, 2].clone()
    cp0 = obj.data.root_pos_w.clone()
    hand_ids = [robot.joint_names.index(n) for n in
                ["R_thumb_proximal_yaw_joint","R_thumb_proximal_pitch_joint","R_index_proximal_joint",
                 "R_middle_proximal_joint","R_ring_proximal_joint","R_pinky_proximal_joint"]]
    pz0 = robot.data.body_pos_w[:, palm, 2].clone()
    d0 = (torch.linalg.norm(robot.data.body_pos_w[:, tips] - obj.data.root_pos_w.unsqueeze(1), dim=-1) - 0.025)
    print("\n===== AFTER SETTLING (policy in its grasp) =====")
    print(f"cube z            : {cz0.mean()*100:.2f} cm  (init {_OBJ_INIT_Z*100:.1f})")
    print(f"palm z            : {pz0.mean()*100:.2f} cm")
    print(f"fingertip surf cm : {np.round(d0.mean(0).cpu().numpy()*100,2)}  [thumb,index,middle,ring,pinky]")

    if args_cli.mode == "squeeze":
        # Drive every hand joint hard shut, well past the policy's own command. If the
        # fingers are genuinely in contact, over-closing MUST disturb the cube (squirt it
        # out, roll it, or push it). If the cube does not move at all, the fingers are not
        # touching it -- no geometry model or sphere approximation involved in that
        # conclusion.
        print(f"\n===== NOW FORCING THE HAND SHUT (hand actions := +{args_cli.squeeze_cmd}) =====")
        for i in range(args_cli.lift_steps):
            with torch.inference_mode():
                a = policy(obs); a = a.clone()
                a[:, 7:] = args_cli.squeeze_cmd          # all 6 hand joints slammed closed
                obs, _, _, _ = env.step(a)
            if (i + 1) % 35 == 0:
                cz = obj.data.root_pos_w[:, 2]
                disp = torch.linalg.norm(obj.data.root_pos_w - obj.data.root_pos_w.new_tensor([0.,0.,0.]) - cp0, dim=1)
                hq = robot.data.joint_pos[:, hand_ids]
                names = ["contact_thumb","contact_index","contact_middle","contact_ring","contact_pinky"]
                tot, nfin = 0.0, 0
                for nm in names:
                    if nm in env.unwrapped.scene.sensors:
                        fm = env.unwrapped.scene.sensors[nm].data.force_matrix_w
                        if fm is not None:
                            m = torch.linalg.norm(fm, dim=-1).sum(dim=(1,2))
                            tot += float(m.mean()); nfin += int((m > 0.1).float().mean() > 0.5)
                fstr = f"filtered_total {tot:.3f}N  fingers_in_contact~{nfin}/5"
                print(f"  step {i+1:3d}  cube moved {disp.mean()*100:+.3f} cm (max {disp.max()*100:+.3f})  "
                      f"cube dz {(cz-cz0).mean()*100:+.3f} cm  |  CONTACT {fstr}")
        disp = torch.linalg.norm(obj.data.root_pos_w - cp0, dim=1)
        print("\n" + "=" * 66)
        print(f"cube total displacement under hard squeeze: mean {disp.mean()*100:.3f} cm, max {disp.max()*100:.3f} cm")
        if disp.mean() * 100 > 0.3:
            print("VERDICT: fingers ARE touching -- squeezing visibly disturbs the cube.")
            print("         Contact exists; the issue is force direction/magnitude, not absence.")
        else:
            print("VERDICT: fingers are NOT touching -- slamming the hand fully shut does")
            print("         not move the cube at all. There is a real air gap; every")
            print("         'penetration' reading is a sphere-model artifact.")
        sys.stdout.flush(); os._exit(0)

    print(f"\n===== NOW FORCING THE ARM UP (shoulder_pitch action += {args_cli.lift_delta}) =====")
    for i in range(args_cli.lift_steps):
        with torch.inference_mode():
            a = policy(obs)
            a = a.clone(); a[:, 0] += args_cli.lift_delta      # arm joint 0 = shoulder_pitch
            obs, _, _, _ = env.step(a)
        if (i + 1) % 35 == 0:
            cz = obj.data.root_pos_w[:, 2]; pz = robot.data.body_pos_w[:, palm, 2]
            print(f"  step {i+1:3d}  palm rose {(pz-pz0).mean()*100:+.2f} cm   "
                  f"cube rose {(cz-cz0).mean()*100:+.2f} cm   "
                  f"cube max {(cz-cz0).max()*100:+.2f} cm   "
                  f"envs with cube >1cm up: {int(((cz-cz0)>0.01).sum())}/{args_cli.num_envs}")

    cz = obj.data.root_pos_w[:, 2]; pz = robot.data.body_pos_w[:, palm, 2]
    palm_rise = (pz - pz0).mean().item() * 100
    cube_rise = (cz - cz0).mean().item() * 100
    print("\n" + "=" * 66)
    print(f"palm rose : {palm_rise:+.2f} cm")
    print(f"cube rose : {cube_rise:+.2f} cm")
    ratio = cube_rise / palm_rise if abs(palm_rise) > 0.1 else 0.0
    print(f"follow ratio (cube/palm): {ratio:.3f}")
    if palm_rise < 0.5:
        print("VERDICT: arm did not actually rise -- test inconclusive, raise --lift_delta")
    elif ratio > 0.5:
        print("VERDICT: GRIP HOLDS -> this is an EXPLORATION problem. Restore arm std.")
    elif ratio > 0.1:
        print("VERDICT: PARTIAL grip -- cube follows weakly, likely slipping.")
    else:
        print("VERDICT: NO GRIP -- hand rises, cube stays. Contact exists but exerts no")
        print("         usable force. No lift reward or arm exploration can fix this;")
        print("         the grasp itself must actually close on the cube.")
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
