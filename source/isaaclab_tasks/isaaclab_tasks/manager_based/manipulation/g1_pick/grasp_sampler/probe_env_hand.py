"""Diagnostic: same measurement as measure_coupling.py, but through the REAL
Isaac-G1-Pick-v0 env's action pipeline (gym.make + env.step + InspireMimicAction),
rather than a bare scene with direct joint targets. Comparing the two is what
originally showed the software mimic ratios in InspireMimicAction are inert on
this USD (Misc./UltraDex.md S3.1, Discovery 1) -- the slave joints have no
position drive, only an underdamped PhysxMimicJointAPI constraint.

Usage:
    conda activate env_isaaclab
    source _isaac_sim/setup_conda_env.sh
    cd grasp_sampler/
    python probe_env_hand.py
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--sweep_steps", type=int, default=5)
parser.add_argument("--settle_steps", type=int, default=180)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401 - triggers gym.register for Isaac-G1-Pick-v0

SLAVE_JOINTS = [
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    "R_index_intermediate_joint", "R_middle_intermediate_joint",
    "R_ring_intermediate_joint", "R_pinky_intermediate_joint",
]
# 6-dim hand action slice within the 13-dim policy action (7 arm + 6 hand, see
# ActionsCfg.right_hand_action in g1_pick_env_cfg.py); arm actions held at 0.
HAND_ACTION_SLICE = slice(7, 13)


def main() -> None:
    env = gym.make("Isaac-G1-Pick-v0", num_envs=1)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    slave_ids = [robot.find_joints(n)[0][0] for n in SLAVE_JOINTS]

    print(f"{'commanded_action':>20} | " + " | ".join(f"{n:>28}" for n in SLAVE_JOINTS))
    for frac in np.linspace(-1.0, 1.0, args.sweep_steps):
        action = torch.zeros(1, 13, device=env.unwrapped.device)
        action[:, HAND_ACTION_SLICE] = frac
        for _ in range(args.settle_steps):
            env.step(action)
        settled = robot.data.joint_pos[0, slave_ids].cpu().numpy()
        print(f"{frac:>20.2f} | " + " | ".join(f"{v:>28.4f}" for v in settled))

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
