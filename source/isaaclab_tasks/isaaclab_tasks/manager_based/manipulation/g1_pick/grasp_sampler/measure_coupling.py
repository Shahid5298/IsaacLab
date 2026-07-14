"""Diagnostic: measure where the USD's slave (mimic) finger joints actually settle
in a bare scene, sweeping the 6 proximal (actuated) joints. This is the script that
produced Discovery 1 in Misc./UltraDex.md S3.1 -- re-run it (and probe_env_hand.py)
first if the USD hand is ever edited, since the synthesis URDF's frozen slave-joint
postures (check_grasps_offline.FROZEN_SLAVE_POSTURES_RAD) depend on these numbers.

Usage:
    conda activate env_isaaclab
    source _isaac_sim/setup_conda_env.sh
    cd grasp_sampler/
    python measure_coupling.py
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--sweep_steps", type=int, default=5)
parser.add_argument("--settle_steps", type=int, default=180)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_cfg import G1_INSPIRE_CFG  # noqa: E402

PROXIMAL_JOINTS = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint", "R_middle_proximal_joint",
    "R_ring_proximal_joint", "R_pinky_proximal_joint",
]
SLAVE_JOINTS = [
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    "R_index_intermediate_joint", "R_middle_intermediate_joint",
    "R_ring_intermediate_joint", "R_pinky_intermediate_joint",
]


@configclass
class BareSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    plane = sim_utils.GroundPlaneCfg()
    light = sim_utils.DomeLightCfg(intensity=2000.0)


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    scene = InteractiveScene(BareSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    prox_ids = [robot.find_joints(n)[0][0] for n in PROXIMAL_JOINTS]
    slave_ids = [robot.find_joints(n)[0][0] for n in SLAVE_JOINTS]

    print(f"{'commanded_proximal':>20} | " + " | ".join(f"{n:>28}" for n in SLAVE_JOINTS))
    for frac in np.linspace(0.0, 1.0, args.sweep_steps):
        target = torch.full((1, len(prox_ids)), frac, device=sim.device)
        for _ in range(args.settle_steps):
            robot.set_joint_position_target(target, joint_ids=prox_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        settled = robot.data.joint_pos[0, slave_ids].cpu().numpy()
        print(f"{frac:>20.2f} | " + " | ".join(f"{v:>28.4f}" for v in settled))

    os._exit(0)


if __name__ == "__main__":
    main()
