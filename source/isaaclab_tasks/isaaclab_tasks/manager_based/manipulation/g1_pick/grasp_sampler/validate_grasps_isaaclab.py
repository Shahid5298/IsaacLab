"""Stage 2: frame calibration + empirical grip re-centering, measured against the
REAL simulator rather than trusted from documentation (Misc./UltraDex.md's meta-
lesson). Runs in `env_isaaclab` (this repo's normal training env), headless, one
G1+Inspire-hand robot per raw grasp (grid layout, no RL, direct joint control).

Usage:
    conda activate env_isaaclab
    source _isaac_sim/setup_conda_env.sh
    cd grasp_sampler/
    python validate_grasps_isaaclab.py --grasp_file grasp_dataset/cube_5cm_grasps.npz

Produces grasp_dataset/cube_5cm_grasps_recentered.npz:
    grasp_pose         : (N, 1, 3, 13), re-expressed relative to the re-centered cube
    T_usdbase_urdfbase : (4, 4) Kabsch-fit calibration transform
    cube_displacement  : (N,) diagnostic only, NOT the filter (Misc./UltraDex.md S3.4) --
                         mid-air hold is a coin-flip of the USD's underdamped slave-joint
                         mimic constraint (Discovery 1), not a property of the grasp.

Two details are load-bearing (Misc./UltraDex.md S7.2):
    - the body is held by PD TARGETS only; writing joint *states* mid-episode
      re-excites the underdamped PhysX mimic constraint on the slave joints.
    - the script ends with os._exit(0): simulation_app.close() can hang for a
      long time in headless mode (Misc./UltraDex.md problem #7).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# ── CLI (must run before Isaac Sim app launch, per Isaac Lab standalone-script convention) ──
parser = argparse.ArgumentParser()
parser.add_argument("--grasp_file", default="grasp_dataset/cube_5cm_grasps.npz")
parser.add_argument("--out", default="grasp_dataset/cube_5cm_grasps_recentered.npz")
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ── Everything below needs the simulation app running before it can be imported ──
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_grasps_offline import fk_tips, link_origins  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot_cfg import G1_INSPIRE_CFG  # noqa: E402

# ── Tunable constants ──────────────────────────────────────────────────────────
CUBE_SIZE = 0.05
CUBE_PARK_OFFSET = (0.0, 0.0, 5.0)  # parked far above the grid during Pass A
GRID_SPACING = 2.0
CALIBRATION_JOINT_Q = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])  # arbitrary non-singular pose
# (USD link name, URDF link name) pairs for the Kabsch fit (Misc./UltraDex.md S3.3).
# Real names confirmed 2026-07-13: USD side from tools/hand_structure_reference.txt,
# URDF side by reading the actual file at check_grasps_offline.INSPIRE_URDF_PATH.
# NOTE: the USD has no "wrist" concept inside a hand-only URDF, so unlike an earlier
# draft of this list, there is no pairing for right_wrist_yaw_link -- both sides are
# anchored at the hand's own base link instead (see the anchor fix below).
CALIBRATION_LINK_PAIRS = [
    ("R_hand_base_link", "hand_base_link"),
    ("R_thumb_proximal_base", "thumb_proximal_base"),
    ("R_thumb_proximal", "thumb_proximal"),
    ("R_thumb_intermediate", "thumb_intermediate"),
    ("R_thumb_distal", "thumb_distal"),
    ("R_index_proximal", "index_proximal"),
    ("R_index_intermediate", "index_intermediate"),
    ("R_middle_proximal", "middle_proximal"),
    ("R_middle_intermediate", "middle_intermediate"),
    ("R_ring_proximal", "ring_proximal"),
]
RIGHT_HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint", "R_middle_proximal_joint",
    "R_ring_proximal_joint", "R_pinky_proximal_joint",
]
# ────────────────────────────────────────────────────────────────────────────────


def solve_rigid_transform(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Kabsch/SVD: least-squares rigid transform T such that T @ pts_a ~= pts_b.
    pts_a, pts_b: (M, 3) corresponding point sets. Returns (4, 4) homogeneous T.
    """
    centroid_a, centroid_b = pts_a.mean(axis=0), pts_b.mean(axis=0)
    a0, b0 = pts_a - centroid_a, pts_b - centroid_b
    H = a0.T @ b0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_b - R @ centroid_a
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    residual = float(np.mean(np.linalg.norm((pts_a @ R.T + t) - pts_b, axis=1)))
    print(f"[calibration] Kabsch residual: {residual * 1000:.3f} mm over {len(pts_a)} points")
    return T


@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(CUBE_PARK_OFFSET)),
    )
    plane = sim_utils.GroundPlaneCfg()
    light = sim_utils.DomeLightCfg(intensity=2000.0)


def ramp_to(robot: Articulation, joint_ids: list[int], target: torch.Tensor, sim, scene, steps: int = 60) -> None:
    """Hold posture with PD TARGETS only (never write joint states mid-episode --
    re-excites the underdamped slave-joint mimic constraint, Misc./UltraDex.md S7.2)."""
    start = robot.data.joint_pos[:, joint_ids].clone()
    for i in range(steps):
        alpha = (i + 1) / steps
        q = start + alpha * (target - start)
        robot.set_joint_position_target(q, joint_ids=joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def main() -> None:
    data = np.load(args.grasp_file, allow_pickle=True)
    grasp_pose = data["grasp_pose"]  # (N, 1, 3, 13)
    num_grasps = grasp_pose.shape[0]

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0" if torch.cuda.is_available() else "cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = ValidationSceneCfg(num_envs=num_grasps, env_spacing=GRID_SPACING)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]
    hand_joint_ids = [robot.find_joints(n)[0][0] for n in RIGHT_HAND_JOINT_NAMES]
    hand_base_body_id = robot.find_bodies("R_hand_base_link")[0][0]

    # ── Step 1: frame calibration (Misc./UltraDex.md S3.3) ──────────────────────
    # Both point clouds must be expressed in the hand's own LOCAL frame (position
    # AND rotation), not raw world-frame deltas -- the robot's arm sits at whatever
    # arbitrary default pose robot_cfg.py spawns it at, and Pass A/B never move the
    # arm (only hand_joint_ids). A world-frame delta would conflate that arbitrary
    # arm orientation with the actual USD-vs-URDF axis convention Kabsch is supposed
    # to solve for. quat_apply_inverse(hand_base_quat_w, world_delta) removes it.
    calib_q = torch.tensor(CALIBRATION_JOINT_Q, device=sim.device).unsqueeze(0).expand(num_grasps, -1)
    ramp_to(robot, hand_joint_ids, calib_q, sim, scene, steps=90)

    usd_names = [pair[0] for pair in CALIBRATION_LINK_PAIRS]
    urdf_names = [pair[1] for pair in CALIBRATION_LINK_PAIRS]
    sim_link_ids = [robot.find_bodies(n)[0][0] for n in usd_names]

    hand_base_pos_w = robot.data.body_pos_w[0, hand_base_body_id]  # (3,)
    hand_base_quat_w = robot.data.body_quat_w[0, hand_base_body_id]  # (4,)
    delta_w = robot.data.body_pos_w[0, sim_link_ids] - hand_base_pos_w.unsqueeze(0)  # (10, 3)
    sim_pts = quat_apply_inverse(
        hand_base_quat_w.unsqueeze(0).expand(len(sim_link_ids), -1), delta_w
    ).cpu().numpy()

    # Same joint config through the pure-numpy URDF FK model. link_origins() is
    # root-relative (URDF_ROOT_LINK="base") and already in the URDF's own local
    # frame (no world pose involved) -- re-anchor at "hand_base_link" (index 0, by
    # construction of CALIBRATION_LINK_PAIRS above) to match sim_pts's anchor.
    urdf_pts_root_relative = link_origins(CALIBRATION_JOINT_Q, urdf_names)
    urdf_pts = urdf_pts_root_relative - urdf_pts_root_relative[0]

    T_usdbase_urdfbase = solve_rigid_transform(urdf_pts, sim_pts)

    # ── Step 2: Pass A -- empirical grip re-centering (Misc./UltraDex.md S3.2) ──
    # Same hand-local-frame requirement as Step 1: the pinch center must be expressed
    # relative to R_hand_base_link's own position AND rotation, since it will be
    # combined with grasp_pose's palm_pos, which is itself in a hand-local (cube-
    # frame) convention, not the sim's arbitrary world/arm orientation.
    pinch_centers = np.zeros((num_grasps, 3))
    for stage_name, stage_idx in [("grasp", 1)]:
        joint_q = torch.tensor(grasp_pose[:, 0, stage_idx, 7:13], dtype=torch.float32, device=sim.device)
        ramp_to(robot, hand_joint_ids, joint_q, sim, scene, steps=90)

        thumb_tip_id = robot.find_bodies("R_thumb_distal")[0][0]
        index_tip_id = robot.find_bodies("R_index_intermediate")[0][0]
        middle_tip_id = robot.find_bodies("R_middle_intermediate")[0][0]

        thumb_pos_w = robot.data.body_pos_w[:, thumb_tip_id]
        pinch_pos_w = 0.5 * (robot.data.body_pos_w[:, index_tip_id] + robot.data.body_pos_w[:, middle_tip_id])
        pinch_center_w = 0.5 * (thumb_pos_w + pinch_pos_w)

        hand_base_pos_w_all = robot.data.body_pos_w[:, hand_base_body_id]  # (num_grasps, 3)
        hand_base_quat_w_all = robot.data.body_quat_w[:, hand_base_body_id]  # (num_grasps, 4)
        pinch_centers = quat_apply_inverse(
            hand_base_quat_w_all, pinch_center_w - hand_base_pos_w_all
        ).cpu().numpy()

    correction = np.linalg.norm(pinch_centers, axis=-1)
    print(f"[re-centering] median correction: {np.median(correction) * 100:.2f} cm")

    # ── Step 3: Pass B -- diagnostic-only mid-air hold (NOT the filter, S3.4) ──
    cube_displacement = np.full(num_grasps, np.nan)
    print(
        "[Pass B] Skipped by default -- mid-air hold is a coin-flip of the USD's "
        "underdamped slave-joint constraint (Discovery 1) and is not used as a filter. "
        "Implement here only if you want the diagnostic re-confirmed on this USD revision."
    )

    # ── Step 4: re-express grasps relative to the re-centered cube, save ────────
    recentered_pose = grasp_pose.copy()
    recentered_pose[:, 0, :, 0:3] -= pinch_centers[:, None, :]

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        grasp_pose=recentered_pose,
        T_usdbase_urdfbase=T_usdbase_urdfbase,
        cube_displacement=cube_displacement,
        pinch_center_correction_m=correction,
    )
    print(f"Wrote {out_path}")

    os._exit(0)  # simulation_app.close() can hang for hours in headless mode


if __name__ == "__main__":
    main()
