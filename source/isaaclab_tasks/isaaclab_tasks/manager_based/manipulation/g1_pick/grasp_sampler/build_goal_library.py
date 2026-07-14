"""Stage 3: filter the re-centered raw grasps down to the goal library.

Keeps a grasp iff (Misc./UltraDex.md S3.4):
    1. geometric soundness: thumb + >= 2 fingers within 2 cm of the cube surface
       in the synthesis model (check_grasps_offline.geometric_soundness).
    2. tray compatibility: nothing reaches below the cube's underside
       (check_grasps_offline.tray_compatible) -- a cube on a tray cannot be
       grasped from below.
Keeps the best MAX_LIBRARY_SIZE by contact quality (mean surface distance,
ascending) and writes cube_5cm_grasps_valid.npz -- the ONE file training reads.

Run (no GPU/Isaac Lab needed -- CPU is fine):
    cd grasp_sampler/
    python build_goal_library.py
    python grasp_selection/select_optimal_grasp.py --library grasp_dataset/cube_5cm_grasps_valid.npz
"""

from __future__ import annotations

import os

import numpy as np

from check_grasps_offline import cube_surface_dist, fk_tips, geometric_soundness, tray_compatible

# ── Tunable constants ──────────────────────────────────────────────────────────
IN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_dataset", "cube_5cm_grasps_recentered.npz")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_dataset", "cube_5cm_grasps_valid.npz")
MAX_LIBRARY_SIZE = 32
STAGE_GRASP = 1  # 0=pregrasp, 1=grasp, 2=squeeze
# ────────────────────────────────────────────────────────────────────────────────


def main() -> None:
    data = np.load(IN_PATH, allow_pickle=True)
    grasp_pose = data["grasp_pose"]  # (N, 1, 3, 13)
    T_cal = data["T_usdbase_urdfbase"]  # (4, 4)
    n = grasp_pose.shape[0]

    kept_indices: list[int] = []
    mean_dists: list[float] = []
    for g in range(n):
        stage1 = grasp_pose[g, 0, STAGE_GRASP, :]
        palm_pos, palm_quat, joint_q = stage1[:3], stage1[3:7], stage1[7:]
        tips = fk_tips(palm_pos, palm_quat, joint_q, T_cal=None)

        if not geometric_soundness(tips):
            continue
        if not tray_compatible(tips, palm_pos):
            continue

        kept_indices.append(g)
        # cube_surface_dist is SIGNED (negative = inside the cube) -- take the mean
        # of the absolute value, not the raw signed value. Sorting the signed mean
        # ascending would rank a grasp with fingers deeply penetrating the cube
        # (very negative) as "better contact" than one lightly touching the surface
        # (near zero), which is backwards: closest-to-surface (either side) is best.
        mean_dists.append(float(np.mean([abs(cube_surface_dist(t)) for t in tips])))

    print(f"{len(kept_indices)} / {n} grasps passed geometric soundness + tray compatibility")
    if not kept_indices:
        raise RuntimeError(
            "No grasps survived filtering. If this is from placeholder/incorrect URDF "
            "geometry, do not proceed to FSWO/training -- fix grasp_sampler/ASSET_SETUP.md "
            "prerequisites first."
        )

    order = np.argsort(mean_dists)  # ascending: closest-to-surface first (see note above)
    best = [kept_indices[i] for i in order[:MAX_LIBRARY_SIZE]]

    valid_pose = grasp_pose[best]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        grasp_pose=valid_pose,
        T_usdbase_urdfbase=T_cal,
        source_indices=np.array(best),
    )
    print(f"Wrote {len(best)} grasps to {OUT_PATH}")


if __name__ == "__main__":
    main()
