"""Stage 1: BODex grasp synthesis for the Inspire Hand on the 5 cm cube.

Thin driver around UltraDexGrasp's GraspSynthesizer (Misc./UltraDex.md S2.3).
Runs in the `ultradex` conda env (torch 2.11 + cu128, see Misc./UltraDex.md S2.1)
-- NOT the `env_isaaclab` env used everywhere else in this pipeline.

Prerequisites (see grasp_sampler/ASSET_SETUP.md for the full checklist):
    - ultradex_repo/ cloned into this directory (git clone of InternRobotics/UltraDexGrasp,
      with third_party/BODex_api and third_party/pytorch3d built per Misc./UltraDex.md S2.1-2.2).
    - The Inspire Hand taught to BODex under ultradex_repo/third_party/BODex_api/src/bodex/
      content/{assets/robot/inspire_hand, configs/robot/inspire_right.yml,
      configs/robot/hand_pose_transfer/inspire.yml, configs/manip/sim_inspire_sim2real/fc_right.yml}
      -- see bodex_config_templates/ in this directory for draft starting points
      (NOT verified against BODex_api's actual parser -- see ASSET_SETUP.md).
    - ultradex_repo/asset/object_mesh/cube/ with mesh/simplified.obj, urdf/coacd.urdf,
      info/simplified.json for the unit cube (scaled to CUBE_SIZE at synthesis time).

Run:
    conda activate ultradex
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
    cd grasp_sampler/
    python synthesize_inspire_grasps.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ── Tunable constants ──────────────────────────────────────────────────────────
CUBE_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultradex_repo", "asset", "object_mesh", "cube")
CUBE_SIZE = 0.05  # m
CUBE_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)  # axis-aligned at synthesis time
NUM_GRASPS = 100
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_dataset", "cube_5cm_grasps.npz")
# ────────────────────────────────────────────────────────────────────────────────


def main() -> None:
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultradex_repo")
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(
            f"'{repo_dir}' not found. Clone UltraDexGrasp there and complete the Inspire-Hand "
            "porting steps in grasp_sampler/ASSET_SETUP.md before running this script."
        )
    sys.path.insert(0, repo_dir)

    from grasp_synthesizer import GraspSynthesizer  # noqa: E402 - provided by ultradex_repo

    synthesizer = GraspSynthesizer(hand=1, hand_type="inspire", dof=6, num_grasp=NUM_GRASPS)
    grasp_pose = synthesizer.synthesize_grasp(CUBE_ASSET, list(CUBE_QUAT_WXYZ), CUBE_SIZE)
    grasp_pose = np.asarray(grasp_pose, dtype=np.float32)  # (NUM_GRASPS, 1, 3, 13)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        grasp_pose=grasp_pose,
        joint_order=np.array(
            ["thumb_yaw", "thumb_pitch", "index", "middle", "ring", "pinky"]
        ),
        stages=np.array(["pregrasp", "grasp", "squeeze"]),
    )
    print(f"Wrote {grasp_pose.shape[0]} raw grasps to {OUT_PATH}")


if __name__ == "__main__":
    main()
