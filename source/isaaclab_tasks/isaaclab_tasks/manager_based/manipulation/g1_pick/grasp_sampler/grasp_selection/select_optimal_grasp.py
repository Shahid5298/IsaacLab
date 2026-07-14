"""Select the mechanically optimal grasp from the 32-sample UltraDexGrasp library
using FSWO force-closure scoring (Misc./optimizer.md), replacing the previous
nearest-palm heuristic.

Run once, offline, on any machine (CPU is sufficient -- no GPU/Isaac Lab needed):

    cd grasp_sampler/
    python grasp_selection/select_optimal_grasp.py \\
        --library grasp_dataset/cube_5cm_grasps_valid.npz \\
        --lam 1.0 \\
        --out grasp_selection/scores.json

Then set `_FIXED_GRASP_IDX` in g1_pick_env_cfg.py to the printed ``best_idx``.
This script's output (scores.json) is not read back by mdp/grasp_goal.py --
only the chosen index matters for the current single-fixed-grasp training mode.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from check_grasps_offline import CUBE_HALF_EDGE, fk_tips
from grasp_selection.fswo import batch_fswo_scores

_STAGE_GRASP = 1  # 0=pregrasp, 1=grasp (fingertips on surface, used for FSWO), 2=squeeze


def cube_contact(p: np.ndarray, cube_half: float = CUBE_HALF_EDGE) -> np.ndarray:
    """Project a fingertip position onto the nearest cube face; return the
    6-vector [contact_x, contact_y, contact_z, normal_x, normal_y, normal_z]
    with the normal pointing INTO the cube (Misc./optimizer.md S3.3)."""
    k = int(np.argmax(np.abs(p) / cube_half))
    contact = p.copy()
    contact[k] = np.sign(p[k]) * cube_half
    normal = np.zeros(3)
    normal[k] = -np.sign(p[k])
    return np.concatenate([contact, normal])


def build_contacts(grasps: np.ndarray) -> np.ndarray:
    """(N, 1, 3, 13) grasp_pose array -> (N, 5, 6) FSWO contact array.

    Uses the grasp (stage-1) pose and the synthesis-URDF FK in
    check_grasps_offline.py. grasp_pose's palm_pos/palm_quat are already
    expressed in the cube frame as BODex synthesized them (cube centered at
    the origin, axis-aligned) -- T_usdbase_urdfbase is a SEPARATE correction
    for the URDF-base vs USD-R_hand_base_link axis convention, applied online
    in mdp/grasp_goal.py when integrating with the simulator, and is
    deliberately NOT applied here: cube_contact's axis-aligned-cube-at-origin
    assumption only holds in the frame grasp_pose already uses.
    """
    n = grasps.shape[0]
    contacts = np.zeros((n, 5, 6))
    for g in range(n):
        stage1 = grasps[g, 0, _STAGE_GRASP, :]  # (13,)
        palm_pos, palm_quat, joint_q = stage1[:3], stage1[3:7], stage1[7:]
        tips = fk_tips(palm_pos, palm_quat, joint_q, T_cal=None)
        for i, tip in enumerate(tips):
            contacts[g, i] = cube_contact(tip)
    return contacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="grasp_dataset/cube_5cm_grasps_valid.npz")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="grasp_selection/scores.json")
    args = ap.parse_args()

    data = np.load(args.library, allow_pickle=True)
    grasps = data["grasp_pose"]  # (32, 1, 3, 13)

    contacts = build_contacts(grasps)  # (32, 5, 6)
    scores = batch_fswo_scores(contacts, lam=args.lam)  # (32,)

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            "Non-finite FSWO score(s) -- a fingertip likely landed far from the cube "
            "surface. Check the URDF path/geometry and T_cal loading before trusting "
            "the ranking (see grasp_sampler/ASSET_SETUP.md)."
        )

    best_idx = int(np.argmax(scores))
    ranking = np.argsort(scores)[::-1].tolist()

    result = {
        "best_idx": best_idx,
        "best_score": float(scores[best_idx]),
        "ranked_indices": ranking,
        "scores": scores.tolist(),
        "lam": args.lam,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Best grasp: index {best_idx}  (FSWO = {scores[best_idx]:.5f})")
    print(f"Ranking (best -> worst): {ranking}")
    if len(scores) > 15:
        print(f"Previous nearest-palm heuristic index 15: FSWO = {scores[15]:.5f}")
    print(f"Wrote {args.out}")
    print(f"\nNext: set _FIXED_GRASP_IDX = {best_idx} in g1_pick_env_cfg.py")


if __name__ == "__main__":
    main()
