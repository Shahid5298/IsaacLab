"""Generic Inspire-Hand forward kinematics + cube-surface distance.

Used by build_goal_library.py (geometric-soundness filtering) and
grasp_selection/select_optimal_grasp.py (FSWO contact extraction, Misc./optimizer.md).

This module is intentionally asset-agnostic: it contains no hand geometry of its
own. It evaluates whatever URDF is placed at INSPIRE_URDF_PATH using
pytorch_kinematics (`pip install pytorch-kinematics` -- a lightweight, CPU-only
URDF-FK library, unrelated to the heavy ultradex/BODex CUDA build in
Misc./UltraDex.md S2.1). See grasp_sampler/ASSET_SETUP.md for exactly what that
URDF needs to contain and where to source it from -- this file will raise
FileNotFoundError with that pointer rather than silently fabricate geometry.
"""

from __future__ import annotations

import os

import numpy as np
import torch

try:
    import pytorch_kinematics as pk
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "check_grasps_offline.py needs pytorch_kinematics (`pip install pytorch-kinematics`)."
    ) from exc

# ── Tunable constants ──────────────────────────────────────────────────────────
# Path to the synthesis URDF -- see grasp_sampler/ASSET_SETUP.md for what must be
# placed here before this module is usable. This is the SAME file BODex itself reads
# during synthesis (its robot config's urdf_path is relative to this content root,
# see bodex_config_templates/inspire_right_sim2real.yml) -- deliberately not a separate
# copy, so there is exactly one URDF to keep correct rather than two that can drift apart.
INSPIRE_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ultradex_repo", "third_party", "BODex_api", "src", "bodex", "content",
    "assets", "robot", "inspire_hand", "inspire_hand_right.urdf",
)

# Root link BODex expresses grasp palm poses relative to. CONFIRMED (2026-07-13)
# against the actual file now in place at INSPIRE_URDF_PATH -- a dex_urdf-derived
# model (LICENSE.txt: derived from Inspire Robotics' own official STEP file). Root
# is a zero-size "base" link, offset from "hand_base_link" by a fixed joint.
# NOTE: this file uses bare joint/link names (no "R_"/"L_" prefix) -- different
# convention from the USD, unlike the unitree_ros DFQ file discussed earlier.
URDF_ROOT_LINK = "base"

# End-effector link per finger, policy order [thumb, index, middle, ring, pinky].
# CONFIRMED: this file already has proper zero-size "_tip" links (thumb_tip,
# index_tip, ...), each attached via a fixed joint with a real measured offset from
# its parent's distal/intermediate link -- better than approximating with the
# terminal link itself.
FINGERTIP_LINKS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

# The 6 policy-controlled (actuated) joints, in mdp.grasp_goal.HAND_JOINT_NAMES order.
# CONFIRMED real names in this file (bare, no "R_" prefix -- differs from the USD's
# mdp.grasp_goal.HAND_JOINT_NAMES, which DO have the "R_" prefix; fk_tips() joins the
# two via this list's order, not by name, so no mapping code is needed here).
ACTUATED_JOINT_NAMES = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]

# CONFIRMED (2026-07-13): the file now at INSPIRE_URDF_PATH already has these joints
# baked in as type="fixed" with exactly these origin rpy z-values -- Misc./UltraDex.md
# S7.5's modification #3 (freeze slave joints at empirically measured USD postures)
# is ALREADY DONE in this file. No <mimic> tags remain for them. Kept here only as a
# record for measure_coupling.py / probe_env_hand.py to re-verify against, and in case
# the USD hand is ever changed (which would invalidate these).
FROZEN_SLAVE_POSTURES_RAD = {
    "thumb_intermediate_joint": -0.16,
    "thumb_distal_joint": -0.24,
    "index_intermediate_joint": 1.15,
    "middle_intermediate_joint": 1.15,
    "ring_intermediate_joint": 1.15,
    "pinky_intermediate_joint": 1.15,
}

# CAVEAT (2026-07-13): this file's index/middle/ring/pinky_proximal_joint upper limit
# is 1.47 rad (84.2 deg) vs the USD's actual 1.7 rad (97.40 deg, tools/
# hand_structure_reference.txt) -- BODex will therefore search a smaller-than-real
# proximal joint range. Not a correctness bug (synthesized grasps will just be a
# conservative subset of what the real hand can do), but worth knowing if synthesis
# never proposes a fully-curled grasp. thumb_proximal_yaw/pitch ranges also differ
# slightly (this file: [0, 1.308]/[0, 0.6]; USD: [-0.1, 1.3]/[0, 0.5] rad).

CUBE_HALF_EDGE = 0.025  # m, 5 cm cube
# ────────────────────────────────────────────────────────────────────────────────


_whole_chain = None  # pytorch_kinematics.Chain, exposes every link in the URDF


def _load_whole_chain(urdf_path: str = INSPIRE_URDF_PATH):
    global _whole_chain
    if _whole_chain is not None:
        return _whole_chain
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(
            f"Inspire hand URDF not found at '{urdf_path}'.\n"
            "This repo does not fabricate hand geometry. See grasp_sampler/ASSET_SETUP.md for "
            "what needs to be there and how to verify it before running any part of this "
            "offline pipeline."
        )
    with open(urdf_path, "rb") as f:
        urdf_bytes = f.read()
    _whole_chain = pk.build_chain_from_urdf(urdf_bytes)
    return _whole_chain


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def link_origins(
    joint_q: np.ndarray,
    link_names: list[str],
    urdf_path: str = INSPIRE_URDF_PATH,
) -> np.ndarray:
    """URDF_ROOT_LINK-relative world-frame origins of arbitrary named links, at the
    given 6 actuated joint angles (policy order). General-purpose FK query used by
    validate_grasps_isaaclab.py's frame-calibration step (needs several non-fingertip
    link origins, e.g. R_thumb_proximal, to fit T_usdbase_urdfbase) -- ``fk_tips`` is
    a thin wrapper around this for the 5 fingertip links specifically.

    Args:
        joint_q: (6,) proximal joint angles, policy order (ACTUATED_JOINT_NAMES).
        link_names: link names to query, must exist in the URDF.
        urdf_path: override INSPIRE_URDF_PATH.

    Returns:
        (len(link_names), 3) positions, root-relative.
    """
    chain = _load_whole_chain(urdf_path)
    all_joint_names = chain.get_joint_parameter_names()
    q = np.asarray(joint_q, dtype=np.float64)

    th = torch.zeros(len(all_joint_names), dtype=torch.float64)
    for j, name in enumerate(all_joint_names):
        if name in ACTUATED_JOINT_NAMES:
            th[j] = q[ACTUATED_JOINT_NAMES.index(name)]
    tf = chain.forward_kinematics(th)  # dict: link_name -> pytorch_kinematics.Transform3d

    origins = np.zeros((len(link_names), 3))
    for i, name in enumerate(link_names):
        origins[i] = tf[name].get_matrix()[0, :3, 3].numpy()
    return origins


def fk_tips(
    palm_pos: np.ndarray,
    palm_quat: np.ndarray,
    joint_q: np.ndarray,
    T_cal: np.ndarray | None = None,
    urdf_path: str = INSPIRE_URDF_PATH,
) -> np.ndarray:
    """Fingertip positions for a grasp's palm pose + 6 proximal joint angles.

    Args:
        palm_pos: (3,) palm position, in whatever frame the caller supplies
            (cube frame for library entries, Misc./UltraDex.md S3.1).
        palm_quat: (4,) palm orientation [qw, qx, qy, qz], same frame.
        joint_q: (6,) proximal joint angles, policy order (ACTUATED_JOINT_NAMES).
        T_cal: optional (4, 4) calibration transform (synthesis-URDF base frame
            -> USD R_hand_base_link frame, Misc./UltraDex.md S3.3). Applied
            after FK if given; omit for raw URDF-frame fingertip positions.
        urdf_path: override INSPIRE_URDF_PATH.

    Returns:
        (5, 3) fingertip positions, order [thumb, index, middle, ring, pinky].
    """
    tips = link_origins(joint_q, FINGERTIP_LINKS, urdf_path)

    # FK above is root-relative (URDF_ROOT_LINK); apply the grasp's palm pose.
    R_palm = _quat_to_matrix(np.asarray(palm_quat, dtype=np.float64))
    tips = tips @ R_palm.T + np.asarray(palm_pos, dtype=np.float64)

    if T_cal is not None:
        T_cal = np.asarray(T_cal, dtype=np.float64)
        tips = tips @ T_cal[:3, :3].T + T_cal[:3, 3]

    return tips


def cube_surface_dist(p: np.ndarray, cube_half: float = CUBE_HALF_EDGE) -> float:
    """Signed distance from point ``p`` to the nearest face of an axis-aligned
    cube of half-edge ``cube_half`` centered at the origin. Negative = inside.
    """
    d = np.abs(p) - cube_half
    outside = float(np.linalg.norm(np.clip(d, 0.0, None)))
    inside = float(min(np.max(d), 0.0))
    return outside + inside


def geometric_soundness(
    tips: np.ndarray,
    cube_half: float = CUBE_HALF_EDGE,
    max_surface_dist: float = 0.02,
    min_sound_fingers: int = 2,
) -> bool:
    """Misc./UltraDex.md S3.4 filter (1): thumb + >= min_sound_fingers fingers
    within max_surface_dist of the cube surface. ``tips`` is (5, 3), thumb first.
    """
    dists = np.array([cube_surface_dist(t, cube_half) for t in tips])
    thumb_ok = dists[0] <= max_surface_dist
    fingers_ok = int(np.sum(dists[1:] <= max_surface_dist))
    return bool(thumb_ok and fingers_ok >= min_sound_fingers)


def tray_compatible(tips: np.ndarray, palm_pos: np.ndarray, cube_half: float = CUBE_HALF_EDGE) -> bool:
    """Misc./UltraDex.md S3.4 filter (2): nothing reaches below the cube's
    underside (a cube resting on a tray cannot be grasped from below).
    """
    underside_z = -cube_half
    return bool(np.min(tips[:, 2]) >= underside_z and palm_pos[2] >= underside_z)
