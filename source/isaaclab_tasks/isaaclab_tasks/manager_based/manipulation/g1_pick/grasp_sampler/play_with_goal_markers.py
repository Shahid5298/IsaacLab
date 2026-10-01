"""Play a trained g1_pick checkpoint WITH the UltraDexGrasp goal grasp visualized.

Draws, every step, in the viewer / recorded video, for every env:
  - a coordinate FRAME at the goal grasp pose (position + orientation from the
    32-grasp library, live-tracked to the cube)
  - a ghost hand at the GOAL pose: the real Inspire-hand visual meshes, FK'd to the
    goal's joint configuration and anchored at the goal root pose, translucent and
    color-tinted (bright red as of 2026-08-21, was green) so it reads clearly against
    the actual (naturally-rendered, opaque) robot hand. No separate ghost is drawn for
    the actual hand -- it's already visible as the real robot mesh.

Also prints the palm→goal distance to the console, and (env 0 only) saves two
diagnostic graphs after the run: palm pose error over the episode, and each
fingertip's distance to its own `grasp_reach_reward` target contact point.

Usage (mirrors play.py; supports the same hydra camera overrides):
  conda activate env_isaaclab && source _isaac_sim/setup_conda_env.sh
  python .../grasp_sampler/play_with_goal_markers.py \
    --task Isaac-G1-Pick-Play-v0 --num_envs 1 --headless --video --video_length 400 \
    --enable_cameras \
    --checkpoint /home/umar/IsaacLab/logs/rsl_rl/g1_pick/2026-07-15_13-49-59/model_6998.pt \
    'env.viewer.origin_type=env' 'env.viewer.eye=[1.5,-1.1,1.4]' 'env.viewer.lookat=[0.35,0.0,1.0]'
"""
import argparse
import json
import os
import sys

# make the rsl_rl helper `cli_args` importable (it lives next to play.py)
_RSL_RL_DIR = os.path.join(os.environ.get("ISAACLAB_PATH", "/home/umar/IsaacLab"),
                           "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL_RL_DIR)

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402  (from the rsl_rl scripts dir)

parser = argparse.ArgumentParser(description="Play g1_pick with UltraDexGrasp goal markers.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Play-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--reach_thresh", type=float, default=0.05,
                    help="Palm-goal distance (m) below which the goal marker turns green.")
parser.add_argument("--video_fps", type=int, default=15,
                    help="Frame rate written into the MP4. Lower = slower playback (sim runs ~30 Hz).")
parser.add_argument("--real_time", action="store_true", default=False, help="Run at real-time speed.")
parser.add_argument("--episodes", type=int, default=0,
                    help="Stop after env 0 completes this many episodes (reset/success/failure), "
                         "instead of running to --video_length regardless. Episode boundaries are "
                         "marked with vertical dashed lines on the diagnostic graphs so multiple "
                         "episodes stay readable. 0 = disabled (old behavior: run to "
                         "--video_length). --video_length still applies as an upper bound.")
parser.add_argument("--single_episode", action="store_true", default=False,
                    help="Shorthand for --episodes 1.")
parser.add_argument("--no_markers", action="store_true", default=False,
                    help="Skip drawing the goal-frame coordinate-axis overlay -- for a 'plain' "
                         "video using this same script's episode-counting, instead of "
                         "approximating episode count via --video_length in the generic play.py. "
                         "Independent of --no_ghost_mesh and the always-on goal_centroid dot -- "
                         "see those for the other two goal-visualization layers.")
parser.add_argument("--no_ghost_mesh", action="store_true", default=False,
                    help="Skip drawing the translucent green ghost-hand mesh at the goal pose "
                         "(added 2026-08-21). Independent of --no_markers (the coordinate frame) "
                         "-- the goal_centroid red dot stays on regardless of either flag.")
parser.add_argument("--no_hold_after_success", action="store_true", default=False,
                    help="By default this script disables the target_lifted termination (in "
                         "THIS script's own env_cfg instance only -- never touches a live "
                         "training run's config) so a successful episode keeps running to the "
                         "natural timeout instead of cutting the instant the cube crosses the "
                         "success height. That's normally what you want for watching playback "
                         "(see the hold, not just the instant of crossing). Pass this flag to "
                         "restore the immediate-termination-on-success behavior instead.")
parser.add_argument("--tray_distractors", action="store_true", default=False,
                    help="Restore the ORIGINAL tray-clutter distractor layout (positions right "
                         "around the cube's own spawn, physically in the hand's approach path) "
                         "instead of the table-margin layout the live env_cfg's __post_init__ "
                         "currently applies for no-interference training. Scoped to THIS "
                         "script's own env_cfg instance only -- never touches a live training "
                         "run's config.")
parser.add_argument("--table_distractors", action="store_true", default=False,
                    help="Move all 10 distractors to the table margins (the 'empty env' / "
                         "no-interference layout used before the live env_cfg default was "
                         "switched back to full tray clutter) -- for an 'empty env' baseline "
                         "comparison. Scoped to THIS script's own env_cfg instance only.")
parser.add_argument("--name_prefix", type=str, default="rl-video",
                    help="Prefix for the recorded video filename (gym.wrappers.RecordVideo "
                         "writes '<name_prefix>-step-0.mp4'). Set this to something unique "
                         "(e.g. include the checkpoint number) when running multiple "
                         "instances of this script in parallel against the same log_dir --"
                         "the default 'rl-video' collides across concurrent runs, and one "
                         "run's rename can steal the file out from under another's (found "
                         "2026-08-19 running two parallel checkpoint-eval batches).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.single_episode:
    args_cli.episodes = max(args_cli.episodes, 1)
sys.argv = [sys.argv[0]] + hydra_args  # hand the rest to hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import time
import numpy as np
import yaml
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_error_magnitude
import isaaclab.sim as sim_utils

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

_HERE = os.path.dirname(os.path.abspath(__file__))
_BODEX = os.path.join(_HERE, "ultradex_repo", "third_party", "BODex_api", "src", "bodex", "content")
_URDF = os.path.join(_BODEX, "assets", "robot", "inspire_hand", "inspire_hand_right.urdf")
_ROBOT_YML = os.path.join(_BODEX, "configs", "robot", "inspire_right.yml")
_VALID_NPZ = os.path.join(_HERE, "grasp_dataset", "cube_5cm_grasps_valid.npz")
_ACT = ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "index_proximal_joint",
        "middle_proximal_joint", "ring_proximal_joint", "pinky_proximal_joint"]

# same cache mdp.grasp_reach_reward reads -- see MDP_REPORT.md Sec 5.2.4
_FINGERTIP_CONTACTS_FILE = os.path.join(os.path.dirname(_HERE), "grasp_selection", "fingertip_contacts.json")
_FINGERTIP_BODIES = ["R_thumb_distal", "R_index_intermediate",
                     "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate"]
_FINGERTIP_ORDER = ["thumb", "index", "middle", "ring", "pinky"]  # matches the cache's order

def _cube_surface_gap(p_world, cube_pos, cube_quat_wxyz, sphere_radii, half_edge=0.025):
    """Real geometric gap between a set of collision-sphere SURFACES and the cube's
    box surface -- negative means the sphere genuinely overlaps the cube (not a camera
    occlusion illusion). Same signed-distance formula grasp_selection/hand_model.py's
    cube_surface_dist uses, just applied to the robot's LIVE simulated pose each step
    instead of an offline synthesis pose.

    p_world: (n,3) sphere centers in world frame.
    cube_pos: (3,), cube_quat_wxyz: (4,) -- the cube's LIVE pose this step.
    sphere_radii: (n,) each sphere's own radius (subtracted so this is a SURFACE gap,
        not a center-to-surface distance).
    Returns: (n,) array, gap[i] < 0 <=> sphere i's surface is inside the cube.
    """
    Rm = R.from_quat(cube_quat_wxyz, scalar_first=True).as_matrix()  # world <- cube-local
    p_local = (p_world - cube_pos) @ Rm  # world point -> cube-local frame
    d = np.abs(p_local) - half_edge
    outside = np.linalg.norm(np.clip(d, 0.0, None), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    sdf = outside + inside  # >0 outside the box, <=0 inside (center-to-surface, signed)
    return sdf - sphere_radii


# ---- EDIT ME: extra rotation applied to the ghost goal hand for eyeballing ----
# Euler angles in degrees applied in the palm-LOCAL frame (order xyz).
# e.g. (0, 0, 90) twists the goal hand 90 deg about its approach/finger axis.
_GHOST_ROT_EULER_DEG = (0.0, 0.0, 0.0)
_GHOST_ROT = R.from_euler("xyz", np.deg2rad(_GHOST_ROT_EULER_DEG)).as_matrix()


class GoalHandGhost:
    """Draws the Inspire hand's collision-sphere silhouette at the goal grasp pose.

    Uses the ~40 collision spheres from inspire_right.yml, forward-kinematics'd to
    the goal finger configuration, anchored at the goal root pose, with the
    URDF->USD calibration applied. Result: a translucent 'ghost hand' at the
    UltraDexGrasp target showing both palm pose AND finger shape.
    """

    def __init__(self):
        self.joints = self._parse_urdf(_URDF)
        cfg = yaml.safe_load(open(_ROBOT_YML))["robot_cfg"]["kinematics"]["collision_spheres"]
        self.spheres = []  # list of (link, center(3,), radius)
        for link, lst in cfg.items():
            for s in lst:
                self.spheres.append((link, np.array(s["center"], float), float(s["radius"])))
        self.radii = np.array([r for _, _, r in self.spheres])
        d = np.load(_VALID_NPZ, allow_pickle=True)
        self.T = d["T_usdbase_urdfbase"] if "T_usdbase_urdfbase" in d else np.eye(4)
        self.n = len(self.spheres)
        self.mesh_links = self._parse_urdf_visual_meshes(_URDF)  # link -> abs .glb path

    @staticmethod
    def _parse_urdf_visual_meshes(path):
        """link_name -> absolute path of its visual mesh, as a USD file.

        The URDF points at .glb files, which USD's Sdf.Layer can't open directly
        (no glTF file-format plugin registered) -- so this resolves to the
        pre-converted cache in grasp_sampler/mesh_cache/ (built once via
        scripts/tools/convert_mesh.py) instead of the raw .glb path.
        All visual origins in this URDF are identity (checked directly), so no
        extra per-link visual offset is needed on top of the link FK transform.
        """
        root_dir = os.path.dirname(path)
        mesh_cache = os.path.join(_HERE, "mesh_cache")
        out = {}
        for link in ET.parse(path).getroot().findall("link"):
            vis = link.find("visual")
            if vis is None:
                continue
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            glb_name = os.path.splitext(os.path.basename(mesh.get("filename")))[0]
            usd_path = os.path.join(mesh_cache, f"{glb_name}.usd")
            if not os.path.isfile(usd_path):
                raise FileNotFoundError(
                    f"No cached USD mesh for link '{link.get('name')}' at {usd_path}. "
                    f"Run scripts/tools/convert_mesh.py on the .glb files in "
                    f"{os.path.join(root_dir, 'meshes', 'visual')} first."
                )
            out[link.get("name")] = usd_path
        return out

    @staticmethod
    def _parse_urdf(path):
        joints = {}
        for j in ET.parse(path).getroot().findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()])
            ax = j.find("axis")
            joints[j.get("name")] = dict(
                type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=xyz, rpy=rpy, axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None)
        return joints

    def _fk_all(self, q6):
        qmap = dict(zip(_ACT, q6))
        poses = {"base": np.eye(4)}
        changed = True
        while changed:
            changed = False
            for name, j in self.joints.items():
                if j["parent"] in poses and j["child"] not in poses:
                    T = np.eye(4)
                    T[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix()
                    T[:3, 3] = j["xyz"]
                    if j["type"] == "revolute":
                        a = j["axis"] / np.linalg.norm(j["axis"])
                        Tj = np.eye(4); Tj[:3, :3] = R.from_rotvec(a * qmap.get(name, 0.0)).as_matrix()
                        T = T @ Tj
                    poses[j["child"]] = poses[j["parent"]] @ T
                    changed = True
        return poses

    def adjusted_goal_mat(self, goal_pos_w, goal_quat_w):
        """Goal pose (4x4) with the editable _GHOST_ROT applied in the palm-local frame."""
        goal = np.eye(4)
        goal[:3, :3] = R.from_quat(goal_quat_w, scalar_first=True).as_matrix() @ _GHOST_ROT
        goal[:3, 3] = goal_pos_w
        return goal

    def world_centers(self, goal_pos_w, goal_quat_w, q6):
        """Return (n,3) world positions of all collision spheres for one goal."""
        link_poses = self._fk_all(np.asarray(q6, float))
        goal = self.adjusted_goal_mat(goal_pos_w, goal_quat_w)
        base = goal @ self.T  # world <- USD-handbase <- URDF-base
        out = np.zeros((self.n, 3))
        for i, (link, c, _) in enumerate(self.spheres):
            Tlink = base @ link_poses.get(link, np.eye(4))
            out[i] = (Tlink @ np.array([c[0], c[1], c[2], 1.0]))[:3]
        return out

    def link_world_poses(self, goal_pos_w, goal_quat_w, q6, link_order):
        """World (pos(3,), quat_wxyz(4,)) for each link in link_order, for the mesh ghost."""
        link_poses = self._fk_all(np.asarray(q6, float))
        goal = self.adjusted_goal_mat(goal_pos_w, goal_quat_w)
        base = goal @ self.T
        positions = np.zeros((len(link_order), 3))
        quats = np.zeros((len(link_order), 4))
        for i, link in enumerate(link_order):
            Tlink = base @ link_poses.get(link, np.eye(4))
            positions[i] = Tlink[:3, 3]
            quats[i] = R.from_matrix(Tlink[:3, :3]).as_quat(scalar_first=True)
        return positions, quats


def _make_markers(mesh_links: dict[str, str]):
    # goal pose as an RGB coordinate frame (shows target orientation)
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.prim_path = "/Visuals/goal_frame"
    frame_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
    goal_frame = VisualizationMarkers(frame_cfg)

    # ghost hand at the GOAL: the REAL Inspire-hand visual meshes (not collision spheres),
    # one USD-referenced .glb prototype per link, translucent green so it reads clearly
    # against the actual (naturally-rendered, opaque) robot hand. No ghost is drawn for the
    # actual hand -- it's already visible as the real robot mesh, a synthetic overlay for it
    # would be redundant. (Briefly tried bright red + higher opacity on 2026-08-21, reverted
    # the same day -- the user found the full mesh+frame overlay hard to read; see the
    # separate goal_centroid marker below instead, a single bright-red dot at the goal palm
    # position, kept on regardless of --no_markers.)
    link_order = list(mesh_links.keys())
    ghost_mesh = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/goal_hand_mesh",
        markers={
            link: sim_utils.UsdFileCfg(
                usd_path=mesh_links[link],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.9, 0.25), opacity=0.5),
            )
            for link in link_order
        },
    ))

    # Simple bright-red dot at the goal palm position (2026-08-21) -- a much easier-to-read
    # stand-in for "where is the goal" than the full ghost mesh/frame, per the user's
    # explicit request. Always drawn, independent of --no_markers (which still gates only
    # goal_frame/ghost_mesh below).
    goal_centroid_marker = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/goal_centroid",
        markers={
            "centroid": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=1.0),
            )
        },
    ))
    return goal_frame, ghost_mesh, goal_centroid_marker, link_order


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    if not args_cli.no_hold_after_success:
        # This env_cfg instance belongs only to this script's own gym.make() call below -- a
        # live training process builds its own separate instance from the same registry entry,
        # so disabling a termination here cannot affect it. Without this, target_lifted ends
        # env 0's episode (and the vec-env auto-resets it) on the SAME step the cube first
        # crosses _SUCCESS_Z, so there is no way to "keep watching" after that point within the
        # normal step loop -- the only way to see the hold is to not terminate on it at all,
        # letting the episode run to the time_out termination instead.
        env_cfg.terminations.target_lifted = None

    if args_cli.tray_distractors:
        # Original per-distractor spawn positions from SceneCfg (before the V2 __post_init__
        # override relocates them to the table margins) -- ring 1 (0.32-0.42, +/-0.08-0.12)
        # sits directly in the approach path to the cube at [0.35, 0.0], ring 3 further out.
        # Same tray height (_OBJ_INIT_Z = 0.845) all ten originally shared.
        _ORIGINAL_TRAY_POS = {
            1: (0.38, 0.08), 2: (0.38, -0.08), 3: (0.42, 0.0), 4: (0.32, 0.12), 5: (0.32, -0.12),
            6: (0.48, 0.12), 7: (0.48, -0.12), 8: (0.28, 0.0), 9: (0.52, 0.0), 10: (0.35, 0.18),
        }
        _ORIGINAL_TRAY_Z = 0.845
        for i in range(1, 11):
            d_cfg = getattr(env_cfg.scene, f"distractor_{i}")
            x, y = _ORIGINAL_TRAY_POS[i]
            d_cfg.init_state.pos = (x, y, _ORIGINAL_TRAY_Z)

    if args_cli.table_distractors:
        # No-interference / "empty env" layout: all 10 distractors on the table margins, clear
        # of the tray and the cube's approach path entirely. Same positions used before the live
        # env_cfg default was switched back to full tray clutter.
        _TABLE_DISTRACTOR_POS = {
            1: (0.16, -0.45), 2: (0.28, -0.45), 3: (0.40, -0.45), 4: (0.52, -0.45), 5: (0.64, -0.45),
            6: (0.16, 0.45), 7: (0.28, 0.45), 8: (0.40, 0.45), 9: (0.52, 0.45), 10: (0.64, 0.45),
        }
        _TABLE_REST_Z = 0.825
        for i in range(1, 11):
            d_cfg = getattr(env_cfg.scene, f"distractor_{i}")
            x, y = _TABLE_DISTRACTOR_POS[i]
            d_cfg.init_state.pos = (x, y, _TABLE_REST_Z)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        # RecordVideo writes the MP4 at env.metadata["render_fps"]; lowering it
        # makes the same motion play back in slow motion.
        env.metadata["render_fps"] = args_cli.video_fps
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(log_dir, "videos", "play_markers"),
                                       step_trigger=lambda step: step == 0,
                                       video_length=args_cli.video_length, disable_logger=True,
                                       name_prefix=args_cli.name_prefix)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    hand_viz = GoalHandGhost()
    goal_frame, ghost_mesh, goal_centroid_marker, ghost_link_order = _make_markers(hand_viz.mesh_links)
    num_envs = env.unwrapped.num_envs
    n_links = len(ghost_link_order)
    # which prototype (link) each of the num_envs*n_links mesh instances is -- constant
    # across steps: env 0's 13 links, then env 1's 13 links, etc.
    ghost_marker_indices = np.tile(np.arange(n_links), num_envs)
    robot = env.unwrapped.scene["robot"]
    fingertip_body_ids = [robot.body_names.index(n) for n in _FINGERTIP_BODIES]

    # fingertip target contact points (grasp_reach_reward's own cache -- MDP_REPORT.md 5.2.4)
    with open(_FINGERTIP_CONTACTS_FILE) as f:
        _fc = json.load(f)
    assert _fc["fingertip_order"] == _FINGERTIP_ORDER
    fingertip_contacts_obj = torch.tensor(
        _fc["contact_pos_cube_frame"], dtype=torch.float32, device=env.unwrapped.device)  # (5,3)

    # env-0 history for the diagnostic graphs, saved after the loop
    hist_palm_pos_err, hist_palm_orient_err, hist_finger_dist, hist_min_gap = [], [], [], []
    episode_boundaries = []  # step indices (in the logged history) where an episode ended
    episode_count = 0

    # env-0 per-term reward history, for the reward-breakdown-over-episode plot (added
    # 2026-08-19). Keyed by RewardManager.active_terms so it automatically tracks whatever
    # terms are actually configured (including any that are weight-zeroed for the current
    # curriculum phase -- those just log as a flat 0 line, which is itself a useful sanity
    # check that they're genuinely inactive).
    reward_term_names = list(env.unwrapped.reward_manager.active_terms)
    hist_reward_terms = {name: [] for name in reward_term_names}
    hist_reward_total = []
    reward_episode_boundaries = []

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # Episode counting is independent of the goal-mimicking diagnostics below: IsaacLab's
        # RewardManager SKIPS calling a term's function entirely when its weight is 0.0 (a
        # documented "micro-optimization" in reward_manager.py), so if grasp_goal_palm is
        # zeroed (e.g. during an early from-scratch curriculum phase), _grasp_goal_term never
        # gets cached on the env and `term` below stays None for the whole run -- which used
        # to mean episode_count never incremented and --episodes never triggered the early
        # stop (discovered 2026-08-19 when a Phase-1 checkpoint's episode counter stuck at 0
        # and the run silently continued to --video_length instead of stopping at 4 episodes).
        episode_ended_this_step = args_cli.episodes > 0 and bool(dones[0].item())
        if episode_ended_this_step:
            episode_count += 1
            # reward_episode_boundaries marks indices into hist_reward_total (which is
            # appended EVERY step, unlike hist_palm_pos_err below which skips reset steps --
            # so this needs its own boundary list, taken before this step's reward gets
            # appended, same convention as episode_boundaries' "mark where the NEXT ep starts").
            reward_episode_boundaries.append(len(hist_reward_total))

        # Per-term reward logging (env 0). RewardManager.compute() runs INSIDE env.step(),
        # strictly before any auto-reset, so _step_reward/_reward_buf here are always for the
        # transition that just happened -- no reset-timing discontinuity to guard against
        # (contrast with the palm/finger pose history below, which DOES need the
        # episode_ended_this_step skip since those read post-reset robot/object state).
        rm = env.unwrapped.reward_manager
        step_reward_0 = rm._step_reward[0].cpu().numpy()  # (num_terms,), already weight-applied
        for name, val in zip(reward_term_names, step_reward_0):
            hist_reward_terms[name].append(float(val))
        hist_reward_total.append(float(rm._reward_buf[0].item()))

        # the reward term caches the goal-sampler instance on the env after step 1
        term = getattr(env.unwrapped, "_grasp_goal_term", None)
        if term is not None:
            goal_pos = term.goal_pos_w
            goal_quat = term.goal_quat_w
            goal_centroid_marker.visualize(translations=goal_pos)  # always on, see _make_markers
            palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
            dist = torch.norm(palm_pos - goal_pos, dim=1)
            # palm -> cube center distance
            cube_pos = env.unwrapped.scene[term._object_name].data.root_pos_w
            dist_cube = torch.norm(palm_pos - cube_pos, dim=1)
            # wrist ORIENTATION error (palm quat vs goal quat)
            palm_quat = robot.data.body_quat_w[:, term._palm_body_idx]
            orient_err = quat_error_magnitude(palm_quat, goal_quat)      # rad
            # FINGER config error (current 6 proximal joints vs goal grasp)
            q_now = robot.data.joint_pos[:, term._hand_joint_ids]
            q_err = torch.norm(q_now - term.goal_hand_q, dim=1)          # rad, L2 over 6 joints
            q_err_per = (q_now - term.goal_hand_q).abs()                 # per-joint |error|

            # ghost hand at the goal, for EVERY env: FK the mesh links (for the visual) and
            # the collision spheres (for the numeric penetration diagnostic below) to each
            # env's own goal config. CPU numpy loop per env (GoalHandGhost is a single-
            # hand, single-pose visualizer) -- fine at diagnostic-video env counts.
            # ACTUAL hand (no ghost drawn -- it's already the real, naturally-rendered robot
            # mesh -- but its collision-sphere FK is still needed for the penetration check).
            goal_pos_np = goal_pos.cpu().numpy()
            goal_quat_np = goal_quat.cpu().numpy()
            goal_hand_q_np = term.goal_hand_q.cpu().numpy()
            actual_palm_pos_np = palm_pos.cpu().numpy()
            actual_palm_quat_np = palm_quat.cpu().numpy()
            actual_q_np = q_now.cpu().numpy()
            all_adj_quats, all_actual_centers = [], []
            all_link_pos, all_link_quat = [], []
            for e in range(num_envs):
                gp_e, gq_e = goal_pos_np[e], goal_quat_np[e]
                lp, lq = hand_viz.link_world_poses(gp_e, gq_e, goal_hand_q_np[e], ghost_link_order)
                all_link_pos.append(lp)
                all_link_quat.append(lq)
                all_adj_quats.append(
                    R.from_matrix(hand_viz.adjusted_goal_mat(gp_e, gq_e)[:3, :3]).as_quat(scalar_first=True)
                )
                all_actual_centers.append(
                    hand_viz.world_centers(actual_palm_pos_np[e], actual_palm_quat_np[e], actual_q_np[e])
                )
            link_pos = np.concatenate(all_link_pos, axis=0)   # (num_envs * n_links, 3)
            link_quat = np.concatenate(all_link_quat, axis=0)  # (num_envs * n_links, 4)
            adj_quats = np.stack(all_adj_quats, axis=0)        # (num_envs, 4)
            actual_centers = np.concatenate(all_actual_centers, axis=0)

            # ghost_mesh (translucent green hand silhouette at the goal) -- gated on its own
            # flag (--no_ghost_mesh, added 2026-08-21) independent of --no_markers (the
            # coordinate FRAME). goal_centroid_marker (the red dot) stays on regardless of
            # either flag.
            if not args_cli.no_ghost_mesh:
                ghost_mesh.visualize(
                    translations=torch.tensor(link_pos, dtype=torch.float32, device=env.unwrapped.device),
                    orientations=torch.tensor(link_quat, dtype=torch.float32, device=env.unwrapped.device),
                    marker_indices=ghost_marker_indices)
            if not args_cli.no_markers:
                goal_frame.visualize(
                    translations=goal_pos,
                    orientations=torch.tensor(adj_quats, dtype=torch.float32, device=env.unwrapped.device))

            # fingertip -> its own target contact point (grasp_reach_reward's signal, §5.2.4)
            tips_w = robot.data.body_pos_w[:, fingertip_body_ids]          # (N,5,3)
            n_e = tips_w.shape[0]
            contacts_b = fingertip_contacts_obj.unsqueeze(0).expand(n_e, -1, -1).reshape(-1, 3)
            cube_quat_full = env.unwrapped.scene[term._object_name].data.root_quat_w
            cube_quat_exp = cube_quat_full.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
            cube_pos_exp = cube_pos.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 3)
            finger_targets_w = (quat_apply(cube_quat_exp, contacts_b) + cube_pos_exp).view(n_e, 5, 3)
            finger_dist = torch.norm(tips_w - finger_targets_w, dim=-1)    # (N,5)

            # REAL geometric gap between the robot's actual live collision spheres and the
            # cube's box surface -- ground truth for "is the hand mesh genuinely overlapping
            # the cube", not a camera-occlusion illusion. Negative = real interpenetration.
            cube_pos0_np = cube_pos[0].cpu().numpy()
            cube_quat0_np = cube_quat_full[0].cpu().numpy()
            gaps0 = _cube_surface_gap(all_actual_centers[0], cube_pos0_np, cube_quat0_np, hand_viz.radii)
            min_gap0 = float(gaps0.min())

            # env-0 history for the post-run graphs. Skip logging on the step where env 0's
            # episode just ended: IsaacLab auto-resets INSIDE env.step() when done fires, so
            # everything read above already reflects the NEXT episode's reset state, not the
            # terminal moment of this one -- logging it would put a false spike/discontinuity.
            # (episode_ended_this_step and episode_count are computed/incremented above, right
            # after env.step() -- independent of this term-diagnostics block existing at all.)
            if not episode_ended_this_step:
                hist_palm_pos_err.append(dist[0].item())
                hist_palm_orient_err.append(np.degrees(orient_err[0].item()))
                hist_finger_dist.append(finger_dist[0].cpu().numpy() * 100.0)  # cm
                hist_min_gap.append(min_gap0 * 100.0)  # cm
            else:
                episode_boundaries.append(len(hist_palm_pos_err))  # mark where the NEXT ep starts

            if timestep % 15 == 0:
                tag = "PENETRATING" if min_gap0 < 0 else ""
                print(f"[goal] step {timestep:4d}  min hand-sphere<->cube surface gap = "
                      f"{min_gap0*100:+6.2f} cm  {tag}", flush=True)

            # one-time: distance between the goal pose and the cube center
            if timestep == 0:
                goal_cube = torch.norm(goal_pos - cube_pos, dim=1)[0].item()
                print(f"[goal] goal→cube distance = {goal_cube*100:.1f} cm "
                      f"(fixed offset of the goal grasp above/around the cube)", flush=True)

            if timestep % 15 == 0:
                d0 = dist[0].item()
                dc0 = dist_cube[0].item()
                oe0 = np.degrees(orient_err[0].item())         # wrist orientation error
                qe0 = np.degrees(q_err[0].item())              # finger config error (L2)
                per = np.degrees(q_err_per[0].cpu().numpy())   # [th_yaw,th_pitch,idx,mid,ring,pinky]
                tag = "REACHED" if d0 < args_cli.reach_thresh else "       "
                print(f"[goal] step {timestep:4d}  palm→goal = {d0*100:5.1f} cm   "
                      f"palm→cube = {dc0*100:5.1f} cm   wrist_err = {oe0:5.1f}°   "
                      f"finger_err = {qe0:5.1f}°  {tag}", flush=True)
                print(f"        finger err per joint [th_yaw,th_pitch,idx,mid,ring,pinky] = "
                      f"{np.round(per, 1)}", flush=True)

        timestep += 1
        if args_cli.episodes > 0 and dones[0].item() > 0:
            print(f"[goal] env 0 episode {episode_count}/{args_cli.episodes} ended at "
                  f"step {timestep}", flush=True)
            if episode_count >= args_cli.episodes:
                break
        if args_cli.video and timestep == args_cli.video_length:
            break

        if args_cli.real_time:
            dt = env.unwrapped.step_dt
            wait = dt - (time.time() - start)
            if wait > 0:
                time.sleep(wait)

    graph_dir = os.path.join(log_dir, "videos", "play_markers")

    # close the env first so RecordVideo finalizes/writes the MP4 ...
    env.close()
    if args_cli.video:
        print(f"[goal] video written to {graph_dir}", flush=True)

    # Shared helpers for BOTH diagnostic-graph blocks below -- moved out of the
    # `if hist_palm_pos_err:` gate (2026-08-19) since the reward-breakdown plot needs them
    # too, and hist_palm_pos_err stays empty whenever term is None (e.g. Phase 1/2 of the
    # from-scratch curriculum, where grasp_goal_palm is weight-zeroed).
    n_ep = f"{len(episode_boundaries) + 1} episode(s)" if episode_boundaries else "1 episode"

    def _mark_episodes(ax):
        for b in episode_boundaries:
            ax.axvline(b, color="0.4", linestyle="--", linewidth=0.8, alpha=0.7)

    # env-0 per-step reward breakdown: every active term as its own line, dotted TOTAL on top.
    if hist_reward_total:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(graph_dir, exist_ok=True)
        r_steps = np.arange(len(hist_reward_total))
        r_n_ep = (f"{len(reward_episode_boundaries) + 1} episode(s)"
                  if reward_episode_boundaries else "1 episode")
        fig_r, ax_r = plt.subplots(figsize=(12, 6))
        # tab10 (matplotlib's default cycle) only has 10 colors -- with 11 reward terms, the
        # 11th would silently wrap around and reuse the 1st term's color. tab20 has 20.
        term_colors = plt.get_cmap("tab20").colors
        for i, name in enumerate(reward_term_names):
            ax_r.plot(r_steps, hist_reward_terms[name], label=name, linewidth=1.1,
                       color=term_colors[i % len(term_colors)])
        ax_r.plot(r_steps, hist_reward_total, label="TOTAL", color="black",
                   linewidth=2.0, linestyle=":")
        ax_r.axhline(0.0, color="0.6", linewidth=0.8)
        ax_r.set_xlabel("control step")
        ax_r.set_ylabel("per-step reward (weighted)")
        ax_r.grid(alpha=0.25)
        for b in reward_episode_boundaries:
            ax_r.axvline(b, color="0.4", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
        ax_r.set_title(f"Per-step reward breakdown -- env 0, {r_n_ep} (dashed = episode reset, "
                        f"dotted black = total)")
        fig_r.tight_layout()
        p_r = os.path.join(graph_dir, "reward_breakdown.png")
        fig_r.savefig(p_r, dpi=120, bbox_inches="tight")
        plt.close(fig_r)
        print(f"[goal] graph written -> {p_r}", flush=True)

    # env-0 diagnostic graphs: palm pose error + per-fingertip contact-point distance
    if hist_palm_pos_err:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(graph_dir, exist_ok=True)
        steps = np.arange(len(hist_palm_pos_err))
        grasp_idx = int(term.goal_grasp_idx[0].item()) if term is not None else -1

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(steps, np.array(hist_palm_pos_err) * 100, color="tab:blue", linewidth=1.2)
        ax1.set_xlabel("control step")
        ax1.set_ylabel("palm position error (cm)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.25)
        _mark_episodes(ax1)
        ax2 = ax1.twinx()
        ax2.plot(steps, hist_palm_orient_err, color="tab:orange", linewidth=1.2)
        ax2.set_ylabel("palm orientation error (deg)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        fig.suptitle(f"Palm (R_hand_base_link) pose error vs. goal grasp #{grasp_idx} -- "
                    f"env 0, {n_ep} (dashed = episode reset)")
        fig.tight_layout()
        p1 = os.path.join(graph_dir, "palm_pose_error.png")
        fig.savefig(p1, dpi=120)
        plt.close(fig)

        fig2, ax = plt.subplots(figsize=(11, 5))
        finger_arr = np.stack(hist_finger_dist, axis=0)  # (T, 5) cm
        for i, name in enumerate(_FINGERTIP_ORDER):
            ax.plot(steps, finger_arr[:, i], label=name, linewidth=1.2)
        ax.set_xlabel("control step")
        ax.set_ylabel("fingertip -> target contact point distance (cm)")
        ax.legend()
        ax.grid(alpha=0.25)
        _mark_episodes(ax)
        ax.set_title(f"Per-fingertip distance to its grasp_reach target contact point "
                    f"(grasp #{grasp_idx}) -- env 0, {n_ep} (dashed = episode reset)")
        fig2.tight_layout()
        p2 = os.path.join(graph_dir, "fingertip_contact_distance.png")
        fig2.savefig(p2, dpi=120)
        plt.close(fig2)

        # REAL geometric penetration check: min gap between the robot's actual live
        # collision spheres and the cube's box surface. <0 = genuine interpenetration
        # of the collision geometry, not a camera-occlusion illusion.
        fig3, ax3 = plt.subplots(figsize=(11, 5))
        gap_arr = np.array(hist_min_gap)
        ax3.plot(steps, gap_arr, color="tab:red", linewidth=1.2)
        ax3.axhline(0.0, color="black", linewidth=1.0)
        ax3.fill_between(steps, gap_arr, 0.0, where=(gap_arr < 0), color="red", alpha=0.25,
                         label="interpenetrating")
        ax3.set_xlabel("control step")
        ax3.set_ylabel("min hand-sphere <-> cube surface gap (cm)")
        ax3.grid(alpha=0.25)
        _mark_episodes(ax3)
        n_pen = int((gap_arr < 0).sum())
        ax3.set_title(f"Min collision-sphere-to-cube surface gap -- env 0, {n_ep} "
                    f"({n_pen}/{len(gap_arr)} steps show real interpenetration) "
                    f"(dashed = episode reset)")
        if n_pen:
            ax3.legend()
        fig3.tight_layout()
        p3 = os.path.join(graph_dir, "hand_cube_penetration.png")
        fig3.savefig(p3, dpi=120)
        plt.close(fig3)

        print(f"[goal] graph written -> {p1}", flush=True)
        print(f"[goal] graph written -> {p2}", flush=True)
        print(f"[goal] graph written -> {p3}", flush=True)
        print(f"[goal] penetration summary: {n_pen}/{len(gap_arr)} steps show min gap < 0 "
              f"(worst = {gap_arr.min():+.2f} cm)", flush=True)

    # ... then hard-exit. simulation_app.close() hangs indefinitely in headless
    # mode AFTER the run has finished, which looks like the script freezing.
    os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
