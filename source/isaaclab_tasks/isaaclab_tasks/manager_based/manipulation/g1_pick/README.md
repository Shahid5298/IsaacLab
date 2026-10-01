# G1 Pick — Dexterous Cube Grasping

G1 humanoid + Inspire dexterous right hand, learning to pick up a 5cm cube with RSL-RL
PPO in Isaac Lab, with sim-to-real transfer as the end goal. The lower body is frozen;
only the right arm + hand are actuated.

For the detailed experiment history, diagnoses, and what did/didn't work, see
[`REWARD_CURRICULUM_LOG.md`](REWARD_CURRICULUM_LOG.md), [`CONTEXT.md`](CONTEXT.md),
[`PROJECT_REFERENCE.md`](PROJECT_REFERENCE.md), and [`MDP_REPORT.md`](MDP_REPORT.md).

## Environment registration

| Task ID | Scene |
|---|---|
| `Isaac-G1-Pick-Empty-v0` / `-Empty-Play-v0` | 10 distractor objects parked at the table margins, out of the way. **Use this one.** |
| `Isaac-G1-Pick-Stage1-v0` / `-Stage1-Play-v0` | 5 distractors spread wide on the tray, 5 parked at the margins. |
| `Isaac-G1-Pick-Stage2-v0` / `-Stage2-Play-v0` | All 10 distractors on the tray in an even 2x5 grid. |
| `Isaac-G1-Pick-v0` / `-Play-v0` | Original clutter task. **Do not use** unless told otherwise — distractors are clustered around the target and corrupt both the visuals and the lift-rate metric. |

The non-`-Play-` variant is the training config (default 2048+ parallel envs); `-Play-`
is the evaluation config (frozen difficulty, small `num_envs`). Neither
`play_with_goal_markers.py` nor `scripts/reinforcement_learning/rsl_rl/play.py` defaults
to the Empty variant — always pass `--task Isaac-G1-Pick-Empty-Play-v0` explicitly.

## Prerequisites

All training/inference commands run inside the `shahid_g1pick` Docker container. If it's
stopped:

```bash
docker start shahid_g1pick
```

## Training

```bash
docker exec shahid_g1pick bash -lc '
  cd /workspace/isaaclab
  /workspace/isaaclab/_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Pick-Empty-v0 --headless --num_envs 2048 \
    --max_iterations 1000 --device cuda:0 --run_name <descriptive_name>'
```

Resume from a checkpoint:

```bash
docker exec shahid_g1pick bash -lc '
  cd /workspace/isaaclab
  /workspace/isaaclab/_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Pick-Empty-v0 --headless --num_envs 2048 \
    --max_iterations 1000 --device cuda:0 \
    --resume --load_run <run_dir_name> --checkpoint model_<N>.pt \
    --run_name <descriptive_name>'
```

Runs land in `logs/rsl_rl/g1_pick/<timestamp>_<run_name>/`, checkpointed every
`save_interval` iterations (`agents/rsl_rl_ppo_cfg.py`, `G1PickPPORunnerCfg`).

Training is long-running — launch it inside its own `tmux` session so it survives a
disconnect:

```bash
tmux new -s g1_train
# ...run the docker exec command above...
# detach: Ctrl+B then D
tmux attach -t g1_train
```

To check whether training is still running, don't `pgrep -f "rsl_rl/train.py"` directly
from a wrapper whose own command line also contains that string — it will match itself.
Use the bracket trick: `pgrep -f "[r]sl_rl/train.py"`.

## Inference / recording a video

Use `grasp_sampler/play_with_goal_markers.py` — it overlays the UltraDexGrasp goal pose
(a translucent ghost hand at the target grasp configuration) on top of the policy
rollout, and dumps reward/pose-error/contact-distance diagnostic plots alongside the
video.

```bash
docker exec shahid_g1pick bash -lc '
  cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
  /workspace/isaaclab/_isaac_sim/python.sh play_with_goal_markers.py \
    --task Isaac-G1-Pick-Empty-Play-v0 --num_envs 1 --headless \
    --video --video_length 500 --enable_cameras --episodes 2 \
    --checkpoint /workspace/isaaclab/logs/rsl_rl/g1_pick/<run_dir_name>/model_<N>.pt \
    --device cuda:0 \
    "env.viewer.origin_type=env" "env.viewer.eye=[0.72,-0.42,1.02]" "env.viewer.lookat=[0.40,0.0,0.88]"'
```

The `env.viewer.eye`/`lookat` pair above is the "zoomed" camera used throughout this
project; drop them for the default wide view. Output video + plots land in
`<run_dir>/videos/play_markers/`.

Two companion diagnostics, same `--task`/`--checkpoint`/`--device` pattern:

- `contact_check.py` — per-fingertip minimum distance to the cube surface over a
  rollout (reads body origins at the knuckle, so compare across checkpoints, not
  against zero).
- `lift_capability_probe.py --mode lift --lift_delta -1.5` — forces the arm to lift and
  reports the cube/palm "follow ratio": near 0 means the fingers aren't exerting real
  grip force even if they look close in the video; `--mode squeeze` instead forces all
  hand joints closed and reports cube displacement.

`grasp_sampler/record_all_checkpoints.sh <run_dir_name>` automates all of the above
(video + both diagnostics) across every checkpoint in a run, in its own tmux session —
use it as a template for new sweeps.

## Grasp generation pipeline

The policy is trained to reach for a specific, mechanically-verified grasp pose rather
than an arbitrary one. That grasp comes from an offline pipeline, run once per cube/hand
geometry and cached to disk — training itself never touches BODex, Isaac Sim collision
meshes, or the optimizer; it only reads the cached library and the cached fingertip
targets.

```
UltraDexGrasp/BODex synthesis  →  Isaac Lab validation + recentering  →  feasibility +
wrap-quality filtering (uses the FSWO optimizer)  →  FSWO re-ranks the survivors  →
single best grasp index + cached fingertip targets, both read at env-config import time
```

### 1. Generate raw grasp candidates (BODex / UltraDexGrasp)

This step needs its own conda env and a separate repo clone — it does not run inside
the Isaac Lab container.

```bash
# one-time setup, in a NEW conda env (not env_isaaclab)
conda create -n ultradex python=3.10 -y
conda activate ultradex
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
cd grasp_sampler
pip install -r ultradex_repo/requirements.txt
# PyTorch3D, cuRobo, and BODex_api — see ultradex_repo/README.md "Getting Started" for
# the full build-from-source steps (compiles C++ extensions, so follow it exactly).
```

`ultradex_repo/` is **gitignored** (its own license, large assets) — it's a copy of
[UltraDexGrasp](https://github.com/InternRobotics/UltraDexGrasp) (ICRA 2026), adapted
from [BODex](https://github.com/JYChen18/BODex), with an added `hand_type == 'inspire'`
branch (see `ultradex_repo/util/bodex_util.py`) and the Inspire-hand configs in
`bodex_config_templates/` copied into
`ultradex_repo/third_party/BODex_api/src/bodex/content/configs/...` at the paths named
in each template's header comment. If `ultradex_repo` is missing from your checkout,
pull it from another checkout on the same machine, or re-clone + re-apply the templates
following `bodex_config_templates/README.md`.

Generate candidates for the 5cm cube:

```bash
cd grasp_sampler
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ~/miniconda3/envs/ultradex/bin/python synthesize_inspire_grasps.py
```

Writes `grasp_dataset/cube_5cm_grasps.npz` — 300 raw grasps, each a (pregrasp, grasp,
squeeze) stage x (root position + root quaternion + 6 Inspire proximal joint angles),
in the cube's object frame.

### 2. Validate + grip-recenter in Isaac Lab

Needs the real USD hand (mimic joints, actual link frames), so this step runs inside
Isaac Lab/Isaac Sim, not the BODex env:

```bash
conda activate env_isaaclab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler/validate_grasps_isaaclab.py --headless
```

For each candidate: holds the arm fixed, places the cube at the grasp-relative pose,
drives pregrasp → grasp → squeeze, lets gravity act, and checks the cube stays near the
palm. Writes `grasp_dataset/cube_5cm_grasps_recentered.npz` (grip-recentered to account
for the USD hand's mimic-joint offsets vs. the URDF BODex optimized against).

### 3. Build the final goal-grasp library

```bash
cd grasp_sampler
python build_goal_library.py
```

Filters the recentered grasps down to ones that are both **task-feasible** (no
fingertip/palm below the cube's underside, near-top-down approach — the G1 arm reaches
a top-down pose far more reliably than a side approach) and **mechanically sound**
(≥3 distinct fingers with collision spheres actually touching the cube on opposing
faces — i.e., force-closable, via the FSWO optimizer from step 4). Ranks survivors by
wrap quality and keeps the top 32. Writes `grasp_dataset/cube_5cm_grasps_valid.npz` —
this is the file `mdp/grasp_goal.py` actually loads at training time.

### 4. Run the FSWO optimizer to pick the single best grasp

The Frictionless Self-balancing Wrench Optimizer (FSWO) scores force-closure quality
from first principles — no Isaac Sim, no GPU, CPU-only, runs in ~1s for 32 grasps. It's
used internally by `build_goal_library.py`, but can also be run standalone to inspect
or re-select the active grasp:

```bash
cd grasp_selection
python selftest.py                      # sanity-check the FSWO math itself
python rank_grasps_sphere_fswo.py       # rank the 32-grasp library, refresh scores.json
python rank_grasps_sphere_fswo.py --sweep         # check robustness of the top pick
python rank_grasps_sphere_fswo.py --compare-v1    # vs. the older fingertip-only scorer
```

For each grasp: forward-kinematics the full hand at its 6 joint angles, samples ~41
collision spheres across 13 links against the cube SDF (not just 5 fingertip points, so
finger thickness counts), gates out grasps with fewer than 3 fingers touching or a weak
FSWO score (contacts on non-opposing faces), then ranks survivors by
`(n_fingers, n_spheres, fswo)`. Writes `scores.json` with `best_idx` plus the full
ranking and per-grasp stats — see [`grasp_selection/README.md`](grasp_selection/README.md)
for the scoring details and known deviations from the spec.

`g1_pick_env_cfg.py` reads `scores.json`'s `best_idx` at import time
(`get_optimal_grasp_idx()`), falling back to the cache instead of recomputing whenever
possible (the BODex URDF/sphere assets it would need are gitignored). Override with
`G1_PICK_GRASP_IDX=<idx>` to pin a specific grasp (e.g. to reproduce an older
checkpoint trained against a different pick).

### 5. Cache the per-fingertip reach targets

```bash
cd grasp_selection
python cache_fingertip_contacts.py
```

For the active grasp (same `scores.json` by default), FK's the 5 fingertip links and
projects each onto its nearest cube face, writing `fingertip_contacts.json`. This is
what the `grasp_reach` reward in `mdp/grasp_goal.py` reads at training time — it never
needs the URDF itself. Re-run steps 4-5 any time the grasp library changes or you want
to force a different grasp.

## File structure

```
g1_pick/
├── __init__.py                      # Gym task registration (table above)
├── g1_pick_env_cfg.py               # Env config: scene, actions, observations, rewards, events, terminations
├── robot_cfg.py                     # G1 + Inspire hand articulation config
├── agents/rsl_rl_ppo_cfg.py         # PPO hyperparameters (RSL-RL)
├── mdp/
│   ├── grasp_goal.py                # Grasp-pose/contact/lift reward terms, reads the cached grasp library
│   ├── events.py                    # Domain randomization, resets
│   └── ...
├── grasp_sampler/
│   ├── ultradex_repo/               # UltraDexGrasp/BODex clone (gitignored)
│   ├── bodex_config_templates/      # Inspire-hand BODex configs to copy into ultradex_repo
│   ├── synthesize_inspire_grasps.py # Step 1: raw grasp candidates
│   ├── validate_grasps_isaaclab.py  # Step 2: physical validation + grip recentering
│   ├── build_goal_library.py        # Step 3: feasibility + wrap-quality filtering -> final library
│   ├── play_with_goal_markers.py    # Inference + video recording with goal-grasp overlay
│   ├── contact_check.py             # Diagnostic: fingertip-to-cube distance
│   ├── lift_capability_probe.py     # Diagnostic: real grip-force follow ratio
│   └── grasp_dataset/               # npz outputs of steps 1-3
├── grasp_selection/
│   ├── fswo.py                      # Step 4: force-closure scorer
│   ├── hand_model.py                # FK + cube SDF + collision spheres
│   ├── rank_grasps_sphere_fswo.py   # Step 4: CLI, writes scores.json
│   ├── cache_fingertip_contacts.py  # Step 5: writes fingertip_contacts.json
│   └── scores.json / fingertip_contacts.json   # Cached outputs consumed at training time
└── REWARD_CURRICULUM_LOG.md         # Full experiment history: what worked, what didn't
```
