# Grasp sampler pipeline — end to end

Read `ASSET_SETUP.md` first if you haven't already — two external assets
(the UltraDexGrasp/BODex clone, and real Inspire-hand URDF geometry) are not
in this repo and must be supplied before stage 1/2 below can run.

```
STAGE 1 (ultradex env, GPU)      STAGE 2 (env_isaaclab, GPU)         STAGE 3 (either env, CPU)
synthesize_inspire_grasps.py  →  validate_grasps_isaaclab.py     →  build_goal_library.py
  100 raw grasps                   calibration + re-centering         32-grasp goal library
  cube_5cm_grasps.npz               cube_5cm_grasps_recentered.npz     cube_5cm_grasps_valid.npz
                                                                              │
                                                                              ▼
                                                          grasp_selection/select_optimal_grasp.py (CPU)
                                                              FSWO scores → best_idx → scores.json
                                                                              │
                                                                              ▼
                                                   set _FIXED_GRASP_IDX in g1_pick_env_cfg.py
                                                                              │
                                                                              ▼
                                                         train.py  →  play.py (evaluate)
```

## Commands

```bash
# --- Stage 1: BODex synthesis (needs ultradex_repo/ + the Inspire URDF, ASSET_SETUP.md) ---
conda activate ultradex
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
cd grasp_sampler/
python synthesize_inspire_grasps.py
# -> grasp_dataset/cube_5cm_grasps.npz  (100 raw grasps)

# --- Stage 2: calibration + empirical grip re-centering (needs a live Isaac Lab sim) ---
conda activate env_isaaclab
source _isaac_sim/setup_conda_env.sh
python validate_grasps_isaaclab.py --grasp_file grasp_dataset/cube_5cm_grasps.npz
# -> grasp_dataset/cube_5cm_grasps_recentered.npz

# --- Stage 3: filter to the 32-grasp goal library (CPU only) ---
python build_goal_library.py
# -> grasp_dataset/cube_5cm_grasps_valid.npz   ★ the one file training reads

# --- Grasp selection: FSWO force-closure ranking, replaces the nearest-palm heuristic ---
mkdir -p grasp_selection  # already created; harmless if it exists
python grasp_selection/select_optimal_grasp.py \
    --library grasp_dataset/cube_5cm_grasps_valid.npz --lam 1.0 --out grasp_selection/scores.json
# prints best_idx -- copy it into g1_pick_env_cfg.py:
#     _FIXED_GRASP_IDX = <best_idx>
#     _ENABLE_GRASP_GOALS = True   (already the default)

# --- Diagnostics (only needed if the USD hand is ever edited) ---
python measure_coupling.py     # bare-scene slave-joint sweep
python probe_env_hand.py       # same sweep through the real Isaac-G1-Pick-v0 env

# --- Train (from the IsaacLab repo root, not this directory) ---
cd /path/to/IsaacLab
conda activate env_isaaclab
source _isaac_sim/setup_conda_env.sh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Pick-v0 --headless --num_envs 1024 --max_iterations 5000

# --- Resume a crashed/stopped run ---
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Pick-v0 --headless --num_envs 1024 \
    --resume --load_run <timestamp_folder> --checkpoint model_<N>.pt

# --- Evaluate ---
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-G1-Pick-Play-v0 --num_envs 64 --checkpoint /path/to/model_<N>.pt
```

## What to watch in TensorBoard

Same as `Misc./UltraDex.md` S5, with one addition: `Curriculum/grasp_goal_ramp`
(logged automatically — `CurriculumManager.reset()` reports every term's
`__call__` return value under `Curriculum/<term_name>`) should rise smoothly
from 0 toward 1 as `Episode_Reward/task_reward` climbs, then flatten near 1
once the base policy reliably reaches the cube. If it's still 0 well past
iteration ~1000, `phi_lo`/`phi_hi` in `CurriculumCfg.grasp_goal_ramp` (in
`g1_pick_env_cfg.py`) are probably too high for how good the base reach reward
actually gets — lower them.

## Training without grasp goals

Set `_ENABLE_GRASP_GOALS = False` at the top of `g1_pick_env_cfg.py` to train
the base policy (task_reward + 5 penalties, unchanged from before this work)
with none of the above wired in — useful before the goal library exists, or as
an ablation baseline once it does.
