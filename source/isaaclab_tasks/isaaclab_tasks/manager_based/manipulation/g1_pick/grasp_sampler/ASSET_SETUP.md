# Asset setup — what's missing and where it has to come from

## 0. Read this first — session summary, status, next steps

**Baseline before this work started:** `g1_pick` had a working, manually-tuned
RL policy (`g1_pick_env_cfg.py`'s `compute_task_reward` + 5 penalty terms,
reach→grasp→lift shaped by hand, no grasp-synthesis pipeline at all — see
`Misc./PROJECT_REFERENCE.md`). Everything below was added in one continuous
session on top of that, and none of it has been run yet (no GPU was available
during the session — this machine is that GPU).

### 0.1 Complete changelog (new session, chronological by feature)

**Plan A — curriculum-gated grasp-goal shaping rewards:**
- New `mdp/grasp_goal_curriculum.py`: `GraspGoalCurriculum`, a `CurriculumTermCfg`
  term exposing `env._ggc_ramp` in [0,1] — a smoothstep ramp driven by a rolling
  window of palm-to-cube reach quality, accumulated every step by
  `grasp_goal_palm_reward` via `GraspGoalCurriculum.accumulate(...)`. Fully
  independent of `mdp/curriculum.py::PickingCurriculumScheduler` (which is dead
  code, not wired into any active `EnvCfg`, and was left untouched).
- New `mdp/grasp_goal.py`: `SampleGraspGoal` (reset event, loads the goal
  library once, single-fixed-grasp mode only), `live_goal_pose` (recomputes the
  goal's world pose from the cube's *current* pose every call — no cache, so
  the staleness bug documented as Problem 14 in `Misc./UltraDex.md` can't recur
  by construction), `grasp_goal_palm_reward`, `grasp_goal_hand_config_reward`.
- `mdp/__init__.py`: exports both new modules.
- `g1_pick_env_cfg.py`: new `CurriculumCfg` (`grasp_goal_ramp` term), two new
  `RewardsCfg` terms (`grasp_goal_palm` w=1.0, `grasp_goal_hand` w=0.3), one new
  `EventCfg` term (`sample_grasp_goal`, placed after `reset_target_object`).
  New constants: `_ENABLE_GRASP_GOALS` (bool, **currently `False`** so the
  pre-existing working env is unaffected), `_FIXED_GRASP_IDX` (currently `-1`,
  a deliberately invalid placeholder), `_GRASP_LIBRARY_PATH`,
  `_GRASP_GOAL_PALM_BODY`. `task_reward` and the 5 original penalty terms are
  byte-for-byte unchanged.

**Plan B — FSWO grasp selector (replaces the nearest-palm heuristic):**
- New `grasp_sampler/grasp_selection/fswo.py`: `fswo_score`, `batch_fswo_scores`.
  Implements Lightning Grasp's FSWO (Yin & Abbeel, arXiv:2511.07418, Eq. 1).
- New `grasp_sampler/grasp_selection/select_optimal_grasp.py`: loads the goal
  library, scores all 32 grasps, writes `scores.json` with `best_idx`.
- New `grasp_sampler/check_grasps_offline.py`: generic (asset-agnostic) URDF FK
  engine via `pytorch_kinematics` — `link_origins()` (arbitrary-link FK, whole
  hand), `fk_tips()` (5 fingertips, thin wrapper over `link_origins`),
  `cube_surface_dist`, `geometric_soundness`, `tray_compatible`.
- New `grasp_sampler/build_goal_library.py` (stage 3: filter raw grasps to the
  32-grasp library), `synthesize_inspire_grasps.py` (stage 1: BODex synthesis
  driver), `validate_grasps_isaaclab.py` (stage 2: Kabsch calibration + grip
  re-centering, real Isaac Lab scene), `measure_coupling.py` / `probe_env_hand.py`
  (diagnostics, ported from `Misc./UltraDex.md`'s description).
- New `grasp_sampler/README.md` (end-to-end run commands), this file, and
  `grasp_sampler/bodex_config_templates/{inspire_right_sim2real.yml,
  hand_pose_transfer_inspire.yml, fc_right.yml}` — now also **copied into the
  real BODex content tree** at `ultradex_repo/third_party/BODex_api/src/bodex/
  content/configs/...` (see §1 below).
- `ultradex_repo/util/bodex_util.py` patched with the `elif hand_type ==
  'inspire': self.bodex_2_sim_q_idx = [0,1,2,3,4,5]` branch.
- Built `ultradex_repo/asset/object_mesh/cube/` from scratch (unit cube mesh +
  `info/simplified.json` + `urdf/coacd.urdf`, same layout as upstream's `bowl/`
  example) — this directory didn't exist at all before.

**Bugs found and fixed during a full pipeline + math re-audit (requested
explicitly, before touching the server):**
1. **FSWO math bug (real, verified against the actual paper PDF).** The NNLS
   reduction used raw `Q` as the design matrix instead of `Q`'s PSD square root
   `L`. Verified numerically against brute-force SLSQP on random contact sets:
   the raw-`Q` version is measurably wrong whenever the true optimum has an
   active non-negativity constraint. Fixed in `fswo.py`; corrected the same
   error in `Misc./optimizer.md` §2.2 and `Misc./quick_ref.md`.
2. `validate_grasps_isaaclab.py`'s frame-calibration step was using a **zero
   placeholder** instead of real URDF FK (before `link_origins()` existed) —
   now uses real FK.
3. `validate_grasps_isaaclab.py`'s calibration and re-centering were anchored
   to the robot's **pelvis** (`root_pos_w`) instead of the hand's own base link,
   and didn't correct for the hand's world orientation — fixed with
   `quat_apply_inverse` so both measurements are in the hand's own local frame,
   matching the URDF FK's implicit convention.
4. `build_goal_library.py`'s ranking criterion used **signed**
   `cube_surface_dist` — would have ranked a grasp with fingers penetrating deep
   into the cube as "better contact" than one lightly touching the surface.
   Fixed to rank by `abs()`.
5. `g1_pick_env_cfg.py`'s `grasp_goal_hand` reward's `SceneEntityCfg` was
   missing `preserve_order=True` — `joint_ids` could silently come out in the
   articulation's internal order rather than `HAND_JOINT_NAMES` order. Fixed
   (would likely have worked by coincidence on this specific hand, not by
   guarantee).
6. Config templates (`bodex_config_templates/*.yml`) were never actually copied
   into BODex's real `content/configs/` tree — fixed (copied into place;
   `fc_right.yml`'s `world:` block confirmed dead code for our call path by
   tracing `GraspSolverConfig.load_from_robot_config`, documented rather than
   deleted).

**Real Inspire Hand URDF (external asset, added by Shahid mid-session):**
- A dex_urdf-derived `inspire_hand_right.urdf` (+ meshes) is now in place at
  `ultradex_repo/third_party/BODex_api/src/bodex/content/assets/robot/
  inspire_hand/`. Verified against the USD (see §2 below) — root link `base`,
  bare joint/link names (no `R_`/`L_` prefix), slave joints **already** frozen
  at the exact empirically-measured USD postures, real `*_tip` links present.
  `check_grasps_offline.py` and the config templates were updated to match its
  real, confirmed names.

**Manual-curriculum artifact check (requested, before any of the above):**
Confirmed via git log + diffing the three env-cfg snapshots that the *current*
`g1_pick_env_cfg.py` reward is byte-identical to the last manually-tuned stage
(`10cube_clutter_pick` + `new_cfg_jerk_penalty`) — no stale reward logic is
silently active. `mdp/rewards.py`'s ~11 unused functions are harmless leftover
exploration, confirmed via grep to be referenced by no `RewardsCfg`.

**Git / repo setup:**
- Forked `daatsi-aeres/IsaacLab` → `Shahid5298/IsaacLab`, set **private**.
  `origin` = the fork (push target), `upstream` = the original.
- Stripped the 4 nested `.git` dirs inside `ultradex_repo/` (itself +
  `pytorch3d`, `curobo`, `BODex_api`) so they track as plain files instead of
  broken gitlinks.
- Removed a leftover `ultradex_repo/.gitignore` (from when it was a standalone
  repo) that was silently excluding `third_party/` and other real content.
- Full `.gitignore` audit across all 8 gitignore files in the repo (root +
  the ones inside `BODex_api`/`curobo`/`pytorch3d`) — confirmed nothing else
  important is currently hidden; the only thing excluded under `g1_pick/` is
  `__pycache__/`. One thing to remember for later: the root `.gitignore` has a
  blanket `**/*.usd` rule — if you ever add a *new* `.usd` file anywhere in
  this repo, it'll need `git add -f`.
- Two commits pushed to `origin/main`: the full grasp-goal+FSWO pipeline, then
  a follow-up including `third_party/`.

**Server access (this session):**
- Laptop → server: SSH key `~/.ssh/id_ed25519_unicorn1`, `~/.ssh/config` alias
  `unicorn1` (`User shahid` — lowercase; `Shahid` does not work). Working.
- Server → GitHub: new key generated on the server, added to `Shahid5298`'s
  GitHub account. Working (`ssh -T git@github.com` confirmed).
- VSCode Remote-SSH connected successfully.

### 0.2 Current status (as of this message)

All code is written, bug-audited, committed, and pushed to
`github.com/Shahid5298/IsaacLab` (private, branch `main`). SSH access
(laptop→server and server→GitHub) is confirmed working. **Nothing has been run
yet** — no clone on the server confirmed complete, no conda envs built, no
BODex synthesis attempted, no training resumed. `_ENABLE_GRASP_GOALS = False`,
so if you ran training right now with none of the steps below done, you'd just
get the original (pre-session) working policy back — the new pipeline is
fully inert until deliberately turned on.

### 0.3 Immediate next steps, in order

1. `git clone git@github.com:Shahid5298/IsaacLab.git` on the server (if not
   already done).
2. Check the server's GPU: `nvidia-smi`. The torch/CUDA versions in
   `Misc./UltraDex.md` §2.1 were chosen for a laptop RTX 5060 (Blackwell,
   sm_120) — confirm whether this server's GPU needs different versions before
   following those install steps verbatim.
3. Set up the `env_isaaclab` conda env (Isaac Sim + Isaac Lab) — standard
   NVIDIA install, independent of git (Isaac Sim itself is gitignored,
   `_isaac_sim*`).
4. Set up the `ultradex` conda env per `Misc./UltraDex.md` §2.1–2.2: torch
   (version per step 2 above), the CUDA toolkit inside the env, `pip install
   -e .` for `pytorch3d` and `BODex_api` (now present under
   `ultradex_repo/third_party/`, no need to re-clone), `conda install coal`,
   the `wp.torch.*`→`wp.*` patch in `world_mesh.py`, the coal `-std=c++17` fix.
5. Run `gen_spheres.py` (in `ultradex_repo`) against the real collision meshes
   now present at `.../inspire_hand/meshes/collision/*.obj` to fill in
   `collision_spheres: {}` in `bodex_config_templates/inspire_right_sim2real.yml`
   (already copied to its real path — edit it there, or re-copy after editing
   the template). This was the one piece structurally blocked until the real
   URDF existed; it's unblocked now.
6. Run the offline pipeline in order (`grasp_sampler/README.md` has exact
   commands): `synthesize_inspire_grasps.py` → `validate_grasps_isaaclab.py` →
   `build_goal_library.py` → `grasp_selection/select_optimal_grasp.py`.
   Sanity-check along the way per `grasp_sampler/README.md` / this file's §1–2
   (Kabsch residual near 0.3mm is a good sign; watch for the narrower finger
   joint range noted in §2).
7. Set `_FIXED_GRASP_IDX` (to the printed `best_idx`) and
   `_ENABLE_GRASP_GOALS = True` in `g1_pick_env_cfg.py`.
8. Train (`grasp_sampler/README.md` has the exact `train.py` invocation),
   watching `Curriculum/grasp_goal_ramp` and `Episode_Reward/grasp_goal_*` in
   TensorBoard per that doc's guidance.

Before trusting synthesis output: diff `bodex_config_templates/fc_right.yml`'s
`seeder_cfg`/`grasp_contact_strategy`/`grasp_cfg` numbers (currently copied
from xhand's own file, unverified for this hand+cube) against what actually
gets produced — if synthesis silently fails, converges to one pose, or never
proposes tight grasps, that's the first place to look.

### 0.4 High-level plan (longer horizon, from `Misc./PROJECT_REFERENCE.md`)

1. Grasp sampler + FSWO optimizer (this session's work) → train with
   goal-shaping active, confirm `target_lifted` starts firing.
2. Goal-conditioned policy: extend observation 96→109 dims (append selected
   goal — palm pose + 6 joint targets) to remove the hidden-goal ceiling
   (Problem 17, `Misc./UltraDex.md`) and re-enable per-episode grasp selection
   over the full 32-grasp library instead of the current single fixed index.
3. Teacher→student distillation: point-cloud observation, DP3 backbone,
   following the ClutterDexGrasp pipeline (`Research Papers/clutterdex.pdf`).
4. Bimanual extension.

---

Everything in `grasp_sampler/` that is pure orchestration, math, or Isaac-Lab-side
code (the FSWO scorer, the curriculum-gated reward integration in `mdp/`, the
synthesis/validation/library-building driver scripts, the generic URDF-FK engine in
`check_grasps_offline.py`) is implemented and correct regardless of hand geometry.

Two things are genuinely external assets this repo does not have and cannot
fabricate without risking silent, undetectable errors (exactly the "silent
divergence between models" failure mode `Misc./UltraDex.md` S6 identifies as the
root cause of most problems the original porting effort hit):

## 1. `ultradex_repo/` — the UltraDexGrasp + BODex_api clone

**Status (2026-07-13): fully cloned.** `UltraDexGrasp` + `third_party/{pytorch3d,curobo,BODex_api}`
are all present at `grasp_sampler/ultradex_repo/`. Still needed before running anything:
the CUDA 12.8/torch 2.11/sm_120 build adjustments in `Misc./UltraDex.md` S2.1 (the
upstream README's own `torch==2.4.1+cu118` install command needs overriding on this
GPU) and the warp/coal patches in S2.2 (`pip install -e .` for pytorch3d/BODex_api,
`conda install coal`, etc. haven't been run yet -- only `git clone` has).

`ultradex_repo/util/bodex_util.py` already had a real, working `hand_type == 'xhand'`
branch to model the Inspire one on -- I've added `elif hand_type == 'inspire':
self.bodex_2_sim_q_idx = [0, 1, 2, 3, 4, 5]` directly (identity mapping, matching
`Misc./UltraDex.md` S2.2's description). That part of "teaching BODex the Inspire
Hand" is done; no longer a TODO.

I also found real example configs (`configs/robot/xhand_right_sim2real.yml`,
`configs/robot/hand_pose_transfer/xhand.yml`, `configs/manip/sim_xhand_sim2real/fc_right.yml`)
and used them to rewrite `bodex_config_templates/*.yml` against BODex's actual schema
instead of a guess — see `bodex_config_templates/README.md` for what's still
genuinely unverified in them (mostly: collision sphere geometry and exact dex-urdf
link names, both blocked on point 2 below).

## 2. `ultradex_repo/third_party/BODex_api/src/bodex/content/assets/robot/inspire_hand/inspire_hand_right.urdf` — real hand geometry

**Status (2026-07-13): added, and verified good.** A real `inspire_hand_right.urdf`
(+ `_left.urdf`, `_glb` variants, and `meshes/{visual,collision}/`) is now in place.
Its `LICENSE.txt` identifies it as dex_urdf's own model, "derived from the step file
provided on the inspire official website," so this is the generic dexsuite/dex-urdf
model discussed earlier (not the `unitree_ros` DFQ file I'd pointed at) — but it
checks out well against our actual robot:

- Root is a zero-size `base` link (offset from the real palm body `hand_base_link`
  by a fixed joint) — no `R_`/`L_` prefix on any joint/link name (different
  convention from the USD, matched by order not by name — see
  `check_grasps_offline.ACTUATED_JOINT_NAMES`'s comment).
- **The slave joints are ALREADY frozen to `type="fixed"`, at origin rpy z-values of
  exactly `-0.16`, `-0.24`, `1.15`, `1.15`, `1.15`, `1.15`** for
  thumb_intermediate/thumb_distal/{index,middle,ring,pinky}_intermediate — an
  *exact* match to the empirically-measured USD postures in `Misc./UltraDex.md` S3.1
  (Discovery 1) / `check_grasps_offline.FROZEN_SLAVE_POSTURES_RAD`. Modification #3
  from `Misc./UltraDex.md` S7.5 is already done in this file — nothing further
  needed there. (These are oddly specific numbers to appear by coincidence; this
  file's slave-joint freeze was almost certainly done against this exact USD
  already, whether by whoever prepared it upstream or by following the recipe I
  wrote in the previous version of this doc.)
- It already has proper zero-size `*_tip` links (`thumb_tip`, `index_tip`, ...),
  each a fixed joint off the real terminal link with a measured offset — better
  than the "use the terminal link itself" fallback I'd assumed before this existed.
- `thumb_proximal_yaw_joint`/`index_proximal_joint` upper limits are close to but not
  identical to the USD's (this file: 1.308/1.47 rad; USD: 1.3/1.7 rad,
  `tools/hand_structure_reference.txt`) — **the 4 finger proximal joints in
  particular are noticeably narrower here (1.47 vs 1.7 rad, 84° vs 97°).** Not a
  correctness bug — BODex will just search a smaller-than-real range, so synthesized
  grasps will be a conservative subset of what the physical hand can actually do
  (never a fully-curled grip). Worth knowing if synthesis never proposes tight
  grasps; not worth blocking on.

`check_grasps_offline.py` (`URDF_ROOT_LINK`, `FINGERTIP_LINKS`,
`ACTUATED_JOINT_NAMES`) and `bodex_config_templates/{inspire_right_sim2real,
fc_right}.yml` have been updated to match this file's real, confirmed names —
nothing left to rename before running synthesis.

**Still TODO, now unblocked:** `collision_spheres` in
`bodex_config_templates/inspire_right_sim2real.yml` — real collision meshes exist
now (`meshes/collision/*.obj`), so `gen_spheres.py` (in `ultradex_repo`, once its
build steps from §1 are done) can actually be run against them. That's an execution
step I can't do without a running environment; still marked `{}` in the template.
BODex's internal optimization-schedule numbers (`seeder_cfg`/
`grasp_contact_strategy`/`grasp_cfg` in `fc_right.yml`) also remain copied from
xhand's own file as a starting point, not verified for this hand + cube.

## What does NOT need this

`mdp/grasp_goal.py`, `mdp/grasp_goal_curriculum.py`, and the FSWO scorer
(`grasp_selection/fswo.py`, `select_optimal_grasp.py`) only depend on the
**documented npz schema** (`grasp_pose` (N,1,3,13), `T_usdbase_urdfbase` (4,4)) —
that contract is stable regardless of how the library was generated, so all of the
RL-training-side code is real, complete, and ready to run the moment a valid
`cube_5cm_grasps_valid.npz` exists (whether you generate it here or copy it over
from your friend's laptop).
