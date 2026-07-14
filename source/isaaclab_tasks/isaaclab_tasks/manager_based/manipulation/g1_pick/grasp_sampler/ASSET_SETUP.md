# Asset setup — what's missing and where it has to come from

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
