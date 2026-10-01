# G1 Pick — Reward Curriculum Log

Running record of the manual reward curriculum: what was tried, what happened, and why.
Kept separate from `MDP_REPORT.md` (which documents the *current* MDP as a reference
spec) — this file is a chronological decision log instead, specifically so this
knowledge doesn't get lost the way it did with Siddharth's earlier manual-curriculum
attempt (he no longer remembers the details and can't transfer them). Update this file
every time a phase or reward change is tried, not just when something works.

**How to use this file**: each attempt gets its own entry under "What Didn't Work" (or
"What Worked" if validated first try, which is rare). When an entry that started in
"What Didn't Work" is later confirmed to actually work (after a fix, a retry, or more
training), move its entry up to "What Worked" — don't duplicate it, don't delete the
history of what was tried before it worked.

---

## THE PLAN (current) — one phase, from scratch

**Owner of this plan: Sufian (PhD student, adviser). Adopted 2026-08-27.**

Everything below this section under "PREVIOUS ATTEMPTS — WHAT DID NOT WORK" is retained as
history only. **Do not take reward or penalty ideas from it.** Per Sufian, the inherited
stack (grown out of Siddharth's original) is too unstable to build on, and reusing pieces of
it risks pulling the policy back toward the same behaviours. The record supports that
judgment: it had grown to 15 active reward terms and never once produced a lift.

Sufian has run this exact recipe before with his own students and had the cube being picked
up reliably **within ~500 iterations**.

### Reward — exactly three terms, plus exactly one penalty

| # | term | what it is |
|---|---|---|
| 1 | `grasp_pose` | palm to the UltraDexGrasp goal pose (position + orientation), weight 2.0 |
| 2 | `lift` | cube height above rest, LINEAR (not tanh), clamped 20cm, weight 20.0 |
| 3 | `fingertip_contact` | **real PhysX contact force** between the 5 fingertips and the cube, weight 2.0 |
| — | `action_smoothness` | the ONLY penalty, weight -3.0 |

Nothing else exists. `grasp_goal_hand`, `grasp_reach`, `finger_contact`, `grasp_envelope`,
`lift_attempt_velocity`, `lift_progress`, `sustained_reach_bonus`, `joint_speed`,
`target_object_accel`, `distractor_accel`, `fingertip_impact` and `task_reward`'s internal
reach/grasp/success terms are all gone.

Term 3 is the one that addresses the root cause found on 2026-08-26: every reward this
project ever used was GEOMETRIC (fingertip position, joint angle), and the hand joints are
position-controlled, so a finger commanded to where it already rests exerts no force. A hand
hovering at zero distance scored identically to one gripping. `fingertip_contact` reads
newtons from a PhysX ContactSensor filtered to the cube and **cannot be earned without
actually pressing on it**.

### Penalties → terminations

The deleted penalties become hard terminations, because a penalty is a gradient the policy
can trade against reward (and demonstrably did) while a termination is a constraint it
cannot buy past:

- `excessive_joint_speed` — any policy joint above 10 rad/s (20x the actuator limit).
- `cube_disturbed` — cube shoved >10cm **laterally** from its episode start. Lateral, not
  total, because vertical motion is the goal. Sized from measurement: slamming the hand shut
  moved the cube 7.3cm mean / 19.7cm max sideways while lifting it 0.09cm.

### Physics / environment

- **Friction domain-randomised 0.2–0.8** on hand and cube (`startup` mode). Both a widening
  and a lowering: the cube was pinned at 1.0 and the hand's was **never set at all**.
- **Cube mass randomised 200–500 g** per reset (`abs` operation).
- **Action clipping [-1, 1]** — previously unset entirely. Measured beforehand: the trained
  policy sustained mean |action| 1.32, 67% above 1.0, peak 4.65.
- **Joint velocity limit 0.5 rad/s** on arm and hand (arms were 0.6).
- **`contact_offset` 0.02 / `rest_offset` 0.002** on hand and cube, to bound interpenetration
  to 1–2 mm. Measured before: 445/478 steps interpenetrating, worst −1.71 cm.
- **Empty environment only** (`Isaac-G1-Pick-Empty-v0`) — distractors at the table margins,
  never on the tray.

### THREE setup bugs found and fixed while wiring this up

Every one of them silently produces a permanently-zero contact reward — no crash, no
warning in the training summary — and would have wasted the whole run:

1. The ContactSensor `prim_path` cannot use `.*` to cross `/` boundaries. The fingertip
   bodies sit three levels under `/Robot`, so the pattern must be fully qualified. The
   failure mode is misleading — it raises *"could not find any bodies with contact reporter
   API"*, which reads as though `activate_contact_sensors` is missing even when it is set.
2. `activate_contact_sensors=True` is required on **both** sides of a filtered contact pair —
   the robot **and** the cube.
3. **A filtered ContactSensor must cover exactly ONE body.** A single 5-body sensor over
   2048 envs makes PhysX expect `5*2048 = 10240` filter prims while only 2048 cubes exist:
   `Filter pattern ... did not match the correct number of entries (expected 10240, found
   2048)`. **This is not fatal** — training proceeds normally and `force_matrix_w` stays all
   zeros forever. Fixed by using five separate single-body sensors
   (`contact_thumb/index/middle/ring/pinky`), giving 1:1 counts.

Bug 3 nearly slipped through: an earlier check read `filtered=0.000N` with the correct-looking
shape `(32, 5, 1, 3)` and non-zero `net_forces_w` (~29 N), and that was misread as "the filter
works and is truthfully reporting the fingertips press the tray, not the cube". It was not —
the filter was dead. **A plausible-looking zero is indistinguishable from a broken sensor
unless you force the condition that must produce a non-zero.**

Verified working before launch, by forcing the hand shut so contact is guaranteed:
filtered force reads **0.835 N / 0.838 N / 0.692 N** across steps with no filter error. The
reward is live. (The values are small, which correctly reflects fingertips that only graze
the cube — exactly the deficiency this reward exists to fix.)

### Run

`sufian_v1_scratch` — **from scratch** (no `--resume`), `Isaac-G1-Pick-Empty-v0`,
**2048 envs, 1000 iterations**.

**What success looks like**: `fingertip_contact` rising off zero at all is the first real
milestone — it is the first metric in this project that cannot be faked by geometry. Then
`Episode_Termination/target_lifted`. Judge by those, not by `grasp_pose`, which has
repeatedly risen while nothing physical improved.

---

# PREVIOUS ATTEMPTS — WHAT DID NOT WORK

Everything below is historical record, kept per the "don't delete the history" rule in the
header. **Do not source reward or penalty designs from it** (see the current plan above).
The single most useful thing in it is the diagnostic tooling and the measured facts —
particularly the 2026-08-26 finding that proximity was never the blocker and grip force was.

## (historical) The Plan — 5 Phases

The roadmap this whole curriculum is working toward. "What Worked"/"What Didn't Work"
below track individual *attempts* within a phase; this section is the fixed target each
attempt is aimed at. Implemented entirely via `RewardsCfg` weights/params in
`g1_pick_env_cfg.py` — every term for every phase already exists in the file from the
start, each currently-zeroed one carrying a `# target: X -- Phase N` comment. No RewTerms
get added or removed between phases, only their weights change, and advancing is a manual
decision (watch TensorBoard converge, not a fixed iteration count). Full current-state
formulas for every term live in `MDP_REPORT.md` §5 — this section only restates the
formula for whatever's *newly added* at each phase, so it reads standalone.

Task stays `Isaac-G1-Pick-Empty-v0` (no distractors near the workspace) through Phases
1-4; Phase 5 is the only phase that changes task, ramping through the three
distractor-difficulty stages in `MDP_REPORT.md` §9a.

### Phase 1 — Reach

**Status: done**, handed off to Phase 2 at checkpoint 250 of
`phase1_reach_5cm_holdsteps10` (2026-08-19) — see "What Worked" below for the full
validation and attempt history.

Adds: `task_reward` with only `reach_weight=1.0` on, `action_smoothness`, and
`joint_speed` (the "absolute-velocity penalty").

$$
\bar{\mathbf p}_{\text{tip}} = \frac{1}{5}\sum_{i=0}^{4}\mathbf p_{\text{tip},i}
\qquad
r_{\text{reach}} = 1 - \tanh\!\left(\frac{\lVert \bar{\mathbf p}_{\text{tip}} - \mathbf p_c \rVert}{0.25}\right)
$$

$$
p_{\text{smooth}} = 0.005\sum_i (a_i - a_i^{\text{prev}})^2
\qquad
p_{\text{joint\_speed}} = \tanh\!\left(\frac{\sum_{j=1}^{13} \dot q_j^2}{\text{threshold}}\right)
$$

**Not in the original plan, added mid-Phase-1 (2026-08-19) once training exposed real
gaps**: `target_object_accel` (moved up from Phase 4 — the reach_rew exploit of
disturbing the cube doesn't need `lift_weight`/`success_weight` active, so leaving this
penalty off until Phase 4 left that exploit open the whole time) and
`sustained_reach_bonus`/`sustained_reach` termination (Phase 1 had no positive-outcome
termination at all otherwise). See "What Didn't Work" for the full story on both.

### Phase 2 — Grasp

**Status: done** (2026-08-21) — merged with Phase 3 and run together from the fixed
Phase 1 foundation (`phase1_reach_palmpos_cuberot45`), rather than either sequential
ordering tried earlier. See "[Phase 2+3 merged]" under "What Worked" for the full data.
The `grasp_weight` formula below was also superseded by the pairwise-coverage redesign
(2026-08-20, see "Proposed fix: pairwise coverage reward") before this merged run — the
thumb/finger-split formula documented just below is what Phase 2 ran with historically,
not what's active now.

Older history, kept for the record: originally ran 2026-08-19 from Phase 1's checkpoint
250 and plateaued cleanly by iteration ~600-650 (see "What Worked" above); was then
reordered to run AFTER Phase 3 (2026-08-20) after the pose-mimicking terms declined when
run alongside it — see "What Didn't Work" for that data. Also carries a metric change to
`sustained_reach_termination` (centroid-to-center → average per-fingertip distance to the
cube surface, 1cm threshold) — see "What Worked" above.

Adds: `task_reward.grasp_weight` (thumb/finger proximity to the cube surface).

$$
d_{\text{thumb}} = \max(\lVert\mathbf p_{\text{tip},0}-\mathbf p_c\rVert - 0.025,\ 0)
\qquad
d_{\text{finger}} = \max\!\Big(\tfrac14\textstyle\sum_{i=1}^4\lVert\mathbf p_{\text{tip},i}-\mathbf p_c\rVert - 0.025,\ 0\Big)
$$

$$
r_{\text{grasp}} = \frac{1}{2}\Big[\big(1-\tanh\tfrac{d_{\text{thumb}}}{0.055}\big) + \big(1-\tanh\tfrac{d_{\text{finger}}}{0.055}\big)\Big]
$$

Counted into `task_reward` at weight $2$. This is the same soft AND gate
($g = (1-\tanh\tfrac{d_{\text{thumb}}}{0.06})(1-\tanh\tfrac{d_{\text{finger}}}{0.06})$)
that Phase 4's lift/success terms are gated on, so getting this phase genuinely solid
(not just reach_rew-adjacent) directly de-risks Phase 4.

### Phase 3 — Optimal grasp pose mimicking

**Status: done** (2026-08-21) — merged with Phase 2, see Phase 2's status note above and
"[Phase 2+3 merged]" under "What Worked" for the full data. `grasp_goal_palm` converged
cleanly this time (0.045 -> ~1.4, held flat/noisy, no decline) once run on top of the
Phase 1 root-cause fix (position-only `grasp_goal_palm` active from Phase 1 itself, plus
the cube spawn-yaw fix) instead of either sequential ordering tried earlier.

Older history, kept for the record: REORDERED (2026-08-20) to run BEFORE Phase 2 instead
of after — a first attempt ran these terms alongside `grasp_weight` already active (per
the original plan order) and `grasp_goal_palm`'s own reward DECLINED over 500 iterations
instead of converging; see "What Didn't Work" for the full data and diagnosis. That
reorder run resumed from Phase 1's own checkpoint 250 (not Phase 2's), with
`grasp_weight` held at 0, and was itself inconclusive (see "What Didn't Work") before the
Phase 1 root-cause investigation superseded both attempts.

Adds: `grasp_goal_palm` (weight $2.0$), `grasp_goal_hand` (weight $1.0$), `grasp_reach`
(weight $1.0$), and flips `task_reward.pose_gated_success` to `True`. Pulls the hand
toward a specific pre-vetted grasp (currently grasp #12 from the UltraDexGrasp/BODex
library) instead of just "close to the cube center" — full mechanism in `MDP_REPORT.md`
§5.2.

$$
r_{\text{palm}} = 0.5\big(1-\tanh\tfrac{d_{\text{pos}}}{0.15}\big) + 0.5\big(1-\tanh\tfrac{d_{\text{ang}}}{0.6}\big)
$$

$$
r_{\text{hand}} = \underbrace{\operatorname{clamp}(1-\tanh\tfrac{d_{\text{pos}}}{0.20},\,0,\,\infty)}_{\text{palm-proximity gate}} \cdot \big(1-\tanh\tfrac{\lVert\mathbf q-\mathbf q^\star_{\text{hand}}\rVert}{0.5}\big)
$$

$$
r_{\text{grasp\_reach}} = \frac15\sum_{i} \big(1-\tanh\tfrac{\lVert\mathbf p_{\text{tip},i}-\mathbf c^\star_i\rVert}{0.05}\big)
\qquad \text{(each fingertip toward its own Cartesian contact point on the cube)}
$$

This is the phase `CONTEXT.md` flags as the open problem worth watching closely:
`grasp_goal_palm` sat flat at $\approx 0.03$ across 6000+ iterations in the old
warm-started curriculum despite dense weight-$2.0$ reward — the policy picked reliably
using its own discovered grasp, never converging toward #12's pose. `grasp_reach` exists
specifically as a task-space (not joint-space) attack on that same problem.

### Phase 4 — Pick up (lift + success)

**Status: in progress**, launched 2026-08-21 from checkpoint 699 (final) of
`phase2and3_merged_from_ckpt200` — see "[Phase 4] Lift + success" under "What Worked" (or
"What Didn't Work" if it doesn't hold up) for the launch rationale and data.

Adds: `task_reward.lift_weight` and `task_reward.success_weight` (both $0 \to 1$).

$$
\Delta h = \operatorname{clamp}(p_{c,z}-0.845,\ 0,\ 0.30)
\qquad
r_{\text{lift}} = 2.0\,\Delta h \cdot g \ \in [0,\ 0.6]
$$

$$
r_{\text{success}} = \mathbb 1[p_{c,z} > 1.134]\cdot g \cdot \big(0.2 + 0.8\cdot\text{pose\_match}\big)\cdot 1000
$$

where $g$ is Phase 2's grasp gate and $\text{pose\_match}$ reuses Phase 3's palm/orient
error against the live goal (`MDP_REPORT.md` §5.1.6) — lifting always pays at least 200,
lifting in grasp #12's pose pays up to 1000. Both terms are gated on $g$ so the policy
can't earn lift/success by shoving the cube with an open palm — this is exactly the
loophole `target_object_accel` (pulled forward into Phase 1) exists to close a second,
independent way.

### Phase 5 — Introduce clutter

Adds: `fingertip_impact` (weight $-2.0$), `distractor_off_tray` (weight $-10.0$),
`distractor_drop` (weight $-100.0$), and ramps the task itself through
`Isaac-G1-Pick-Stage1-v0` → `Isaac-G1-Pick-Stage2-v0` → `Isaac-G1-Pick-v0` (full tray
clutter) — see `MDP_REPORT.md` §9a for what each stage's distractor layout looks like.

**On when clutter should start** (the user asked for an opinion on this while the plan
was being drawn up): not before Phase 5, i.e. only after grasp + pose-mimicking + lift +
success are all solid on the empty task. Reasoning: the entire point of doing this
curriculum from-scratch with one term added at a time (vs. the old warm-started
multi-term approach) is that a training stall can be attributed to whatever was *just*
added — see the intro to `MDP_REPORT.md` §9c. Introducing distractor interference earlier
than Phase 5 would confound that attribution: a stall could be "the grasp-pose mimicking
isn't working" or it could be "a distractor is in the way," and disentangling the two
costs exactly the debugging clarity this whole staged approach was built to preserve.
Phase 1 already broke this discipline once out of necessity (`target_object_accel` and
`sustained_reach_bonus` both got pulled forward mid-phase, see above) — that was a
reactive fix to a discovered exploit/gap, not a planned early introduction of Phase-5-style
complexity, and it's worth not making a habit of it.

---

## What Worked

### [Phase 1] Reach-only — validated 2026-08-19, handed off to Phase 2 at checkpoint 250

**Dates**: 2026-08-18 → 2026-08-19 (three attempts before validation — see below for the
full history, moved up from "What Didn't Work" per this file's own convention rather than
duplicated).
**Final task/run**: `Isaac-G1-Pick-Empty-v0`, trained from scratch,
`2026-08-19_20-47-28_phase1_reach_5cm_holdsteps10`.

**What actually converged.** `task_reward` (reach_weight=1.0 only) + `action_smoothness` +
`joint_speed` (bounded, rescoped to the 13 policy-controlled joints — see attempt 2 below)
+ `target_object_accel` + a hard `velocity_limit_sim` clamp on the arm (0.6 rad/s) and
hand (0.5 rad/s) actuators (`robot_cfg.py`). Pulled directly from TensorBoard
(`Policy/mean_noise_std`, `Episode_Reward/*`, `Train/mean_reward`) across the final run:

| Iter | task_reward | Train/mean_reward | Policy/mean_noise_std | joint_speed |
|---|---|---|---|---|
| 200 | 0.707 | 4.82 | 0.128 | -0.094 |
| 300 | 0.705 | 4.88 | 0.106 | -0.086 |
| 400 | 0.703 | 5.06 | 0.093 | -0.067 |
| 499 | 0.705 | 5.06 | 0.088 | -0.059 |

Task performance (`task_reward`, `Train/mean_reward`) is flat from ~iteration 200 onward —
the policy stops improving at the task after that point. `Policy/mean_noise_std` (PPO's
actual action-noise std) keeps shrinking the whole way, still declining at 499 (0.128 →
0.088, ~31% reduction) — meaning later checkpoints trade away exploration budget for
calmer motion (`joint_speed`/`action_smoothness` keep improving too) without any further
task benefit. A dwell-time diagnostic (`speed_reach_diag.py`, new standing tool, 64 envs ×
700 steps) on checkpoint 450 confirmed the reach behavior itself is a genuine, stable
equilibrium, not still-evolving: episodes spend ~all of their ~235 steps within 10cm of
the cube (mean dwell 221.5 steps), and >50% spend 10+ consecutive steps within 7cm — this
is solid, reliable, well-converged reach behavior.

**Checkpoint 250 chosen for the Phase 2 hand-off** (user's decision, reasoned through
together): since task performance is already flat by 200 and only exploration budget is
still being spent between 200 and 499 for no task gain, handing Phase 2 a less-converged
checkpoint (250, not 499) preserves more of that action-noise budget for discovering the
NEW finger-curling behavior `grasp_weight` needs — behavior Phase 1 never incentivized at
all. Same principle applied to the Phase 2/3 merge question (kept separate, see the
project's own `MDP_REPORT.md` §5.2.3 precedent of `grasp_goal_palm` sitting flat for
6000+ iterations when a coarser reward already paid out): don't over-converge a phase
before handing off to the next, more specific one.

**The sustained_reach bonus never fired, and that's not actually a Phase 1 failure.**
Across all three attempts (see full history below), the +10 sparse bonus for holding the
fingertip *centroid* within a shrinking threshold of the cube *center* never fired once —
not even after `distance_threshold` went 10cm→5cm and `hold_steps` went 30→10. A
dwell-time diagnostic on checkpoint 450 (128 episodes) explained why: the policy reliably
sits within 10cm for nearly the entire episode, but the longest any episode ever held
under 5cm was 8 consecutive steps. The centroid-to-center metric was the actual problem —
it's satisfiable by fingers hovering AROUND the cube, spread out, with none of them
actually close to the surface, which is exactly the shape Phase 1's own reward (reach only,
no grasp incentive) naturally produces. Rather than keep chasing threshold/hold_steps, the
metric itself was redefined at Phase 2's start to average per-fingertip distance to the
cube SURFACE (1cm threshold — see `sustained_reach_termination`'s docstring in
`g1_pick_env_cfg.py`), a genuinely more grasp-relevant signal now that `grasp_weight` is
active to reward exactly the finger-closing behavior this metric requires.

**Full attempt history** (kept for the record, per this file's stated purpose):

<details>
<summary>Attempt 1 (2026-08-18→19): unbounded <code>joint_speed_penalty</code> — root cause of an early training collapse</summary>

**The plan.** Phase 1 = reach only. Active reward terms:
- `task_reward` with only `reach_weight=1.0` on (`grasp_weight`, `lift_weight`,
  `success_weight` all zeroed) — dense reward for closing the distance between the
  fingertip centroid and the cube.
- `action_smoothness` (action-rate + action-L2 penalty) — always-on.
- `joint_speed_penalty` — a **raw, unbounded** `sum(joint_vel²) * 0.001`, weight `-1.0`.
  Added specifically to front-load a "move gently" prior before the policy has any grasp
  skill, pre-empting the kind of violent motion that caused earlier (pre-from-scratch,
  warm-started) training collapses in this project.
- `distractor_accel` — left on but expected to be inert, since no distractor sits
  anywhere near the workspace in this task.

**Formulas, v1 (as actually run).** Let $\bar{\mathbf p}_{\text{tip}} = \frac{1}{5}\sum_{i=0}^{4}\mathbf p_{\text{tip},i}$
be the fingertip centroid (mean of the 5 tracked fingertip body positions) and
$\mathbf p_c$ the cube center.

$$
r_{\text{reach}} = 1 - \tanh\!\left(\frac{\lVert \bar{\mathbf p}_{\text{tip}} - \mathbf p_c \rVert}{0.25}\right)
\qquad\Rightarrow\qquad
r_{\text{task}} = 1.0 \cdot r_{\text{reach}}
$$
(posture, grasp, lift, success sub-weights all $0$ this phase — see `MDP_REPORT.md` §5.1
for the full `compute_task_reward` mechanism `r_{\text{task}}` is one term of.)

$$
p_{\text{smooth}} = 0.005\sum_{i=1}^{13}(a_i-a_i^{\text{prev}})^2
\qquad\text{(action-L2 term zeroed too, so this is rate-only)}
$$

$$
p_{\text{joint\_speed}}^{(v1)} = 0.001 \sum_{j=1}^{13} \dot q_j^2
\qquad\longleftarrow\quad\textbf{unbounded — this is the bug}
$$

$$
p_{\text{accel}} = \frac{1}{10}\sum_{d=1}^{10}\tanh\!\left(\frac{\lVert\Delta \mathbf v_d\rVert}{2.0}\right)
$$

$$
R^{(1,v1)} = 1.0\, r_{\text{reach}} \;-\; 3.0\, p_{\text{smooth}} \;-\; 1.0\, p_{\text{joint\_speed}}^{(v1)} \;-\; 3.0\, p_{\text{accel}}
$$

Ran to iteration 500 target (stopped manually around iteration 150-157 once the problem
below became clear).

**What we observed.** By iteration ~50-60, the policy's hand motion became visibly
*more* aggressive, not less — a fast, sweeping motion toward the cube rather than a
controlled reach. Initial hypothesis (before investigating): the policy was exploiting
`reach_rew` by knocking the cube toward the hand instead of moving the hand to the cube
(same failure *shape* as the earlier checkpoint-5500 cube-throw exploit, since
`reach_rew` only measures distance, not how it closed).

**Root cause (confirmed, not guessed)**: built a new inference-time diagnostic — a
per-step, per-reward-term plot with a dotted total line (`play_with_goal_markers.py`,
now a standing feature) — and ran it on checkpoint 150. The dominant line by a huge
margin was `joint_speed`, spiking to **-35 per step** (`sum(joint_vel²) ≈ 35,000` at the
worst point), vs. `task_reward` capped near +0.6. That's a ~35-70x magnitude mismatch.
The real problem wasn't that the penalty failed to discourage the behavior — it's that
an unbounded outlier this large corrupts the *learning signal itself*: it inflates
return variance enough to degrade the critic's value predictions (and therefore GAE
advantage estimates), and it triggers PPO's adaptive-KL learning-rate schedule to
throttle down (large noisy advantages look like they risk destabilizing updates). Net
effect: the presence of the huge penalty was actively slowing down learning of
*everything*, including learning to avoid the penalty itself. The original "it's brief,
so it should net out fine against sustained task reward" hypothesis assumed a
well-scaled penalty; it doesn't hold once the penalty's own magnitude swamps the
objective it's supposed to be balanced against.

Two side notes from the same investigation, for whoever revisits this:
- The reward-breakdown plot initially *looked* like `distractor_accel` was the runaway
  term, not `joint_speed` — that was a legend color-collision artifact (11 term-lines
  through matplotlib's 10-color default cycle silently wrapping colors). Fixed by
  switching to `tab20`. If a future plot looks like the wrong term is spiking, check
  colors aren't colliding before trusting the visual.
- `--episodes N` early-stopping in `play_with_goal_markers.py` was silently broken
  whenever `grasp_goal_palm` has weight 0.0 (true for Phase 1/2): IsaacLab's
  `RewardManager` skips calling a term's function entirely when its weight is 0 (a
  documented "micro-optimization"), so `_grasp_goal_term` never got cached and the
  script's episode-counter — which depended on that cache — stuck at 0 and never
  triggered the stop. Fixed by decoupling episode-counting from that dependency.

**Fix applied**:
1. `joint_speed_penalty` rewritten from the raw quadratic to `tanh(sum(joint_vel²) /
   3000) * weight(-1.0)` — same bounded-magnitude pattern already used by
   `distractor_accel`/`target_object_accel`/`fingertip_impact`, so it can't dominate the
   total reward regardless of how violently an early policy flails. `threshold=3000` is
   a starting value, not a measured optimum.
2. `target_object_accel` moved up from its originally-planned Phase 4 to Phase 1
   (weight `0.0` → `-5.0`). Reasoning: the reach_rew exploit path (disturb the cube
   instead of moving the hand) doesn't actually require `lift_weight`/`success_weight`
   to be active — it can happen through `reach_rew` alone — so deferring this penalty to
   Phase 4 left that path open the whole time. Direct penalty on the cube's own sudden
   velocity change closes it regardless of which later terms are on.
3. New `sustained_reach` termination + `sustained_reach_bonus` reward (weight `10.0`):
   fingertip centroid within 10cm of the cube for 30 consecutive control steps (~1s at
   30Hz) now ends the episode with a genuine positive signal. Before this, Phase 1 had
   no positive-outcome termination at all — every episode ended via `time_out` or
   `target_dropped`, so a policy that reached and then drifted away looked identical,
   reward-wise, to one that never reached. `distance_threshold=0.10m`/`hold_steps=30`/
   `weight=10.0` are starting defaults, not measured optima.

**Formulas, v2 (the fix).** $r_{\text{reach}}$, $r_{\text{task}}$, and $p_{\text{smooth}}$
are unchanged from v1 above. The changes:

$$
p_{\text{joint\_speed}}^{(v2)} = \tanh\!\left(\frac{\displaystyle\sum_{j=1}^{13} \dot q_j^2}{3000}\right)
\qquad\longleftarrow\quad\textbf{now bounded to }[0,1]\textbf{, matching every other impact-style penalty}
$$

$$
p_{\text{target\_accel}} = \tanh\!\left(\frac{\lVert\Delta \mathbf v_{\text{cube}}\rVert}{2.0}\right)
\qquad\text{(weight }-5.0\text{, was }0.0\text{ — now active)}
$$

$$
\text{counter}_t = \begin{cases}
\text{counter}_{t-1} + 1 & \lVert \bar{\mathbf p}_{\text{tip}} - \mathbf p_c \rVert < 0.10 \\
0 & \text{otherwise}
\end{cases}
\qquad
r_{\text{sustained\_reach}} = \mathbb 1\big[\text{counter}_t \ge 30\big]
$$

$$
R^{(1,v2)} = 1.0\, r_{\text{reach}} \;+\; 10.0\, r_{\text{sustained\_reach}}
\;-\; 3.0\, p_{\text{smooth}} \;-\; 1.0\, p_{\text{joint\_speed}}^{(v2)}
\;-\; 3.0\, p_{\text{accel}} \;-\; 5.0\, p_{\text{target\_accel}}
$$

**Status at the time**: fix applied, fresh from-scratch run launched 2026-08-19
(`phase1_reach_from_scratch_v2`) with all old logs cleared.

</details>

<details>
<summary>Attempt 2 (2026-08-19): v2 fix converges reach cleanly, but the 5cm sustained_reach bonus never fires — root cause traced to a real 6-7cm equilibrium, not a bug</summary>

The v2 fix (attempt 1) resolved the joint_speed blowup — checkpoint 300+ of
`phase1_reach_from_scratch_v2` looked genuinely good (hand approaching and hovering near
the cube, no more sweeping/flailing), and TensorBoard's task/total reward converged
starting ~iteration 300, peaking slightly higher at 400 and staying stable after. The
hand was observed hovering ABOVE the cube to trigger the 10cm sustained_reach bonus,
confirming that mechanism worked as designed — but at 10cm the hand was clearly too far
from the cube to be useful preparation for later grasp phases.

**Threshold tightened 10cm → 5cm, trained from scratch again**
(`phase1_reach_5cm_threshold`). After 500 iterations, `Episode_Termination/sustained_reach`
was 0.0 for the entire run — the bonus never fired once. Initial hypothesis (the user's
own, confirmed correct by investigation): not enough exploration, or the threshold was
simply outside what a reach-only policy could reach.

A dedicated diagnostic (`speed_reach_diag.py`, new standing tool — 64 parallel envs, runs
a checkpoint headless and logs per-joint velocity + fingertip-centroid-to-cube-center
closest-approach per episode, no video/rendering overhead) on checkpoint 499 (128
completed episodes) confirmed it quantitatively: closest approach clustered at
**mean=6.27cm, p10=5.42cm** — only 0.5% of episodes ever touched <5cm, vs. 83.3% touching
<7cm and 100% touching <10cm. Cube is 5cm (half-width 2.5cm) and the centroid averages 5
OPEN, un-clenched fingertips (Phase 1 has no grasp/finger-closing reward), so <5cm was
essentially demanding a pre-grasp hand shape this phase never trains for — not a random
exploration failure, a threshold sitting just past the reachable envelope for this
reward's own incentive structure.

**First response (2026-08-19): threshold loosened to 7cm.** Reverted almost immediately
at the user's explicit direction: later phases need genuine sub-5cm contact-range hand
positioning for grasp/lift anyway, so it's worth Phase 1 actually learning that precision
now via the reach mechanism itself, rather than lowering the bar. Went back to
`distance_threshold=0.05`.

**Second response: `hold_steps` 30 → 10.** Reasoning: the dense `reach_rew =
1-\tanh(dist/0.25)` is actually close to LINEAR (not flattening) in the 5-10cm range, so
gradient toward 5cm genuinely exists — the likely blocker was `target_object_accel`
(weight -5.0) punishing the near-contact proximity getting under 5cm requires, PLUS the
sparse bonus needing 30 CONSECUTIVE sub-5cm steps to ever fire once, when individual
step-samples <5cm were only ~0.5% — the probability of a 30-in-a-row run was effectively
zero, so the bonus had literally never been sampled once in the whole run, leaving PPO
with no gradient at all telling it the accel-penalty risk was worth pushing through.
Also added, in the same pass: a hard `velocity_limit_sim` clamp on the arm (0.6 rad/s)
and right-hand (0.5 rad/s) actuators (`robot_cfg.py`) — sized from the same diagnostic's
per-joint velocity percentiles (right_arm p99=0.85, max=5.5 rad/s; right_hand p99=0.50,
max=4.5 rad/s) — and rescoped `joint_speed_penalty` from summing over all 29 articulation
joints down to just the 13 policy-controlled ones, after discovering the locked
legs/waist/left-arm joints were producing a ~709 rad/s one-step reset-frame velocity
READING artifact (position teleports on reset; velocity briefly reads as if it did too)
that was spuriously saturating the penalty every single reset step across up to 2048
parallel envs, independent of how gently the actual arm/hand moved.

Trained from scratch again (`phase1_reach_5cm_holdsteps10`) — this is the run validated
above. Even with `hold_steps=10`, the bonus STILL never fired (confirmed via the
dwell-time extension to the same diagnostic on checkpoint 450 — max consecutive dwell
under 5cm across 192 episode-samples was 8 steps). That result is what motivated
redefining the termination METRIC itself (centroid-to-center → average
fingertip-to-surface) rather than continuing to chase threshold/hold_steps — see the
validated entry above for the reasoning and MDP_REPORT.md/`g1_pick_env_cfg.py` for the
final formula.

</details>

---

## What Didn't Work

### [Phase 3, attempt 1] Pose-mimicking run alongside `grasp_weight` (original plan order) — `grasp_goal_palm` declined instead of converging

**Dates**: 2026-08-19 → 2026-08-20.
**Task/run**: `Isaac-G1-Pick-Empty-v0`, resumed from Phase 2's checkpoint 749
(`2026-08-19_23-15-23_phase2_grasp_from_ckpt250`), run name
`2026-08-20_05-19-15_phase3_grasp_pose_from_ckpt749`, 500 iterations (749→1248).

**The plan (as originally laid out).** Follow the 5-phase plan in order: `grasp_weight`
(Phase 2) already active from the prior phase, add `grasp_goal_palm` (weight $2.0$),
`grasp_goal_hand` (weight $1.0$), `grasp_reach` (weight $1.0$), and flip
`task_reward.pose_gated_success` to `True`. No new penalty terms — considered
`fingertip_impact` but left it for Phase 5 per the plan, reasoning Phase 1's existing
`target_object_accel` + `joint_speed` + `velocity_limit_sim` already covered the
jab/slam risk without evidence of an actual problem.

**What we observed.** Watching checkpoint 750 through 1248 in the recorded videos: fingers
curl, but essentially randomly — different episodes settle the hand in different positions
around the cube, fingertips generally nearby but with no consistent envelope/wrap shape
and no bias toward grasp #12's specific ~65° side approach. Visually indistinguishable
from Phase 2 alone (see the Phase 1 "What Worked" entry's dwell-time diagnostic for that
baseline).

**Root cause (confirmed via TensorBoard, not guessed):**

| Iter | `grasp_goal_palm` | `grasp_goal_hand` | `grasp_reach` | `task_reward` | `Policy/mean_noise_std` |
|---|---|---|---|---|---|
| 800 | 0.147 | 0.0060 | 0.158 | 1.576 | 0.109 |
| 1000 | 0.144 | 0.0050 | 0.163 | 1.609 | 0.112 |
| 1150 | 0.139 | 0.0045 | 0.162 | 1.591 | 0.113 |
| 1248 | 0.137 | 0.0038 | 0.167 | 1.616 | 0.115 |

`grasp_goal_palm` (max possible $\approx 1.0$) doesn't plateau — it **declines** after an
early jump, and `grasp_goal_hand` (near-zero throughout) declines too. This is a worse
outcome than the historical precedent in `MDP_REPORT.md` §5.2.3 (that older, warm-started
run at least sat FLAT at $\approx 0.03$ for 6000+ iterations rather than trending
backward). `grasp_reach` (the one task-space, per-fingertip-to-contact-point term) is the
sole exception — it kept improving slowly (0.158 → 0.167) across the same run.
`Policy/mean_noise_std` also **rose** slightly through this run (0.107 → 0.115) instead of
the usual monotonic decay — a sign of genuine optimization conflict, not smooth
refinement. `Episode_Termination/sustained_reach` (now the 1cm-surface-contact metric)
stayed at 0.0 the entire run.

**Diagnosis**: `r_grasp` (`grasp_weight`, still active this whole run) is direction-agnostic
— any fingertip-near-cube-surface configuration scores well regardless of approach angle
or opposition, so once established (during Phase 2) it's an easy, "good enough" reward
the policy has no pressure to abandon for grasp #12's one specific pose. `grasp_goal_palm`
has to actively fight that already-comfortable local optimum, and by this data, loses.
`grasp_reach` improving anyway makes sense under this diagnosis: fingertip-to-specific-
contact-point distance correlates loosely with what `grasp_weight` already rewards
(closer fingertips generally), so it isn't as directly opposed as `grasp_goal_palm`'s
whole-hand position+orientation target is.

**Response (2026-08-20, user's proposal, reasoned through together)**: reorder the
curriculum — run Phase 3's pose-mimicking terms BEFORE `grasp_weight` rather than after,
resuming from Phase 1's checkpoint 250 (not Phase 2's) with `grasp_weight` held at $0.0$.
Rationale this is viable, not just a guess: unlike the sparse `sustained_reach_bonus`,
`grasp_goal_palm` is **dense and ungated** — by its own original design intent (§5.2.1 in
`MDP_REPORT.md`) it "pays from anywhere in the workspace, purely as a function of how
close the hand's pose is to the goal, every single step." It doesn't need `reach_rew` or
`grasp_weight` to bootstrap it, and `grasp_goal_hand`'s palm-proximity gate opens
naturally once `grasp_goal_palm` gets the palm close, with no dependency on `grasp_weight`
either. The "coarse-to-fine" ordering the original plan assumed makes sense when the
coarse objective is a stepping stone toward the fine one; here it was instead acting as a
decoy. If `grasp_goal_palm`'s own curve now climbs instead of declining, that confirms
`grasp_weight` was the specific conflict — it gets reintroduced afterward, to tighten
distance within the now-established correct approach shape rather than compete with it.
Fallback if this ALSO fails (the user's own stated plan): revert to grasp-before-pose
ordering and design an explicit enveloping/opposition mechanism directly into Phase 2's
`r_grasp` instead of relying on Phase 3 to supply direction.

**Status**: reorder run (`phase3_pose_before_grasp_from_ckpt250`, resumed from Phase 1
checkpoint 250) killed by the user at iteration ~680, also inconclusive/stalled.
`grasp_goal_palm` did NOT decline this time (unlike attempt 1: 0.172@270 → 0.180@450,
holding/slightly up rather than reversing) and `grasp_reach` kept improving and even
accelerating late (0.085@430 → 0.097@450), with `Policy/mean_noise_std` decaying normally
(0.116→0.099) instead of attempt 1's anomalous rise -- genuinely different symptoms from
attempt 1, not a repeat of the same failure. But `grasp_goal_palm`/`grasp_goal_hand` were
still essentially flat through iteration 450 and the user's own read at iteration 680 was
"pretty much not learned anything," so removing `grasp_weight` alone was not sufficient to
produce real pose convergence either. Per the user's own stated fallback (see above):
moving to a redesigned `r_grasp` with an explicit opposition/envelope term, rather than
relying on either ordering of grasp_weight vs. pose-mimicking to produce it implicitly --
see the proposed design at the bottom of this file.

---

## Proposed fix: pairwise coverage reward (2026-08-20)

Two earlier designs were tried and abandoned (opposition-based, then a whole-hand
"spread" score) — both are cleared out in favor of this one rather than kept as
history, per request. This section explains the final design in plain language first,
with the math underneath as a reference, not the primary explanation.

### The problem in one sentence

Every formula tried so far can be satisfied by the hand doing something OTHER than a
real grasp — usually all 5 fingertips (and the palm) getting close to the cube from the
SAME side (e.g. pressing down on top) — because nothing in any earlier formula actually
checks whether the close points are on different sides of the object. They only ever
ask "is each point close," in one form of averaging or another, and clustering
everything on one face satisfies that just fine.

### The idea, in plain language

Look at the hand not as 6 independent points that each individually want to be close to
the cube (palm + 5 fingertips), but as **15 PAIRS** of points (any 2 of the 6). For each
pair, ask two questions:

1. Are BOTH points in this pair close to the cube's surface?
2. Are they on genuinely DIFFERENT sides of the cube (not the same side)?

A pair only scores well if the answer to BOTH is yes. Two fingers both touching the
cube but on the SAME face score close to zero as a pair — not because they aren't
close, but because together they aren't covering anything new. Two fingers that are
both close AND on different/opposite sides score near the maximum.

The final reward is just the average score across all 15 pairs.

### Why this can't be tricked by clustering

Every earlier formula rewarded "is each point close," aggregated somehow (average,
whole-hand spread score, whatever) — clustering makes every individual point close, so
it always scored well no matter how the aggregation was done. This design never asks
"is this ONE point close" by itself — it only ever asks "are these TWO points close AND
on different sides," as one combined question, for every pair. Pile all 6 points onto
the same face and every one of the 15 pairs is "close, but same side" — each pair
scores near zero because the separation half of the question fails, even though the
proximity half succeeds. There is no path to a high reward through proximity alone.

### Concrete numbers, so this isn't abstract

Using the actual formula (proximity scale = 5.5cm, same as everywhere else in this
file):

| Scenario | both close? | different sides? | pair score |
|---|---|---|---|
| Both fingers 1cm from surface, same face | high (0.82 each) | no (~0) | **~0.00** |
| Both fingers 1cm from surface, opposite faces | high (0.82 each) | yes (~1) | **~0.67** (near max) |
| One finger 1cm away, other 10cm away, opposite faces | mixed (0.82 × 0.05) | yes (~1) | **~0.04** (low — the far finger drags it down) |
| Both fingers 10cm away, opposite faces | low (0.05 each) | yes (~1) | **~0.003** (very low) |

The middle two rows are the important comparison: same distance, same proximity score,
but "same face" scores essentially nothing while "opposite faces" scores 0.67 — pure
separation is what moves the number, not extra closeness.

### The formula, for reference

$$
r_{\text{grasp}} = \frac{1}{15}\sum_{\text{all 15 pairs }(k,j)}\ \underbrace{\Big(1-\tanh\tfrac{d_k}{0.055}\Big)\Big(1-\tanh\tfrac{d_j}{0.055}\Big)}_{\text{both points close}} \ \times\ \underbrace{\frac{1-\hat{\mathbf u}_k\cdot\hat{\mathbf u}_j}{2}}_{\text{on different sides}}
$$

- $d_k$ = point $k$'s distance to the cube SURFACE (same as every other distance term
  in this file: raw distance to center, minus the cube's 2.5cm half-width, floored at 0).
- $\hat{\mathbf u}_k$ = the direction from the cube's center out to point $k$ (a unit
  vector). Two points on the same side point almost the same way
  ($\hat{\mathbf u}_k\cdot\hat{\mathbf u}_j\approx 1$, separation score $\approx 0$); two
  points on opposite sides point almost opposite ways
  ($\hat{\mathbf u}_k\cdot\hat{\mathbf u}_j\approx -1$, separation score $\approx 1$).
- The 6 points are the palm (`right_wrist_yaw_link`) and the 5 fingertips. No finger is
  named or assigned a role anywhere in this formula — it treats all 6 identically and
  only ever asks questions about pairs of them.

### Why this should create the "positive spiral" being asked for

This test starts from Phase 1's checkpoint 250, where reach behavior is already solid
and several fingertips are already sitting within a few cm of the cube (measured
directly: index/middle around 1.3-1.6cm, per an earlier diagnostic). That means several
pairs ALREADY have a decent proximity score going in — what's missing is separation. If,
through ordinary exploration noise, one fingertip happens to drift toward a different
side while staying reasonably close, the pairs involving that fingertip get a real,
immediate reward bump (proximity was already there, separation just improved), and that
specific improvement gets reinforced. The mechanism doesn't require randomly
discovering the entire envelope shape in one shot — just one pair at a time getting a
little more separated while staying close, with each such improvement individually,
immediately rewarded.

### Two parameter decisions (2026-08-20)

- **No floor/blending term.** The earlier designs blended proximity and shape quality
  with a tunable floor (`(1-β) + β·[shape term]`) so a policy with poor shape still
  banked some "just being close" reward regardless. That floor was exactly the escape
  hatch that let both earlier formulas settle for "close but not spread out" — this
  design drops it entirely (equivalent to setting that earlier β to 1). Given the
  starting checkpoint already has real proximity established (see above), there's no
  cold-start concern that would justify keeping a floor this time.
- **`grasp_weight` raised from 2.0 to 3.0** (task_reward's outer multiplier on this
  whole term) — so that once the policy does discover genuine multi-sided coverage, the
  payoff is bigger and more decisive, making the harder coordination this formula
  demands clearly worth it once found.

### Implementation notes

- Same 6 tracked points already available in `compute_task_reward` (`palm_pos`,
  `tips_pos`) -- no new body tracking needed.
- Replaces `grasp_rew`'s computation entirely — neither the earlier thumb/finger split
  nor the whole-hand spread-score design are layered underneath this; this formula
  stands alone.
- Recommended test path, same as every attempt logged in this file: resume from Phase 1
  checkpoint 250, this new formula as `grasp_weight`'s only content, pose-mimicking
  terms still off, keeping this one attributable variable against everything already
  tried.

**Status (2026-08-21)**: implemented and sanity-tested (both `grasp_rew` inside
`compute_task_reward` and the new standalone `grasp_envelope` RewTerm below confirmed
computing real, non-NaN values), launched from Phase 1 checkpoint 250 for 500
iterations. Killed early by the user before reaching a conclusion, not because this
formula was found wrong — see "Root cause reassessment" below for why: the user
identified a more fundamental problem one level down, in Phase 1 itself, that likely
affected every Phase 2/3 attempt including this one. This design isn't abandoned, just
paused — see the reassessment for the plan to retry it once the new Phase 1 foundation
is validated.

---

## Root cause reassessment: Phase 1 lacked directional consistency (2026-08-21)

Every Phase 2/3 attempt logged in this file so far — the original `grasp_weight`, both
orderings of pose-mimicking against it, and the pairwise-coverage redesign above — was
built on checkpoints descending from the same Phase 1 run
(`2026-08-19_20-47-28_phase1_reach_5cm_holdsteps10`). The user's observation, reviewing
videos across all of them: the hand consistently approaches and settles from directly
**above** the cube, never learning to envelope from the sides, regardless of which
Phase 2 reward was tried.

**Diagnosis**: `reach_rew` (Phase 1's only real behavioral driver) is
`1 - tanh(||fingertip_centroid - cube_center|| / 0.25)` — completely direction-agnostic.
It rewards closing the distance from ANY approach angle equally. Given 500 iterations of
training with no competing directional signal, and given this arm's kinematics
apparently make a top-down reach the easiest/most natural motion (the OLD, now-disabled
`posture_rew` term independently confirms this — it hardcoded "palm 8cm above the
cube," and `MDP_REPORT.md` documents it was disabled specifically because grasp #12 is
a ~65° SIDE approach that a top-down anchor would fight), Phase 1 plausibly converged
hard on a top-down approach purely because it was the path of least resistance. Every
downstream Phase 2/3 checkpoint inherited that same commitment before any
grasp-shaping or pose-mimicking reward ever got a chance to matter. This would explain
the recurring struggle independent of how Phase 2's reward was designed — the
grasp-shaping terms were never the foundational problem; the starting point they were
all built on top of was.

### The rejected fix: a fixed "N cm above the cube" position target

First proposed: add a dense reward pulling the palm to a fixed point (e.g. 5cm) above
the cube's center, active in Phase 1, removed again once Phase 3 begins. **Rejected**
before implementation: this doesn't fix the diagnosed problem, it re-encodes it. "Above
the cube" is itself a top-down-specific target — adding it would make the same bias
*more* deliberate and *more* strongly reinforced (an explicit dense reward instead of an
emergent tendency), recreating the exact conflict `posture_rew` was disabled for in the
first place, just earlier in the curriculum and harder to undo later.

### The actual fix: activate `grasp_goal_palm` in Phase 1, position-only

Instead of inventing a new fixed anchor, reuse the existing live-goal infrastructure
(`grasp_goal_palm_reward`, already tracking grasp #12's target pose re-attached to the
cube's current position every step) and turn it on early, in Phase 1, rather than
waiting for Phase 3. Whatever directional habit Phase 1 now forms is the SAME one
Phase 3 will eventually want — nothing needs to be unlearned at any phase handoff.

**Position-only, not full pose** (position + orientation), was a deliberate scope
decision, reasoned through with the user: the diagnosed problem is specifically about
*which side* the hand approaches from — a positional question. Wrist orientation is a
separate, higher-dimensional target, and demanding it this early stacks a second new
objective onto a policy still learning basic arm control from scratch. There's also
reason to think orientation may partially follow position for free — physically
reaching a point on the side of the cube likely requires a wrist angle already closer
to grasp #12's target than "flat, facing straight down" was, without needing an
explicit orientation reward to force it. Implemented via
`grasp_goal_palm_reward(orient_weight=0.0)`, which cleanly reduces to the position term
alone (`(1-0)*pos_rew + 0*orient_rew`) — no new function needed. `orient_weight`
returns to its default (0.5, full pose) once Phase 3 reactivates this term for real.

### What changed operationally

- `grasp_goal_palm`: weight `0.0 -> 2.0`, now active from Phase 1, `orient_weight`
  forced to `0.0` (position-only) until Phase 3.
- `task_reward.grasp_weight` (pairwise-coverage `grasp_rew`) and the standalone
  `grasp_envelope` RewTerm: both set back to `0.0` — Phase 2 as a whole is paused, not
  because either formula is suspect, but to test the NEW Phase 1 foundation on its own
  first, isolated from any Phase 2 grasp-shaping, before layering Phase 2 back on top.
- All existing `logs/rsl_rl/g1_pick/` run directories and checkpoints cleared, TensorBoard
  reset — every prior checkpoint descends from the now-suspect Phase 1 foundation, so
  none of them are useful resume points for what comes next.
- New training launched from scratch (random init), Phase 1 config only (reach_rew +
  position-only grasp_goal_palm + the existing Phase 1 safety net), 500 iterations.

**Status**: launched 2026-08-21. Watch specifically for whether the hand's approach
direction diversifies away from top-down — that's the direct signal this fix is
addressing, more informative early on than `task_reward` itself (which Phase 1 already
handled well before, per the original `phase1_reach_5cm_holdsteps10` results).

**Update (2026-08-21, same day)**: the `phase1_reach_palmpos_anchor` run above confirmed
the fix works exactly as intended -- `grasp_goal_palm` climbed monotonically the whole
run (0.005 -> 1.62, weight 2.0, so ~81% of max) with no decline, unlike every prior
attempt, and by checkpoint 350 the palm was reaching the goal position to within ~1cm.
But reviewing the videos with the goal-mesh visualization surfaced a NEW, more
fundamental problem underneath that success: the cube was spawning with NO yaw
randomization at all (`reset_target_object`'s `pose_range` never had a `yaw` key, and
`reset_root_state_uniform` treats a missing key as always-zero) -- every episode used
the exact same fixed cube orientation. Since grasp #12 is defined relative to the
cube's own frame, a fixed cube orientation means a fixed WORLD-frame goal too, and the
user identified that this particular fixed direction is kinematically awkward for this
arm -- the hand has to reach around into an unnatural configuration to get there, which
is what the "approaching from a weird side" behavior in the videos actually was. Not a
training failure at all; a scene/task setup issue upstream of every reward term tried
so far in this whole file.

**Fix**: added a fixed (not randomized -- same every episode) +45 degree yaw offset to
`reset_target_object`'s `pose_range`, applied on top of the existing default orientation
(`reset_root_state_uniform` composes sampled yaw as a delta via quat_mul, not an
absolute reset). +45 degrees about +Z is counter-clockwise viewed from above. This does
NOT change any reward term, weight, or the grasp_goal_palm design validated above --
only the cube's spawn orientation. Sanity-tested (3 iterations, clean), then all
logs/checkpoints cleared and training relaunched from scratch with the exact same Phase
1 setup (reach_rew + position-only grasp_goal_palm + existing safety net), just with
the cube rotated. If this run reproduces the same clean `grasp_goal_palm` convergence
AND the approach direction now looks natural in the videos, that's confirmation this
was the real remaining issue.

**Confirmed (2026-08-21)**: `phase1_reach_palmpos_cuberot45` validated cleanly -- checked
via inference video with the goal-mesh overlay at checkpoints 350/400/499, approach
direction now looks natural (no more "reaching around" from an awkward side), and
`grasp_goal_palm` converged smoothly with no decline. Handed off to Phase 2+3 (merged,
see below) at checkpoint 200 -- chosen over the fully-converged 499 by the same
over-convergence logic as every other handoff in this file: `grasp_goal_palm` had
already reached 98.5% of its final value by iteration 200
($0.0448 \to \sim 1.11$ pulled from TensorBoard at handoff time), while
`Policy/mean_noise_std` was still meaningfully higher at 200 (0.309) than at 499
(0.177) with no further metric gain in between -- preserving that exploration budget
for Phase 2+3's new objectives rather than spending it on marginal refinement of a
metric that had already plateaued.

---

### [Phase 2+3 merged] Pairwise-coverage grasp + pose-mimicking together, from the fixed Phase 1 foundation -- validated 2026-08-21

**Dates**: 2026-08-21 (single continuous run).
**Task/run**: `Isaac-G1-Pick-Empty-v0`, resumed from checkpoint 200 of
`phase1_reach_palmpos_cuberot45`, run name
`2026-08-21_08-48-51_phase2and3_merged_from_ckpt200`, 500 iterations (200 -> 699).

**The decision to merge (not reorder again).** After the Phase 1 root-cause fix above,
the user's read of the resulting videos was that the hand's fingers were already
settling into roughly grasp-appropriate shapes as an emergent side effect of good
positioning -- i.e. Phase 1 no longer hands Phase 2/3 a blank slate the way it did for
every earlier attempt in this file. That changes the risk calculus behind the original
"keep Phase 2 and Phase 3 separate" decision (see the Phase 3 reordering entries above,
and `MDP_REPORT.md` §5.2.3's precedent of a coarse reward letting the policy ignore a
specific pose target): the concern was always that `grasp_weight`'s coarse,
direction-agnostic reward gives the policy an easy "good enough" alternative to the
specific pose-mimicking target. With Phase 1 already biasing toward the right general
shape, that easy alternative is less available. Reasoned through with the user, decision
was to merge and test directly rather than doing another sequential reorder.

**Setup**: all of `grasp_weight` (pairwise-coverage `grasp_rew`, raised 2.0 -> 3.0),
`grasp_envelope` (weight 2.5, new standalone whole-hand coverage bonus), `grasp_goal_hand`
(weight 1.0), `grasp_reach` (weight 1.0) activated together, and `grasp_goal_palm`'s
`orient_weight` restored from Phase 1's position-only 0.0 back to its full-pose default
0.5. `task_reward.pose_gated_success` set `True` (inert until Phase 4). No new penalty
terms -- `fingertip_impact` still held for Phase 5 per the original plan.

**What actually converged**, pulled directly from TensorBoard across the full run:

| Iter | task_reward | grasp_envelope | grasp_goal_palm | grasp_goal_hand | grasp_reach | Train/mean_reward | Policy/mean_noise_std |
|---|---|---|---|---|---|---|---|
| 200 | 0.036 | 0.020 | 0.045 | 0.002 | 0.002 | 0.95 | 0.308 |
| 250 | 0.831 | 0.746 | 1.124 | 0.037 | 0.138 | 19.83 | 0.285 |
| 300 | 0.889 | 0.857 | 1.260 | 0.047 | 0.175 | 22.34 | 0.266 |
| 350 | 0.922 | 0.940 | 1.353 | 0.045 | 0.187 | 24.33 | 0.255 |
| 400 | 0.948 | 0.997 | 1.326 | 0.045 | 0.198 | 25.14 | 0.246 |
| 450 | 0.960 | 1.000 | 1.378 | 0.045 | 0.197 | 25.58 | 0.240 |
| 500 | 0.973 | 1.020 | 1.357 | 0.046 | 0.200 | 26.13 | 0.235 |
| 550 | 0.977 | 1.003 | 1.408 | 0.048 | 0.202 | 27.27 | 0.230 |
| 600 | 0.996 | 1.034 | 1.426 | 0.051 | 0.213 | 27.53 | 0.224 |
| 650 | 1.016 | 1.042 | 1.395 | 0.053 | 0.217 | 27.22 | 0.220 |
| 699 | 1.023 | 1.070 | 1.387 | 0.053 | 0.217 | 27.28 | 0.218 |

No decline anywhere -- unlike attempt 1 of the old ordering, `grasp_goal_palm` climbed
fast (0.045 -> 1.35 by iter 350) and then held flat/noisy through 699 rather than
reversing. `grasp_envelope` (the standalone coverage bonus, raw max 1.0, so weighted
max 2.5) climbed to ~1.0-1.07 by iter 400 and stayed there -- i.e. the raw shape-quality
signal itself is sitting around 0.4-0.43 of its own max, not fully saturated, but stable.
`grasp_reach` kept a slow, steady climb the whole run (0.138 -> 0.217) without
plateauing as hard as the others. `Policy/mean_noise_std` declined smoothly and
monotonically the whole way (0.308 -> 0.218, ~29% reduction) -- normal decay, no sign of
the "rising noise = optimization conflict" symptom seen in the old ordering's attempt 1.

**Confirmed via inference video + a live-goal mesh overlay** (checkpoints 350, 400, 450
recorded with both the translucent green goal-pose hand mesh and a bright-red goal-palm
centroid marker): palm pose error converges within the first ~50-100 control steps of
each episode to ~3-5cm position / ~15-20deg orientation and holds there for the rest of
the episode (not just a brief pass-through). Per-fingertip distance to each finger's own
`grasp_reach` target settles to 4-6cm for the four non-thumb fingers, with the thumb
tightening further over the course of the run.

**Reward-breakdown plot caveat** (checkpoint 350): the per-step weighted total (dotted
line) sits much lower (~0.1-0.15) than `task_reward`/`grasp_envelope`/`grasp_goal_palm`
individually (~1.0-1.4) once each has converged -- not because those terms are failing,
but because `joint_speed` is a large, repeatedly-spiking negative penalty (-0.5 to -1.0)
throughout the episode that drags the aggregate down. Worth remembering when reading the
raw `Train/mean_reward` curve in isolation -- the individual grasp/pose terms can be
genuinely converged and high while the headline aggregate still looks unimpressive.

**Grasp-readiness diagnostic (new, 2026-08-21)**: none of the four curves above directly
measure the thing Phase 4's `is_grasped` gate actually needs -- how close the fingers
get to the cube SURFACE, not to a goal pose or a coverage score. Ran `contact_check.py`
(64 envs x 300 steps, measures per-fingertip minimum surface distance over the rollout)
at checkpoints 350/500/600/699:

| Ckpt | thumb mean (cm) | index/middle/ring/pinky mean (cm) | thumb envs <1cm buffer | computed is_grasped (approx) |
|---|---|---|---|---|
| 350 | 1.48 | 3.26 / 2.81 / 2.88 / 3.51 (avg 3.12) | 29/64 | ~0.12 |
| 500 | 1.31 | 2.68 / 2.35 / 2.57 / 3.36 (avg 2.74) | 38/64 | ~0.16 |
| 600 | 1.38 | 2.46 / 2.11 / 2.28 / 2.84 (avg 2.42) | 37/64 | ~0.19 |
| 699 | 1.31 | 2.33 / 2.10 / 2.25 / 2.49 (avg 2.29) | 39/64 | ~0.21 |

(`is_grasped` approximated from these means via the actual formula:
$(1-\tanh(\text{thumb\_dist}/0.03))\cdot(1-\tanh(\text{finger\_dist}/0.03))$, distances
in meters, half-width already subtracted by `contact_check.py`.) Thumb was already
mostly converged by checkpoint 350 (~1.3-1.5cm the whole way); the four-finger average
kept closing steadily and had NOT plateaued by 699 (still improving iter-over-iter, no
sign of leveling off the way `grasp_goal_palm`/`grasp_envelope` had by ~iter 400). This
is the key fact that drove the Phase 4 checkpoint choice below: the metric this next
phase actually depends on was still making real progress at the last checkpoint, unlike
every prior phase handoff in this file where the target metric had already plateaued
before handoff.

**Checkpoint 699 (final) chosen for the Phase 4 hand-off** -- deliberately NOT an
earlier "preserve exploration budget" checkpoint this time. The over-convergence
principle used at every other handoff in this file (Phase 1 -> 2+3 above, Phase 1's own
`sustained_reach` history) applies when the target metric has stopped improving and only
`Policy/mean_noise_std` keeps shrinking for no further gain -- that's not what's
happening here for the finger-closing behavior specifically. `Policy/mean_noise_std`
also only declined ~29% over the full run (0.308 -> 0.218), a smaller reduction than
Phase 1's own final run (~31% and flagged as still leaving useful exploration budget),
so there's no strong exploration-budget argument for an earlier checkpoint either.
**Status**: done, handed off to Phase 4.

---

### [Phase 4] Lift + success, launched from checkpoint 699 -- 2026-08-21

**Note on how this entry was written**: the user asked to run this phase fully
autonomously overnight (choose the checkpoint, set up Phase 4, launch it, log it) without
stopping to ask questions. Everything above this point in the file was reviewed/validated
with the user directly; this entry and everything after it was decided and executed
autonomously following the same reasoning patterns and conventions established
throughout the rest of this file. Flagging this distinction so a future reader knows
which decisions had direct human sign-off at the time vs. were made by extrapolating this
file's own established practice.

**Setup**: `task_reward.lift_weight` and `task_reward.success_weight` both flipped
`0.0 -> 1.0` -- the only change from Phase 2+3's config. No other RewTerm or weight
touched. `pose_gated_success` was already `True` (set in Phase 2+3, inert until now) so
`success_bonus` is pose-gated from the moment this phase starts, not added later --
`success_floor=0.2` means an ungated-pose lift still pays 200 of the 1000 max, a
correctly-posed one pays the full 1000. Both `lift_cont_rew` and `success_bonus` were
already fully implemented in `compute_task_reward` from the start of this from-scratch
curriculum (2026-08-13) -- see the function body in `g1_pick_env_cfg.py` -- so this
activation needed no new reward code, only the two weight flips plus the phase-tracking
comment at the top of `RewardsCfg`.

**Checkpoint choice**: 699 (the final checkpoint of the merged Phase 2+3 run), not an
earlier one -- see the grasp-readiness diagnostic and reasoning in the "[Phase 2+3
merged]" entry above. Summary: a direct measurement of finger-to-cube-SURFACE distance
(not any of the four RewTerm curves, none of which track this quantity directly) showed
the four-finger average was still closing steadily at the last checkpoint (2.29cm at 699
vs 3.12cm at 350, no plateau), and the computed `is_grasped` gate value roughly doubled
across the same span (~0.12 -> ~0.21) -- meaning more Phase 2+3 training was still buying
genuine progress on the exact quantity Phase 4's lift/success terms are gated on, unlike
every earlier phase handoff in this file where the over-convergence principle favored an
earlier, less-converged checkpoint.

**Sanity check**: resumed from checkpoint 699, 3 iterations, 64 envs -- clean, no NaNs,
all reward terms (including the newly-active lift/success contribution baked into
`task_reward`) computing plausible values. Sanity run directory deleted after
confirmation, consistent with every other phase transition in this file.

**Launch**: `2026-08-21_11-17-01_phase4_lift_success_from_ckpt699`, 2048 envs, 500
iterations (699 -> 1198 following this project's established 0-indexed final-checkpoint
convention: final = start + max_iterations - 1). Confirmed running at full scale shortly
after launch. Appending to the same `logs/rsl_rl/g1_pick/` TensorBoard root as every
other run in this curriculum -- no logs cleared for this phase.

**What to watch for once this run has enough data** (for whoever reviews this next,
human or otherwise): `Episode_Termination/target_lifted` was 0.0 at the end of Phase 2+3
(expected -- lift/success were both weight-0 and the sparse `sustained_reach` termination
never fires either, see Phase 1's entry on why that specific metric is hard to satisfy
even with good reach/grasp behavior). The real signal for whether Phase 4 is working is
whether `target_lifted` starts firing at all, and whether `task_reward` gets a
step-change (not just gradual drift) once lift/success first start paying out -- lift is
worth up to 0.6 per step while grasped-and-rising, success is worth up to 1000 once, both
much larger than anything active in Phase 2+3, so a working Phase 4 should look
qualitatively different from more-of-the-same gradual curve growth. If `target_lifted`
stays at 0.0 for a long stretch (a few hundred iterations) with no movement, that would
mirror the historical `sustained_reach` non-firing pattern in Phase 1 and might mean the
gate (`is_grasped`, still only ~0.21 at handoff) needs more headroom before lift becomes
reachable at all -- worth a contact_check.py-style diagnostic on the live checkpoint
before assuming the reward design itself is wrong.

**Status**: DID NOT WORK -- `target_lifted` stayed at exactly 0.0 for all 500 iterations
(699 -> 1198). `task_reward` barely moved (1.023 at handoff -> 1.058 final, essentially
the same plateau Phase 2+3 already reached) and `Policy/mean_noise_std` stopped decaying
and drifted slightly UP (0.218 -> 0.224) instead of continuing its normal decline -- see
"[Phase 4 v2]" below for the diagnosis and what was tried next.

---

### [Phase 4 v1 -> v2] From gated-and-stuck to a bootstrap-then-taper fix -- 2026-08-21

**The user's own diagnosis, stated plainly**: "it got too comfortable holding the grasp
and never experienced what it's like to increase z ... the policy doesn't even realize
it'll get a reward if it increases z." Investigated and confirmed, then fixed in two
rounds.

**Round 1 -- remove the `is_grasped` gate from `lift_cont_rew` (2026-08-21).**
`lift_cont_rew` was `lift_height * 2.0 * is_grasped`, and `is_grasped` was only ~0.21 at
Phase 4 handoff (see the grasp-readiness diagnostic in "[Phase 2+3 merged]" above) -- so
even a genuine small lift attempt paid a reward on the order of 0.001, far below any
learnable gradient. The gate's original purpose (block the "shove the cube for a lucky
proximity frame" exploit from the checkpoint-5500+ regression, documented in
`target_object_acceleration_penalty`'s docstring) is already covered independently by
`target_object_accel` (-5.0, penalizes the cube's own sudden Δv regardless of grip
state), and this is a FORWARD curriculum -- grasp-quality terms (`grasp_goal_palm` +
`grasp_envelope` + `grasp_weight`, ~7-8 combined weight) are already dense, converged,
and pulling toward one specific pose, so abandoning the real grasp to shove/scoop instead
fights all of that for a capped 0.6 payoff, a bad trade the policy has little incentive
to take. Reasoned through with the user, who flagged the one residual risk (a SLOW push
that avoids the jerk penalty, since that penalty only fires on sudden Δv) as low-priority
enough to check visually rather than block on. Gate removed
(`lift_cont_rew = lift_height * lift_scale`, no `is_grasped` factor).

Relaunched from checkpoint 699 (`phase4_lift_ungated_from_ckpt699`) -- **but initially,
by mistake, from checkpoint 1198 (Phase 4 v1's own dead-end final checkpoint) instead**.
Caught by the user ("why you starting from this... checkpoint... I wanted you to start
from the checkpoint where you will start phase four"): checkpoint 1198 carried forward
500 iterations of a run that never learned anything about lifting, so restarting from
it only reinforces the same "hold still" equilibrium further for no benefit, instead of
giving the ungated reward a clean shot from the actual Phase 2+3 handoff point.
Corrected -- killed, that dead-end run directory deleted, relaunched from 699.

**Round 1 result: still nothing.** After 358 logged iterations (699 -> 1056, well over
half the planned run), `Episode_Termination/target_lifted` was still exactly 0.0 across
every single one, and `Policy/mean_noise_std` had moved from 0.2182 to 0.2185 -- de facto
frozen. `task_reward` had already re-converged to ~1.01-1.02 within the first ~10
iterations, matching Phase 4 v1's plateau almost exactly. Ungating fixed the reward
*magnitude* problem but not the deeper one: the policy's action noise never happened to
produce a coordinated upward motion in the first place, so a bigger reward for that
motion never got sampled. Confirmed and killed by the user's explicit instruction.

**Root cause, restated precisely**: two separate, both-necessary problems, not
alternatives -- (1) `lift_cont_rew` is linear in height (`lift_height * lift_scale`), so
a tiny accidental height gain (e.g. 1mm) pays a tiny reward (`0.001 * 2.0 = 0.002`) no
matter what multiplies it, likely below PPO's advantage-estimation noise floor; and (2)
after three prior phases of convergence, `Policy/mean_noise_std` (~0.22) produces mostly
small, uncorrelated per-joint noise, and a coordinated multi-joint vertical lift is
unlikely to emerge from that by chance in any single episode.

**The `target_object_accel` question -- does it need to be relaxed, or removed further
back in Phase 1?** Checked the exact formula (`target_object_acceleration_penalty`):
`penalty = tanh(|Δv_cube| / 2.0)`, where `Δv_cube` is the cube's PER-STEP velocity
change. This penalizes jerk (sudden Δv), not velocity or motion itself. Worked through
the numbers: a smooth lift ramping to even 0.2-0.3 m/s over 5-10 steps produces a
per-step Δv of ~0.02-0.06, giving `tanh(0.03/2.0) ≈ 0.015` -- essentially free. Only a
near-instant velocity spike (a slap/throw) saturates this term. **Conclusion: a
smoothly-executed lift does not meaningfully fight this penalty, so there is no need to
weaken or remove it in Phase 1, Phase 2+3, or Phase 4** -- doing so would reopen exactly
the exploit it was added to close (see the checkpoint-5500+ regression story in its own
docstring), for a problem it isn't actually the cause of. The real interaction: EXPLORATION
noise is inherently jerky (uncoordinated per-joint noise across 13 joints), so undiscovered,
noisy first attempts at lifting are exactly the kind of motion this penalty bites hardest --
it doesn't block a mature lift, but it can plausibly suppress the noisy discovery of one.
That's a reason to fix exploration/discovery, not a reason to touch the penalty itself.

**Round 2 -- bootstrap-then-taper (2026-08-21, the user's design, reasoned through
together).** Rather than trying to make a single reward term simultaneously easy-to-
discover AND safe-to-hold-indefinitely (a hard needle to thread), split it into a
TEMPORARY bootstrap phase that gets manually switched off once discovery is confirmed:

1. `lift_scale` (new param, split out of `lift_cont_rew`'s previously-hardcoded `2.0`
   multiplier): raised to `6.0`. Steepens the reward slope near zero height so tiny
   accidental gains are large enough to register.
2. `lift_attempt_velocity` (new RewTerm, weight `0.3`): rewards the cube's own
   instantaneous positive z-VELOCITY via `tanh(clamp(v_z, min=0)/0.15)`, not its height --
   cheaper for exploration noise to produce a brief upward nudge than a lasting net height
   gain. Orthogonal to `target_object_accel` by construction (that's a Δv/jerk penalty;
   this is plain v -- a smooth sustained lift pays this fully while incurring near-zero
   jerk penalty).
3. Considered and REJECTED relying on the tanh bound alone to keep this safe
   indefinitely: bounding the ceiling only removes the incentive to exceed `vel_scale`, it
   does NOT remove the incentive to treat `vel_scale` as the default cruising speed for
   the whole lift, the longer the term stays active and gets reinforced (the user's own
   objection, and correct -- a well-tuned bound protects against unlimited speed, not
   against a faster-than-necessary HABIT). Instead: run it only until first discovery,
   then remove the incentive entirely rather than trying to make it perfectly safe to
   leave on.
4. `lift_progress` (new RewTerm, weight `1.0`, logging-only): mean cube height above
   resting, clamped at 3cm. Needs a non-zero weight to be computed/logged at all
   (IsaacLab's RewardManager skips any term with weight exactly `0.0` -- the same behavior
   that broke `--episodes N` early-stopping earlier in this project) but contributes at
   most ~0.03 to the total reward, negligible next to every other term. Purely so the
   milestone is visible as a live TensorBoard curve.
5. **Planned manual cutover** (not automated -- consistent with every other phase
   transition in this file, and deliberately keeping a human check in the loop given the
   stakes of baking in a bad habit): once `lift_progress` shows the cube reliably reaching
   ~1-1.5cm, set `lift_attempt_velocity`'s weight to `0.0` and dial `lift_scale` back down
   toward `2.0`, then resume from whatever checkpoint first shows the milestone. IsaacLab
   does have a `CurriculumManager` for automating this kind of in-training weight change
   (unused everywhere in this env so far) -- deliberately not used here, to keep this
   transition manually reviewed like every other one in this project.

**Checkpoint choice for this relaunch: 450, not 699.** The original reasoning for
preferring 699 (finger-closing/`is_grasped` tightness still improving at the final
checkpoint, see "[Phase 2+3 merged]" above) was built around the OLD gated
`lift_cont_rew`, which no longer applies now that the gate is removed. With grasp
quality nearly identical between the two (`grasp_goal_palm` 1.33@400 vs 1.39@699, ~4%
relative) but `Policy/mean_noise_std` meaningfully higher at 450 (0.240) than 699
(0.218), the leftover exploration budget matters more than the marginal grasp-tightness
699 bought, especially given Round 1's evidence that exploration collapse -- not grasp
quality -- is the binding constraint.

**Launch**: `2026-08-21_19-49-54_phase4v2_bootstrap_from_ckpt450`, 2048 envs, 500
iterations, resumed from checkpoint 450 of `phase2and3_merged_from_ckpt200`. Sanity
checked first (3 iterations, 64 envs) -- both new terms (`lift_attempt_velocity`,
`lift_progress`) confirmed computing and logging non-zero, non-NaN values. Launched
inside the `g1_train_5cm` tmux session this time (Round 1's launches used a detached
`docker exec -d`, invisible to the user's own tmux pane -- corrected after the user
flagged it).

**Round 2 result: a faint signal, but not enough to compound.** Watched live via the
`lift_progress`/`lift_attempt_velocity` TensorBoard curves per the user's own monitoring.
By iteration 850 (well past the checkpoint originally planned to check at ~800):
`lift_progress` peaked at a mere 0.06mm (iteration 513) and had already receded to
0.03mm by 599, never recovering; `lift_attempt_velocity` grew slowly but stayed tiny
(peak weighted 0.0033, implying a peak actual cube velocity of only ~1.6mm/s);
`Episode_Termination/target_lifted` fired exactly ONCE across the whole run (0.00049 at
iteration 463 -- roughly 1 of 2048 envs crossing the threshold for a single instant,
never recurring); `task_reward` settled back to the same ~1.0-1.02 plateau as every prior
attempt. Unlike Round 1's complete flatline, `Policy/mean_noise_std` did keep declining
(0.240 -> 0.217), but this looks like ordinary continued convergence rather than evidence
of exploring anything new -- nothing else moved with it. Killed by the user's explicit
instruction at iteration 850 ("still no lift... try stronger rewards").

**Round 3 -- recalibrate to the actual exploration magnitude, not the eventual target
(2026-08-21).** The key diagnosis: Round 2's `vel_scale=0.15` (tuned for a real ~15cm/s
lift speed) meant the term was barely off its own saturation floor even at its best
moment -- `tanh(0.0016/0.15) ~= 0.011`, i.e. ~1% of the term's own max reward for the
BEST motion the policy produced in 400+ iterations. Raising the *weight* alone would only
have multiplied "almost nothing" by a bigger number. Instead:
- `lift_scale`: `6.0 -> 12.0` (further steepens the near-zero-height slope).
- `lift_attempt_velocity` weight: `0.3 -> 1.0`.
- `lift_attempt_velocity`'s `vel_scale`: `0.15 -> 0.02`, RECALIBRATED to match the actual
  mm/s-scale motion being explored rather than a real lift's eventual speed -- the exact
  same ~1.6mm/s motion that scored ~0.011 raw under the old scale now scores
  `tanh(0.0016/0.02) ~= 0.08`, roughly 7x more sensitive at the magnitude that's actually
  occurring, combined with the 3.3x weight increase for a combined ~25x stronger signal
  for the same tiny motions already being produced.

Sanity-checked (3 iterations, 64 envs) before relaunch: `lift_attempt_velocity` read
0.0119 at the equivalent point in the sanity rollout vs. 0.0007 for the same checkpoint
under Round 2's parameters -- roughly 17x higher on a tiny sample, consistent with the
recalibration engaging as intended. Relaunched from checkpoint 450 (same as Round 2 --
the checkpoint choice reasoning is unaffected by this parameter change), inside the
`g1_train_5cm` tmux session, run name `phase4v2r2_bootstrap_from_ckpt450`.

**Round 3 result: recalibration worked exactly as designed, and that's what proved
reward-shaping alone had hit its ceiling.** Watched live: `lift_attempt_velocity` rose
fast from 0.0005 to ~0.03-0.04 by iteration ~560 (roughly 10x higher than Round 2's peak
at the equivalent point, matching the recalibration's intent), but then PLATEAUED there
through iteration 658 rather than continuing to climb. `lift_progress` stayed negligible
throughout (max 0.05mm). `target_lifted` never fired once this round (worse than Round
2's single blip). The clean interpretation: making the reward MORE SENSITIVE to small
motions worked (the same tiny motions score far higher now), but the motions THEMSELVES
never got any bigger -- confirming the bottleneck is exploration magnitude, not reward
magnitude, exactly as flagged as the fallback above.

---

### [Phase 4 v2 Round 4] Targeted arm-joint noise boost -- 2026-08-21

**Mechanism check first**: confirmed via `OnPolicyRunner.load()`'s source
(`self.alg.policy.load_state_dict(...)`) that resuming from a checkpoint restores the
ENTIRE policy including its trained noise std -- so changing `agent_cfg.init_noise_std`
and resuming would do nothing, the checkpoint's own saved value overwrites it
immediately. Actually raising exploration for a resumed run requires directly patching
the checkpoint file's `model_state_dict["std"]` tensor (shape `(13,)`, one scalar per
policy-controlled joint, `noise_std_type="scalar"` -- confirmed via
`ActorCritic.__init__`) and resuming from THAT modified file.

**A discovery from actually doing this**: the un-patched checkpoint 450's per-joint std
was starkly asymmetric, not a single collapsed number:

| Joint group | std values |
|---|---|
| Arm (shoulder x3, elbow, wrist x3) | 0.044, 0.050, 0.062, 0.049, 0.124, 0.134, 0.087 |
| Hand (thumb x2, index, middle, ring, pinky) | 0.503, 0.418, 0.409, 0.368, 0.446, 0.421 |

The arm -- the joints that would actually have to move to raise the hand+cube in z --
had collapsed to near-zero (as low as 0.044), 4-10x below the hand joints. This is a
more precise explanation for Rounds 1-3's failure than "noise_std is generally low": the
arm specifically was nearly frozen by three phases of converging on a precise static
pose, while the hand retained substantial exploration the whole time.

**The user's proposal, reasoned through together**: rather than a uniform boost across
all 13 dims (which would have actually LOWERED most hand joints, since several were
already above 0.35), boost the arm specifically -- the single joint most responsible for
z-height higher, the rest of the arm to about half that (since a real lift needs
whole-arm coordination, not one joint swinging in isolation), and leave the hand
untouched.

**Identifying the responsible joint -- two attempts, the first one flawed.**

*Attempt 1 (rejected by the user, correctly):* a single-step (5-step) perturbation test
on the CONVERGED GRASP POSE (cube already held), measuring resulting `cube_z` delta.
Result: `shoulder_pitch` cube_z=+0.011cm, `shoulder_yaw` cube_z=+0.032cm -- appeared to
favor yaw. The user pushed back: `cube_z` deltas that small are dominated by noisy,
weak grip-cube contact coupling (only 2 of 7 joints even registered a non-zero cube_z at
all), not a reliable kinematic signal, and a single 5-step perturbation isn't a "good
amount" of swing to see a joint's real effect. Asked for the HAND's own (x, y, z) -- not
the cube's -- tracked through a proper SUSTAINED swing instead.

*Attempt 2 (the real answer):* 40-step sustained swings (not single-step) of
`shoulder_pitch` and `shoulder_yaw` individually, both directions, tracking palm (x, y,
z) directly rather than the cube. Also cross-referenced against the actual video
(`phase4_lift_success_from_ckpt699/videos/play_markers/rl-video-step-0_ckpt1000_mesh.mp4`
at 1:02, frame 930 of 960 at 15fps -- confirmed this lands ~210 steps into the 4th
recorded episode) to confirm the settled test pose (side-approach, hand curling in from
above-and-beside the cube) visually matches the kind of grasp being reasoned about:

| Joint, delta | dz (cm) | dx (cm) | dy (cm) |
|---|---|---|---|
| shoulder_pitch, -1.0 | **+3.21** | +0.34 | +2.33 |
| shoulder_pitch, +1.0 | -2.12 | -3.98 | -1.14 |
| shoulder_yaw, +1.0 | -0.34 | -2.75 | +1.17 |
| shoulder_yaw, -1.0 | +0.33 | +0.87 | -1.01 |

`shoulder_pitch` moves the palm ~3.2cm vertically at this swing magnitude, a clean
sustained climb per the step-by-step trajectory (not a blip); `shoulder_yaw` moves it
only ~0.33cm vertically -- an order of magnitude less, and yaw's effect is
overwhelmingly horizontal (2-3cm of dx/dy) rather than vertical, exactly matching the
expected kinematics of a roughly-vertical rotation axis. Attempt 1's `cube_z` numbers
were noise; `shoulder_pitch` (action index 0) is confirmed as the primary z-joint.

**Patch applied**: `std[0]` (shoulder_pitch) `0.044 -> 0.35`; `std[1..6]` (the other six
arm joints) `-> 0.175` (half); `std[7..12]` (hand) left at their original per-joint
values. Saved as `model_450_std_armboost.pt`. Sanity-checked (3 iterations, 64 envs) --
`Mean action noise std` read 0.30 (matches the hand-computed weighted average of the
patched vector), no NaNs, Round 3's bootstrap reward terms (`lift_scale=12`,
`lift_attempt_velocity` weight=1.0/vel_scale=0.02) still active and logging correctly on
top of this.

**On grip tightness** (raised as a caveat after Attempt 1 -- even a real palm z-shift
barely moved the cube, suggesting the grip may not be tight enough to carry it along):
the user's call was to not pre-solve this and let the policy discover it once it sees
even a little reward for actually increasing cube z -- reasonable given `is_grasped`-
independent shaping (`lift_attempt_velocity`, `lift_progress`) is already active and
should reward incidental height gain regardless of grip quality; revisit only if height
gain shows up without ever translating into sustained cube movement.

**Launch**: `2026-08-21_23-48-37_phase4v2r4_armboost_from_ckpt450`, 2048 envs, 500
iterations, resumed from the patched checkpoint. Round 3's bootstrap reward terms
unchanged, only the noise std patched.

**Round 4 result: also no growth.** `lift_progress` peaked at iteration 471 (0.16mm) and
*declined* from there to 0.06mm by iteration 790 -- no growth trend despite the targeted
exploration boost. `target_lifted` fired once (iteration 562, ~5 of 2048 envs
momentarily), never again. `Policy/mean_noise_std` decayed normally (0.305 -> 0.259)
this time (unlike Round 2's frozen signature) -- meaning the policy DID have more raw
exploration available and was consuming it in the ordinary course of convergence, but
still never discovered or reinforced a growing lift trend. Combining "properly targeted,
empirically-justified exploration boost" with "recalibrated, amplified dense reward" --
arguably the most principled version of the reward/exploration-magnitude approach --
still produced nothing after 790 iterations. Killed by the user's own assessment ("I
know it will not converge to an actual pickup... I don't think there's any option of
giving it another chance").

---

### [Phase 4 v2 Round 5] Reference State Initialization (RSI) -- attempted, blocked by an unresolved Isaac Sim hang, 2026-08-21/22

**The idea**: rather than only reshaping reward or exploration noise, give the policy
direct experience of already-elevated hand+cube states by overriding a fraction of
episode resets to start the arm/hand in the policy's own converged grasp joint
configuration (measured empirically via `capture_grasp_joints.py`, median over 64 real
rollouts of checkpoint 450) with a `shoulder_pitch` offset that raises the hand by a
randomly sampled 1-8cm, and the cube pre-positioned to match. Implemented as
`reset_arm_hand_object_with_rsi` in `mdp/events.py`, wired in as a new `EventTerm`
(`rsi_arm_hand_object`, mode `"reset"`, 30% probability) in `EventsCfg`, inserted between
`reset_target_object` and `sample_grasp_goal` (the latter reads the post-reset cube pose,
so ordering matters). Used the exact same `write_joint_state_to_sim`/
`write_root_pose_to_sim` API calls as the existing, working `reset_joints_by_offset`/
`reset_root_state_uniform` functions.

**The hang**: the instant this `EventTerm` was registered, environment creation hung
indefinitely -- no error, no crash, no traceback, CPU actively pegged (~110%,
confirmed not deadlocked/idle) but no progress, reproduced across 4-env and 64-env
scenes alike, and independent of the term's param VALUES (bisected down to a
minimal-but-complete params dict of flat scalars, still hung). Disabling the term
(commenting it out) restored normal, fast (~15s) environment creation immediately.
`py-spy`/`strace` were unavailable to get a live stack trace -- this container lacks the
`SYS_PTRACE` capability both tools need, and it cannot be added to an already-running
container without recreating it.

**Root cause found (2026-08-22), via Python's built-in `faulthandler` module** --
`faulthandler.dump_traceback_later(45, exit=False)`, which self-dumps every thread's
Python stack via a signal handler from *inside* the process, requiring no ptrace
capability at all (unlike py-spy/gdb). Re-enabled the RSI term, armed the dump, and
re-ran the same test. The captured trace:

```
get_dof_velocities()                          [PhysX tensor API, C++/CUDA]
  <- ArticulationData.joint_vel / .joint_acc   [articulation_data.py]
  <- ArticulationData.update()
  <- Articulation.update()
  <- InteractiveScene.update()
  <- ManagerBasedEnv.__init__()
```

The hang is **not inside `reset_arm_hand_object_with_rsi`'s own body at all** -- it
occurs earlier, inside `ManagerBasedEnv.__init__`'s own automatic first
`InteractiveScene.update()` call, stuck inside PhysX's `get_dof_velocities()` GPU tensor
query (via `joint_acc`'s finite-difference of `joint_vel`), before any `"reset"`-mode
`EventTerm` -- including this one -- ever actually executes. This means the bug was never
in this function's reset logic; it's a low-level interaction between Isaac Sim's PhysX
GPU tensor API and having an 18th `"reset"`-mode term registered (regardless of what that
term does), most plausibly a race condition or GPU-buffer-readiness ordering issue
triggered by however `EventManager._prepare_terms()` allocates per-term buffers relative
to PhysX's own tensor-view initialization. Not fixable from the Python/environment-config
level available in this project -- would need Isaac Sim engineering support or
lower-level GPU/CUDA debugging tooling than this container has (no `SYS_PTRACE`, no
`py-spy`, no `strace`, and cannot recreate the container to add them mid-session).

**Status**: RSI shelved, not because the underlying idea (reference-state initialization)
is wrong, but because this specific implementation path is blocked by a simulator-level
issue outside this project's reach right now. `reset_arm_hand_object_with_rsi` is left in
`mdp/events.py`, and its `EventTerm` registration in `EventsCfg` left commented out with
the full trace above, for whoever has the tooling to pick this back up. The user's own
call, after being shown this evidence: accept it as genuinely blocked and move to a
different exploration mechanism (RND) instead -- see "[Phase 4 v3]" below.

---

**Diagnosis, reframed one more level**: the user's own root-cause read, arrived at
independently: `Policy/mean_noise_std` being nontrivial (0.24-0.31 average) does NOT
mean the arm is exploring net displacement in z, because PPO's per-step Gaussian action
noise is memoryless -- resampled independently every control step, uncorrelated across
time. Around an already-converged, STATIONARY mean action (the resting grasp), iid noise
produces jitter that mostly cancels out over any time window; it does not produce
SUSTAINED, correlated drift in one direction. Discovering "hold this direction for many
consecutive steps" requires either (a) noise that's temporally correlated (not available
in vanilla PPO without algorithm-level changes), or (b) a positive reward specifically
for STATE NOVELTY (not outcome) that makes deviating from the converged equilibrium
itself worth doing, regardless of whether that particular deviation happens to increase
height. This reframing is what motivated the switch to RND below, and is also what ruled
out two further ideas raised and rejected in discussion (kept for the record, not
implemented):

1. **Warm-starting from an old pre-existing checkpoint** (an older, non-from-scratch
   "perfect pick" policy, and a "checkpoint 5450" policy with correct hand pose too, both
   predating this whole from-scratch curriculum). Rejected for two independent reasons:
   (a) mechanically, any state-injection approach needs the same custom EventTerm
   machinery that RSI's implementation hung on (see "[Phase 4 v2 Round 5]" further above
   for the unresolved technical wall), so it likely inherits that exact risk; (b) even if
   implemented, naively mixing an old policy's actions into PPO's on-policy training
   would bias the current policy toward literally copying the old policy's specific
   behavior (PPO's surrogate objective, given actions it didn't itself sample, degenerates
   toward behavior-cloning that action) -- and the user specifically doesn't want that,
   since the old "perfect pick" policy was independently flagged (by the user and their
   advisor) as too aggressive, which is exactly the behavior the from-scratch forward
   curriculum was designed to avoid in the first place.
2. **A penalty that's large while the cube is at rest and fades/flips positive as height
   increases**, on the theory that a sudden reward-level drop would make the policy
   "notice something changed" and search harder. Rejected on RL-theoretic grounds: a
   smoothly-fading penalty has the identical GRADIENT (with respect to height) as the
   already-tried `lift_cont_rew`, since PPO's advantage estimates are computed from
   *differences* in returns -- a constant offset added uniformly to every step's reward
   doesn't change which actions look relatively better, so this is mathematically the
   same mechanism as Rounds 2-4's dense reward, already shown insufficient at 6x-12x
   amplification. The step-threshold variant (penalty removed once cube crosses some
   height) reduces to the earlier-rejected "lower `_SUCCESS_Z`" idea, with the same
   bootstrapping problem (needs the cube to already cross the threshold once, which has
   never happened even to 1mm). The one real, valid mechanism buried in the proposal --
   PPO's critic being transiently miscalibrated right after a reward-function change,
   producing a brief burst of uniformly-biased advantage -- is real but short-lived (the
   critic re-fits within tens of iterations) and undirected (doesn't specifically point
   toward "try lifting" over any other deviation), so it isn't a substitute for a
   sustained, targeted exploration mechanism.

---

### [Phase 4 v3] Random Network Distillation (RND), arm-joint-only novelty -- 2026-08-21

**The mechanism**: RSL-RL has native support for Random Network Distillation (Burda et
al., 2018) as an algorithm-level intrinsic reward -- a frozen random "target" network and
a trained "predictor" network both embed the current state; the L2 distance between their
embeddings is high for states the predictor hasn't learned to match yet (novel) and low
for familiar states. This is genuinely different in kind from every reward tried so
far in Phase 4: it is not a function of height, velocity, or any task outcome at all --
it rewards visiting unfamiliar STATES, independent of whether that visit helps the task.
Configured entirely via `agents/rsl_rl_ppo_cfg.py` (`RslRlPpoAlgorithmCfg.rnd_cfg`) and a
new `ObservationsCfg` group -- does not touch `EventsCfg`/`EventManager` at all, so it
does not inherit RSI's unresolved hang (see "[Phase 4 v2 Round 5]" above).

**Scoping the RND state -- two design questions raised by the user, both changed the
design for the better:**

1. *"What if the policy never bothers to change the cube's pose?"* The first draft of
   this idea scoped RND to the cube's own position. That has a bootstrapping problem: if
   the cube has never moved, "cube always at rest" is trivially predictable, so the
   intrinsic reward computed over cube-state would stay at ~zero forever -- there's
   nothing novel to detect in a state that never varies. Fixed by scoping RND to the ARM
   JOINT POSITIONS instead (not the cube) -- arm angles already have some natural
   per-step variance from ordinary action noise, so RND has something to compute novelty
   over from step one, and it directly targets the actual precursor behavior needed
   (novel arm configurations) rather than depending on the cube already having moved.
2. *"I would refrain from having fingers in that observation space... grasp pose matters
   to me a lot."* Confirmed correct and incorporated directly: the RND state is scoped to
   ONLY the 7 right-arm joints (shoulder x3, elbow, wrist x3) -- explicitly excluding the
   6 hand/finger joints, so novelty-seeking cannot push the trained grasp SHAPE around at
   all. The existing grasp-quality reward terms (`grasp_goal_palm`, `grasp_envelope`,
   `grasp_weight`) remain the only influence on finger configuration, untouched by this
   change.

**Implementation**:
- New `ObservationsCfg.RndStateCfg` group (`g1_pick_env_cfg.py`): a single `ObsTerm`
  (`mdp.joint_pos_rel`, scoped via `SceneEntityCfg(joint_names=[7 arm joints])`), 7-dim
  output. (Caught one bug while sanity-checking: the class needs its own `@configclass`
  decorator like every other `ObsGroup` subclass -- without it, `ObsTerm` fields don't
  register at all, and the group resolves to an empty/zero-shape observation. Fixed
  before launch.)
- `agents/rsl_rl_ppo_cfg.py`: `obs_groups = {"policy": ["policy"], "critic": ["policy"],
  "rnd_state": ["rnd_state"]}` and `RslRlRndCfg(weight=10.0, state_normalization=True,
  reward_normalization=True, num_outputs=16)` on the algorithm config. `weight=10.0` gets
  multiplied by `env.step_dt` (~0.033) internally by RSL-RL, giving an effective ~0.33
  per-step contribution -- a starting value, not a measured optimum.
- **Checkpoint-compatibility snag**: `OnPolicyRunner.load()` unconditionally tries to
  load `rnd_state_dict` from the resumed checkpoint whenever RND is enabled in the
  current config -- but checkpoint 450 predates RND entirely and has no such key,
  producing a `KeyError` on resume. Fixed the same way the arm-joint noise std was
  patched earlier: loaded `model_450.pt`, constructed a fresh `RandomNetworkDistillation`
  module + Adam optimizer matching the exact `RslRlRndCfg` params above, added their
  freshly-initialized `state_dict()`s as `rnd_state_dict`/`rnd_optimizer_state_dict`
  keys, saved as `model_450_rnd_init.pt`.

**Sanity check** (3 iterations, 64 envs, from the patched checkpoint): confirmed
`Active Observation Terms in Group: 'rnd_state' (shape: (7,))` -- correctly scoped,
arm-only; `Mean rnd loss` printed and changing each iteration (0.24 -> 0.24 -> 0.18),
confirming the predictor network is actually training against the frozen target's
embeddings; no NaNs, clean exit.

**Two ideas raised and explicitly ruled out before settling on RND** (both discussed at
length with the user, kept for the record -- see "Diagnosis, reframed one more level"
above for the full reasoning): warm-starting from an old pre-curriculum "perfect pick"
checkpoint (rejected: inherits RSI's technical risk AND risks the policy mimicking that
checkpoint's independently-flagged aggressive behavior, which this whole from-scratch
curriculum exists to avoid), and a penalty-while-at-rest that fades as height increases
(rejected: mathematically equivalent in gradient to the already-tried and insufficient
dense `lift_cont_rew`, since PPO's advantage estimation is offset-invariant to constant
per-step reward shifts).

**Launch**: `2026-08-22_04-34-33_phase4v3_rnd_from_ckpt450`, 2048 envs, 500 iterations,
resumed from `model_450_rnd_init.pt`. Round 3's bootstrap reward (`lift_scale=12`,
`lift_attempt_velocity` weight=1.0/vel_scale=0.02) left active alongside RND -- RND
drives exploration into new arm configurations, the existing shaped reward reinforces
staying/progressing once a configuration happens to increase height. Confirmed running
at full scale with `Mean rnd loss` actively computing (0.11 -> 0.09 in the first two
logged iterations).

**Status**: DID NOT WORK -- ran the full 500 iterations (450 -> 949).
`Rnd/mean_intrinsic_reward` started at 5.78, peaked at 8.57 (iteration 456), then decayed
to 0.05 by the end -- `Loss/rnd` dropped from 0.236 to ~0.00002 over the same span,
meaning the predictor network fully learned to match the target network's embeddings for
whatever LIMITED region of arm-joint space the policy was actually sampling. The intrinsic
reward engaged exactly as designed (a real, measurable novelty signal existed and was
being optimized), but the region it found novel was too small/local to ever compound into
a large enough excursion to raise the cube -- `lift_progress` never exceeded 0.06mm,
`target_lifted` fired once (iteration 479, ~1 of 2048 envs). Killed by the user's own
assessment ("even that didn't work").

---

### [Phase 4 v2 Round 6] Same arm-boosted setup, lift_attempt_velocity weight x10 -- 2026-08-22

**The user's own last-resort proposal, explicitly accepting the instability risk**: after
RND also failed to produce growth, reproduce Round 4's exact setup (arm-boosted noise --
`std[shoulder_pitch]=0.35`, other arm joints `=0.175`, hand untouched -- `lift_scale=12`)
but raise `lift_attempt_velocity`'s weight another 10x, from `1.0` to `10.0` (`vel_scale`
unchanged at `0.02`). Explicitly framed as a one-shot bet: accept a real risk of unstable
or aggressive motion at this magnitude, on the chance it produces even a crude pickup --
then immediately strip this term back out and fall back to the position-based rewards to
smooth the behavior out, rather than keep this weight permanently.

**Implementation**:
- RND (Phase 4 v3) disabled: `algorithm.rnd_cfg = None` in `agents/rsl_rl_ppo_cfg.py`.
  `obs_groups` and `ObservationsCfg.RndStateCfg` left in place, unused, in case RND is
  revisited later with a different scoping.
- `lift_attempt_velocity` weight `1.0 -> 10.0` in `RewardsCfg` (`g1_pick_env_cfg.py`).
- Resumed from `model_450_std_armboost.pt` (Round 4's arm-boosted checkpoint, still on
  disk) -- NOT the RND-patched checkpoint, since RND is off for this run.

**Sanity check** (3 iterations, 64 envs): `lift_attempt_velocity` weight correctly showed
`10.0` in the printed reward-term table; `Episode_Reward/lift_attempt_velocity` scaled up
proportionally (0.19 by iteration 452, vs. ~0.02 at the equivalent point under weight
1.0); `Mean action noise std` correctly read 0.30, matching the arm-boosted checkpoint;
no NaNs, clean exit.

**Launch note**: this run's scene setup was unusually slow (~6 minutes of CPU-only
activity with zero GPU memory growth before PhysX buffers finally allocated and training
began normally) -- initially indistinguishable from another hang like RSI's. Confirmed
NOT a repeat via `nvidia-smi` memory tracking (flat at baseline the whole time, then a
clean jump to 3.1GB right as training started) rather than killing prematurely. Likely
explained by host GPU 0 running an unrelated user's job at ~55-60% utilization throughout
this session, plausibly causing some cross-GPU driver-level contention during CPU-bound
scene setup -- not investigated further since it self-resolved.

**Launch**: `2026-08-22_07-15-20_phase4v2r6_from_ckpt450`, 2048 envs, 500 iterations.
Confirmed running normally at full scale once past the slow startup (14s/iteration,
matching every other run this project).

**Result, checked at checkpoints 650 and 700 via inference recording + TensorBoard**:
for the first time in this whole Phase 4 saga, something measurably different happened --
`lift_progress` reached 2.0-2.3mm (15-40x any prior round's peak), `Episode_Termination/
target_dropped` went nonzero for the first time ever (peak 0.325% of envs at iteration
494 -- meaning a genuine grasp-then-lose-it cycle, not nothing), and `Policy/
mean_noise_std` INCREASED (0.305 -> 0.34) instead of decaying, the first time exploration
grew rather than settling. But the reward-breakdown plot showed why: `lift_attempt_
velocity` was slamming into its saturated ceiling (~10, matching weight=10) repeatedly
and continuously throughout every episode, not as a single sustained push, alongside
persistently large `joint_speed`/`target_object_accel` penalties (up to -0.75/-0.40 per
step) the entire episode. The `palm_pose_error` plot confirmed it visually: position
error swung up to 30cm and orientation error up to 175 degrees WITHIN single episodes at
checkpoint 650 (improving to ~12-14cm / 55-65 degrees by 700, but still large) -- the
policy was thrashing through large regions of pose space chasing the velocity spike, not
executing a controlled lift. `lift_progress` plateaued at ~2mm across both checkpoints,
not growing further. The user's own read, confirmed by this data: "the policy has not
learned how to pick up. It has learned just to move the cube around... deters his hand as
much as possible to increase the velocity reward." Killed at iteration ~732.

**Status**: DID NOT WORK as hoped -- produced real signal (unprecedented lift_progress,
first-ever drops, growing exploration) but via reward-farming through large, poorly-
controlled motion rather than a genuine pickup attempt. This was the last idea in the
user's own list before requesting a full stop -- see the wrap-up entry below.

---

## Session wrap-up (2026-08-22) -- Phase 4 paused, RSI re-confirmed blocked, all attempt logs cleared

After Round 6, the user called a stop: "the policy has not learned how to pick up... just
stop the training. Clear all the trainings." Actions taken:

1. Killed the Round 6 training process.
2. Deleted every Phase 4 attempt's run directory (v1 gated, v2 ungated, v2 bootstrap
   Rounds 2-4, v3 RND, v2 Round 6) and their derived/patched checkpoint artifacts
   (`model_450_std_armboost.pt`, `model_450_rnd_init.pt`). `logs/rsl_rl/g1_pick/` now
   contains only `phase1_reach_palmpos_cuberot45` (the validated Phase 1 foundation) and
   `phase2and3_merged_from_ckpt200` (Phase 2+3, source of checkpoint 450 -- the handoff
   point for every Phase 4 attempt so far).
3. The user asked to revisit RSI one more time before giving up on it entirely
   ("push back on the RSI idea if you can... if it's absolutely not possible, then I'm
   completely at a loss"). Re-enabled `rsi_arm_hand_object`, instrumented the reproduction
   script with `faulthandler.dump_traceback_later` (works without the `SYS_PTRACE`
   capability that blocked `py-spy`/`strace` earlier), and got a real stack trace this
   time -- see "[Phase 4 v2 Round 5]" above for the full trace and analysis. Confirmed:
   the hang is a low-level Isaac Sim PhysX GPU tensor API issue triggered merely by
   registering an 18th `"reset"`-mode `EventTerm`, occurring before any such term's body
   ever runs -- genuinely not fixable at the Python/config level available here. RSI is
   re-confirmed blocked, not abandoned prematurely. Re-disabled the term.
4. Phase 4's reward/algorithm config (`lift_attempt_velocity` weight=10.0,
   `lift_scale=12`, `rnd_cfg=None`, the arm-only `RndStateCfg` observation group, the
   `obs_groups` mapping) is left as-is in code -- reflects the state of the last attempt,
   not a recommendation, and will need to be redesigned before the next Phase 4 attempt.

**Where this leaves things**: five distinct approaches have now been tried and ruled out
for getting the policy to discover lifting from checkpoint 450 (gated dense reward,
ungated dense reward, escalated+recalibrated dense reward, targeted arm-exploration
noise, RND intrinsic novelty, and a 10x reward escalation) plus RSI blocked at the
simulator level. The common thread across every reward-shaping attempt: something CAN be
made to move (lift_attempt_velocity, RND's intrinsic reward, arm noise_std) but none of
it compounds into a controlled, sustained lift -- either it plateaus (Rounds 2-4, RND) or
it produces large uncontrolled motion without ever tightening into something purposeful
(Round 6). The root-cause diagnosis that's held up across all of this: PPO's per-step
Gaussian action noise is memoryless (resampled independently every control step), so it
cannot produce SUSTAINED, correlated drift in one direction around an already-converged,
stationary grasp equilibrium -- only local jitter or, once amplified enough, uncontrolled
excursions. A genuinely different remaining avenue, not yet tried: temporally-correlated
exploration noise (e.g. an Ornstein-Uhlenbeck process instead of i.i.d. Gaussian) injected
directly into action sampling via a custom RSL-RL `ActorCritic` subclass -- this attacks
the memorylessness itself, is configurable at the algorithm level (same integration point
already used for RND, via `RslRlPpoActorCriticCfg.class_name`), and doesn't touch
`EventsCfg` at all, so it doesn't inherit RSI's blocked mechanism. Not yet discussed with
the user or implemented -- next candidate to raise.

---

### [Phase 4 v4] Static riser -- cube pre-elevated for the whole episode, 2026-08-22/23

**The user's idea**: spawn the cube 1-2cm above resting each episode, hoping a short
enough drop wouldn't disturb its orientation and would let the policy discover that
increasing cube z pays off, without touching PPO/the policy itself -- explicitly
requested to stay in "vanilla PPO, reward/init changes only" territory after ruling out
the custom-`ActorCritic` (correlated-noise) idea for changing the policy itself.

**Why the idea as literally described wouldn't have worked, and the fix**: a 1-2cm free
fall resolves in ~1-2 control steps (`t = sqrt(2h/g) ~= 55ms` at 30Hz), but the hand
doesn't typically reach the cube until 50-150 steps into an episode (the converged
Phase 2+3 behavior, confirmed in every reward-breakdown plot this whole project). The
elevation-and-fall would be over long before the hand arrives, so nothing about it is
caused by the policy's actions -- PPO has nothing to attribute that reward to. This is
the same failure mode that ruled out scoping RND to cube-state earlier ("[Phase 4 v3]").
Fix: instead of a free-fall, give the cube something to physically REST ON at the
elevated height for the WHOLE episode, so whenever the hand does arrive, it's still up
there -- closing the timing gap entirely.

**Implementation, round 1 (a new static riser prop) -- also hit RSI's exact hang, from
an entirely different cause.** Added a new `riser` `AssetBaseCfg` (a thin kinematic
cuboid, 0.22 x 0.12 x 1.5cm, sized to cover the cube's full randomized spawn footprint
with margin, comfortably clear of every distractor's fixed anchor) plus a matching
`reset_target_object` z pose_range change (`0.0` -> `_RISER_HEIGHT`). The instant this
was tested, environment creation hung -- **the identical stack trace as RSI's**
(`get_dof_velocities` inside the env's first automatic `scene.update()`), even though RSI
itself was confirmed disabled and no `EventTerm` was involved at all this time. This
REVISED the earlier RSI root-cause read: it isn't specifically about `EventTerm` count,
it's something broader.

**Isolation testing that initially pointed the wrong way.** Bisected carefully:
- Riser asset alone (param change reverted): still hung.
- Reverting the riser too, just raising the EXISTING `tray`'s z-position instead (no new
  prim at all): still hung.
- Reverting THAT too, testing the pure `reset_target_object` z-offset param change alone
  on an otherwise completely stock scene: still hung.
- Reverting EVERYTHING back to the exact baseline that worked fine for Round 6: **still
  hung**, even with a triple-confirmed clean GPU/process state (no leftover processes, no
  compute-apps, baseline memory).

This pointed toward the environment itself having degraded independent of any of these
changes -- plausibly from the cumulative effect of dozens of `SIGKILL`s on hung Isaac
Sim/CUDA processes throughout this very long session (each individually left no visible
trace, but `SIGKILL` never lets CUDA release its GPU context gracefully). Restarted the
`shahid_g1pick` container (checked with the user first, given restarting a container is a
more impactful action than killing an individual process) -- **the baseline still hung
immediately after restart**, which seemed to rule out container-scoped state too (Docker
GPU passthrough shares the host's one driver instance; a container restart doesn't reset
host-level driver state).

**The actual, much narrower cause**: before escalating further, tested whether the
standard `train.py` resume path (not the custom repro script) still worked -- it did,
cleanly, 3/3 iterations. The one consistent difference: `train.py`'s sanity checks always
use `--device cuda:0`; every repro script this session had defaulted to `--device cuda:1`
(this project's established convention: cuda:0 for training, cuda:1 for
inference/diagnostics). Re-ran the exact same hanging repro script with `--device cuda:0`
instead -- worked immediately, cleanly, first try. **`cuda:1` (host GPU 2) had
degraded specifically, not the container, not the host broadly, and not any of the
riser/RSI code.** This also retroactively explains the RSI hang from "[Phase 4 v2 Round
5]" -- that investigation also exclusively used `cuda:1` and was never re-tested on
`cuda:0`, so RSI's implementation may well be fine; the earlier "root cause" (PhysX
tensor-API race from added scene/config complexity) was real as an observed symptom but
`cuda:1`'s degraded state was very plausibly the actual underlying trigger, not
complexity/count. Worth retrying RSI on `cuda:0` if this comes up again -- not
re-attempted here since the riser idea was already in flight and the user wanted to keep
moving.

**Verified working on `cuda:0`**: `verify_riser.py` (standalone reset + 30 zero-action
steps, no policy) -- cube spawns at exactly 86.0cm (0.845 + 0.015, matching the design
exactly) and stays there with max drift 6e-6cm over 30 steps -- rock stable, no sinking
through, no bounce. Sanity-checked via the standard 3-iteration/64-env `train.py` resume
-- clean, no NaNs, `lift_progress` reads nonzero from the very first iteration (expected
now: the riser means every episode registers ~0.015 of `lift_progress` baseline
regardless of policy action -- the meaningful signal going forward is whether it climbs
ABOVE that baseline, not just whether it's nonzero).

**Reward config reverted to un-escalated defaults for this attempt**, to test the riser's
effect in isolation rather than confounded with the earlier extreme reward tuning (this
project's one-variable-at-a-time convention): `lift_scale` `12.0 -> 2.0` (original
default), `lift_attempt_velocity` weight `10.0 -> 0.0` (fully disabled -- Round 6 showed
it driving large uncontrolled thrashing rather than controlled lifting; the whole premise
of the riser is that a genuinely reachable elevated cube should make lifting discoverable
through the existing dense `lift_cont_rew` alone, without needing an extra
velocity-chasing incentive).

**Launch**: `2026-08-23_21-18-33_phase4v4_riser_from_ckpt450`, 2048 envs, 500 iterations,
resumed from checkpoint 450. Confirmed running normally at full scale (16s/iteration).

**Coupled changes, note for reverting later** (per the user's own stated plan -- once the
policy learns to lift, go back to a normal resting-height cube to let it perfect the
grasp pose): `tray`'s z-position and `reset_target_object`'s z pose_range must be
reverted TOGETHER. Reverting only one embeds the cube inside the riser height gap or
leaves the tray floating above where the cube expects to land.

**Result**: ran 225 iterations (450 -> 675). `lift_progress` sat at exactly the riser's
own baseline (0.015, first logged at 0.00104 rising immediately to ~0.015 and staying
there -- max 0.01521 at iteration 460) for the entire run, with no growth beyond it.
`target_lifted`/`target_dropped` each fired once (single-env blips, iterations 465/498).
`Policy/mean_noise_std` declined normally (0.240 -> 0.230), no sign of new exploration.
`task_reward` converged to ~1.04, the same plateau as every other round. The policy
learned to simply hold at the free elevation the riser provided and never discovered
pushing beyond it -- confirmed by the user directly watching TensorBoard ("still
absolutely nothing"). Killed at iteration 675.

**Also attempted while this was running**: the user, feeling responsible for having
caused GPU 2's degraded state (many `SIGKILL`s across this session), asked whether it
could be fixed. Ran `nvidia-smi -i 2 -q` diagnostics -- no ECC errors, no retired memory
pages, no thermal throttling, `GPU Reset Status: Reset Required: No` (the driver itself
doesn't consider this a fatal hardware fault, consistent with a recoverable wedged
CUDA-context state rather than permanent damage). Attempted `nvidia-smi --gpu-reset -i 2`
directly -- failed with "Insufficient Permissions" (requires root/sudo, unavailable in
this environment). Left as an action item for the user or a server administrator with
sudo access; not blocking further work here since `cuda:0` remains healthy and is now
used for all inference/diagnostic scripts too (previously `cuda:1` by convention).

**Reverted (2026-08-23), per the user's explicit request**: `tray` z-position back to
`0.81` (from `0.81 + _RISER_HEIGHT`), `reset_target_object`'s z pose_range back to
`(0.0, 0.0)` (from `(_RISER_HEIGHT, _RISER_HEIGHT)`), the now-unused `_RISER_HEIGHT`/
`_RISER_TOP_Z` constants removed. Verified via `verify_riser.py` on `cuda:0`: cube spawns
at exactly 84.5cm (the original `_OBJ_INIT_Z`), stable, no NaN -- environment geometry
confirmed back to the validated Phase 2+3 baseline. Failed run's logs deleted (matching
this project's established practice of clearing dead-end run directories while keeping
Phase 1 and Phase 2+3). `logs/rsl_rl/g1_pick/` now contains only
`phase1_reach_palmpos_cuberot45` and `phase2and3_merged_from_ckpt200` again.

**Status**: reverted. Six distinct approaches now tried and ruled out from checkpoint 450
(gated/ungated/escalated dense reward, targeted arm-exploration noise, RND, 10x reward
escalation, and pre-elevating the cube via a riser) -- the user is stepping back to think
of new ideas rather than continue immediately. One open, not-yet-retried thread worth
remembering: RSI's "blocked" verdict from "[Phase 4 v2 Round 5]" was reached entirely on
the now-confirmed-degraded `cuda:1`, never re-tested on the healthy `cuda:0` -- it may
not actually be blocked at all.

---

## [Phase 2+3 v2] Root cause found: the fingers never touched the cube -- 2026-08-23/24

**How this was found.** After six failed Phase 4 attempts, stepped back and re-measured
the actual physical starting point every one of them shared: checkpoint 450 of
`phase2and3_merged_from_ckpt200`. Ran `contact_check.py` (64 envs x 300 steps,
`--device cuda:0`) to get each fingertip's CLOSEST-EVER approach to the cube surface
across a full rollout -- not the average, the single best moment:

| Finger | Closest ever | Mean (of per-env closest) | Envs actually touching |
|---|---|---|---|
| thumb | +0.18cm | +1.33cm | 0 / 64 |
| index | +1.92cm | +2.68cm | 0 / 64 |
| middle | +1.75cm | +2.34cm | 0 / 64 |
| ring | +1.28cm | +2.57cm | 0 / 64 |
| pinky | +1.72cm | +3.14cm | 0 / 64 |

**Zero of 64 environments ever made contact, on any finger, at their single closest
moment across the whole rollout.** Same result at every other checkpoint checked this
whole project (350, 500, 600, 699) -- this was never specific to 450, it was the
underlying state of the policy the entire time Phase 4 was being attempted.

**Why this explains every single Phase 4 failure mechanically, not just circumstantially.**
A reward can only drive policy-gradient learning through a state if it has a non-zero
gradient with respect to something the policy directly controls, evaluated AT the
policy's current state. `lift_cont_rew` (and everything downstream of cube height --
`lift_progress`, the riser's baseline, `success_bonus`) is a function of the CUBE's
position. With a 1-3cm permanent air gap between every finger and the cube, moving the
arm changes cube height by exactly zero -- not a small gradient, an identically zero one,
for as long as no finger is in contact. Scaling `lift_scale` 2->6->12, or
`lift_attempt_velocity` to weight 10, multiplies zero by a bigger number and gets zero.
RND and every noise-magnitude change couldn't help either, for the same reason
`grasp_goal_palm`'s earlier fix DID help: `grasp_goal_palm` is a function of PALM
position, which has gradient everywhere (any joint movement changes it), so the policy
could hill-climb it from a cold start. `lift_cont_rew` never had that property once the
hand stalled 1-3cm short. The ONLY way to move the cube at all without contact is to
physically bump it -- which is exactly what Round 6's escalated `lift_attempt_velocity`
reward eventually incentivized (large uncontrolled thrashing), because it was the sole
reachable non-zero-gradient path once the reward was cranked hard enough to matter.

**Diagnosing WHY contact never happens: `grasp_goal_hand` was the clearest smoking
gun.** It sat at `Episode_Reward/grasp_goal_hand` ~= 0.05 out of a possible 1.0
(weight=1.0) essentially the entire time Phase 4 was attempted. Working backward through
its own formula (`gate * (1 - tanh(q_err / q_std))`, `q_std=0.5`, gate open at ~0.8 given
the palm is well-positioned): raw term ~0.0625 -> `tanh(q_err/0.5) ~= 0.9375` ->
`q_err ~= 0.86 rad`. The six target hand joints sit roughly 0.86 radians away from grasp
#12's target configuration -- the palm is placed correctly (`grasp_goal_palm` converges
fine), the fingers are simply not curling the rest of the way. Mechanically: at
`q_std=0.5`, an error of 0.86rad already sits in the tanh's near-saturated tail (weak
gradient there), while `action_smoothness` (-3.0) and `joint_speed` (-1.0) charge a real,
unsaturated cost for the joint motion needed to close further -- the policy settled
exactly where marginal reward from closing more equals marginal movement cost, a stable
"standoff claw" equilibrium.

**The fix -- the user's own Phase 1 precedent, applied one level deeper.** The user
independently recalled how Phase 1's own analogous problem was solved: Phase 1 originally
converged to a "fingers curl toward SOME random local optimum near the cube" equilibrium
(not enveloping it), and the fix was to activate an EXISTING, already-implemented
mechanism (`grasp_goal_palm`, position-only) rather than invent a new reward or redesign
anything -- once the palm had a specific target to reach, curling naturally became
purposeful. `grasp_goal_hand` is the exact same kind of already-implemented, currently
under-weighted mechanism for the FINGERS specifically. Applied the identical pattern:
- `grasp_goal_hand` weight: `1.0 -> 4.0` (comparable to `grasp_weight=3.0`, so finger
  closure competes properly against the smoothness/speed penalties instead of losing to
  them).
- `q_std`: `0.5 -> 0.3` (sharpens the gradient specifically in the currently-plateaued
  region -- the measured 0.86rad error sits in the near-saturated tail at std=0.5, but on
  the steep part of the curve at std=0.3, giving real pressure to keep closing rather than
  settling early).

Single, targeted, attributable change to one term -- not a multi-formula redesign,
consistent with this project's own established discipline and the specific precedent the
user pointed to.

**Also reverted for this run**: `task_reward.lift_weight`/`success_weight` back to `0.0`
(Phase 4 off) -- this is explicitly a Phase 2+3 REDO, not a Phase 4 attempt, and Phase 4
shouldn't be touched again until contact is genuinely established and verified via
`contact_check.py`.

**Sanity check** (3 iterations, 64 envs, resumed from Phase 1 checkpoint 250 --
`cuda:0`): `grasp_goal_hand` weight correctly reads `4.0` in the printed reward-term
table, no NaNs, clean resume.

**Launch**: `2026-08-23_23-11-02_phase23v2_handclose_from_p1ckpt250`, 2048 envs, 500
iterations, resumed from Phase 1's checkpoint 250 (the original, validated Phase 2+3
handoff point -- NOT checkpoint 450 of the old `phase2and3_merged_from_ckpt200` run,
since that entire lineage is what produced the never-touching policy in the first place).
Confirmed running at full scale (12s/iteration).

**What to check once this has real data**: run `contact_check.py` on the resulting
checkpoints before touching Phase 4 again. Success criterion, objective and already
tooled: surface distance at or below zero (genuine geometric contact) for the thumb plus
at least two other fingers, across a meaningful fraction of environments -- not just
`grasp_goal_hand`'s reward value looking better in isolation.

**Result -- checkpoint selection, full batch review.** Ran the full inference/recording/
plot pipeline across all 11 checkpoints (250-749) plus `contact_check.py` (64 envs, 300
steps) across 300/400/500/600/700/749, alongside the existing 450 measurement:

| Ckpt | thumb mean (cm) | index | middle | ring | pinky | thumb envs <1cm |
|---|---|---|---|---|---|---|
| 300 | 1.63 | 3.42 | 3.34 | 3.72 | 4.46 | 6/64 |
| 400 | 0.95 | 2.84 | 2.41 | 2.38 | 2.93 | 34/64 |
| 450 | 0.92 | 2.80 | 2.36 | 2.30 | 2.80 | 32/64 |
| 500 | 0.81 | 2.75 | 2.25 | 2.18 | 2.64 | 43/64 |
| **600** | **0.72** | 2.82 | 2.21 | 2.12 | 2.70 | **46/64** |
| 700 | 0.78 | 2.73 | 2.14 | 2.11 | 2.72 | 41/64 |
| 749 | 0.81 | 2.57 | 2.02 | 2.06 | 2.69 | 40/64 |

**Checkpoint 600 is the objective peak** on the metric that actually matters (real
geometric proximity, not aggregate reward) -- best thumb distance and most envs with
thumb inside the 1cm buffer. Past 600, thumb performance regresses slightly even though
`grasp_goal_hand`'s own reward keeps climbing all the way to 749 -- whatever's still
improving after 600 isn't fingertip proximity specifically. Honest caveat: index/ring/
pinky are essentially flat from ~400 onward at every checkpoint (2.0-2.8cm), never
achieving genuine contact at any checkpoint -- this run produced real, substantial
progress (thumb converged from never-improving ~1.3cm to a stable ~0.7cm, and the
`hand_cube_penetration` check showed ~92% of steps with real hand-cube geometric contact
via the palm/knuckles, sustained across full episodes -- both unprecedented), but it is
not a fully-solved five-finger grasp. Checkpoint 600 chosen for the Phase 4 handoff.

---

### [Phase 4 v5] Un-escalated original design, from the hand-closing fix -- 2026-08-24

**The test this run is actually asking**: does the ORIGINAL, never-modified Phase 4
reward design (ungated `lift_cont_rew`, plain `lift_scale=2.0`, no
`lift_attempt_velocity`/RND/riser -- none of Rounds 2-6's escalations) work on its own,
now that the actual prerequisite -- genuine finger-cube contact -- is finally met? Every
earlier Phase 4 attempt was diagnosed after the fact as having an exactly-zero reward
gradient because no finger ever touched the cube; this is the first attempt where that
precondition is no longer violated (see "[Phase 2+3 v2]" above for the full diagnosis and
checkpoint comparison).

**Config**: `task_reward.lift_weight`/`success_weight` `0.0 -> 1.0`. `lift_scale` left at
`2.0` (the original default, not re-escalated). `lift_attempt_velocity` weight left at
`0.0` (off). RND (`rnd_cfg`) left at `None` (off). No riser, no scene changes. This is
deliberately the SAME reward configuration every prior Phase 4 round tried and failed
with -- the only thing that's changed is the checkpoint it starts from.

**Sanity check** (3 iterations, 64 envs, resumed from checkpoint 600 -- `cuda:0`): clean,
no NaNs.

**Launch**: `2026-08-24_04-42-40_phase4v5_from_p23v2ckpt600`, 2048 envs, 500 iterations,
resumed from checkpoint 600 of `phase23v2_handclose_from_p1ckpt250`. Confirmed running at
full scale (13s/iteration).

**What to watch for**: `Episode_Termination/target_lifted` firing at all (it never fired
more than a single-env blip in any prior round) would be the clearest signal that real
contact is translating into real lift capability. Also watch whether `Policy/
mean_noise_std` behaves differently than the flat/frozen pattern seen in Rounds 2 and 3 --
if the policy now has something to actually push against, exploration should look
qualitatively different from before.

**Result**: `Train/mean_reward`/`Loss/surrogate`/`Policy/mean_noise_std` all froze from
iter ~660 onward -- nothing was being learned. `hand_state_probe.py` on `model_1099`:
best cube height ever seen across 15,360 sampled steps was 1.83cm (mean 0.03mm),
`target_lifted` never fired. Root cause traced to `grasp_goal_hand`, not to lift/success
-- see "[Phase 2+3 v3]" below.

---

### [Phase 2+3 v3] q_std retune attempt #2 -- a regression, not a fix -- 2026-08-24

**Diagnosis of why v5 learned nothing**: `hand_state_probe.py` (new diagnostic --
measures `grasp_goal_hand`'s live gate/q_err/per-joint state and where that error sits
on the reward's own tanh curve) on `model_1099`: q_err=0.898rad, and at `q_std=0.3,
weight=4.0` that error sits at u=2.99 -- 99.5% into the tanh's saturated tail. Measured
`d(rew)/d(q_err)=0.109`, actually WEAKER than the pre-"[Phase 2+3 v2]" config's
`q_std=0.5, weight=1.0` (`d(rew)/d(q_err)=0.171`) despite the 4x weight increase. The
"[Phase 2+3 v2]" fix tightened `q_std` to sharpen the gradient but sized it to the error
we wanted, not the error we had -- exactly backwards.

**Fix attempted**: `q_std` 0.3 -> 1.0 (weight kept at 4.0), putting the 0.90rad error
near the tanh's steepest region (measured gradient 1.60, a 14.7x improvement on paper).
Sanity check (3 iter, resumed from `model_1099`) showed `grasp_goal_hand` jump from
~0.03 to 0.13-0.26 within 2 iterations -- looked immediately promising.

**Launched wrong**: first launch mistakenly resumed from checkpoint 1099 (Phase 4 v5's
own dead-end endpoint) instead of checkpoint 250 (the actual Phase 1->2+3 handoff) --
caught by the user, killed, deleted, relaunched correctly from `model_250.pt` of
`phase1_reach_palmpos_cuberot45`. Run: `2026-08-24_09-27-23_phase23v3_qstd1_from_p1ckpt250`,
2048 envs, 500 iterations (250->749).

**Result -- the fix regressed real contact.** `Episode_Reward/grasp_goal_hand` climbed
to 1.34 by the end (vs never breaking 0.03 in "[Phase 2+3 v2]") -- looked like a clear
win on the reward curve. `contact_check.py` across all 11 checkpoints said the opposite:

| ckpt | thumb mean | index | middle | ring | pinky |
|---|---|---|---|---|---|
| 250 | 2.16cm | 4.61 | 4.65 | 4.91 | 5.24 |
| 300 | 2.56 | 4.64 | 4.61 | 4.86 | 5.30 |
| 400 | 3.56 | 4.96 | 4.59 | 4.71 | 5.25 |
| 500 | 4.05 | 5.40 | 5.00 | 5.07 | 5.50 |
| 600 | 3.96 | 5.90 | 6.00 | 6.04 | 6.29 |
| 700 | 4.19 | 6.05 | 6.51 | 6.54 | 6.74 |
| 749 | 4.07 | 5.73 | 6.35 | 6.40 | 6.80 |

Every finger got monotonically WORSE over training, ending roughly 2x farther from the
cube than "[Phase 2+3 v2]"'s best checkpoint (thumb 0.72cm at its ckpt 600). `hand_state_probe.py`
on `model_749` confirmed it independently: palm gate 0.548 (vs 0.817 for phase23v2's
continuation), q_err 1.095rad (vs 0.898) -- worse by the term's own internal metric too,
despite the higher logged episodic reward.

**Why**: `grasp_goal_hand`'s `gate_dist=0.20` is loose (20cm), much looser than
`grasp_goal_palm`'s `pos_std=0.15`. With the gradient now 14.7x stronger, the policy could
extract most of the now-large `grasp_goal_hand` reward by matching the target JOINT ANGLES
while merely being *somewhere* within the wide 20cm gate ball -- trading precise palm
alignment (the thing that actually determines whether curled fingers land on the real cube)
for joint-angle conformity. This is a real overshoot from the "[Phase 2+3 v2]" fix, not a
math error in isolation -- the theoretical gradient argument was correct, but a much
stronger term dominating a comparatively weaker, tighter positional term created a new
exploit. Lesson: strengthening a partial/proxy reward's gradient can make it dominate
and get gamed, even when the strengthening is individually well-reasoned.

**q_std reverted to 1.0 kept in the codebase for the record but this run's checkpoints are
not used going forward.** No further q_std retuning attempted -- see "[Control]" below for
the direction taken instead.

---

### [Control] Testing task_reward's native grasp mechanism without pose-imitation -- 2026-08-24

**New context surfaced**: `CONTEXT.md` section 12 (dated 2026-07-29, predates this
session's whole Phase 1-4 curriculum) documents an experiment on this same task where
`Episode_Termination/target_lifted` reached **94%** (and a follow-up with
`pose_gated_success` on reached 93%) using ONLY `task_reward`'s own native
`grasp_rew`/`is_grasped`/`lift_cont_rew`/`success_bonus` mechanism -- `grasp_goal_palm`
and `grasp_goal_hand` sat completely flat (~0.029 / ~0.001) for 6000+ combined iterations
across two runs. Documented diagnosis: the terminal 1000-point `success_bonus` dwarfs
additive pose-imitation shaping, and the two objectives actively compete (risking a
working grasp to chase pose-matching is a bad trade), so the policy correctly ignores
pose-imitation and optimizes its own grasp. There is no reward-math dependency forcing
pose-imitation to succeed before lift can -- this whole session's working assumption
("fix grasp_goal_hand first, lift second") was never actually required by the reward
structure. A "control experiment" to verify this on the current repo/task was proposed
in that document but never run.

**Caveats carried forward, not dismissed**: (1) that 94% result predates this session's
tighter pairwise-coverage `is_grasped` (2026-08-20), so it may not reproduce as cleanly
today. (2) The same document reports real, measured hand-cube interpenetration (up to
-2.79cm) in a related checkpoint -- a high raw success rate can come from fingers
clipping through the cube rather than a genuine grasp, which matters for this project's
explicit hardware-transfer goal. `hand_cube_penetration` (via `play_with_goal_markers.py`)
needs checking alongside `target_lifted`, not treated as a secondary detail.

**Config**: resumed from Phase 1 checkpoint 250 (same handoff point, keeps the validated
reach/approach behavior). `task_reward.lift_weight`/`success_weight` `0.0 -> 1.0`
(grasp_weight=3.0 and pose_gated_success=True unchanged -- both are task_reward's own
machinery, not UltraDex-specific). `grasp_goal_hand` and `grasp_reach` weights `-> 0.0`
(both depend on the UltraDex per-grasp library, the thing being isolated out).
`grasp_goal_palm` and `grasp_envelope` left ON -- neither is implicated in either
diagnosis above, and grasp_goal_palm is what fixed Phase 1's approach-angle problem
originally.

**Sanity check** (3 iterations, 64 envs, resumed from checkpoint 250 -- `cuda:0`): clean,
`grasp_goal_hand`/`grasp_reach` confirmed at exactly 0.0, no NaNs.

**Launch**: `control_native_grasp_from_p1ckpt250`, 2048 envs, 500 iterations, resumed
from checkpoint 250 of `phase1_reach_palmpos_cuberot45`.

**What decides this experiment**: `Episode_Termination/target_lifted` actually rising
above the ~0.6% noise floor phase23v2/v5 showed would be the clearest positive signal.
If it does, cross-check `hand_cube_penetration` before calling it a real grasp -- per the
caveat above, a rising success rate with heavy interpenetration would mean the mechanism
works but produces a hardware-unusable grasp, a different problem than the one this
session has been solving.

**Result**: did NOT reproduce the historical 94%. `target_lifted` stayed at 0.5-2% noise
for the whole run with no trend (peaked at iter 266, drifted down to 0.4% by 749).
Approach quality did improve cleanly (`grasp_goal_palm` 0.03 -> 1.04), so the arm learns
to position itself -- it just never lifts. Whatever produced the 2026-07 result does not
survive in the current repo/task.

**Follow-up -- the IK oracle test, and why its result was discarded.** Ran
`collect_demos.py` (a scripted oracle: differential IK drives the palm to the goal grasp,
fingers close through the exact BODex angles, arm lifts -- no policy, no reward involved)
as a reachability probe, the test `CONTEXT.md` section 9 proposed and never ran. Result:
**0 successes in 512 episodes**, with `diagnose_oracle.py` showing the palm never getting
closer than ~10cm to the goal pose.

**That result is NOT evidence about reachability and was discarded.** The oracle clamps
its arm actions to [-1,1] with `ARM_SCALE=0.3` and `use_default_offset=True`, i.e. it can
only move each arm joint +/-0.3 rad (~17 degrees) from the default posture -- a crippling
restriction baked into the script. The trained policy has no such clamp (`clip_actions` is
unset). So the oracle was hobbled, not the robot. Per the user, video review clearly shows
the arm reaching the grasp pose with good finger placement around the cube. **Reachability
of grasp #12 is settled as a non-issue; do not re-litigate it from this script's output
without first removing that clamp.**

---

### [Phase 2+3 v4] Gated per-finger contact reward -- 2026-08-25

**The diagnosis this acts on.** Three independent measurements agree that the problem is
the three non-thumb fingers simply never closing:

- `mimic_track_probe.py` (new -- measures joints during a real rollout with
  `InspireMimicAction` active): middle/ring/pinky proximal joints sit at **~0% of their
  travel**, with the policy actively commanding them BELOW their lower limit, while the
  thumb uses 79-88% of its range and index ~18%.
- `measure_coupling.py`: those same joints **track position commands fine up to 1.0 rad**.
  So they are mechanically capable -- this is a reward problem, not a hardware one.
- The coupled intermediate joints sitting at ~1.15 rad regardless of command is **known and
  already compensated for**, not a new bug: `grasp_sampler/README.md` row 8 documents the
  slave joints being effectively inert and the synthesis URDF being frozen at exactly those
  empirical postures ("fingers ~=1.15 rad"). Measured 0.89/1.14/1.15/1.22 -- matches.

**Measurement caveat now on the record**: `contact_check.py` reads each finger link's BODY
ORIGIN, which sits at the knuckle joint, not at the fingertip pad ~2-3cm further along. Its
absolute centimetre values are therefore inflated by a roughly constant offset. Trends
across checkpoints remain valid (the offset is identical across them); absolute "is it
touching" claims from it are not. This is why checkpoint scoring below uses a SUM of
distances (offset-robust ranking) rather than an absolute contact threshold.

**Also on the record, from the user**: the FSWO/UltraDexGrasp optimizer chooses contact
points by PROJECTING each fingertip onto the cube surface and solving force closure, so a
valid grasp does **not** require every finger to physically touch -- some target points sit
at a standoff by construction. "All five fingers touching" is therefore the wrong success
criterion and has been dropped. The goal is a stable, consistent, repeatable grasp for
sim-to-real, which is the whole reason the grasp is prescribed rather than left to the
policy (its self-discovered grasps were inconsistent between episodes).

**Config**:
- NEW `finger_contact` (`mdp.finger_contact_reward`), weight **3.0**: proximity-gated pull
  of middle/ring/pinky toward their own cube-surface contact points. `std=0.02`,
  `gate_dist=0.12`. Structurally immune to the "[Phase 2+3 v3]" hack because it scores
  against points ON THE CUBE, so correct hand shape in the wrong place pays nothing.
  `gate_dist` was set to 0.08 first, but the sanity run showed the term reading exactly
  0.0000 -- the policy operates ~8cm from the goal, where a 0.08 gate opens to only ~0.24.
  Loosened to 0.12 (opens to ~0.55 there). The gate's only job here is to stop premature
  curling during approach; it does not need to be the anti-hack mechanism.
- `grasp_goal_hand` **reverted to its original weight 1.0 / q_std 0.5**. Both retunes
  failed in opposite directions (0.3 = saturated gradient, 1.0 = dominant and hacked), so
  it goes back to the known-neutral baseline and keeps only its shape-holding job.
- `grasp_reach` restored to weight 1.0 (all five fingers, keeps the overall grasp aligned).
- lift/success remain 0.0 -- Phase 4 turns them on.

**Launch**: `2026-08-25_07-05-11_phase23v4_fingercontact_from_p1ckpt250`, 2048 envs, 500
iterations, resumed from Phase 1 checkpoint 250.

**Result: FAILED, and Phase 4 was deliberately NOT launched off it.** Full packet kept at
`grasp_sampler/phase4_decision_packet.v4.txt`.

| ckpt | thumb | index | middle | ring | pinky |
|---|---|---|---|---|---|
| 300 | **2.58** | 4.64 | 4.61 | 4.83 | 5.26 |
| 450 | 4.16 | 5.63 | 4.80 | 4.50 | 4.57 |
| 550 | 4.34 | 4.92 | **4.33** | **4.53** | **4.20** |
| 749 | **4.44** | 5.08 | 4.40 | 5.21 | 4.87 |

Two separate failures, both traceable to specific choices made when configuring v4:

**1. `finger_contact`'s `std` was sized wrong -- the same mistake this log already
documents for `grasp_goal_hand`'s `q_std`, made a second time.** `std=0.02` was chosen for
the error we WANT (1-2cm); the measured error is ~4.5cm. That puts the term at u=2.25, deep
in tanh's flat tail: value 0.022 of a possible 1.0, gradient `(1/std)*sech^2(u)` = 2.16/m.
Consequence: the term's reward rose **340x** over the run (2.2e-05 -> 0.0074) and still
finished at **0.25% of its maximum**, and middle/ring/pinky ended flat (4.61/4.83/5.26 ->
4.40/5.21/4.87, with a transient best at ckpt 550). A reward rising 340x while achieving
nothing is what a saturated term looks like -- the multiplier is large, the *slope* is not.

**2. The thumb -- the one finger that was actually close -- regressed badly**, 2.58cm ->
4.44cm, with `thumb_yaw |dq|` going 0.18 (under v2/v5's weight 4.0) -> **1.02**. Cause:
reverting `grasp_goal_hand` to weight 1.0 removed the only strong term holding the thumb at
goal, and `grasp_reach` at weight 1.0 could not take over. Corroborated by `hand_state_probe`:
palm gate fell 0.82 -> 0.56, `q_err` rose 0.90 -> 1.27, and steps with the cube >1cm up went
22 -> 5 -> 1 across ckpts 450/600/749.

**Why Phase 4 was held back**: its precondition -- real fingertip contact, the user's own
stated bar ("try to have each and every finger contact the cube at the end of phase 2+3") --
was not met at any checkpoint (0/64 envs touching, every finger, every checkpoint). Launching
Phase 4 into a no-contact state reproduces the exactly-zero lift gradient that killed rounds
1-6. Worse, Phase 4's `lift_scale=60` on an **ungated** `lift_cont_rew` would pay ~1.2/step
for a cube merely knocked to 2cm -- with no grasp to speak of, the likely outcome is a
learned cube-flick, which is Round 6's thrashing failure with a bigger multiplier. The
orchestrator was stopped before its 60-minute blind fallback could fire; the config was
verified still unswapped (all six `PHASE4_SWAP_*` markers at Phase 2+3 values).

---

### [Phase 2+3 v5] Correct the std sizing and restore the thumb -- 2026-08-25

Two changes, each targeting one of the two v4 failures:

- **`finger_contact` `std` 0.02 -> 0.055.** Sized to the error we HAVE (~4.5cm), putting it
  at u=0.82 -- essentially tanh's steepest point (peak gradient is at u~0.77). Gradient
  9.96/m vs 2.16/m, **4.6x stronger**, and it self-sharpens as distance closes so it should
  not need retuning again.
- **`grasp_reach` weight 1.0 -> 3.0.** This is the thumb fix. Chosen over re-arming
  `grasp_goal_hand` deliberately: `grasp_reach` scores against points ON THE CUBE, so unlike
  the joint-angle term it cannot be satisfied by the right hand shape in the wrong place --
  the exploit that wrecked "[Phase 2+3 v3]". Its `std=0.05` already matches the measured
  ~4.5cm error, so its gradient was already near-optimal; only the weight was too low.
  `grasp_goal_hand` stays at its neutral 1.0 / q_std 0.5.

**Sanity check**: `finger_contact` read **0.0056 after 3 iterations** -- versus 0.0074 after
the entire 500-iteration v4 run. The gradient fix is confirmed working before committing.

**Launch**: `2026-08-25_09-45-10_phase23v5_stdfix_from_p1ckpt250`, 2048 envs, 500 iterations,
from Phase 1 checkpoint 250.

**Result: the gradient fix WORKED, and it made no difference to physical contact.** This is
the most informative negative result of the project so far, because it eliminates reward
shaping as the remaining explanation.

The term genuinely engaged this time -- `finger_contact` rose **276x** (0.0019 -> 0.5163),
reaching **17% of its maximum** versus v4's 0.25%. `grasp_reach` rose 92x. `grasp_envelope`
0.016 -> 0.891. `Train/mean_reward` 0.62 -> 27.3. And yet **every finger ended FURTHER from
the cube**:

| ckpt | thumb | index | middle | ring | pinky |
|---|---|---|---|---|---|
| 300 | **2.60** | 4.78 | 4.78 | 5.00 | 5.40 |
| 500 | 4.16 | 5.62 | 4.92 | 4.83 | 4.26 |
| 749 | **4.92** | 5.79 | 5.50 | 5.85 | 5.63 |

Decomposing the apparent contradiction (both rewards are literally fingertip-to-cube-point
distances, so this looked arithmetically impossible): inverting `grasp_reach`'s own tanh
shows fingertip-to-TARGET went 17.1cm -> 10.5cm (ckpt 300) -> **5.65cm** (ckpt 749). Against
targets that sit 2.57-3.33cm from cube centre, the triangle inequality says the fingertips
started nearly diametrically OPPOSITE their targets and rotated round to the correct side
while staying far away. So the reward was doing exactly its job -- orienting the hand
correctly -- and simply ran out of room. Not a hack, not a mis-sizing. It just cannot
finish.

**The real bottleneck, measured**: the palm stalls **8.53cm from the goal position** (mean;
best 6.74cm) and stops improving. Fingers extend ~8-10cm from the palm; a hand held 8.5cm
off its grasp pose physically cannot put fingertips on the cube, no matter what the finger
rewards pay. Every reward experiment in v2-v5 was downstream of a palm that never arrives.

**Ruled out -- the action space is NOT the limit** (`action_mag_probe.py`, new). The policy
is already straining: per-joint mean |action| `[1.48, 0.69, 2.03, 1.59, 1.79, 0.52, 1.15]`,
**67% of actions exceed 1.0**, 23% exceed 2.0, max 4.65. It is not clipped and not shy. But
the resulting joint offsets `[0.40, 0.21, 0.61, 0.48, 0.53, 0.16, 0.35]` rad fall far short
of the `[0.36, 1.38, 1.21, 1.04, 1.22, 0.03, 1.17]` rad an unclamped IK oracle needs -- short
by 2-6x on five of seven joints. Raising `scale` was considered and rejected on this
evidence: the policy would have to sustain |action| ~3.0 where it currently sustains 1.3,
and nothing is stopping it from doing so already.

**The deeper finding, and the one that should be resolved before ANY further RL** (from
`diagnose_oracle_unclamped.py` -- the corrected, unclamped rerun of the oracle whose earlier
clamped result was correctly discarded): a scripted differential-IK controller, driving the
palm straight to the goal and closing the fingers through the exact BODex pregrasp -> grasp
-> squeeze angles, gets the palm to **3.04cm** -- far better than the policy's 8.53cm -- and
**still does not move the cube at all** (84.50cm -> 84.51cm, i.e. 0.1mm, across every
episode). Note it also does not reach 0: even perfect IK stalls ~3cm short.

That reframes the problem. If near-perfect scripted execution of this grasp does not disturb
the cube, then the blocker is not exploration, not reward shaping, and not the action space
-- it is that **this grasp configuration does not close on the cube in simulation**. No
reward curriculum can fix that. The decisive test remains the static one: force the hand to
`goal_pos_w` / `goal_quat_w` / `goal_hand_q` exactly and measure fingertip-to-cube geometry
directly, with no controller in the loop.

**Phase 4 was again NOT launched**, and the orchestrator was stopped before its blind
fallback could fire. Config verified still unswapped (all six `PHASE4_SWAP_*` markers at
Phase 2+3 values). Launching lift rewards onto a hand that has never touched the cube would
reproduce rounds 1-6 for the seventh time, and `lift_scale=60` on an ungated
`lift_cont_rew` would additionally invite a learned cube-flick.

**Status**: complete. Superseded -- see the correction immediately below.

**CORRECTION (2026-08-25), important**: the "palm stalls at 8.53cm" conclusion above was
measured on v5, which was ITSELF a regression introduced in v3/v4/v5. Re-measuring the
earlier `[Phase 2+3 v2]` run gives a completely different picture:

| checkpoint | thumb | index | middle | ring | pinky | palm->goal |
|---|---|---|---|---|---|---|
| **phase23v2 ckpt 600** | **0.72cm** (46/64 in 1cm buffer) | 2.82 | 2.21 | 2.12 | 2.70 | **3.07cm** (min 0.66) |
| phase23v2 ckpt 500 | 0.81 (43/64) | 2.75 | 2.25 | 2.18 | 2.64 | -- |
| phase23v2 ckpt 700 | 0.78 (41/64) | 2.73 | 2.14 | 2.11 | 2.72 | -- |
| phase23v5 ckpt 300 (best v5) | 2.60 | 4.78 | 4.78 | 5.00 | 5.40 | ~8.5cm |

**phase23v2 ckpt 600 achieves palm 3.07cm -- matching the unclamped IK oracle's 3.04cm.**
The palm is not structurally stuck; the best policy places it as well as a scripted IK
controller does. Every reward change made after v2 (q_std retunes, finger_contact,
grasp_reach reweighting) was a net regression against it. **phase23v2 ckpt 600 is the best
state this project has reached and is the correct Phase 4 handoff.**

---

### [Phase 4 v6] Grasp-free lift, pose mimicry removed entirely -- 2026-08-25

**The plan (user's).** Learn the PICK first, with whatever grasp the policy finds on its
own, then re-introduce pose mimicry as a SECOND stage to converge that grasp onto the
UltraDex pose. This is the sequence that demonstrably worked for the user previously: the
policy found its own grasp, and a pose-error term afterwards made it consistent. The reason
a prescribed grasp is wanted at all is sim-to-real *consistency* -- self-discovered grasps
varied between episodes -- and that consistency can be imposed after the pick exists.

**Why the evidence supports it.** `CONTEXT.md` section 6 records **94% target_lifted** on
this task while `grasp_goal_palm`/`grasp_goal_hand` sat flat at 0.029/0.001 for the entire
run -- the successful run was already effectively pose-free. The 2026-08-24 `[Control]` run
looks like a counterexample but is not: it kept `grasp_goal_palm` ON at weight 2.0, so the
policy was still chasing a prescribed pose it demonstrably cannot fully reach.

**Config** -- all four UltraDex pose terms to 0.0 (`grasp_goal_palm` 2.0->0,
`grasp_goal_hand` 1.0->0, `grasp_reach` 3.0->0, `finger_contact` 3.0->0), plus the six
Phase 4 swaps (`lift_weight`/`success_weight` 0->1, `lift_scale` 2->60, `_SUCCESS_Z`
1.134->0.865, `joint_speed` -1.0->-0.3, `target_object_accel` -5.0->-2.0). Retained:
`task_reward`'s own `reach_weight` 1.0 / `grasp_weight` 3.0 (pairwise coverage, library-
independent) and `grasp_envelope` 2.5 (whole-hand coverage, also library-independent) --
these still shape a real grasp without prescribing WHICH grasp.

**`pose_gated_success` True -> False, and this one is load-bearing.** With the pose terms
off, leaving it True would have silently defeated the whole phase: `success_scale =
success_floor + (1-floor)*pose_match` scores `pose_match` against the very goal pose we are
no longer pursuing, so the 1000-point bonus would stay pinned near its 0.2 floor -- the
policy would be penalised for not matching a pose nothing else asks it to match.

**Why the flick exploit is not a serious risk here**, despite `lift_scale=60` on an ungated
`lift_cont_rew`. At ckpt 600, `is_grasped ~= 0.30` (thumb 0.72cm, ring 2.12cm). With
`pose_gated_success=False` and success at 2cm, a cube merely flicked to 2cm pays
`lift_cont_rew` alone ~1.2/step, while a genuine grasp-and-lift pays
`is_grasped * 1000` ~= **300/step** -- 250x more, and the bonus is grasp-gated. The
incentive strongly favours a real grasp. Watch `Episode_Termination/target_lifted` against
`Episode_Reward/lift_progress`: lift_progress rising while target_lifted stays flat would
be the flick signature.

**Launch**: `phase4v6_graspfree_from_p23v2ckpt600`, 2048 envs, 500 iterations, from
`phase23v2` checkpoint 600. Backup of the pre-swap config at `g1_pick_env_cfg.py.phase23.bak`.

**Result: stopped early by the user at iteration ~890/1100.** `target_lifted` did move --
0.0308 at the first iteration to **0.0394** at stop, the highest sustained value of any run
so far, and `task_reward` reached 0.852 with `grasp_envelope` 0.671. But ~4% after 90
minutes was judged too slow to be worth continuing. Notably `Policy/mean_noise_std` had
already decayed to 0.23 -- a warm-started policy has little exploration left to spend.

---

### [Scratch v1] From-scratch retrain with lift active from iteration 0 -- 2026-08-25

**The idea (user's, explicitly a last-ditch attempt).** Every attempt so far has bolted the
lift reward onto an already-converged policy whose exploration noise has collapsed
(`mean_noise_std` 0.23-0.28). Instead, retrain **from random init** with the pick reward
present from the very first iteration, so the policy spends its entire high-noise phase in
an MDP that already pays for lifting -- giving accidental lifts a chance to be discovered
and reinforced while exploration is still wide.

**The premise checks out quantitatively**: `init_noise_std=0.8` versus the 0.23-0.28 every
warm-started run has been operating at -- roughly **3x more exploration**, sustained over
hundreds of iterations rather than absent from the start.

**Config = the EXACT Phase 1 reward set, plus lift/success.** Read from the phase1 run's own
`params/env.yaml` rather than reconstructed from memory, which caught two details that would
otherwise have been wrong:
- `grasp_goal_palm` `orient_weight` is **0.0** in Phase 1 (POSITION-ONLY). Full-pose
  orientation was a Phase 3 addition; the current file had drifted to 0.5.
- `grasp_weight` is **0.0** in Phase 1 (pairwise-coverage grasp_rew is a Phase 2 term), and
  `grasp_envelope` is **0.0**.

So: `reach_weight` 1.0, `grasp_weight` 0.0, `grasp_envelope` 0.0, `grasp_goal_palm` 2.0
(orient_weight 0.0), `grasp_goal_hand`/`grasp_reach`/`finger_contact` 0.0,
`action_smoothness` -3.0, `joint_speed` **-1.0**, `target_object_accel` **-5.0**,
`sustained_reach_bonus` 10.0, `pose_gated_success` False (already False in Phase 1).

`joint_speed` and `target_object_accel` were deliberately returned to their Phase 1 values
rather than kept at the Phase 4 tuning (-0.3 / -2.0), per "exactly the same plus lift".
Checked before doing so: `target_object_accel` measured only -0.0133 episodic at weight -2.0,
so even at -5.0 it costs ~1-2% of return, and it penalises JERK rather than sustained motion
-- a smooth lift barely pays it. `joint_speed` at -1.0 is the more material of the two
(~-3.4 of return); it is the first thing to relax if this run stalls.

Lift terms carried over from the Phase 4 tuning: `lift_weight`/`success_weight` 1.0,
`lift_scale` **60.0**, `lift_height` clamp 0.05, `_SUCCESS_Z` **0.865 (a 2cm lift)**. The
2cm threshold is what makes "stumble into it by accident" plausible at all -- the original
29cm was never once approached in any run.

**Launch**: `scratch_p1_withlift`, 2048 envs, **1000 iterations**, NO `--resume` (random
init). Longer than the usual 500 deliberately, since the whole point is to give exploration
time to find a lift.

**Next leg if this works**: repeat for Phase 2+3 (restore `grasp_weight` 3.0,
`grasp_envelope` 2.5, `grasp_goal_hand`, `grasp_reach`, and `grasp_goal_palm`'s
`orient_weight` 0.5) with lift still active throughout.

**IMPORTANT CALIBRATION, from this run's iteration 0**: at **random initialisation**, with
an untrained policy flailing at `mean_noise_std=0.79`, `Episode_Termination/target_lifted`
already reads **0.0303**. That is the random-chance baseline for knocking a cube 2cm off a
tray -- it represents no learning whatsoever.

This retroactively reinterprets every "weak positive signal" read off `target_lifted` in
this log. `[Phase 4 v6]`'s 0.0308 -> 0.0394 was **at, and then barely above, the noise
floor** -- essentially nothing was learned about lifting, rather than "slow progress". Same
for the 1-2% values in v3/v4/v5 and the `[Control]` run's 0.5-2%: all at or below what
random actions produce.

**The bar for this run, therefore, is `target_lifted` climbing clearly and persistently
above ~0.03**, not merely being non-zero. And it must be cross-checked against
`is_grasped`/contact, since the 2cm threshold is demonstrably reachable by pure disturbance:
`lift_progress` rising while contact stays flat would mean the policy learned to knock the
cube, not lift it.

**Result: stopped at iteration ~466. The high-noise exploration hypothesis was tested and
did not produce a lift.** `target_lifted` sat at the random-chance floor for the entire run,
straight through the high-noise window:

| iter | `mean_noise_std` | `target_lifted` |
|---|---|---|
| 0 | 0.79 | 0.030 |
| 100 | 0.54 | 0.037 |
| 200 | 0.34 | 0.029 |
| 450 | 0.245 | 0.036 |

The extra exploration was real and measurable (0.79 vs the 0.23-0.28 of every warm-started
run) -- it simply never found a lift. Inference recordings at ckpts 300/400 show why, and
the mechanism is unambiguous: **the fingers never move at all.** Per-joint finger errors are
identical to the decimal across every step of every episode
(`[*, 22.0, 21.9, 9.0, 16.4, 37.4]`, only thumb-yaw varying), all five fingertips sit on a
flat line 5.5-10.5cm from their contact targets, and **0/399 steps show any hand-cube
contact** at either checkpoint. `lift_progress` is flat zero.

The cause is structural, not tuning: **the Phase 1 reward set contains no grasp reward of
any kind** (`grasp_weight`, `grasp_envelope`, `grasp_goal_hand`, `grasp_reach`,
`finger_contact` all 0.0 -- correct for Phase 1, whose job was reaching). So "Phase 1 +
lift" produces a policy with nothing telling it to close its hand; the lift reward
multiplies zero contact, the same trap arriving by a new route. Exploration cannot help
because there is no grasp to discover a lift *with*.

Genuine positive from this run: **palm->goal reached 0.6-0.7cm** ("REACHED"), the best palm
placement of any run to date (vs 3.07cm for phase23v2 ckpt600), with `grasp_goal_palm`
saturated at 1.94/2.0. Phase 1's own objective was solved cleanly -- easier here because
Phase 1 uses `orient_weight=0.0`, i.e. position-only.

---

### [Scratch v2] Same config on the EMPTY environment -- 2026-08-25

User stopped v1 on seeing tray clutter in the recordings and asked for the same config on
the empty environment (`Isaac-G1-Pick-Empty-v0`: all 10 distractors relocated to the table
margins at y=+/-0.45, clear of the tray and approach path, but still in the scene and the
96-dim observation space so nothing else changes). v1's logs/TensorBoard deleted.

**This turned out to matter far more than the reward accounting suggested, and corrects an
assessment made when relaunching.** The distractor PENALTY was indeed negligible
(`distractor_accel` -0.0005 to -0.0019 episodic, ~0.1% of total reward; `distractor_dropped`
0.0005-0.007), which is what "clutter was not the bottleneck" was based on. But the clutter
was corrupting the *success metric itself*:

| environment | `target_lifted` at iteration 0 (random init) |
|---|---|
| cluttered (`Isaac-G1-Pick-v0`) | **0.0303** |
| empty (`Isaac-G1-Pick-Empty-v0`) | **0.0000** |

Identical reward config, identical random initialisation. **The ~3% "random-chance floor"
was not chance at all -- it was tray distractors being knocked into the target cube and
popping it over the 2cm `_SUCCESS_Z` threshold.** With the distractors moved off the tray,
nothing can disturb the cube except the hand, and an untrained policy scores exactly zero.

Consequences, and they are significant:
- Every `target_lifted` reading from a cluttered run is contaminated by distractor-mediated
  cube disturbance, including the 3-4% values in `[Phase 4 v6]` and the 1-2% in v3/v4/v5.
- On the empty env the metric is clean: **any sustained non-zero `target_lifted` now means
  the hand actually moved the cube.** No noise floor to clear.

**Launch**: `scratch_p1_withlift_empty`, `Isaac-G1-Pick-Empty-v0`, 2048 envs, 1000
iterations, random init, reward config byte-identical to v1 (verified via the printed reward
table).

**Caveat carried forward, unchanged**: the frozen-finger problem above is a property of the
Phase 1 reward set, not of the clutter, so this run is still expected to show a hand that
never closes. The recommended one-line fix -- run the same from-scratch, lift-from-step-0
experiment on the **Phase 2+3** reward set instead, so the fingers have a reason to close
during the high-noise window -- has been put to the user and is not yet actioned.

**Status**: stopped early by the user in favour of "[Scratch v3]" below, which keeps the
high-noise premise but pairs it with a reward set that can actually close the hand.

---

### [Scratch v3] Phase 2+3 from ckpt 250 with noise RESET to 0.8 -- 2026-08-25

**The synthesis (user's).** Resume Phase 2+3 from the canonical Phase 1 checkpoint 250 as
usual, but forcibly re-initialise the policy's exploration noise to 0.8 -- the from-scratch
value -- instead of inheriting the converged value. This keeps the high-noise exploration
premise from v1/v2 while fixing what made them futile: **Phase 2+3's reward set actually
contains grasp terms**, so the fingers have a reason to close during the high-noise window.
Lift/success stay active throughout, per the original plan.

**Why the noise reset is the crux, and what the checkpoint actually held**: `--resume` loads
the entire policy including the learned per-joint `std`, so every warm-started run so far
inherited a nearly-dead explorer. Checkpoint 250's `std`, per joint:

```
arm  [0.064, 0.088, 0.128, 0.067, 0.196, 0.218, 0.189]
hand [0.491, 0.416, 0.363, 0.368, 0.427, 0.397]
```

The **arm** joints are at 0.064-0.22 -- shoulder-pitch 0.064 and elbow 0.067 are essentially
frozen. That is a concrete, previously unmeasured explanation for why the palm converged to
a fixed offset and stopped improving in every warm-started run: the arm had almost no
exploration left to find anything better with.

**Patch** (`model_250_std08.pt`, written alongside the original -- the original is
untouched): all 13 `std` entries set to 0.8, **and the Adam `state` dict cleared**. The
second part is necessary, not cosmetic: Adam's `exp_avg` for the `std` parameter still
carries the accumulated downward gradient that drove std to 0.06-0.49 in the first place,
and would pull a naive reset straight back down within a few updates. `param_groups`
(learning rate etc.) are preserved; Adam re-initialises per-parameter state lazily.
Verified in a 3-iteration run: `Mean action noise std: 0.80`, holding across iterations.

**Config = the EXACT Phase 2+3 reward set** (read from
`2026-08-21_08-48-51_phase2and3_merged_from_ckpt200/params/env.yaml`, not reconstructed):
`grasp_weight` 3.0, `grasp_envelope` 2.5, `grasp_goal_palm` 2.0 with `orient_weight` **0.5**
(full pose -- the Phase 3 value, vs Phase 1's position-only 0.0), `grasp_goal_hand` 1.0
(q_std 0.5), `grasp_reach` 1.0 (std 0.05), `pose_gated_success` **True**,
`action_smoothness` -3.0, `joint_speed` -1.0, `target_object_accel` -5.0,
`sustained_reach_bonus` 10.0. `finger_contact` stays 0.0 (a later invention, not part of the
authentic Phase 2+3 set). Plus lift: `lift_weight`/`success_weight` 1.0, `lift_scale` 60.0,
`_SUCCESS_Z` 0.865.

`pose_gated_success` is left at its authentic Phase 2+3 value of True. It is coherent here
(unlike in "[Phase 4 v6]", where the pose terms were off): Phase 2+3 actively pursues the
UltraDex pose, and `success_floor=0.2` still pays 200 points for any genuinely grasped lift
regardless of pose quality, so the discovery signal is not suppressed.

**Environment**: `Isaac-G1-Pick-Empty-v0`, carried forward from v2 -- with distractors off
the tray, `target_lifted` has a clean zero baseline, so any sustained non-zero reading is a
real hand-driven lift rather than distractor-mediated knocking.

**Launch**: `2026-08-26_00-35-09_phase23v6_std08_from_p1ckpt250`, 2048 envs, 1000 iterations
(250 -> 1249), `Isaac-G1-Pick-Empty-v0`.

**Infrastructure note -- the first launch attempt crashed, and it was NOT our fault.**
`malloc(): invalid size (unsorted)`, core dumped during scene setup. Diagnosis: **host RAM
exhaustion**, not GPU and not anything in the config. `free -g` showed 31GB total / 0 free /
25GB swapped, because another user (`vincent`) had a distillation pipeline running in a
separate container since 11:07 holding 7836MiB of host GPU 0 plus ~2GB RSS. Our container
only sees the other two GPUs (both ~7.5GB free), so this was purely host memory contention.
That job was left alone -- it is someone else's work. A straight retry at the same 2048 envs
succeeded, so the failure was transient contention. Related: the 118 `<defunct>` processes
in our container are zombies from repeated `pkill -9`; they hold **0 kB RSS/VSZ**, cannot be
killed (already dead), and persist only because the container's PID 1 is `sleep infinity`,
which never reaps. They are harmless and were NOT a contributor to the crash -- the fix, if
ever wanted, is `docker run --init` at container-creation time.

**Progress during the run** (in-flight readings): `grasp_goal_hand` 0.0016 -> **0.3116**
(~200x) and `grasp_reach` 0.0014 -> **0.2791** (~200x), both far above anything the
frozen-finger Phase 1 runs produced -- the fingers are engaging this time.
`Policy/mean_noise_std` held at 0.80 as patched. `target_lifted` stayed ~0.001-0.002 against
the empty env's true-zero baseline.

**Status**: complete (checkpoints 250-1249). Evaluation and Phase 4 handoff delegated to
`grasp_sampler/run_phase4_after_v6.sh`.

---

### [Phase 4 v7] Automated handoff -- queued 2026-08-26

Handled by `grasp_sampler/run_phase4_after_v6.sh` in the `g1_orchestrator` tmux session
(survives the assistant's session ending). Per the user: pick the best checkpoint and start
Phase 4 -- **and if v6 did not really change anything versus previous Phase 2+3 sessions,
reset the noise std to 0.8 again before starting Phase 4**, so Phase 4 itself runs with live
exploration rather than a dead explorer.

**What "Phase 4" still means from here.** Lift has been active since "[Scratch v1]", so
`lift_weight`/`success_weight` 1.0, `lift_scale` 60.0 and `_SUCCESS_Z` 0.865 are ALREADY in
the config. The only remaining Phase 4 delta is relaxing the two penalties that were tuned
for a hold-still objective and now oppose lifting: `joint_speed` -1.0 -> **-0.3** and
`target_object_accel` -5.0 -> **-2.0**.

**The improved/nothing decision rule**, calibrated against measurements already in this log
rather than guessed: best checkpoint is the one with the lowest thumb distance (the finger
that has consistently led), and the verdict is `improved` iff **thumb <= 1.5cm AND the mean
of the other four <= 3.5cm**. Validated against real historical sweeps before arming:
phase23v2 ckpt600 (thumb 0.72, others 2.12-2.82) -> `improved`; phase23v5 ckpt300
(thumb 2.60, others 4.78-5.40) -> `nothing`. `improved` hands off normally; `nothing`
patches the chosen checkpoint's `std` to 0.8 (and clears Adam state, as in v3) first.

The orchestrator verifies both penalty swaps and the four already-set lift lines are present
and that the file parses, restoring a backup and refusing to launch otherwise. It also
retries once at 1024 envs if startup hits the `malloc` crash again, since that failure mode
is now known to be host-RAM contention with another user's job. A 45-minute window is left
for the assistant to override the auto-choice by writing `<ckpt> <std08|nostd>` to
`phase4_v6_decision.txt`.

**RESULT OF THE v6 SWEEP -- the first genuine, monotonic contact improvement of the whole
project.** Every finger improves at every checkpoint, which no previous run has ever done
(v3/v4/v5 all degraded monotonically in the wrong direction):

| ckpt | thumb | index | middle | ring | pinky | thumb in 1cm buffer |
|---|---|---|---|---|---|---|
| 400 | 2.08 | 2.11 | 2.32 | 2.56 | 3.24 | 7/64 |
| 600 | 1.59 | 1.90 | 2.24 | 2.21 | 2.25 | 18/64 |
| 800 | 1.01 | 1.77 | 2.04 | 2.01 | 1.29 | 39/64 |
| 1000 | 0.85 | 1.73 | 1.96 | 1.97 | 1.35 | 54/64 |
| **1100** | **0.79** | **1.71** | 1.96 | **1.94** | 1.34 | **57/64** |
| 1249 | 0.87 | 1.68 | 1.89 | 1.98 | 1.40 | 50/64 |

Against the previous best ever (`phase23v2` ckpt600: thumb 0.72, index 2.82, middle 2.21,
ring 2.12, pinky 2.70, 46/64):

- thumb essentially matched (0.79 vs 0.72), buffer count better (57/64 vs 46/64)
- **index 2.82 -> 1.71 and pinky 2.70 -> 1.34** -- the fingers that were the entire problem,
  stuck at 2-6cm in every prior lineage, have closed by more than a centimetre each
- the hand now closes *uniformly* rather than thumb-only

**What actually produced this**: the noise reset. Nothing else changed versus the authentic
Phase 2+3 reward set. Warm-starting had been silently inheriting a dead explorer, and
restoring `std=0.8` let the fingers find contact they had never been able to search for.

**Verdict**: `improved` (thumb 0.79 <= 1.5, other-four mean 1.74 <= 3.5), so per the user's
rule Phase 4 hands off WITHOUT another std reset. Checkpoint **1100** chosen -- best thumb
distance and best buffer count; 1249 is marginally better on index/middle but worse on thumb
and buffer, and the assistant confirmed the auto-pick rather than overriding it. The 45-min
override window was short-circuited by writing the decision immediately, to avoid idling the
GPU.

**IMPORTANT CAVEAT for Phase 4 -- the ARM explorer is dead again, even though the hand's is
not.** At ckpt 1100 the per-joint std has split sharply:

```
arm  [0.051, 0.037, 0.211, 0.073, 0.266, 0.261, 0.099]   (mean ~0.14)
hand [0.597, 0.653, 0.639, 0.663, 0.593, 0.588]          (mean ~0.62)
```

The hand kept exploring for the whole run -- which is exactly why the fingers closed -- but
the arm's noise re-collapsed to 0.037-0.099 on the shoulder/elbow joints. **Lifting is an ARM
motion.** So Phase 4 is being handed a policy that can still search with its fingers but
barely at all with its arm.

The argument for proceeding anyway (and why `nostd` is still right): with contact finally
real, `lift_cont_rew` should have a genuinely non-zero gradient with respect to arm motion
for the first time in the project. Gradient ascent does not need large noise -- only
*discovering a discontinuous behaviour* does. If Phase 4 stalls with `target_lifted` flat,
**the first thing to try is resetting the ARM std alone** (leaving the hand's intact, so the
hard-won finger closure is not disturbed) rather than any further reward surgery.

For reference, `target_lifted` stayed at 0.001-0.005 throughout v6 (lift rewards were active
the whole time), so no lifting has emerged yet -- unsurprising, since contact only became
good in the last few hundred iterations.

**Launched**: `phase4v7_from_v6ckpt1100_nostd`, from `model_1100.pt`, config verified
(`joint_speed` -0.3, `target_object_accel` -2.0, `lift_scale` 60.0, `_SUCCESS_Z` 0.865,
lift/success 1.0), `Isaac-G1-Pick-Empty-v0`, 2048 envs, 1000 iterations.

**Result: completed all 1000 iterations (1100 -> 2099). Grasp improved further; NO LIFT.**
`target_lifted` 0.0022, `lift_progress` 0.0002 (mean cube height above rest ~0.15mm), both
flat across the whole run. Contact at ckpt 2099 is nevertheless **the best of the project**:

```
thumb  0.50cm  (64/64 envs inside the 1cm buffer -- every single environment)
ring   1.17cm  (10/64)     pinky 1.49cm (4/64)
index  1.65cm              middle 1.80cm
```

The arm/hand std split predicted in the previous entry held exactly: arm ended at
`[0.079,0.041,0.125,0.072,0.183,0.132,0.070]` (mean 0.101, dead) while hand stayed at
mean 0.557.

**But the pre-registered "reset the arm std" remedy is WRONG, and a direct measurement
proved it before another 5 hours were spent on it.** New diagnostic
`grasp_sampler/lift_capability_probe.py` lets the policy settle into its grasp, then
OVERRIDES the arm action to drive the shoulder upward while leaving the policy's hand action
untouched, and watches the cube:

```
palm rose : +2.50 cm
cube rose : -0.01 cm
follow ratio (cube/palm): -0.005
VERDICT: NO GRIP
```

(First attempt used `lift_delta=+1.5` and the palm went DOWN 4.37cm -- shoulder_pitch's sign
is inverted for "up"; `-1.5` is the raising direction. Worth remembering.)

So the hand has near-perfect *proximity* and **exactly zero grip force**. This was never an
exploration problem, and no arm-std reset, lift weight, or success threshold could ever have
fixed it.

---

### [Squeeze v1] Reward grip FORCE, not just proximity -- 2026-08-26

**The structural error, and it runs through the entire project.** Every reward tried so far
-- `grasp_reach`, `finger_contact`, `grasp_envelope`, `grasp_rew`, `grasp_goal_hand` -- is
GEOMETRIC: fingertip position, or joint angle. The hand joints are **position-controlled**,
so commanding a finger to exactly where it already rests produces **no force**. A hand
hovering at zero distance scores identically to one squeezing hard -- in fact slightly
better, since squeezing jostles the cube and pays `target_object_accel`. Nothing anywhere
rewarded grip force. That is why proximity kept improving while lift stayed at zero: the
policy was optimising precisely what it was asked for.

**The fix uses infrastructure that already existed and was never wired in.** The grasp
library stores three stages per grasp -- pregrasp / grasp / **squeeze** -- and
`sample_grasp_goal` already loads all three (`goal_hand_q_pre`, `goal_hand_q`,
`goal_hand_q_squeeze`; only `collect_demos.py` ever used squeeze). `grasp_goal_hand` has
always targeted the *grasp* stage, which is a **touching** pose. The squeeze stage closes
each finger further:

| joint | grasp | squeeze | delta |
|---|---|---|---|
| th_yaw | 1.049 | 1.280 | +0.231 |
| th_pitch | 0.383 | 0.430 | +0.046 |
| index | 0.383 | 0.557 | +0.174 |
| middle | 0.157 | 0.314 | +0.157 |
| ring | 0.286 | 0.391 | +0.105 |
| pinky | 0.653 | 0.613 | -0.040 |

L2 |squeeze - grasp| = 0.351 rad. Commanding it drives the fingers PAST the cube surface;
PhysX stops them at the surface and the PD controller then generates real normal force --
which is what friction, and therefore lifting, requires.

**Config**: `grasp_goal_hand_config_reward` gains `use_squeeze` (default False, so nothing
else changes); set True here. Weight **1.0 -> 3.0**, and that is not arbitrary: moving the
target to squeeze raises q_err ~0.35 -> ~0.70 rad, which at `q_std=0.5` *drops* the tanh
gradient `(1/q_std)*sech^2(q_err/q_std)` from 1.264 to 0.434 -- ~3x weaker exactly when it
needs to be stronger. Weight 3.0 restores the effective gradient to ~1.30. `q_std` stays at
0.5 rather than widening to the gradient-maximising ~0.9, because wide q_std at high weight
is what got REWARD-HACKED in "[Phase 2+3 v3]".

**Sanity check**: the term reads 0.0104 on resume (lower than the grasp target's 0.393, as
expected from the larger q_err) and climbs to 0.1865 within three iterations -- real
gradient, and the policy responds immediately.

**Launch**: `squeeze_v1_from_p4v7ckpt2099`, from the best-contact checkpoint, 2048 envs,
1000 iterations, `Isaac-G1-Pick-Empty-v0`.

**How to judge it -- do NOT use the reward curve.** Re-run `lift_capability_probe.py` on the
resulting checkpoints. The number that matters is the **follow ratio**: >0.5 means the grip
holds and lifting becomes an ordinary gradient-ascent problem; ~0 means the fingers still
aren't pressing and the next lever is a genuine contact-FORCE reward (PhysX ContactSensor)
rather than any further geometric shaping.

**Result: the squeeze target was pursued, and it did NOT produce grip.** Stopped at ~ckpt
3000 of 3099. `grasp_goal_hand` rose to **1.775** (weight 3.0, raw 0.59, vs 0.393 raw under
the grasp target), so the policy really did chase the squeeze configuration -- but
`target_lifted` stayed at **0.0000** and `lift_progress` at 0.0005.
`lift_capability_probe.py` on ckpt 3000: palm rose +1.90cm, cube rose **-0.06cm**, follow
ratio **-0.030**. Still no grip.

**Why squeeze alone was not enough, and it corrects an overstatement in the entry above.**
Fingertip distances at settle: `[thumb 0.60, index 2.47, middle 2.35, ring 1.61, pinky
1.83]`. Only the thumb is near the cube; the other four are 1.6-2.5cm away. Commanding
squeeze angles therefore **curls those fingers shut in empty space beside the cube** rather
than onto it. Force needs opposing contact, and there is nothing for them to press against.

The previous entry claimed "no lift reward or arm exploration can fix this". That was too
strong. Force cannot come from geometry alone -- true -- but **the arm determines WHERE the
hand sits, and a different palm placement could put the cube between the fingers so that
closing traps it.** Arm exploration changes the geometry, and changed geometry can create
the contact that squeeze then converts into force. So the two interventions are
complementary, not alternatives.

---

### [ArmStd v1] Restore ARM exploration only, keep the hand's -- 2026-08-26

**The user's call, and the reasoning is sound.** The noise reset is the one intervention
that has ever produced a step change in this project ("[Phase 2+3 v6]"), and the arm is
where exploration is now dead while the hand's is healthy. At `phase4v7` ckpt 2099:

```
arm  [0.079, 0.041, 0.125, 0.072, 0.183, 0.132, 0.070]   mean 0.101  <- dead
hand [0.508, 0.661, 0.539, 0.626, 0.556, 0.454]          mean 0.557  <- healthy
```

**Patch**: `model_2099_armstd08.pt` -- indices 0-6 (the 7 arm joints) set to 0.8, indices
7-12 (the 6 hand joints) **left exactly as they were**, so the hard-won finger closure is
not disturbed. Adam `state` cleared as before, since its `exp_avg` for `std` carries the
downward gradient that killed the arm noise and would undo the reset within a few updates.
Verified on resume: `Mean action noise std: 0.69`, which is exactly
`(0.8*7 + 3.344)/13 = 0.688` -- confirming arm raised and hand preserved.

The squeeze reward change from "[Squeeze v1]" is deliberately KEPT (`use_squeeze=True`,
weight 3.0). The two are complementary: arm exploration searches for a palm placement where
the cube actually sits between the fingers, and squeeze converts that contact into force.
Reverting squeeze would remove the only mechanism that can generate normal force once the
geometry is right.

**Launch**: `armstd08_from_p4v7ckpt2099`, 2048 envs, 1000 iterations,
`Isaac-G1-Pick-Empty-v0`.

**How to judge it**: `lift_capability_probe.py` follow ratio, NOT the reward curve. Also
worth watching whether the non-thumb fingertip distances (currently 1.6-2.5cm) come down --
that, not `grasp_goal_hand`'s value, is what would indicate the arm found a placement where
the hand can actually close around the cube.

**Status**: launched, awaiting data.

---

### [Phase 4 v6] Queued to auto-launch -- config decided 2026-08-25

Handled by `grasp_sampler/run_phase4_when_ready.sh`, running in its own `g1_orchestrator`
tmux session so it survives the assistant's session ending (the 2026-08-24 attempt used a
session-scoped monitor and Phase 4 silently never started overnight).

**Checkpoint choice is a REASONED decision, not an argmin** (per the user, 2026-08-25).
Rev 1 auto-picked the lowest summed fingertip distance; that rule is too blind -- run
against the phase23v3 data it would have selected ckpt 350 purely because every later
checkpoint was worse, which is an argmin over a degrading run, not a judgement. Rev 2
instead gathers a **decision packet** (`phase4_decision_packet.txt`): the full
`contact_check.py` sweep across 9 checkpoints, `hand_state_probe.py` internals on 450/600/749,
and the reward curves for `finger_contact`/`grasp_reach`/`grasp_goal_hand`/`grasp_goal_palm`/
`target_lifted`. The assistant is woken with that packet, picks a checkpoint, and writes it
to `phase4_decision.txt`; the orchestrator validates the file (digits only, checkpoint must
exist) and proceeds.

**The specific thing to weigh when choosing**: whether `finger_contact` rising corresponds
to measured fingertip distance actually *falling*. If the term climbs while contact_check
worsens, the new reward is being gamed the same way `grasp_goal_hand` was in
"[Phase 2+3 v3]", and the right choice is an earlier checkpoint from before the divergence
-- or a decision not to proceed to Phase 4 at all.

A blind fallback pick still exists but fires **only** if no decision arrives within 60
minutes, purely so Phase 4 cannot silently fail to start. After the decision, it applies
the six marked config swaps, verifies all six applied and that the file still parses
(restoring a backup and refusing to launch if not), then starts Phase 4.

**The reward changes, and the arithmetic behind them** (all per the user):

| what | from | to | why |
|---|---|---|---|
| `lift_scale` | 2.0 | **60.0** | At 2.0 a 1cm lift held all episode paid `0.01*2.0*8s` = **+0.16** of return out of ~27 (+0.6%), while `joint_speed` alone already cost **-1.80**. Lifting was a strictly bad trade. 60.0 makes that same 1cm lift worth **~5.0**, the target the user set. |
| `lift_height` clamp | 0.30 | **0.05** | With success now at 2cm, a dense term paying out to 30cm would be worth ~144 at full extension and would dominate every grasp term -- the "fling it upward" failure Round 6 produced. Saturating just past the success threshold keeps the gradient where it matters. |
| `_SUCCESS_Z` | 1.134 (29cm) | **0.865 (2cm)** | 29cm was never approachable: best cube height ever observed across 15,360 sampled steps was **1.83cm**, so the 1000-point bonus sat ~16x beyond anything exploration ever reached and contributed zero gradient for the entire project. |
| `joint_speed` | -1.0 | **-0.3** | Tuned for Phase 1, where holding still was the objective; under Phase 4 it directly taxes the target behavior. |
| `target_object_accel` | -5.0 | **-2.0** | Lifting necessarily accelerates the cube, so the largest-magnitude penalty in the config was taxing exactly what Phase 4 wants. Reduced, not removed -- it still closes the shove/knock exploit. |

**The open question this run actually tests** (user's own framing: *why would the policy
ever bother?*): PPO does not need to discover a whole lift. `lift_cont_rew` is linear in
cube height, so it pays for one millimetre -- if the fingers genuinely grip, a tiny upward
motion produces a tiny cube rise produces a tiny reward increase, and gradient ascent walks
up from there. That requires two things simultaneously, and every prior round had at most
one: **(a) real contact**, or the cube height derivative w.r.t. arm motion is *exactly*
zero and any weight multiplies zero (this is why 6x/12x/RND/10x all failed identically),
and **(b) a slope that beats the penalties**, which is what the table above buys. Phase
2+3 v4 is meant to deliver (a); this run supplies (b).

**Status**: queued, will auto-launch on Phase 2+3 v4 completion.

---
