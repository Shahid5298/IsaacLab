#!/bin/bash
# Overnight orchestrator (2026-08-25, rev 2): waits for the "[Phase 2+3 v4]" run to finish,
# gathers a RICH diagnostic packet across checkpoints, then hands the checkpoint decision to
# the assistant. Falls back to an automatic pick only if no decision arrives in time, so
# phase 4 always starts (the 2026-08-24 attempt silently never launched overnight).
#
# Design note: rev 1 auto-picked by a single summed-distance metric. That is too blind --
# on the phase23v3 data that rule would have chosen ckpt 350 purely because every later
# checkpoint was worse, which is not a judgement, just an argmin over a degrading run.
# The sweep now also captures per-finger detail, grasp_goal_hand internals, and the reward
# curves, so the choice can weigh "did the fingers actually close" against "did the term
# get gamed again" -- the exact failure mode that made the reward curve untrustworthy.
#
# Runs in its own tmux session so it survives the assistant's session ending.

set -uo pipefail

G1DIR=/home/shahid/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick
CFG=$G1DIR/g1_pick_env_cfg.py
LOGDIR=/home/shahid/IsaacLab/logs/rsl_rl/g1_pick
LOG=$G1DIR/grasp_sampler/phase4_orchestrator.log
PACKET=$G1DIR/grasp_sampler/phase4_decision_packet.txt
DECISION=$G1DIR/grasp_sampler/phase4_decision.txt
RUN_NAME_P23="$1"
CANDIDATES="300 400 450 500 550 600 650 700 749"
PROBE_CKPTS="450 600 749"          # heavier hand_state_probe only on a few
DECISION_WAIT_MIN=60               # how long to wait for an assistant decision

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

rm -f "$DECISION"
say "=== orchestrator (rev 2) started, watching $RUN_NAME_P23 ==="

# ---------------------------------------------------------------- 1. wait for training
# Match on "rsl_rl/train.py" ONLY. Do NOT match on the run name: this script receives the
# run name as $1, so its own cmdline contains it, and `pgrep -f <run name>` matches THIS
# process -- an infinite wait in which phase 4 never launches (the exact silent failure
# seen on 2026-08-24). "rsl_rl/train.py" cannot appear in this script's own cmdline.
train_running() { pgrep -f "rsl_rl/train.py" >/dev/null 2>&1; }

for _ in $(seq 1 60); do
  train_running && break
  sleep 10
done
if ! train_running; then
  say "FATAL: never observed a running rsl_rl/train.py -- aborting, NOT launching phase 4"
  exit 1
fi
say "observed phase 2+3 training running; waiting for it to finish"

while train_running; do
  sleep 60
done
say "phase 2+3 training process is gone; verifying checkpoints landed"

RUN_DIR=$LOGDIR/$RUN_NAME_P23
if [ ! -d "$RUN_DIR" ]; then
  say "FATAL: run dir $RUN_DIR does not exist -- aborting, NOT launching phase 4"
  exit 1
fi
if ! ls "$RUN_DIR"/model_*.pt >/dev/null 2>&1; then
  say "FATAL: no checkpoints at all -- aborting, NOT launching phase 4"
  exit 1
fi
[ -f "$RUN_DIR/model_749.pt" ] || say "WARNING: model_749.pt missing -- run may have died early"

# ------------------------------------------------- 2. gather the decision packet
say "=== gathering diagnostic packet (this takes ~20-30 min) ==="
: > "$PACKET"
{
  echo "############################################################"
  echo "# PHASE 4 CHECKPOINT DECISION PACKET"
  echo "# run: $RUN_NAME_P23"
  echo "# generated: $(date '+%F %T')"
  echo "############################################################"
  echo
  echo "## available checkpoints"
  ls "$RUN_DIR" | grep -o 'model_[0-9]*\.pt' | sort -t_ -k2 -n | tr '\n' ' '
  echo; echo
} >> "$PACKET"

say "  [1/3] contact_check sweep"
{ echo "## contact_check.py -- min per-fingertip distance to cube SURFACE, 64 envs x 300 steps"
  echo "## NOTE: these read finger-link BODY ORIGINS (at the knuckle), not fingertip pads"
  echo "## ~2-3cm further along. Absolute values are inflated by a ~constant offset;"
  echo "## compare ACROSS checkpoints, don't read them as absolute contact."
  echo; } >> "$PACKET"
for C in $CANDIDATES; do
  [ -f "$RUN_DIR/model_${C}.pt" ] || continue
  say "    contact_check ckpt $C"
  echo "=== ckpt $C ===" >> "$PACKET"
  docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh contact_check.py \
      --task Isaac-G1-Pick-Empty-Play-v0 --num_envs 64 --steps 300 --headless \
      --checkpoint /workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME_P23/model_${C}.pt \
      --device cuda:0" 2>&1 | grep -E "^(thumb|index|middle|ring|pinky): min" >> "$PACKET"
done
echo >> "$PACKET"

say "  [2/3] hand_state_probe on $PROBE_CKPTS"
{ echo "## hand_state_probe.py -- grasp_goal_hand internals + actual cube height reached"
  echo "## (palm gate, hand joint error q_err, per-joint breakdown; joint order is"
  echo "##  [thumb_yaw, thumb_pitch, index, middle, ring, pinky])"
  echo; } >> "$PACKET"
for C in $PROBE_CKPTS; do
  [ -f "$RUN_DIR/model_${C}.pt" ] || continue
  say "    hand_state_probe ckpt $C"
  echo "=== ckpt $C ===" >> "$PACKET"
  docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh hand_state_probe.py \
      --task Isaac-G1-Pick-Empty-Play-v0 --num_envs 64 --steps 240 --headless \
      --checkpoint /workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME_P23/model_${C}.pt \
      --device cuda:0" 2>&1 | sed -n '/palm gate/,/steps with cube/p' >> "$PACKET"
  echo >> "$PACKET"
done

say "  [3/3] reward curves"
{ echo "## TensorBoard curves -- watch finger_contact (did the new term actually engage?)"
  echo "## against contact_check above. If finger_contact climbs while measured distance"
  echo "## worsens, the term is being gamed like grasp_goal_hand was in phase23v3."
  echo; } >> "$PACKET"
docker exec shahid_g1pick bash -lc "
  /workspace/isaaclab/_isaac_sim/python.sh \
    /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler/tb_dump_series.py \
    /workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME_P23 \
    'Episode_Reward/finger_contact' 'Episode_Reward/grasp_reach' \
    'Episode_Reward/grasp_goal_hand' 'Episode_Reward/grasp_goal_palm' \
    'Episode_Reward/grasp_envelope' 'Episode_Termination/target_lifted' \
    'Policy/mean_noise_std' 'Train/mean_reward'" 2>&1 | grep -v "^\[" >> "$PACKET"

# fallback suggestion only -- explicitly NOT the decision
FALLBACK=$(python3 - "$PACKET" <<'PY'
import re, sys
best, cur, vals = None, None, {}
def flush():
    global best
    if cur is not None and len(vals) == 5:
        s = sum(vals.values())
        if best is None or s < best[1]:
            best = (cur, s)
for line in open(sys.argv[1]):
    m = re.match(r"=== ckpt (\d+) ===", line.strip())
    if m:
        flush(); cur = int(m.group(1)); vals = {}; continue
    m = re.match(r"(\w+): min=\S+ mean=([-+0-9.]+)cm", line.strip())
    if m:
        vals[m.group(1)] = float(m.group(2))
flush()
print(best[0] if best else "")
PY
)
[ -n "$FALLBACK" ] || FALLBACK=600
[ -f "$RUN_DIR/model_${FALLBACK}.pt" ] || FALLBACK=749
{ echo; echo "## fallback auto-pick (lowest summed mean distance): $FALLBACK"
  echo "## This is a BLIND argmin and is used ONLY if no decision arrives within"
  echo "## ${DECISION_WAIT_MIN} min. Prefer a reasoned choice."; } >> "$PACKET"

say "=== DECISION PACKET READY at $PACKET (fallback would be $FALLBACK) ==="
say "=== waiting up to ${DECISION_WAIT_MIN} min for a checkpoint decision in $DECISION ==="

# ------------------------------------------------- 3. wait for the assistant's decision
BEST=""
for _ in $(seq 1 $((DECISION_WAIT_MIN * 6))); do
  if [ -f "$DECISION" ]; then
    D=$(tr -cd '0-9' < "$DECISION")
    if [ -n "$D" ] && [ -f "$RUN_DIR/model_${D}.pt" ]; then
      BEST=$D; say "decision received: checkpoint $BEST"; break
    else
      say "decision file present but invalid ('$D') -- ignoring, still waiting"
      rm -f "$DECISION"
    fi
  fi
  sleep 10
done
if [ -z "$BEST" ]; then
  BEST=$FALLBACK
  say "no decision within ${DECISION_WAIT_MIN} min -- falling back to auto-pick $BEST"
fi
say "=== selected checkpoint $BEST for phase 4 handoff ==="

# ------------------------------------------------------------- 4. swap config to phase 4
say "=== applying phase 4 reward config ==="
cp "$CFG" "$CFG.phase23v4.bak"
sed -i \
  -e 's/^_SUCCESS_Z.*# PHASE4_SWAP_SUCCESS_Z/_SUCCESS_Z    = 0.865  # PHASE4_SWAP_SUCCESS_Z/' \
  -e 's/"lift_weight": 0\.0,  # PHASE4_SWAP_LIFT_WEIGHT/"lift_weight": 1.0,  # PHASE4_SWAP_LIFT_WEIGHT/' \
  -e 's/"success_weight": 0\.0,  # PHASE4_SWAP_SUCCESS_WEIGHT/"success_weight": 1.0,  # PHASE4_SWAP_SUCCESS_WEIGHT/' \
  -e 's/"lift_scale": 2\.0,  # PHASE4_SWAP_LIFT_SCALE/"lift_scale": 60.0,  # PHASE4_SWAP_LIFT_SCALE/' \
  -e 's/weight=-1\.0,  # PHASE4_SWAP_JOINT_SPEED/weight=-0.3,  # PHASE4_SWAP_JOINT_SPEED/' \
  -e 's/weight=-5\.0,  # PHASE4_SWAP_TARGET_ACCEL/weight=-2.0,  # PHASE4_SWAP_TARGET_ACCEL/' \
  "$CFG"

say "swapped values now read:"
grep -n "PHASE4_SWAP" "$CFG" | tee -a "$LOG"

EXPECTED=("_SUCCESS_Z    = 0.865" '"lift_weight": 1.0' '"success_weight": 1.0' \
          '"lift_scale": 60.0' "weight=-0.3," "weight=-2.0,")
for e in "${EXPECTED[@]}"; do
  if ! grep -qF "$e" "$CFG"; then
    say "FATAL: phase 4 swap failed to apply '$e' -- restoring config, NOT launching"
    cp "$CFG.phase23v4.bak" "$CFG"
    exit 1
  fi
done
if ! python3 -c "import ast;ast.parse(open('$CFG').read())" 2>>"$LOG"; then
  say "FATAL: config no longer parses after swap -- restoring, NOT launching"
  cp "$CFG.phase23v4.bak" "$CFG"
  exit 1
fi
say "all 6 phase 4 swaps verified, config parses"

# --------------------------------------------------------------------- 5. launch phase 4
say "=== launching phase 4 from $RUN_NAME_P23 / model_${BEST}.pt ==="
tmux send-keys -t g1_train_5cm "docker exec shahid_g1pick bash -lc 'cd /workspace/isaaclab && /workspace/isaaclab/_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-Pick-v0 --headless --num_envs 2048 --max_iterations 500 --device cuda:0 --resume --load_run $RUN_NAME_P23 --checkpoint model_${BEST}.pt --run_name phase4v6_from_p23v4ckpt${BEST}'" Enter

sleep 120
if tmux capture-pane -t g1_train_5cm -p | grep -q "Learning iteration"; then
  say "=== PHASE 4 IS RUNNING (from ckpt $BEST) ==="
else
  say "!!! phase 4 did not reach 'Learning iteration' within 120s -- check tmux g1_train_5cm"
  tmux capture-pane -t g1_train_5cm -p | tail -30 | tee -a "$LOG"
fi
say "=== orchestrator done ==="
