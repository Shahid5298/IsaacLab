#!/bin/bash
# Overnight orchestrator (2026-08-26): after "[Phase 2+3 v6]" (std=0.8 reset) finishes,
# measure real contact, pick the best checkpoint, decide whether the run actually IMPROVED
# on contact, and launch Phase 4 accordingly.
#
# Per the user: if v6 did NOT meaningfully change anything versus previous Phase 2+3
# sessions, re-inject exploration by resetting the policy's noise std to 0.8 again before
# starting Phase 4 -- otherwise hand off normally.
#
# Runs in its own tmux session so it survives the assistant's session ending.

set -uo pipefail

G1DIR=/home/shahid/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick
CFG=$G1DIR/g1_pick_env_cfg.py
LOGDIR=/home/shahid/IsaacLab/logs/rsl_rl/g1_pick
LOG=$G1DIR/grasp_sampler/phase4_v6_orchestrator.log
SWEEP=$G1DIR/grasp_sampler/phase4_v6_sweep.txt
DECISION=$G1DIR/grasp_sampler/phase4_v6_decision.txt   # assistant may write "<ckpt> <std08|nostd>"
RUN_NAME="$1"
CANDIDATES="400 500 600 700 800 900 1000 1100 1249"
DECISION_WAIT_MIN=45

# "Improved" thresholds, calibrated against real measurements in REWARD_CURRICULUM_LOG.md:
#   phase23v2 ckpt600 (best ever): thumb 0.72cm, other four 2.12-2.82cm
#   phase23v5 (regression):        thumb 2.60-4.92cm, other four 4.2-5.9cm
# So thumb<=1.5 and others<=3.5 sits clearly in "as good as our best" territory, and
# anything worse counts as "nothing really changed" -> reset std to 0.8 for phase 4.
THUMB_OK=1.5
OTHERS_OK=3.5

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
rm -f "$DECISION"
say "=== phase4-after-v6 orchestrator started, watching $RUN_NAME ==="

# NOTE the [r] bracket: `pgrep -f "rsl_rl/train.py"` run inside `bash -lc '...'` MATCHES ITS
# OWN WRAPPER, because the wrapper's command line contains that literal string -- it always
# reports "running" and this loop would never exit, silently never launching phase 4. The
# regex [r]sl_rl/train.py matches a real trainer but not the literal text "[r]sl_rl/..."
# in this command's own cmdline. Verified both forms before arming.
train_running() { docker exec shahid_g1pick bash -lc 'pgrep -f "[r]sl_rl/train.py" >/dev/null' 2>/dev/null; }

# wait for the v6 run to end (it may already have finished before this script started)
for _ in $(seq 1 12); do train_running && break; sleep 5; done
while train_running; do sleep 60; done
say "no training running; verifying checkpoints"

RUN_DIR=$LOGDIR/$RUN_NAME
if [ ! -d "$RUN_DIR" ] || ! ls "$RUN_DIR"/model_*.pt >/dev/null 2>&1; then
  say "FATAL: no run dir or no checkpoints at $RUN_DIR -- NOT launching phase 4"; exit 1
fi

# ---------------------------------------------------------------- 1. measure contact
say "=== contact_check sweep (this takes ~15-20 min) ==="
: > "$SWEEP"
for C in $CANDIDATES; do
  [ -f "$RUN_DIR/model_${C}.pt" ] || continue
  say "  contact_check ckpt $C"
  echo "=== ckpt $C ===" >> "$SWEEP"
  docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh contact_check.py \
      --task Isaac-G1-Pick-Empty-Play-v0 --num_envs 64 --steps 300 --headless \
      --checkpoint /workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME/model_${C}.pt \
      --device cuda:0" 2>&1 | grep -E "^(thumb|index|middle|ring|pinky): min" >> "$SWEEP"
done
say "sweep complete:"; cat "$SWEEP" | tee -a "$LOG"

# ------------------------------------------------- 2. pick best ckpt + improved verdict
read -r AUTO_CKPT AUTO_VERDICT <<< "$(python3 - "$SWEEP" "$THUMB_OK" "$OTHERS_OK" <<'PY'
import re, sys
sweep, thumb_ok, others_ok = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows, cur = {}, None
for line in open(sweep):
    m = re.match(r"=== ckpt (\d+) ===", line.strip())
    if m: cur = int(m.group(1)); rows[cur] = {}; continue
    m = re.match(r"(\w+): min=\S+ mean=([-+0-9.]+)cm", line.strip())
    if m and cur is not None: rows[cur][m.group(1)] = float(m.group(2))
rows = {k: v for k, v in rows.items() if len(v) == 5}
if not rows:
    print("NONE nothing"); sys.exit()
# best = lowest thumb distance (the finger that has always led), tiebreak on total
best = min(rows, key=lambda c: (rows[c]["thumb"], sum(rows[c].values())))
r = rows[best]
others = (r["index"] + r["middle"] + r["ring"] + r["pinky"]) / 4.0
verdict = "improved" if (r["thumb"] <= thumb_ok and others <= others_ok) else "nothing"
print(f"{best} {verdict}")
PY
)"
say "auto-pick: checkpoint=$AUTO_CKPT verdict=$AUTO_VERDICT (thumb<=$THUMB_OK & others<=$OTHERS_OK => improved)"

# ------------------------------------------------- 3. allow the assistant to override
say "waiting up to ${DECISION_WAIT_MIN} min for an override in $DECISION (format: '<ckpt> <std08|nostd>')"
CKPT=""; MODE=""
for _ in $(seq 1 $((DECISION_WAIT_MIN * 6))); do
  if [ -f "$DECISION" ]; then
    read -r d_ckpt d_mode < "$DECISION"
    if [ -n "${d_ckpt:-}" ] && [ -f "$RUN_DIR/model_${d_ckpt}.pt" ] \
       && { [ "${d_mode:-}" = "std08" ] || [ "${d_mode:-}" = "nostd" ]; }; then
      CKPT=$d_ckpt; MODE=$d_mode; say "override accepted: ckpt=$CKPT mode=$MODE"; break
    fi
    say "invalid override ('${d_ckpt:-} ${d_mode:-}') -- ignoring"; rm -f "$DECISION"
  fi
  sleep 10
done
if [ -z "$CKPT" ]; then
  CKPT=$AUTO_CKPT
  [ "$AUTO_VERDICT" = "improved" ] && MODE=nostd || MODE=std08
  say "no override; using auto: ckpt=$CKPT mode=$MODE"
fi
[ "$CKPT" != "NONE" ] || { say "FATAL: could not score any checkpoint -- NOT launching"; exit 1; }

# ------------------------------------------------- 4. optional std reset to 0.8
LOAD_CKPT="model_${CKPT}.pt"
if [ "$MODE" = "std08" ]; then
  say "=== resetting policy noise std to 0.8 on ckpt $CKPT (v6 did not clearly improve contact) ==="
  LOAD_CKPT="model_${CKPT}_std08.pt"
  docker exec shahid_g1pick bash -lc "
    /workspace/isaaclab/_isaac_sim/python.sh -c \"
import torch
src='/workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME/model_${CKPT}.pt'
dst='/workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME/$LOAD_CKPT'
d=torch.load(src, weights_only=False, map_location='cpu')
old=d['model_state_dict']['std'].clone()
d['model_state_dict']['std']=torch.full_like(old,0.8)
if 'optimizer_state_dict' in d: d['optimizer_state_dict']['state']={}
torch.save(d,dst)
print('old std:',[round(x,3) for x in old.tolist()])
print('new std:',[round(x,3) for x in torch.load(dst,weights_only=False,map_location='cpu')['model_state_dict']['std'].tolist()])
\"" 2>&1 | tee -a "$LOG"
  if [ ! -f "$RUN_DIR/$LOAD_CKPT" ]; then
    say "FATAL: std patch failed to produce $LOAD_CKPT -- NOT launching"; exit 1
  fi
fi

# ------------------------------------------------- 5. apply the phase 4 penalty swaps
# lift_weight/success_weight/lift_scale/_SUCCESS_Z are ALREADY at phase 4 values (lift has
# been active since "[Scratch v1]"). The only remaining phase 4 delta is relaxing the two
# penalties that were tuned for a hold-still objective and now oppose lifting.
say "=== applying phase 4 penalty swaps ==="
cp "$CFG" "$CFG.pre_phase4v7.bak"
sed -i \
  -e 's/weight=-1\.0,  # PHASE4_SWAP_JOINT_SPEED/weight=-0.3,  # PHASE4_SWAP_JOINT_SPEED/' \
  -e 's/weight=-5\.0,  # PHASE4_SWAP_TARGET_ACCEL/weight=-2.0,  # PHASE4_SWAP_TARGET_ACCEL/' \
  "$CFG"
grep -n "PHASE4_SWAP" "$CFG" | tee -a "$LOG"
for e in "weight=-0.3,  # PHASE4_SWAP_JOINT_SPEED" "weight=-2.0,  # PHASE4_SWAP_TARGET_ACCEL" \
         '"lift_weight": 1.0' '"success_weight": 1.0' '"lift_scale": 60.0' "_SUCCESS_Z    = 0.865"; do
  grep -qF "$e" "$CFG" || { say "FATAL: expected config line missing after swap: '$e' -- restoring, NOT launching"
                            cp "$CFG.pre_phase4v7.bak" "$CFG"; exit 1; }
done
python3 -c "import ast;ast.parse(open('$CFG').read())" 2>>"$LOG" \
  || { say "FATAL: config does not parse -- restoring, NOT launching"; cp "$CFG.pre_phase4v7.bak" "$CFG"; exit 1; }
say "phase 4 config verified"

# ------------------------------------------------- 6. launch phase 4
RN="phase4v7_from_v6ckpt${CKPT}_${MODE}"
say "=== launching phase 4: $RN (from $LOAD_CKPT) ==="
tmux clear-history -t g1_train_5cm 2>/dev/null
tmux send-keys -t g1_train_5cm "docker exec shahid_g1pick bash -lc 'cd /workspace/isaaclab && /workspace/isaaclab/_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-Pick-Empty-v0 --headless --num_envs 2048 --max_iterations 1000 --device cuda:0 --resume --load_run $RUN_NAME --checkpoint $LOAD_CKPT --run_name $RN'" Enter

for _ in $(seq 1 40); do
  sleep 15
  if tmux capture-pane -t g1_train_5cm -p | tail -25 | grep -q "Learning iteration"; then
    say "=== PHASE 4 IS RUNNING ($RN) ==="; exit 0
  fi
  if tmux capture-pane -t g1_train_5cm -p | tail -25 | grep -qE "malloc|Aborted|Traceback|CUDA error"; then
    say "!!! phase 4 crashed on startup -- retrying once at 1024 envs (host RAM contention seen 2026-08-25)"
    tmux send-keys -t g1_train_5cm "docker exec shahid_g1pick bash -lc 'cd /workspace/isaaclab && /workspace/isaaclab/_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-Pick-Empty-v0 --headless --num_envs 1024 --max_iterations 1000 --device cuda:0 --resume --load_run $RUN_NAME --checkpoint $LOAD_CKPT --run_name ${RN}_1024'" Enter
    sleep 180
    tmux capture-pane -t g1_train_5cm -p | tail -25 | grep -q "Learning iteration" \
      && say "=== PHASE 4 RUNNING at 1024 envs ===" || say "!!! phase 4 still not running -- needs attention"
    exit 0
  fi
done
say "!!! phase 4 did not reach 'Learning iteration' -- check tmux g1_train_5cm"
tmux capture-pane -t g1_train_5cm -p | tail -30 | tee -a "$LOG"
