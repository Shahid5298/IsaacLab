#!/bin/bash
# Waits for "[Sufian v5]" (scratch, gated blend, std clamp) to finish, then for EVERY
# checkpoint records a zoomed inference video + diagnostic plots, contact_check, and the
# lift-capability follow ratio. ALWAYS Isaac-G1-Pick-Empty-Play-v0 -- that task is the
# DEFAULT in play_with_goal_markers.py, so it must be passed explicitly every call.
#
# Runs in its own tmux session so it survives the assistant's session ending.

set -uo pipefail
G1DIR=/home/shahid/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick
LOGDIR=/home/shahid/IsaacLab/logs/rsl_rl/g1_pick
RUN_NAME="$1"
LOG=$G1DIR/grasp_sampler/record_curriculum.log
SUMMARY=$G1DIR/grasp_sampler/record_curriculum_summary.txt
TASK=Isaac-G1-Pick-Empty-Play-v0
DEVICE=cuda:0

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# [r] bracket: a plain "rsl_rl/train.py" pattern inside `bash -lc '...'` matches this very
# command's own wrapper cmdline and would report "running" forever -- bit us three times
# already this project.
train_running() { docker exec shahid_g1pick bash -lc 'pgrep -f "[r]sl_rl/train.py" >/dev/null' 2>/dev/null; }

say "=== record_v5 started, watching $RUN_NAME ==="
for _ in $(seq 1 12); do train_running && break; sleep 5; done
while train_running; do sleep 60; done
say "training finished"

RUN_DIR=$LOGDIR/$RUN_NAME
VID=$RUN_DIR/videos/play_markers
if [ ! -d "$RUN_DIR" ]; then say "FATAL: no run dir $RUN_DIR"; exit 1; fi

CKPTS=$(ls "$RUN_DIR" | grep -oE '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n)
[ -n "$CKPTS" ] || { say "FATAL: no checkpoints in $RUN_DIR"; exit 1; }
say "checkpoints to process: $(echo $CKPTS | tr '\n' ' ')"

: > "$SUMMARY"
{ echo "############################################################"
  echo "# CURRICULUM STEP (position-only + grasp_reach) -- ALL-CHECKPOINT INFERENCE + RECORDING"
  echo "# run: $RUN_NAME"
  echo "# env: $TASK  (empty -- distractors at table margins)"
  echo "# camera: zoomed (eye=[0.72,-0.42,1.02] lookat=[0.40,0.0,0.88])"
  echo "# generated: $(date '+%F %T')"
  echo "############################################################"; echo; } >> "$SUMMARY"

for C in $CKPTS; do
  CK=/workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME/model_${C}.pt
  say "--- ckpt $C : recording video ---"
  echo "================ ckpt $C ================" >> "$SUMMARY"

  PEN=$(docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh play_with_goal_markers.py \
      --task $TASK --num_envs 1 --headless --video --video_length 500 \
      --enable_cameras --episodes 2 --name_prefix curr_ckpt${C} --checkpoint $CK --device $DEVICE \
      'env.viewer.origin_type=env' 'env.viewer.eye=[0.72,-0.42,1.02]' 'env.viewer.lookat=[0.40,0.0,0.88]'" 2>&1 \
    | grep -E "penetration summary" | tail -1)
  echo "  ${PEN:-penetration summary: (none)}" >> "$SUMMARY"

  for f in reward_breakdown palm_pose_error fingertip_contact_distance hand_cube_penetration; do
    [ -f "$VID/$f.png" ] && cp "$VID/$f.png" "$VID/curr_ckpt${C}_$f.png"
  done

  say "--- ckpt $C : contact_check ---"
  docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh contact_check.py \
      --task $TASK --num_envs 64 --steps 300 --headless \
      --checkpoint $CK --device $DEVICE" 2>&1 \
    | grep -E "^(thumb|index|middle|ring|pinky): min" >> "$SUMMARY"

  say "--- ckpt $C : lift capability (follow ratio) ---"
  docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh lift_capability_probe.py \
      --task $TASK --num_envs 32 --headless --mode lift --lift_delta -1.5 \
      --checkpoint $CK --device $DEVICE" 2>&1 \
    | grep -E "palm rose|cube rose|follow ratio|VERDICT" >> "$SUMMARY"
  echo >> "$SUMMARY"
done

say "=== ALL CHECKPOINTS DONE ==="
{ echo; echo "## follow ratio >0.5 = grip holds; ~0 = no grip."
  echo "## contact_check reads finger-link body ORIGINS (knuckle) -- compare across checkpoints, not vs zero."; } >> "$SUMMARY"
cat "$SUMMARY" | tee -a "$LOG"
