#!/bin/bash
# Records zoomed inference video + contact_check + lift follow ratio for v6 checkpoints
# 900 down to 50, in that order, per the user's explicit request. Runs on cuda:1 so it
# doesn't contend with training (still going on cuda:0). ALWAYS Isaac-G1-Pick-Empty-Play-v0.

set -uo pipefail
G1DIR=/home/shahid/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick
LOGDIR=/home/shahid/IsaacLab/logs/rsl_rl/g1_pick
RUN_NAME="$1"
LOG=$G1DIR/grasp_sampler/record_v6_backwards.log
SUMMARY=$G1DIR/grasp_sampler/record_v6_backwards_summary.txt
TASK=Isaac-G1-Pick-Empty-Play-v0
DEVICE=cuda:1

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

RUN_DIR=$LOGDIR/$RUN_NAME
VID=$RUN_DIR/videos/play_markers
CKPTS="900 850 800 750 700 650 600 550 500 450 400 350 300 250 200 150 100 50"
say "=== recording (descending) for: $CKPTS ==="

: > "$SUMMARY"
{ echo "# V6 CHECKPOINTS 900->50 -- run: $RUN_NAME  env: $TASK  generated: $(date '+%F %T')"; echo; } >> "$SUMMARY"

for C in $CKPTS; do
  CK=/workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME/model_${C}.pt
  # checkpoint may not exist yet if training hasn't reached it (shouldn't happen for
  # 900-and-below since training is nearly done, but skip cleanly rather than error)
  if ! docker exec shahid_g1pick bash -lc "[ -f $CK ]" 2>/dev/null; then
    say "--- ckpt $C : not on disk yet, skipping ---"; continue
  fi
  say "--- ckpt $C : recording video ---"
  echo "================ ckpt $C ================" >> "$SUMMARY"

  PEN=$(docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh play_with_goal_markers.py \
      --task $TASK --num_envs 1 --headless --video --video_length 500 \
      --enable_cameras --episodes 2 --name_prefix v6ckpt${C} --checkpoint $CK --device $DEVICE \
      'env.viewer.origin_type=env' 'env.viewer.eye=[0.72,-0.42,1.02]' 'env.viewer.lookat=[0.40,0.0,0.88]'" 2>&1 \
    | grep -E "penetration summary" | tail -1)
  echo "  ${PEN:-penetration summary: (none)}" >> "$SUMMARY"

  for f in reward_breakdown palm_pose_error fingertip_contact_distance hand_cube_penetration; do
    [ -f "$VID/$f.png" ] && cp "$VID/$f.png" "$VID/v6ckpt${C}_$f.png"
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

say "=== ALL REQUESTED CHECKPOINTS DONE ==="
{ echo; echo "## follow ratio >0.5 = grip holds; ~0 = no grip."
  echo "## contact_check reads finger-link body ORIGINS (knuckle) -- compare across checkpoints, not vs zero."; } >> "$SUMMARY"
cat "$SUMMARY" | tee -a "$LOG"
