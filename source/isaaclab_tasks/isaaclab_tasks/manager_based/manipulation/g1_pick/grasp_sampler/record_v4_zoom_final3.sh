#!/bin/bash
# Records inference video + contact_check + lift-capability probe for every checkpoint that
# currently exists in a run, WITHOUT waiting for training to finish (training is ongoing on
# cuda:0; this runs on cuda:1 to avoid contention). ALWAYS Isaac-G1-Pick-Empty-Play-v0.

set -uo pipefail
G1DIR=/home/shahid/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick
LOGDIR=/home/shahid/IsaacLab/logs/rsl_rl/g1_pick
RUN_NAME="$1"
LOG=$G1DIR/grasp_sampler/record_v4_zoom_final3.log
SUMMARY=$G1DIR/grasp_sampler/record_v4_zoom_final3_summary.txt
TASK=Isaac-G1-Pick-Empty-Play-v0
DEVICE=cuda:1

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

RUN_DIR=$LOGDIR/$RUN_NAME
VID=$RUN_DIR/videos/play_markers
CKPTS="2000 2050 2099"
say "=== recording (in-flight, training may still be running) for: $(echo $CKPTS | tr '\n' ' ') ==="

: > "$SUMMARY"
{ echo "# IN-FLIGHT INFERENCE — run: $RUN_NAME  env: $TASK  generated: $(date '+%F %T')"; echo; } >> "$SUMMARY"

for C in $CKPTS; do
  CK=/workspace/isaaclab/logs/rsl_rl/g1_pick/$RUN_NAME/model_${C}.pt
  say "--- ckpt $C : recording video ---"
  echo "================ ckpt $C ================" >> "$SUMMARY"

  PEN=$(docker exec shahid_g1pick bash -lc "
    cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler
    /workspace/isaaclab/_isaac_sim/python.sh play_with_goal_markers.py \
      --task $TASK --num_envs 1 --headless --video --video_length 500 \
      --enable_cameras --episodes 2 --name_prefix v4zoomckpt${C} --checkpoint $CK --device $DEVICE \
      'env.viewer.origin_type=env' 'env.viewer.eye=[0.72,-0.42,1.02]' 'env.viewer.lookat=[0.40,0.0,0.88]'" 2>&1 \
    | grep -E "penetration summary" | tail -1)
  echo "  ${PEN:-penetration summary: (none)}" >> "$SUMMARY"

  for f in reward_breakdown palm_pose_error fingertip_contact_distance hand_cube_penetration; do
    [ -f "$VID/$f.png" ] && cp "$VID/$f.png" "$VID/v4ckpt${C}_$f.png"
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

say "=== IN-FLIGHT SWEEP DONE ==="
{ echo; echo "## follow ratio >0.5 = grip holds; ~0 = no grip."
  echo "## contact_check reads finger-link body ORIGINS (knuckle) -- compare across checkpoints, not vs zero."; } >> "$SUMMARY"
cat "$SUMMARY" | tee -a "$LOG"
