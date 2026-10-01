import torch

src = "/workspace/isaaclab/logs/rsl_rl/g1_pick/2026-08-21_08-48-51_phase2and3_merged_from_ckpt200/model_450.pt"
dst = "/workspace/isaaclab/logs/rsl_rl/g1_pick/2026-08-21_08-48-51_phase2and3_merged_from_ckpt200/model_450_std_armboost.pt"

# original 13-dim std values (from the unpatched checkpoint, confirmed earlier):
# [0] shoulder_pitch=0.0445 [1] shoulder_roll=0.0497 [2] shoulder_yaw=0.0617
# [3] elbow=0.0492 [4] wrist_roll=0.1241 [5] wrist_pitch=0.1336 [6] wrist_yaw=0.0865
# [7..12] hand joints=0.5033,0.4178,0.4086,0.3676,0.4465,0.4213 -- left untouched

d = torch.load(src, weights_only=False, map_location="cpu")
sd = d["model_state_dict"]
old_val = sd["std"].clone()
print("old:", old_val.tolist())

new_val = old_val.clone()
new_val[0] = 0.35       # shoulder_pitch -- primary z-joint, confirmed empirically
for i in [1, 2, 3, 4, 5, 6]:
    new_val[i] = 0.175  # rest of the arm -- half of the primary joint's value
# indices 7-12 (hand) left at their original values

sd["std"] = new_val
print("new:", new_val.tolist())

torch.save(d, dst)
print(f"Saved to {dst}")
