import torch
import sys

src = "/workspace/isaaclab/logs/rsl_rl/g1_pick/2026-08-21_08-48-51_phase2and3_merged_from_ckpt200/model_450.pt"
dst = "/workspace/isaaclab/logs/rsl_rl/g1_pick/2026-08-21_08-48-51_phase2and3_merged_from_ckpt200/model_450_std035.pt"
target_std = 0.35

d = torch.load(src, weights_only=False, map_location="cpu")
sd = d["model_state_dict"]

std_key = None
for k in sd.keys():
    if k in ("std", "log_std") or k.endswith(".std") or k.endswith(".log_std"):
        std_key = k
        break

if std_key is None:
    print("STD_KEY_NOT_FOUND. Available keys:")
    for k in sd.keys():
        print(" ", k, tuple(sd[k].shape) if hasattr(sd[k], "shape") else sd[k])
    sys.exit(1)

old_val = sd[std_key].clone()
print(f"Found std param at key={std_key!r}, shape={tuple(old_val.shape)}, old values={old_val.tolist()}")

if "log_std" in std_key:
    sd[std_key] = torch.log(torch.full_like(old_val, target_std))
else:
    sd[std_key] = torch.full_like(old_val, target_std)

print(f"New values={sd[std_key].tolist()}")

torch.save(d, dst)
print(f"Saved patched checkpoint to {dst}")
