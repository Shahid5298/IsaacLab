import torch
from rsl_rl.modules.rnd import RandomNetworkDistillation

src = "/workspace/isaaclab/logs/rsl_rl/g1_pick/2026-08-21_08-48-51_phase2and3_merged_from_ckpt200/model_450.pt"
dst = "/workspace/isaaclab/logs/rsl_rl/g1_pick/2026-08-21_08-48-51_phase2and3_merged_from_ckpt200/model_450_rnd_init.pt"

d = torch.load(src, weights_only=False, map_location="cpu")

# Must match agents/rsl_rl_ppo_cfg.py's RslRlRndCfg exactly, or the state_dict shapes
# won't match what OnPolicyRunner.load() expects on the real training run.
rnd = RandomNetworkDistillation(
    num_states=7,
    obs_groups={"rnd_state": ["rnd_state"]},
    num_outputs=16,
    predictor_hidden_dims=[-1],
    target_hidden_dims=[-1],
    activation="elu",
    weight=10.0,
    state_normalization=True,
    reward_normalization=True,
    device="cpu",
)
rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1.0e-3)

d["rnd_state_dict"] = rnd.state_dict()
d["rnd_optimizer_state_dict"] = rnd_optimizer.state_dict()

torch.save(d, dst)
print(f"Saved {dst}")
print("rnd_state_dict keys:", list(d["rnd_state_dict"].keys())[:5], "...")
