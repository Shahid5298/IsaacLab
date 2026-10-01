p = "/isaac-sim/kit/python/lib/python3.11/site-packages/rsl_rl/modules/actor_critic.py"
s = open(p).read()
if "MAX_NOISE_STD" in s:
    print("ALREADY PATCHED"); raise SystemExit
old = """    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")"""
new = """    # --- g1_pick local patch, 2026-08-27 -------------------------------------------
    # Hard ceiling on the policy's exploration noise. RSL-RL has no built-in bound, and
    # without one PPO's entropy bonus grows std without limit whenever the surrogate
    # gradient goes flat: observed std 0.8 -> 2.22 over 1000 iterations while mean_reward
    # sat at ~8.4 and surrogate loss at -0.002. That destroyed fine finger control --
    # measured fingertip-cube contact peaked around iteration 300 and then collapsed.
    # Clamping (rather than only lowering entropy_coef) pins exploration AT the ceiling
    # instead of letting it decay, which is what this task needs.
    # Set to None to restore stock behaviour. Backup: actor_critic.py.pre_stdclamp.bak
    MAX_NOISE_STD = 1.0

    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        if self.MAX_NOISE_STD is not None:
            std = std.clamp(max=self.MAX_NOISE_STD)"""
assert old in s, "anchor not found"
open(p, "w").write(s.replace(old, new, 1))
print("PATCHED rsl_rl std clamp")
