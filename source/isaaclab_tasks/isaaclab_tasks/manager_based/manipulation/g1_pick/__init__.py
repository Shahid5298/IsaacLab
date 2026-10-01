# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 Object Picking Environment with Curriculum Learning.

This environment trains a G1 humanoid robot to pick target objects from a cluttered tray.
The robot stands with frozen lower body and uses its arms and dexterous hands to manipulate objects.
"""

import gymnasium as gym

from . import agents
from .g1_pick_env_cfg import (
    G1PickEnvCfg,
    G1PickEnvCfg_PLAY,
    G1RightArmLiftEnvCfg_V2_Empty,
    G1RightArmLiftEnvCfg_V2_Empty_PLAY,
    G1RightArmLiftEnvCfg_V2_Stage1,
    G1RightArmLiftEnvCfg_V2_Stage1_PLAY,
    G1RightArmLiftEnvCfg_V2_Stage2,
    G1RightArmLiftEnvCfg_V2_Stage2_PLAY,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-G1-Pick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PickEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Pick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PickEnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

# From-scratch 5-phase reward curriculum, Phases 1-4 base env: all 10 distractors on the
# table margins, present in the scene/observation space but physically out of the way.
gym.register(
    id="Isaac-G1-Pick-Empty-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RightArmLiftEnvCfg_V2_Empty,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Pick-Empty-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RightArmLiftEnvCfg_V2_Empty_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

# Clutter-robustness curriculum, stage 1 of 3: 5 distractors spread wide on the tray (not
# clustered), the other 5 parked on the table margins out of the way. See
# G1RightArmLiftEnvCfg_V2_Stage1's docstring in g1_pick_env_cfg.py for the full rationale.
gym.register(
    id="Isaac-G1-Pick-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RightArmLiftEnvCfg_V2_Stage1,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Pick-Stage1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RightArmLiftEnvCfg_V2_Stage1_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

# Clutter-robustness curriculum, stage 2 of 3: all 10 distractors on the tray, evenly spaced in
# a 2x5 grid (not clustered at corners like stage 1, not clustered adversarially like stage 3).
gym.register(
    id="Isaac-G1-Pick-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RightArmLiftEnvCfg_V2_Stage2,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Pick-Stage2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RightArmLiftEnvCfg_V2_Stage2_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickPPORunnerCfg",
    },
)
