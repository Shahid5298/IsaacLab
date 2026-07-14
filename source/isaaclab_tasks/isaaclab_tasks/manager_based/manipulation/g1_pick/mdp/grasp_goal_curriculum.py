# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training-time ramp for the grasp-goal shaping rewards (mdp/grasp_goal.py).

Misc./UltraDex.md S4.2 added two grasp-goal shaping rewards (grasp_goal_palm,
grasp_goal_hand) at full weight from iteration 0. That competes with the base
task_reward before the policy has learned to reach at all. This module adds an
*independent* progression signal -- it does not read or mutate task_reward,
the five penalty terms, or PickingCurriculumScheduler (mdp/curriculum.py),
which is not wired into any active EnvCfg and is left untouched.

Mechanism
---------
Every step, grasp_goal_palm_reward (mdp/grasp_goal.py) calls
``GraspGoalCurriculum.accumulate(...)`` which adds this step's palm-to-cube
reach quality into a per-env running total, ``env._ggc_episode_accum``. When
an env resets, this term's ``__call__`` (registered as a CurriculumTermCfg,
which IsaacLab's CurriculumManager invokes only for the resetting env_ids --
see isaaclab/envs/manager_based_rl_env.py::_reset_idx) drains that env's
accumulated total into a rolling window and recomputes a single scalar ramp

    w_g = smoothstep(clip((mean_reach - phi_lo) / (phi_hi - phi_lo), 0, 1))

cached as ``env._ggc_ramp`` for grasp_goal.py's reward functions to read
directly (no manager traversal needed) and multiply into their own output.
smoothstep (3x^2 - 2x^3) gives a continuously-differentiable ramp instead of
the hard step PickingCurriculumScheduler uses at its phase thresholds.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _reach_quality(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Palm-to-cube reach quality in [0, 1), same functional form as
    compute_task_reward's r_reach in g1_pick_env_cfg.py, computed independently
    (not imported) so this module never touches that file."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    palm_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    dist = torch.linalg.norm(palm_pos - obj.data.root_pos_w, dim=-1)
    return 1.0 - torch.tanh(dist / 0.25)


class GraspGoalCurriculum(ManagerTermBase):
    """Curriculum term exposing ``env._ggc_ramp`` in [0, 1].

    Params (all optional, set via CurriculumTermCfg(params={...})):
        history_size: rolling window of completed episodes. Default 200.
        min_history: minimum completed episodes before ramp can leave 0.
            Default 20 -- avoids reacting to noise from a handful of episodes.
        phi_lo: mean per-step reach quality below which ramp = 0. Default 0.05.
        phi_hi: mean per-step reach quality above which ramp = 1. Default 0.55.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._history: deque[float] = deque(maxlen=cfg.params.get("history_size", 200))
        self._min_history: int = cfg.params.get("min_history", 20)
        self._phi_lo: float = cfg.params.get("phi_lo", 0.05)
        self._phi_hi: float = cfg.params.get("phi_hi", 0.55)

        if not hasattr(env, "_ggc_episode_accum"):
            env._ggc_episode_accum = torch.zeros(env.num_envs, device=env.device)
        env._ggc_ramp = 0.0

    @staticmethod
    def accumulate(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ) -> None:
        """Add this step's reach quality into the running per-env episode total.

        Called once per control step from grasp_goal_palm_reward. Safe to call
        before this term's __init__ has run (lazily creates the buffer) so
        term declaration order in the config doesn't matter.
        """
        if not hasattr(env, "_ggc_episode_accum"):
            env._ggc_episode_accum = torch.zeros(env.num_envs, device=env.device)
        env._ggc_episode_accum += _reach_quality(env, robot_cfg, object_cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
        history_size: int = 200,
        min_history: int = 20,
        phi_lo: float = 0.05,
        phi_hi: float = 0.55,
    ) -> float:
        if len(env_ids) == 0:
            return env._ggc_ramp

        ep_len = max(env.max_episode_length, 1)
        completed = (env._ggc_episode_accum[env_ids] / ep_len).tolist()
        self._history.extend(completed)
        env._ggc_episode_accum[env_ids] = 0.0

        if len(self._history) >= self._min_history:
            mean_reach = sum(self._history) / len(self._history)
            x = (mean_reach - self._phi_lo) / max(self._phi_hi - self._phi_lo, 1e-6)
            x = min(max(x, 0.0), 1.0)
            ramp = x * x * (3.0 - 2.0 * x)  # smoothstep: C1-continuous, no jump at phi_lo/phi_hi
        else:
            ramp = 0.0

        env._ggc_ramp = ramp
        return ramp
