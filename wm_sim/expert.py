"""Scripted expert policy for data collection.

Two-phase strategy:
  1. Navigate behind the block (opposite side from target)
  2. Push the block toward the target

Includes simple obstacle avoidance via repulsive potential fields.
"""

import numpy as np


OBSTACLES = [
    {"pos": np.array([1.0, 0.5]), "radius": 0.55},
    {"pos": np.array([-0.8, -0.5]), "radius": 0.75},
    {"pos": np.array([0.5, -1.2]), "radius": 0.45},
]


class ScriptedExpert:
    """Heuristic policy for the push-block-to-target task."""

    def __init__(
        self,
        target_pos: np.ndarray,
        push_gain: float = 1.5,
        avoid_gain: float = 0.8,
        avoid_radius: float = 0.6,
    ):
        self.target_pos = np.asarray(target_pos, dtype=np.float32)
        self.push_gain = push_gain
        self.avoid_gain = avoid_gain
        self.avoid_radius = avoid_radius

    def _avoidance_force(self, pos: np.ndarray) -> np.ndarray:
        force = np.zeros(2, dtype=np.float32)
        for obs in OBSTACLES:
            diff = pos - obs["pos"]
            d = float(np.linalg.norm(diff))
            if d < self.avoid_radius and d > 1e-6:
                force += self.avoid_gain * diff / (d * d)
        return force

    def __call__(self, state: np.ndarray) -> np.ndarray:
        agent_pos = state[:2]
        block_pos = state[4:6]

        to_target = self.target_pos - block_pos
        to_target_norm = float(np.linalg.norm(to_target))

        if to_target_norm < 1e-6:
            return np.zeros(2, dtype=np.float32)

        to_target_dir = to_target / to_target_norm

        approach_point = block_pos - to_target_dir * 0.4
        to_approach = approach_point - agent_pos
        dist_to_approach = float(np.linalg.norm(to_approach))

        if dist_to_approach > 0.3:
            desired = self.push_gain * to_approach / max(dist_to_approach, 1e-6)
        else:
            desired = self.push_gain * to_target_dir

        desired += self._avoidance_force(agent_pos)
        return np.clip(desired, -1.0, 1.0).astype(np.float32)
