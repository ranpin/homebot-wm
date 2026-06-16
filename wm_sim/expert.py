"""Scripted expert policy for data collection.

Strategy:
  1. Navigate behind the block (opposite side from target)
  2. Push the block toward the target with speed proportional to distance
  3. Slow down when block is close to target to avoid overshooting

Includes simple obstacle avoidance via repulsive potential fields.
"""

import numpy as np


OBSTACLES = [
    {"pos": np.array([1.0, 0.5]), "radius": 0.6},
    {"pos": np.array([-0.8, -0.5]), "radius": 0.8},
    {"pos": np.array([0.5, -1.2]), "radius": 0.5},
]


class ScriptedExpert:
    """Heuristic policy for the push-block-to-target task."""

    def __init__(
        self,
        target_pos: np.ndarray,
        avoid_gain: float = 0.8,
        avoid_radius: float = 0.7,
    ):
        self.target_pos = np.asarray(target_pos, dtype=np.float32)
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
        agent_vel = state[2:4]
        block_pos = state[4:6]

        to_target = self.target_pos - block_pos
        dist_to_target = float(np.linalg.norm(to_target))

        if dist_to_target < 0.3:
            return np.zeros(2, dtype=np.float32)

        to_target_dir = to_target / dist_to_target

        push_point = block_pos - to_target_dir * 0.35
        to_push = push_point - agent_pos
        dist_to_push = float(np.linalg.norm(to_push))

        speed_scale = min(1.0, dist_to_target / 1.5)

        if dist_to_push > 0.25:
            desired = speed_scale * to_push / max(dist_to_push, 1e-6)
        else:
            desired = speed_scale * to_target_dir

        desired += self._avoidance_force(agent_pos)
        return np.clip(desired, -1.0, 1.0).astype(np.float32)
