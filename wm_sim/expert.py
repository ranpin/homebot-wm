"""Scripted expert policy for data collection.

Strategy:
  1. If block is near a wall, first push it away from the wall
  2. Navigate behind the block (opposite side from push direction)
  3. Push the block with speed proportional to distance to target
  4. Slow down when block is close to target to avoid overshooting

Includes obstacle and wall avoidance via repulsive potential fields.
"""

import numpy as np


OBSTACLES = [
    {"pos": np.array([1.0, 0.5]), "radius": 0.6},
    {"pos": np.array([-0.8, -0.5]), "radius": 0.8},
    {"pos": np.array([0.5, -1.2]), "radius": 0.5},
]

WALL_LIMIT = 2.5
WALL_MARGIN = 0.5


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

    def _obstacle_avoidance(self, pos: np.ndarray) -> np.ndarray:
        force = np.zeros(2, dtype=np.float32)
        for obs in OBSTACLES:
            diff = pos - obs["pos"]
            d = float(np.linalg.norm(diff))
            if d < self.avoid_radius and d > 1e-6:
                force += self.avoid_gain * diff / (d * d)
        return force

    def _wall_avoidance(self, pos: np.ndarray, gain: float = 0.5) -> np.ndarray:
        force = np.zeros(2, dtype=np.float32)
        for axis in range(2):
            if pos[axis] > WALL_LIMIT - WALL_MARGIN:
                force[axis] -= gain * (pos[axis] - (WALL_LIMIT - WALL_MARGIN))
            elif pos[axis] < -WALL_LIMIT + WALL_MARGIN:
                force[axis] += gain * ((-WALL_LIMIT + WALL_MARGIN) - pos[axis])
        return force

    def __call__(self, state: np.ndarray) -> np.ndarray:
        agent_pos = state[:2]
        block_pos = state[4:6]

        dist_to_target = float(np.linalg.norm(self.target_pos - block_pos))
        if dist_to_target < 0.3:
            return np.zeros(2, dtype=np.float32)

        block_near_wall = any(
            abs(block_pos[i]) > WALL_LIMIT - WALL_MARGIN for i in range(2)
        )

        if block_near_wall:
            push_dir = self._wall_avoidance(block_pos, gain=1.0)
            push_norm = float(np.linalg.norm(push_dir))
            if push_norm > 1e-6:
                push_dir = push_dir / push_norm
            else:
                push_dir = (self.target_pos - block_pos)
                push_dir /= float(np.linalg.norm(push_dir))
        else:
            push_dir = self.target_pos - block_pos
            push_dir /= float(np.linalg.norm(push_dir))

        push_point = block_pos - push_dir * 0.35
        to_push = push_point - agent_pos
        dist_to_push = float(np.linalg.norm(to_push))

        speed_scale = min(1.0, dist_to_target / 1.5)

        if dist_to_push > 0.25:
            desired = speed_scale * to_push / max(dist_to_push, 1e-6)
        else:
            desired = speed_scale * push_dir

        desired += self._obstacle_avoidance(agent_pos)
        return np.clip(desired, -1.0, 1.0).astype(np.float32)
