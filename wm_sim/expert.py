"""Scripted expert policy for data collection.

Strategy (alignment-gated pushing):
  1. Compute the push direction from block to target.
  2. The agent must sit *behind* the block (opposite the target) and be laterally
     aligned with the block→target line before pushing — otherwise contact is
     off-center and the block deflects away.
  3. If the agent is not behind/aligned, reposition to the push point, swinging
     laterally around the block when the agent is on the wrong side.
  4. Push with a controlled speed; ease off as the block nears the target.

Includes obstacle/wall repulsion for the agent.
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
        avoid_gain: float = 0.6,
        avoid_radius: float = 0.7,
        contact_offset: float = 0.32,
        stop_dist: float = 0.5,
    ):
        self.target_pos = np.asarray(target_pos, dtype=np.float32)
        self.avoid_gain = avoid_gain
        self.avoid_radius = avoid_radius
        self.contact_offset = contact_offset  # block half (0.15) + agent radius (0.15) + margin
        self.stop_dist = stop_dist            # stop pushing when block this close (< success 1.0)

    def _obstacle_avoidance(self, pos: np.ndarray) -> np.ndarray:
        force = np.zeros(2, dtype=np.float32)
        for obs in OBSTACLES:
            diff = pos - obs["pos"]
            d = float(np.linalg.norm(diff))
            if 1e-6 < d < self.avoid_radius:
                force += self.avoid_gain * diff / (d * d)
        return force

    def __call__(self, state: np.ndarray) -> np.ndarray:
        agent_pos = state[:2]
        block_pos = state[4:6]

        to_target = self.target_pos - block_pos
        dist_to_target = float(np.linalg.norm(to_target))
        if dist_to_target < self.stop_dist:
            return np.zeros(2, dtype=np.float32)

        push_dir = to_target / max(dist_to_target, 1e-6)
        tangent = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)

        # Agent position relative to the block, decomposed along / perpendicular
        # to the push direction. `along < 0` means the agent is behind the block.
        to_agent = agent_pos - block_pos
        along = float(np.dot(to_agent, push_dir))
        lateral_vec = to_agent - along * push_dir
        lateral_err = float(np.linalg.norm(lateral_vec))

        speed = min(1.0, dist_to_target / 1.5)

        if along < -0.12 and lateral_err < 0.22:
            # Behind and aligned -> push through the block center toward target.
            desired = speed * push_dir
        else:
            # Reposition to the push point behind the block.
            push_point = block_pos - push_dir * self.contact_offset
            to_point = push_point - agent_pos
            d = float(np.linalg.norm(to_point))
            desired = (to_point / max(d, 1e-6))
            if along > -0.12:
                # Agent is level with / in front of the block: swing around it
                # on the shorter side instead of shoving through it.
                side = -1.0 if float(np.dot(lateral_vec, tangent)) >= 0 else 1.0
                desired = desired + 0.8 * side * tangent
            desired = desired / max(float(np.linalg.norm(desired)), 1e-6)

        desired = desired + self._obstacle_avoidance(agent_pos)
        return np.clip(desired, -1.0, 1.0).astype(np.float32)
