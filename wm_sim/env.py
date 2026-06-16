"""MuJoCo tabletop environment for navigation + manipulation tasks."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ASSETS_DIR = Path(__file__).parent / "assets"


class HomeTabletopEnv(gym.Env):
    """Push a block to a target zone while avoiding obstacles.

    Observation:
        image: (H, W, 3) RGB from overhead camera
        state: [agent_x, agent_y, agent_vx, agent_vy, block_x, block_y]

    Action:
        2D force applied to agent (x, y), range [-2, 2]

    Reward:
        -0.1 * dist(block, target) per step, +10 bonus when block reaches target.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        image_size: int = 84,
        max_steps: int = 200,
        target_pos: np.ndarray | None = None,
        render_mode: str | None = "rgb_array",
    ):
        super().__init__()
        import mujoco

        self._mj = mujoco

        xml_path = xml_path or ASSETS_DIR / "home_tabletop.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.image_size = image_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.target_pos = target_pos if target_pos is not None else np.array([2.0, 2.0])
        self.success_threshold = 0.4

        self._renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self._camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,), dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        self._step_count = 0

    def _get_state(self) -> np.ndarray:
        agent_pos = self.data.qpos[:2]
        agent_vel = self.data.qvel[:2]
        block_pos = self.data.qpos[2:4]
        return np.concatenate([agent_pos, agent_vel, block_pos]).astype(np.float32)

    def _get_image(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._camera_id)
        return self._renderer.render()

    def _get_obs(self) -> dict[str, Any]:
        return {"image": self._get_image(), "state": self._get_state()}

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._mj.mj_resetData(self.model, self.data)

        rng = np.random.default_rng(seed)
        agent_pos = rng.uniform(-2.0, -0.5, size=2)
        block_pos = rng.uniform(-0.5, 0.5, size=2)

        self.data.qpos[0] = agent_pos[0]
        self.data.qpos[1] = agent_pos[1]
        self.data.qpos[2] = block_pos[0]
        self.data.qpos[3] = block_pos[1]

        self._mj.mj_forward(self.model, self.data)
        self._step_count = 0

        obs = self._get_obs()
        return obs, {"target_pos": self.target_pos.copy()}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:2] = action
        self._mj.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        block_pos = self.data.qpos[2:4]
        dist = float(np.linalg.norm(block_pos - self.target_pos))

        reward = -0.1 * dist
        success = dist < self.success_threshold
        if success:
            reward += 10.0

        truncated = self._step_count >= self.max_steps
        info = {"dist_to_target": dist, "success": success}

        return obs, reward, success, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._get_image()
        return None

    def close(self) -> None:
        self._renderer.close()
