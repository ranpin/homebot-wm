"""Trajectory collection and HDF5 storage.

Each episode stores:
  observations/images: (T, H, W, 3) uint8
  observations/states: (T, 6) float32
  actions:             (T, 2) float32
  rewards:             (T,) float32
  dones:               (T,) bool
  success:             scalar bool
"""

from pathlib import Path

import h5py
import numpy as np


def save_trajectory(
    f: h5py.File,
    episode_id: int,
    images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
    success: bool,
) -> None:
    grp = f.create_group(f"episode_{episode_id:05d}")
    grp.create_dataset("observations/images", data=np.stack(images), compression="gzip", compression_opts=4)
    grp.create_dataset("observations/states", data=np.stack(states).astype(np.float32))
    grp.create_dataset("actions", data=np.stack(actions).astype(np.float32))
    grp.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))
    grp.create_dataset("dones", data=np.array(dones, dtype=bool))
    grp.attrs["success"] = success
    grp.attrs["length"] = len(actions)


def collect_episode(env, expert) -> dict:
    """Run one episode with the expert policy, return trajectory dict."""
    obs, info = env.reset()
    target_pos = info["target_pos"]

    images, states, actions, rewards, dones = [], [], [], [], []
    success = False

    while True:
        images.append(obs["image"])
        states.append(obs["state"])

        action = expert(obs["state"])
        obs, reward, terminated, truncated, step_info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        success = success or step_info.get("success", False)

        if terminated or truncated:
            images.append(obs["image"])
            states.append(obs["state"])
            break

    return {
        "images": images,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "success": success,
    }


def collect_dataset(
    output_path: str | Path,
    num_episodes: int = 2000,
    image_size: int = 84,
    max_steps: int = 200,
    seed: int = 42,
) -> Path:
    """Collect expert demonstrations and save to HDF5."""
    from wm_sim.env import HomeTabletopEnv
    from wm_sim.expert import ScriptedExpert

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = HomeTabletopEnv(image_size=image_size, max_steps=max_steps)
    expert = ScriptedExpert(target_pos=env.target_pos)

    rng = np.random.default_rng(seed)
    success_count = 0

    with h5py.File(output_path, "w") as f:
        f.attrs["num_episodes"] = num_episodes
        f.attrs["image_size"] = image_size
        f.attrs["max_steps"] = max_steps

        for i in range(num_episodes):
            ep_seed = int(rng.integers(0, 2**31))
            traj = collect_episode(env, expert)

            save_trajectory(
                f,
                episode_id=i,
                images=traj["images"],
                states=traj["states"],
                actions=traj["actions"],
                rewards=traj["rewards"],
                dones=traj["dones"],
                success=traj["success"],
            )

            if traj["success"]:
                success_count += 1

            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{num_episodes}] success rate: {success_count/(i+1):.2%}")

    env.close()
    print(f"Saved {num_episodes} episodes to {output_path} (success rate: {success_count/num_episodes:.2%})")
    return output_path
