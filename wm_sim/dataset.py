"""PyTorch Dataset for HDF5 trajectory files.

Provides random access to (state, action, next_state) tuples across all episodes.
Images are loaded lazily to keep memory footprint low.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset over HDF5 trajectory files.

    Each sample returns:
        image:      (3, H, W) float32 normalized to [0, 1], or
                    (obs_horizon*3, H, W) when obs_horizon > 1 (frames
                    t-k+1..t concatenated, clamped at the episode start)
        state:      (6,) float32
        action:     (2,) float32
        next_image: (3, H, W) float32
        next_state: (6,) float32
        contact:    bool, whether the block actually moved on this transition

    Args:
        hdf5_path: Path to the trajectory HDF5 file.
        obs_horizon: Number of past frames to stack into ``image``. A single
            frame carries no velocity, so the effect of an action is invisible
            in it; k > 1 lets the latent encode motion.
        contact_eps: Block displacement above which a transition counts as
            "contact" (the action actually moved the block).
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        obs_horizon: int = 1,
        contact_eps: float = 1e-3,
    ):
        self.hdf5_path = Path(hdf5_path)
        self._file: h5py.File | None = None
        self._index: list[tuple[str, int]] = []
        self.obs_horizon = obs_horizon
        self.contact_eps = contact_eps

        with h5py.File(self.hdf5_path, "r") as f:
            for ep_key in sorted(f.keys()):
                if not ep_key.startswith("episode"):
                    continue
                ep = f[ep_key]
                length = int(ep.attrs["length"])
                for t in range(length):
                    self._index.append((ep_key, t))

    def _open_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self._index)

    def _load_image(self, ep, t: int) -> torch.Tensor:
        return torch.from_numpy(ep["observations/images"][t]).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_key, t = self._index[idx]
        f = self._open_file()
        ep = f[ep_key]

        if self.obs_horizon > 1:
            frames = [ep["observations/images"][max(t - dt, 0)]
                      for dt in range(self.obs_horizon - 1, -1, -1)]
            stacked = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            image = stacked.reshape(-1, *stacked.shape[-2:]).float() / 255.0
        else:
            image = self._load_image(ep, t)

        state = torch.from_numpy(ep["observations/states"][t])
        action = torch.from_numpy(ep["actions"][t])

        next_t = min(t + 1, ep["observations/images"].shape[0] - 1)
        next_image = self._load_image(ep, next_t)
        next_state = torch.from_numpy(ep["observations/states"][next_t])

        contact = bool(torch.norm(next_state[4:6] - state[4:6]) > self.contact_eps)

        return {
            "image": image,
            "state": state,
            "action": action,
            "next_image": next_image,
            "next_state": next_state,
            "contact": torch.tensor(contact),
        }

    def contact_weights(self, contact_boost: float = 5.0) -> np.ndarray:
        """Per-sample weights for ``WeightedRandomSampler``.

        Most transitions have the agent still navigating, so the action has no
        effect on the block. Training on them uniformly teaches the dynamics
        model that predicting "no change" is already near-optimal — model
        laziness. Upweighting contact transitions removes that shortcut.
        """
        weights = np.ones(len(self._index), dtype=np.float64)
        by_ep: dict[str, list[int]] = {}
        for i, (ep_key, _) in enumerate(self._index):
            by_ep.setdefault(ep_key, []).append(i)

        with h5py.File(self.hdf5_path, "r") as f:
            for ep_key, idxs in by_ep.items():
                block = np.asarray(f[ep_key]["observations/states"])[:, 4:6]
                for i in idxs:
                    t = self._index[i][1]
                    nt = min(t + 1, len(block) - 1)
                    if float(np.linalg.norm(block[nt] - block[t])) > self.contact_eps:
                        weights[i] = contact_boost
        return weights

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
