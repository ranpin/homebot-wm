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
        image:      (3, H, W) float32, normalized to [0, 1]
        state:      (6,) float32
        action:     (2,) float32
        next_image: (3, H, W) float32
        next_state: (6,) float32
    """

    def __init__(self, hdf5_path: str | Path):
        self.hdf5_path = Path(hdf5_path)
        self._file: h5py.File | None = None
        self._index: list[tuple[str, int]] = []

        with h5py.File(self.hdf5_path, "r") as f:
            for ep_key in sorted(f.keys()):
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_key, t = self._index[idx]
        f = self._open_file()
        ep = f[ep_key]

        image = torch.from_numpy(ep["observations/images"][t]).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(ep["observations/states"][t])
        action = torch.from_numpy(ep["actions"][t])

        next_t = min(t + 1, ep["observations/images"].shape[0] - 1)
        next_image = torch.from_numpy(ep["observations/images"][next_t]).permute(2, 0, 1).float() / 255.0
        next_state = torch.from_numpy(ep["observations/states"][next_t])

        return {
            "image": image,
            "state": state,
            "action": action,
            "next_image": next_image,
            "next_state": next_state,
        }

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
