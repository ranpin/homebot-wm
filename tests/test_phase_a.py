"""Phase-A regression tests: spatial encoder (A1) and dataset (A2).

Runs on CPU with a tiny synthetic HDF5 -- no real dataset, no MuJoCo, no GPU.
Covers the fast invariants; the slower end-to-end diagnostic is exercised by
scripts/diagnose_components.py itself.
"""
import h5py
import numpy as np
import pytest
import torch

from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset

IMG = 84


def _blob(block_xy, agent_xy=(-1.0, -1.0), seed=1):
    rng = np.random.default_rng(seed)
    img = np.full((IMG, IMG, 3), 30, dtype=np.uint8)
    for (x, y), color, rad in [(agent_xy, (220, 40, 40), 7), (block_xy, (40, 80, 220), 9)]:
        cx = int((x + 2.5) / 5.0 * IMG)
        cy = int((y + 2.5) / 5.0 * IMG)
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2] = color
    return np.clip(img + rng.integers(0, 8, img.shape, dtype=np.uint8), 0, 255).astype(np.uint8)


@pytest.fixture()
def synthetic_h5(tmp_path):
    """6 episodes x 24 steps; even episodes move the block (contact), odd do not."""
    path = tmp_path / "synthetic.h5"
    rng = np.random.default_rng(0)
    T = 24
    with h5py.File(path, "w") as f:
        for e in range(6):
            agent = rng.uniform(-2.0, -0.5, size=2)
            block = rng.uniform(-0.5, 0.5, size=2)
            states, images, actions = [], [], []
            for t in range(T + 1):
                states.append(np.concatenate([agent, rng.normal(0, .1, 2), block]).astype(np.float32))
                images.append(_blob(block, agent))
                if t < T:
                    a = rng.uniform(-1, 1, size=2).astype(np.float32)
                    actions.append(a)
                    agent = np.clip(agent + a * 0.15, -2.4, 2.4)
                    if e % 2 == 0:
                        block = np.clip(block + a * 0.10, -2.4, 2.4)
            g = f.create_group(f"episode_{e:05d}")
            g.create_dataset("observations/images", data=np.stack(images))
            g.create_dataset("observations/states", data=np.stack(states))
            g.create_dataset("actions", data=np.stack(actions))
            g.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))
            g.create_dataset("dones", data=np.zeros(T, dtype=bool))
            g.attrs["success"] = bool(e % 2 == 0)
            g.attrs["length"] = T
    return path


# ---------------------------------------------------------------- A1: encoder
def test_spatial_encoder_shapes():
    enc = ResNetEncoder(adapter_dim=32, output_dim=16, pretrained=False, spatial=True, image_size=IMG)
    assert enc._grid == 3
    assert enc.latent_dim == 16 * 3 * 3
    out = enc(torch.rand(2, 3, IMG, IMG))
    assert tuple(out.shape) == (2, 144)


def test_legacy_encoder_shapes():
    enc = ResNetEncoder(adapter_dim=32, output_dim=64, pretrained=False, spatial=False, image_size=IMG)
    assert enc.latent_dim == 64
    assert tuple(enc(torch.rand(2, 3, IMG, IMG)).shape) == (2, 64)


def test_from_config_roundtrip_and_legacy_default():
    spatial_cfg = {"adapter_dim": 32, "spatial": True, "spatial_out_channels": 16,
                   "image_size": IMG, "latent_dim": 144, "pretrained": False}
    legacy_cfg = {"adapter_dim": 32, "latent_dim": 64, "pretrained": False}  # no 'spatial' key
    e1 = ResNetEncoder.from_config(spatial_cfg)
    e2 = ResNetEncoder.from_config(legacy_cfg)
    assert e1.spatial and e1.latent_dim == 144
    assert not e2.spatial and e2.latent_dim == 64
    # a checkpoint saved from a spatial encoder must load back into from_config
    e1.load_state_dict(ResNetEncoder(adapter_dim=32, output_dim=16, pretrained=False,
                                     spatial=True, image_size=IMG).state_dict())


def test_spatial_latent_keeps_position():
    """Pooling destroys where the block is; flattening must not."""
    def enc_one(enc, block_xy):
        t = torch.from_numpy(_blob(block_xy)).permute(2, 0, 1).float().div_(255.0)[None]
        with torch.no_grad():
            return enc(t)

    es = ResNetEncoder(adapter_dim=32, output_dim=16, pretrained=False, spatial=True, image_size=IMG).eval()
    ep = ResNetEncoder(adapter_dim=32, output_dim=64, pretrained=False, spatial=False, image_size=IMG).eval()
    d_s = torch.norm(enc_one(es, (0., 0.)) - enc_one(es, (1.5, 1.5)))
    d_p = torch.norm(enc_one(ep, (0., 0.)) - enc_one(ep, (1.5, 1.5)))
    assert d_s > d_p


# ---------------------------------------------------------------- A2: dataset
def test_dataset_shapes_and_contact(synthetic_h5):
    ds = TrajectoryDataset(synthetic_h5)
    assert len(ds) == 6 * 24
    s = ds[0]
    assert tuple(s["image"].shape) == (3, IMG, IMG)
    assert s["image"].max() <= 1.0 and s["image"].min() >= 0.0
    assert s["contact"].dtype == torch.bool


def test_dataset_obs_horizon(synthetic_h5):
    ds = TrajectoryDataset(synthetic_h5, obs_horizon=3)
    s = ds[0]
    assert tuple(s["image"].shape) == (9, IMG, IMG)
    assert tuple(s["next_image"].shape) == (3, IMG, IMG)


def test_contact_weights_mix(synthetic_h5):
    ds = TrajectoryDataset(synthetic_h5)
    w = ds.contact_weights(contact_boost=5.0)
    assert len(w) == len(ds)
    n_contact = int((w > 1.0).sum())
    # even episodes move the block, odd don't -> expect a strict, non-trivial mix
    assert 0 < n_contact < len(ds)
