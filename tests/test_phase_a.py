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
    # next_image is also a k-frame window (slid forward one step) so that latent
    # and next_latent share a width -- the dynamics model needs matching dims.
    assert tuple(s["next_image"].shape) == (9, IMG, IMG)


def test_contact_weights_mix(synthetic_h5):
    ds = TrajectoryDataset(synthetic_h5)
    w = ds.contact_weights(contact_boost=5.0)
    assert len(w) == len(ds)
    n_contact = int((w > 1.0).sum())
    # even episodes move the block, odd don't -> expect a strict, non-trivial mix
    assert 0 < n_contact < len(ds)


def test_contact_sampler_actually_oversamples(synthetic_h5):
    """Constructing the sampler is not enough -- it must shift the draw distribution.

    Mirrors exactly what train.py / train_dynamics_mlp.py do: weight the full
    dataset, then index by the train Subset's own indices.
    """
    from torch.utils.data import WeightedRandomSampler, random_split

    ds = TrajectoryDataset(synthetic_h5)
    boost = 5.0
    w = ds.contact_weights(contact_boost=boost)
    n_train = int(len(ds) * 0.9)
    train_set, _ = random_split(ds, [n_train, len(ds) - n_train],
                                generator=torch.Generator().manual_seed(0))
    train_w = np.asarray(w)[train_set.indices]
    sampler = WeightedRandomSampler(
        torch.as_tensor(train_w, dtype=torch.double),
        num_samples=len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(0),
    )

    # Expectation derived from the weights themselves, not hardcoded.
    is_contact = train_w > 1.0
    expected = float(train_w[is_contact].sum() / train_w.sum())
    base = float(is_contact.mean())

    # The sampler yields indices into the train Subset (0..len(train_set)-1), so
    # look up contact via train_w, NOT the full dataset.
    drawn = np.array([bool(is_contact[i]) for i in sampler])
    empirical = float(drawn.mean())

    assert empirical > base + 0.15, f"no real shift: {empirical:.3f} vs base {base:.3f}"
    assert abs(empirical - expected) < 0.08, (
        f"empirical {empirical:.3f} != analytic {expected:.3f}")


# ------------------------------------------------- A2b: obs_horizon (temporal)
def test_obs_horizon_encoder_shapes():
    enc = ResNetEncoder(adapter_dim=32, output_dim=16, pretrained=False, spatial=True,
                        image_size=IMG, obs_horizon=2)
    assert enc.latent_dim == 16 * 3 * 3 * 2
    out = enc(torch.rand(2, 6, IMG, IMG))   # 2 frames x 3 channels
    assert tuple(out.shape) == (2, 288)


def test_obs_horizon_from_config_roundtrip():
    cfg = {"adapter_dim": 32, "spatial": True, "spatial_out_channels": 16,
           "image_size": IMG, "latent_dim": 288, "obs_horizon": 2, "pretrained": False}
    enc = ResNetEncoder.from_config(cfg)
    assert enc.obs_horizon == 2 and enc.latent_dim == 288
    # a checkpoint with no obs_horizon key must default to 1 (back-compat)
    old = {"adapter_dim": 32, "spatial": True, "spatial_out_channels": 16,
           "image_size": IMG, "latent_dim": 144, "pretrained": False}
    enc_old = ResNetEncoder.from_config(old)
    assert enc_old.obs_horizon == 1 and enc_old.latent_dim == 144


def test_dataset_obs_horizon_window_is_shifted_by_one(synthetic_h5):
    """next_image must be the image window slid forward one frame, not a single frame."""
    ds = TrajectoryDataset(synthetic_h5, obs_horizon=3)
    s = ds[5]                                  # episode 0, t=5 -> no start clamping
    assert tuple(s["image"].shape) == (9, IMG, IMG)
    assert tuple(s["next_image"].shape) == (9, IMG, IMG)
    # image = frames [t-2,t-1,t]; next_image = [t-1,t,t+1]
    # => next_image's first two frames are image's last two frames
    assert torch.equal(s["next_image"][:6], s["image"][3:9])
    # and the newest frame actually differs (the scene moved)
    assert not torch.equal(s["next_image"][6:9], s["image"][6:9])


def test_encoder_accepts_dataset_window(synthetic_h5):
    """End-to-end shape check: dataset window -> encoder -> latent of the right width."""
    ds = TrajectoryDataset(synthetic_h5, obs_horizon=2)
    enc = ResNetEncoder(adapter_dim=32, output_dim=16, pretrained=False, spatial=True,
                        image_size=IMG, obs_horizon=2).eval()
    s = ds[5]
    with torch.no_grad():
        z = enc(s["image"][None])
        zn = enc(s["next_image"][None])
    assert tuple(z.shape) == (1, enc.latent_dim)
    assert tuple(zn.shape) == tuple(z.shape)   # dynamics needs matching widths

