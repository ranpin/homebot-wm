"""Generate TRT INT8 calibration data (.npy) for the obs_encoder + unet.

TRT native INT8 calibration (IInt8EntropyCalibrator2) needs representative
input batches. We reuse the SAME calibration distribution as the onnxruntime
PTQ that was validated accuracy-free on CPU (quantize_int8.py): real Push-T
observations for the encoder, and (randn sample, uniform timestep, encoded
global_cond) for the unet. Saved as .npy so the Orin-side calibrator (which
has no torch/zarr, only libcudart via ctypes) can load them directly.

Run on pc-3070 from /data/tmp/chenrunbin/diffusion_policy with PYTHONPATH=$PWD:
  ./dp_venv/bin/python gen_calib_data.py \
      --checkpoint data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt \
      --n_calib 128 --out_dir calib_data
"""
import argparse
import pathlib

import numpy as np
import torch
import torchvision.transforms.functional as ttf

from export_onnx import load_policy, ObsEncoderONNX, _traceable_center_crop
from quantize_int8 import load_calib_obs

ttf.center_crop = _traceable_center_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--zarr", default="data/pusht/pusht_cchi_v7_replay.zarr")
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--out_dir", default="calib_data")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg, policy = load_policy(args.checkpoint, device="cpu")
    To = int(cfg.policy.n_obs_steps)
    T = int(cfg.policy.horizon)
    Da = int(cfg.task.shape_meta.action.shape[0])

    images, agent_pos = load_calib_obs(args.zarr, args.n_calib, To)
    np.save(out / "enc_image.npy", images.astype(np.float32))       # (N,To,3,96,96)
    np.save(out / "enc_agentpos.npy", agent_pos.astype(np.float32)) # (N,To,2)

    # unet global_cond: encode the real obs with the pytorch encoder
    enc = ObsEncoderONNX(policy, To)
    enc.eval()
    gc_list = []
    with torch.no_grad():
        for i in range(len(images)):
            gc = enc(torch.from_numpy(images[i:i + 1]), torch.from_numpy(agent_pos[i:i + 1]))
            gc_list.append(gc.numpy())
    global_conds = np.concatenate(gc_list, axis=0).astype(np.float32)  # (N,132)

    rng = np.random.default_rng(1)
    samples = rng.standard_normal((len(images), T, Da)).astype(np.float32)  # (N,T,Da)
    timesteps = rng.integers(0, 100, size=(len(images),)).astype(np.int64)  # (N,)

    np.save(out / "unet_sample.npy", samples)
    np.save(out / "unet_timestep.npy", timesteps)
    np.save(out / "unet_globalcond.npy", global_conds)

    print(f"saved calib data to {out}/:")
    print(f"  enc_image       {images.shape} {images.dtype}")
    print(f"  enc_agentpos    {agent_pos.shape} {agent_pos.dtype}")
    print(f"  unet_sample     {samples.shape} {samples.dtype}")
    print(f"  unet_timestep   {timesteps.shape} {timesteps.dtype}")
    print(f"  unet_globalcond {global_conds.shape} {global_conds.dtype}")


if __name__ == "__main__":
    main()
