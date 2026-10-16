"""INT8 PTQ for the exported ONNX components (obs_encoder + unet).

Static quantization via onnxruntime, calibrated on real Push-T observations
(encoder) and representative diffusion inputs (unet). Runs on CPU (no GPU).
Reports INT8-vs-FP32 output error so the accuracy cost is visible before
building TensorRT engines on the Orin.

Calibration note: the unet is fed randn action samples + uniform timesteps,
which covers the noisy regime of the reverse process. For tighter INT8 accuracy,
refine with forward-diffused real actions (scheduler.add_noise) per timestep.

Example:
  PYTHONPATH=$PWD python quantize_int8.py \
      --checkpoint data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt \
      --onnx_dir onnx_export --out_dir onnx_export_int8 --n_calib 128
"""
import argparse
import pathlib

import numpy as np
import torch
import dill
import zarr
import onnxruntime as ort
import torchvision.transforms.functional as ttf
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat, quant_pre_process,
)

from export_onnx import load_policy, ObsEncoderONNX, _traceable_center_crop

ttf.center_crop = _traceable_center_crop


class CalibReader(CalibrationDataReader):
    def __init__(self, samples):
        self.it = iter(samples)

    def get_next(self):
        return next(self.it, None)


def load_calib_obs(zarr_path, n, To):
    """Real obs in the format predict_action expects: image in [0,1], agent_pos raw."""
    z = zarr.open(zarr_path, mode="r")
    img = z["data/img"]      # (N, 96, 96, 3) float32 in [0,255]
    state = z["data/state"]  # (N, 5)
    N = img.shape[0]
    rng = np.random.default_rng(0)
    idxs = rng.integers(To, N - 1, size=n)
    images, agent_pos = [], []
    for i in idxs:
        frames = [np.moveaxis(np.asarray(img[i - k]), -1, 0) / 255.0
                  for k in range(To - 1, -1, -1)]
        images.append(np.stack(frames, axis=0).astype(np.float32))          # (To,3,96,96)
        apos = [np.asarray(state[i - k])[:2] for k in range(To - 1, -1, -1)]
        agent_pos.append(np.stack(apos, axis=0).astype(np.float32))          # (To,2)
    return np.array(images), np.array(agent_pos)


def run_model(model_path, feeds):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx_dir", default="onnx_export")
    ap.add_argument("--zarr", default="data/pusht/pusht_cchi_v7_replay.zarr")
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--out_dir", default="onnx_export_int8")
    ap.add_argument("--per_channel", action="store_true",
                    help="per-channel weight quantization (usually recovers 1-3 pts)")
    ap.add_argument("--op_types", default="",
                    help="comma-separated op types to quantize (e.g. 'Conv'); empty = all. "
                         "per_channel can crash on 1D weights, so restrict to Conv if so.")
    args = ap.parse_args()
    op_types = [t for t in args.op_types.split(",") if t] or None

    onnx_dir = pathlib.Path(args.onnx_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, policy = load_policy(args.checkpoint, device="cpu")
    To = int(cfg.policy.n_obs_steps)
    T = int(cfg.policy.horizon)
    Da = int(cfg.task.shape_meta.action.shape[0])

    images, agent_pos = load_calib_obs(args.zarr, args.n_calib, To)
    print(f"calib obs: images {images.shape}  agent_pos {agent_pos.shape}")

    # ---- encoder: calibrate + quantize ----
    enc_samples = [{"image": images[i:i + 1], "agent_pos": agent_pos[i:i + 1]}
                   for i in range(len(images))]
    enc_fp32 = onnx_dir / "obs_encoder.onnx"
    quant_pre_process(str(enc_fp32), str(out_dir / "obs_encoder.preproc.onnx"),
                      skip_symbolic_shape=True, auto_merge=True)
    quantize_static(str(out_dir / "obs_encoder.preproc.onnx"),
                    str(out_dir / "obs_encoder_int8.onnx"),
                    CalibReader(enc_samples),
                    quant_format=QuantFormat.QDQ, per_channel=args.per_channel,
                    weight_type=QuantType.QInt8, op_types_to_quantize=op_types)
    print("quantized obs_encoder -> obs_encoder_int8.onnx")

    # ---- unet: build global_cond with the pytorch encoder, then calibrate ----
    enc = ObsEncoderONNX(policy, To)
    enc.eval()
    gc_list = []
    with torch.no_grad():
        for i in range(len(images)):
            gc = enc(torch.from_numpy(images[i:i + 1]), torch.from_numpy(agent_pos[i:i + 1]))
            gc_list.append(gc.numpy())
    global_conds = np.concatenate(gc_list, axis=0)
    print(f"global_cond {global_conds.shape}")

    rng = np.random.default_rng(1)
    unet_samples = []
    for i in range(len(images)):
        unet_samples.append({
            "sample": rng.standard_normal((1, T, Da)).astype(np.float32),
            "timestep": np.array([rng.integers(0, 100)], dtype=np.int64),
            "global_cond": global_conds[i:i + 1].astype(np.float32),
        })
    unet_fp32 = onnx_dir / "unet.onnx"
    quant_pre_process(str(unet_fp32), str(out_dir / "unet.preproc.onnx"),
                      skip_symbolic_shape=True, auto_merge=True)
    quantize_static(str(out_dir / "unet.preproc.onnx"),
                    str(out_dir / "unet_int8.onnx"),
                    CalibReader(unet_samples),
                    quant_format=QuantFormat.QDQ, per_channel=args.per_channel,
                    weight_type=QuantType.QInt8, op_types_to_quantize=op_types)
    print("quantized unet -> unet_int8.onnx")

    # ---- INT8 vs FP32 output error (single sample) ----
    i = 0
    fp32 = run_model(enc_fp32, enc_samples[i])
    int8 = run_model(out_dir / "obs_encoder_int8.onnx", enc_samples[i])
    print(f"obs_encoder INT8-vs-FP32 max_abs_diff: {np.abs(fp32 - int8).max():.3e}")
    fp32 = run_model(unet_fp32, unet_samples[i])
    int8 = run_model(out_dir / "unet_int8.onnx", unet_samples[i])
    print(f"unet        INT8-vs-FP32 max_abs_diff: {np.abs(fp32 - int8).max():.3e}")
    print("INT8 PTQ done")


if __name__ == "__main__":
    main()
