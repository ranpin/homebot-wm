"""Run the Push-T eval with the ONNX (FP32) and INT8 policies on CPU.

The GPU is down, but PushTImageEnv is a pymunk (CPU) sim and onnxruntime runs
on the CPUExecutionProvider, so the INT8 accuracy drop can be measured without a
GPU. Both variants use the SAME DDIM sampler at the SAME step count and the SAME
env seeds + initial-noise seed, so the score difference isolates the INT8 PTQ
effect (not the step-count effect).

The obs_encoder ONNX has the observation normalizer baked in; action
un-normalization and the diffusion sampling loop run outside the graph.

Example:
  PYTHONPATH=$PWD python eval_onnx_pusht.py \
      --checkpoint data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt \
      --onnx_dir onnx_export --int8_dir onnx_export_int8 \
      --steps 16 --n_test 10
"""
import argparse
import pathlib

import numpy as np
import torch
import dill
import hydra
import onnxruntime as ort
import torchvision.transforms.functional as ttf
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from export_onnx import load_policy, _traceable_center_crop

ttf.center_crop = _traceable_center_crop


class OnnxPolicy:
    """Mimics the DiffusionUnetHybridImagePolicy interface using ONNX sessions."""

    def __init__(self, enc_path, unet_path, ref_policy, num_inference_steps, seed=0,
                 scheduler_type="ddim"):
        opts = ort.SessionOptions()
        self.enc = ort.InferenceSession(str(enc_path), opts, providers=["CPUExecutionProvider"])
        self.unet = ort.InferenceSession(str(unet_path), opts, providers=["CPUExecutionProvider"])
        self.ref = ref_policy
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.scheduler_type = scheduler_type
        if scheduler_type == "ddpm":
            # the original stochastic DDPM sampler the model was trained with
            import copy
            self.scheduler = copy.deepcopy(ref_policy.noise_scheduler)
        else:
            # DDIM sampler reusing the trained (epsilon) weights
            c = ref_policy.noise_scheduler.config
            self.scheduler = DDIMScheduler(
                beta_start=c.beta_start, beta_end=c.beta_end, beta_schedule=c.beta_schedule,
                clip_sample=c.clip_sample, set_alpha_to_one=True, steps_offset=0,
                prediction_type=c.prediction_type, num_train_timesteps=c.num_train_timesteps,
            )

    def reset(self):
        pass

    def predict_action(self, obs_dict):
        ref = self.ref
        image = obs_dict["image"]          # (B, To, 3, 96, 96) in [0,1]
        agent_pos = obs_dict["agent_pos"]  # (B, To, 2)
        B = image.shape[0]
        T = ref.horizon
        Da = ref.action_dim

        global_cond = self.enc.run(None, {
            "image": image.numpy(), "agent_pos": agent_pos.numpy()})[0]

        g = torch.Generator().manual_seed(self.seed)
        trajectory = torch.randn(B, T, Da, generator=g)
        self.scheduler.set_timesteps(self.num_inference_steps)
        for t in self.scheduler.timesteps:
            ts = np.full((B,), int(t), dtype=np.int64)
            noise_pred = self.unet.run(None, {
                "sample": trajectory.numpy(), "timestep": ts,
                "global_cond": global_cond})[0]
            if self.scheduler_type == "ddpm":
                # DDPM step is stochastic; pass the seeded generator so the run
                # is reproducible and paired with the DDIM run.
                trajectory = self.scheduler.step(
                    torch.from_numpy(noise_pred), int(t), trajectory, generator=g).prev_sample
            else:
                trajectory = self.scheduler.step(
                    torch.from_numpy(noise_pred), int(t), trajectory).prev_sample

        naction = trajectory[..., :Da]
        action_pred = ref.normalizer["action"].unnormalize(naction)
        start = ref.n_obs_steps - 1
        end = start + ref.n_action_steps
        return {"action": action_pred[:, start:end], "action_pred": action_pred}


def run_eval(cfg, policy, n_test, tag):
    out = pathlib.Path("/tmp/dp_onnx_eval")
    (out / "media").mkdir(parents=True, exist_ok=True)
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out),
                                     n_test=n_test, n_envs=n_test)
    log = runner.run(policy)
    score = log.get("test/mean_score", float("nan"))
    print(f"[{tag}] test/mean_score = {score:.4f}", flush=True)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx_dir", default="onnx_export")
    ap.add_argument("--int8_dir", default="onnx_export_int8")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_fp32", action="store_true")
    args = ap.parse_args()

    cfg, ref_policy = load_policy(args.checkpoint, device="cpu")
    onnx_dir = pathlib.Path(args.onnx_dir)
    int8_dir = pathlib.Path(args.int8_dir)
    results = {}

    if not args.skip_fp32:
        fp32 = OnnxPolicy(onnx_dir / "obs_encoder.onnx", onnx_dir / "unet.onnx",
                          ref_policy, args.steps, seed=args.seed)
        results["fp32"] = run_eval(cfg, fp32, args.n_test, f"FP32 ddim{args.steps}")

    int8 = OnnxPolicy(int8_dir / "obs_encoder_int8.onnx", int8_dir / "unet_int8.onnx",
                      ref_policy, args.steps, seed=args.seed)
    results["int8"] = run_eval(cfg, int8, args.n_test, f"INT8 ddim{args.steps}")

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    if "fp32" in results and "int8" in results:
        print(f"  INT8 drop vs FP32: {results['fp32'] - results['int8']:+.4f}")


if __name__ == "__main__":
    main()
