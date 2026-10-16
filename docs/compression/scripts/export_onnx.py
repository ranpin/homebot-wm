"""Export the image Push-T diffusion policy to ONNX (two components).

The diffusion sampling loop (UNet x num_inference_steps) stays in the runtime;
we export the two heavy compute blocks so they can be turned into TensorRT
engines on the Orin:

  1. obs_encoder  : raw obs -> global_cond   (runs 1x per predict_action)
  2. unet         : (noisy_action, timestep, global_cond) -> noise_pred
                    (runs Nx per predict_action -- the latency bottleneck)

The observation/action normalizers are baked into the encoder wrapper so the
exported graph takes RAW observations (image in [0,1], agent_pos) and returns
global_cond directly. Action un-normalization is a cheap linear op left to the
runtime.

Runs on CPU (no GPU needed). Verifies numerical parity against PyTorch with
onnxruntime before writing.

Example:
  PYTHONPATH=$PWD python export_onnx.py \
      --checkpoint data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt \
      --out_dir onnx_export --opset 17
"""
import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
import dill
import hydra
import torchvision.transforms.functional as ttf
from diffusion_policy.common.pytorch_util import dict_apply

WS_DIR = "/tmp/dp_export_ws"


# torchvision's center_crop does int(round((h - ch)/2)) on symbolic shapes during
# ONNX tracing -> "type Tensor doesn't define __round__". Replace with a traceable
# narrow-based crop (offsets as tensor ops). The eval crop is fixed (96->84 here).
def _traceable_center_crop(img, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    ch, cw = int(output_size[0]), int(output_size[1])
    top = (img.shape[-2] - ch) // 2
    left = (img.shape[-1] - cw) // 2
    out = torch.narrow(img, -2, top, ch)
    out = torch.narrow(out, -1, left, cw)
    return out


ttf.center_crop = _traceable_center_crop


def load_policy(checkpoint, device="cpu"):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    pathlib.Path(WS_DIR).mkdir(parents=True, exist_ok=True)
    workspace = cls(cfg, output_dir=WS_DIR)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return cfg, policy


class ObsEncoderONNX(nn.Module):
    """raw obs -> global_cond (normalize + obs_encoder + reshape)."""

    def __init__(self, policy, n_obs_steps):
        super().__init__()
        self.normalizer = policy.normalizer
        self.obs_encoder = policy.obs_encoder
        self.n_obs_steps = n_obs_steps

    def forward(self, image, agent_pos):
        # image: (B, To, 3, H, W); agent_pos: (B, To, 2)
        obs_dict = {"image": image, "agent_pos": agent_pos}
        nobs = self.normalizer.normalize(obs_dict)
        B = image.shape[0]
        To = self.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)
        return global_cond


class UNetONNX(nn.Module):
    """(noisy_action, timestep, global_cond) -> noise_pred."""

    def __init__(self, policy):
        super().__init__()
        self.model = policy.model

    def forward(self, sample, timestep, global_cond):
        return self.model(sample, timestep, local_cond=None, global_cond=global_cond)


def check_ort(onnx_path, feeds, ref, name, atol=1e-4):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, feeds)[0]
    ok = np.allclose(out, ref, atol=atol)
    maxdiff = float(np.abs(out - ref).max())
    print(f"  [{name}] onnxruntime parity: {'OK' if ok else 'MISMATCH'}  max_abs_diff={maxdiff:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="onnx_export")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, policy = load_policy(args.checkpoint, device="cpu")
    To = int(cfg.policy.n_obs_steps)
    T = int(cfg.policy.horizon)
    Da = int(cfg.task.shape_meta.action.shape[0])
    B = args.batch
    print(f"loaded policy on CPU | n_obs_steps={To} horizon={T} action_dim={Da}")

    # ---- component 1: obs encoder ----
    enc = ObsEncoderONNX(policy, To)
    enc.eval()
    image = torch.rand(B, To, 3, 96, 96)
    agent_pos = torch.rand(B, To, 2)
    with torch.no_grad():
        global_cond_ref = enc(image, agent_pos)
    cond_dim = int(global_cond_ref.shape[1])
    print(f"global_cond dim = {cond_dim}")

    enc_path = out_dir / "obs_encoder.onnx"
    torch.onnx.export(
        enc, (image, agent_pos), str(enc_path),
        input_names=["image", "agent_pos"], output_names=["global_cond"],
        dynamic_axes={"image": {0: "B"}, "agent_pos": {0: "B"}, "global_cond": {0: "B"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"exported {enc_path}")

    # ---- component 2: UNet ----
    unet = UNetONNX(policy)
    unet.eval()
    sample = torch.randn(B, T, Da)
    timestep = torch.tensor([policy.num_inference_steps // 2], dtype=torch.long).expand(B)
    global_cond = global_cond_ref
    with torch.no_grad():
        noise_ref = unet(sample, timestep, global_cond)

    unet_path = out_dir / "unet.onnx"
    torch.onnx.export(
        unet, (sample, timestep, global_cond), str(unet_path),
        input_names=["sample", "timestep", "global_cond"], output_names=["noise_pred"],
        dynamic_axes={"sample": {0: "B"}, "timestep": {0: "B"},
                      "global_cond": {0: "B"}, "noise_pred": {0: "B"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"exported {unet_path}")

    # ---- parity check ----
    print("parity check (onnxruntime vs pytorch):")
    ok1 = check_ort(enc_path, {"image": image.numpy(), "agent_pos": agent_pos.numpy()},
                    global_cond_ref.numpy(), "obs_encoder")
    ok2 = check_ort(unet_path, {"sample": sample.numpy(), "timestep": timestep.numpy(),
                                "global_cond": global_cond.numpy()},
                    noise_ref.numpy(), "unet")
    if ok1 and ok2:
        print("\nALL PARITY CHECKS PASSED")
    else:
        print("\nPARITY MISMATCH -- inspect before building TensorRT")


if __name__ == "__main__":
    main()
