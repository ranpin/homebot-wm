"""TRT INT8-vs-FP16 numerical parity on the Orin (no env, no torch).

Cheap gate BEFORE the full closed-loop eval: run the SAME inputs through
unet_int8 vs unet_fp16 and obs_encoder_int8 vs obs_encoder_fp16, report output
diffs. If INT8 tracks FP16 tightly, the TRT native-calibration INT8 is
numerically faithful, so the closed-loop accuracy validated on the onnxruntime
INT8 path (README table 1/2) carries over to the deployed TRT engines.

Inputs = the real calibration obs (calib_data/). For the unet, global_cond comes
from the FP16 encoder (identical for both unets) to isolate the unet's own INT8
error from the encoder's. noise_pred is the epsilon prediction (~N(0,1)), so its
abs scale is ~1 and max_abs_diff is directly interpretable.

Example (on dog, /home/vision/homebot_wm):
  python3 parity_int8.py --n 64
"""
import argparse
import pathlib

import numpy as np

from bench_predict_action import (Engine, stream_create, stream_sync,
                                  predict_action, ddim_timesteps, make_alphas_cumprod)


def rel_max(a, b):
    """max|a-b| / max|a| -- relative error at the largest-magnitude element."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", default="engines")
    ap.add_argument("--calib_dir", default="calib_data")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps for the full-action parity")
    ap.add_argument("--n_act", type=int, default=32, help="obs count for the (slower) full-action parity")
    args = ap.parse_args()

    ed = pathlib.Path(args.engine_dir)
    cd = pathlib.Path(args.calib_dir)
    stream = stream_create()
    enc16 = Engine(ed / "obs_encoder_fp16.engine", stream,
                   {"image": (1, 2, 3, 96, 96), "agent_pos": (1, 2, 2)})
    enc8 = Engine(ed / "obs_encoder_int8.engine", stream,
                  {"image": (1, 2, 3, 96, 96), "agent_pos": (1, 2, 2)})
    u16 = Engine(ed / "unet_fp16.engine", stream,
                 {"sample": (1, 16, 2), "timestep": (1,), "global_cond": (1, 132)})
    u8 = Engine(ed / "unet_int8.engine", stream,
                {"sample": (1, 16, 2), "timestep": (1,), "global_cond": (1, 132)})

    images = np.load(cd / "enc_image.npy")[:args.n]
    apos = np.load(cd / "enc_agentpos.npy")[:args.n]
    samples = np.load(cd / "unet_sample.npy")[:args.n]
    tsteps = np.load(cd / "unet_timestep.npy")[:args.n]
    n = len(images)

    # ---- encoder parity: same obs -> global_cond ----
    enc_abs, enc_rel, gcs = [], [], []
    for i in range(n):
        enc16.set_input("image", images[i:i + 1]); enc16.set_input("agent_pos", apos[i:i + 1])
        enc16.run(); stream_sync(stream); g16 = enc16.get_output().astype(np.float32)
        enc8.set_input("image", images[i:i + 1]); enc8.set_input("agent_pos", apos[i:i + 1])
        enc8.run(); stream_sync(stream); g8 = enc8.get_output().astype(np.float32)
        enc_abs.append(np.abs(g16 - g8).ravel()); enc_rel.append(rel_max(g16, g8)); gcs.append(g16)
    enc_abs = np.concatenate(enc_abs)

    # ---- unet parity: global_cond from FP16 encoder (same for both unets) ----
    u_abs, u_rel = [], []
    for i in range(n):
        gc = gcs[i]
        for eng in (u16, u8):
            eng.set_input("global_cond", gc); eng.set_input("sample", samples[i:i + 1])
            eng.set_input("timestep", tsteps[i:i + 1])
        u16.run(); stream_sync(stream); e16 = u16.get_output().astype(np.float32)
        u8.run(); stream_sync(stream); e8 = u8.get_output().astype(np.float32)
        u_abs.append(np.abs(e16 - e8).ravel()); u_rel.append(rel_max(e16, e8))
    u_abs = np.concatenate(u_abs)

    gc_mag = float(np.abs(np.concatenate([g.ravel() for g in gcs])).mean())
    print(f"=== TRT INT8-vs-FP16 single-step parity (n={n}, real calib obs) ===")
    print(f"obs_encoder global_cond : max_abs={enc_abs.max():.4e}  mean_abs={enc_abs.mean():.4e}  "
          f"rel mean={np.mean(enc_rel):.4e}  worst={np.max(enc_rel):.4e}  (|gc| mean={gc_mag:.3f})")
    print(f"unet noise_pred (eps~N01): max_abs={u_abs.max():.4e}  mean_abs={u_abs.mean():.4e}  "
          f"rel mean={np.mean(u_rel):.4e}  worst={np.max(u_rel):.4e}")

    # ---- full-action parity: does INT8 change the DDIM trajectory OUTPUT? ----
    # Same obs + same initial noise, unet_fp16 vs unet_int8 (enc_fp16 for both).
    # This is the real gate: a small per-step eps error can accumulate over N steps.
    alphas_cp = make_alphas_cumprod(100)
    ts, step_ratio = ddim_timesteps(args.steps)
    n_act = min(args.n_act, n)
    act_abs, act_rel = [], []
    a16 = None
    for i in range(n_act):
        img = images[i:i + 1]
        ag = apos[i:i + 1]
        a16 = predict_action(enc16, u16, stream, img, ag, ts, step_ratio, alphas_cp,
                             np.random.default_rng(123)).astype(np.float32)
        a8 = predict_action(enc16, u8, stream, img, ag, ts, step_ratio, alphas_cp,
                            np.random.default_rng(123)).astype(np.float32)
        act_abs.append(np.abs(a16 - a8).ravel())
        act_rel.append(float(np.abs(a16 - a8).max() / (np.abs(a16).max() + 1e-12)))
    act_abs = np.concatenate(act_abs)
    print(f"=== full-action parity (DDIM-{args.steps}, n={n_act}, same noise seed) ===")
    print(f"action (denoised, |a| mean={np.abs(a16).mean():.3f}): max_abs={act_abs.max():.4e}  "
          f"mean_abs={act_abs.mean():.4e}  rel mean={np.mean(act_rel):.4e}  worst={np.max(act_rel):.4e}")

    enc16.free(); enc8.free(); u16.free(); u8.free()


if __name__ == "__main__":
    main()
