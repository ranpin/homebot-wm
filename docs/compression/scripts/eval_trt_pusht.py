"""Closed-loop Push-T eval of the TRT engines on the Orin (no torch).

Reproduces the diffusion_policy PushTImageRunner closed loop standalone:
MultiStepWrapper obs stacking (n_obs_steps=2) + action execution
(n_action_steps=8) + coverage scoring (max per-episode reward), driving the env
with the TRT engines via bench_predict_action.predict_action.

Faithfulness knobs (each verified, see METHODOLOGY / PROBLEMS):
  - action unnormalize = exact linear map from the checkpoint's action
    normalizer (probes matched): a_raw = a_norm*[249.5,243] + [261.5,268].
  - DDIM step uses clip_sample=True (verified vs diffusers DDIMScheduler 2.8e-08);
    alphas_cumprod + timesteps match diffusers to 1e-7.
  - env is the extracted PushTImageEnv (pymunk 7.3.1 on dog; the only patch is
    skipping the info-only collision handler removed in pymunk>=7).

Compares INT8 vs FP16 ON THE SAME platform (isolates the INT8 effect). The
absolute score may differ from the pc-3070 onnxruntime baseline (0.814) due to
pymunk version (7.3.1 vs 6.2.1) + noise realization; the FP16 run is the sanity
check, INT8-vs-FP16 is the deliverable.

Example (on dog, /home/vision/chenrunbin/homebot_wm):
  SDL_VIDEODRIVER=dummy python3 eval_trt_pusht.py --unet unet_fp16 --steps 50 --n_test 50
  SDL_VIDEODRIVER=dummy python3 eval_trt_pusht.py --unet unet_int8 --steps 50 --n_test 50

Demo recording (opt-in; the scoring path is byte-identical with or without it):
  SDL_VIDEODRIVER=dummy python3 eval_trt_pusht.py --unet unet_fp16 --steps 16 \
      --n_test 12 --init_noise torch_noise_seed0.npy --record_dir rec_fp16
Writes 512x512 PNG frames to <record_dir>/seed<N>/ plus a manifest.json holding
per-seed score / step count / reward trajectory (for labelling the reel).
"""
import argparse
import json
import os
import pathlib
import sys
import time
from collections import deque

import cv2
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(pathlib.Path(__file__).parent / "pusht_standalone"))

import pygame
pygame.init()  # headless Surface/surfarray

from bench_predict_action import (Engine, stream_create, stream_sync, predict_action,
                                  ddim_timesteps, make_alphas_cumprod)
from pusht_image_env import PushTImageEnv

# action normalizer (limits mode, [-1,1] -> pixel coords), from checkpoint probes
NORM_SCALE = np.array([249.5, 243.0], dtype=np.float32)
NORM_OFFSET = np.array([261.5, 268.0], dtype=np.float32)

N_OBS_STEPS = 2
N_ACTION_STEPS = 8
MAX_EPISODE_STEPS = 200

# --- demo recording -----------------------------------------------------------
# The env draws onto a 512x512 pygame canvas and only then downsamples to
# render_size. render_size ALSO fixes the observation shape (3,96,96) that the
# exported encoder expects, so it must stay 96 for the policy. capture_frame
# therefore swaps it to the native canvas size for the duration of one render
# call and restores it immediately -- the obs/policy path is never touched.
# render_action is disabled because the env hardcodes the marker coordinate as
# action/512*96 (correct only at render_size=96); we redraw it at the right
# scale ourselves.
RECORD_SIZE = 512
_MARKER_SIZE = int(8 / 96 * RECORD_SIZE)   # 42 -- same ratio the env uses at 96
_MARKER_THICK = int(1 / 96 * RECORD_SIZE)  # 5


def capture_frame(env):
    """High-res (RECORD_SIZE^2) RGB frame; the obs path stays 96x96.

    PushTImageEnv.render() returns a render_cache populated by _get_obs() at the
    96x96 obs resolution, so we bypass it and call PushTEnv._render_frame directly
    (PushTImageEnv does not override it): a fresh canvas resized to render_size,
    no cache, no physics step. render_action is False (set in PushTImageEnv's
    __init__), so the parent draws no action marker; we add ours at the correct
    scale below.
    """
    old_rs = env.render_size
    env.render_size = RECORD_SIZE
    try:
        img = env._render_frame("rgb_array")
    finally:
        env.render_size = old_rs
    img = np.ascontiguousarray(img)
    act = getattr(env, "latest_action", None)
    if act is not None:
        xy = np.asarray(act, dtype=np.float64).reshape(-1)[:2]
        coord = (int(np.clip(xy[0], 0, RECORD_SIZE - 1)),
                 int(np.clip(xy[1], 0, RECORD_SIZE - 1)))
        cv2.drawMarker(img, coord, color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                       markerSize=_MARKER_SIZE, thickness=_MARKER_THICK)
    return img


def save_frame(img_rgb, path):
    """Write an RGB array as PNG (cv2.imwrite expects BGR)."""
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
# -----------------------------------------------------------------------------


def stack_last_n_obs(obs_list, n):
    """MultiStepWrapper.stack_last_n_obs: stack last n obs, pad front by repeating."""
    result = {}
    for key in obs_list[-1]:
        arrs = [o[key] for o in obs_list]
        out = np.zeros((n,) + arrs[-1].shape, dtype=arrs[-1].dtype)
        k = min(n, len(arrs))
        out[n - k:] = np.array(arrs[-k:])
        if n > len(arrs):
            out[:n - k] = out[n - k]
        result[key] = out
    return result


def run_episode(seed, enc, unet, stream, ts, step_ratio, alphas_cp, noise_seed, init_noise=None,
                record_dir=None):
    env = PushTImageEnv(render_size=96)
    env.seed(seed)
    obs = env.reset()
    obs_buf = deque([obs], maxlen=N_OBS_STEPS + 1)
    rewards = []
    n_env_steps = 0
    done = False
    if record_dir is not None:
        record_dir = pathlib.Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        save_frame(capture_frame(env), record_dir / "frame_0000.png")
    while not done and n_env_steps < MAX_EPISODE_STEPS:
        stacked = stack_last_n_obs(list(obs_buf), N_OBS_STEPS)
        image = stacked["image"][None].astype(np.float32)          # (1,2,3,96,96)
        agent_pos = stacked["agent_pos"][None].astype(np.float32)  # (1,2,2)
        # OnnxPolicy reseeds the initial noise to a fixed value on EVERY call
        traj = predict_action(enc, unet, stream, image, agent_pos, ts, step_ratio,
                              alphas_cp, np.random.default_rng(noise_seed),
                              clip_sample=True, init_noise=init_noise)  # (1,16,2) normalized
        traj = traj.astype(np.float32) * NORM_SCALE + NORM_OFFSET   # unnormalize
        start = N_OBS_STEPS - 1
        actions = traj[0, start:start + N_ACTION_STEPS]             # (8,2)
        for act in actions:
            obs, reward, done, info = env.step(act)
            obs_buf.append(obs)
            rewards.append(float(reward))
            n_env_steps += 1
            if record_dir is not None:
                save_frame(capture_frame(env), record_dir / f"frame_{n_env_steps:04d}.png")
            if done or n_env_steps >= MAX_EPISODE_STEPS:
                break
    env.close()
    return (float(np.max(rewards)) if rewards else 0.0), n_env_steps, rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", default="engines")
    ap.add_argument("--unet", default="unet_int8")
    ap.add_argument("--enc", default="obs_encoder_fp16")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--test_start_seed", type=int, default=10000)
    ap.add_argument("--noise_seed", type=int, default=0)
    ap.add_argument("--init_noise", default=None,
                    help=".npy of the exact initial noise (e.g. the baseline's torch.randn seed0)")
    ap.add_argument("--record_dir", default=None,
                    help="opt-in demo recording: write 512x512 PNG frames per episode "
                         "under <dir>/seed<N>/ plus manifest.json (scoring is unaffected)")
    args = ap.parse_args()

    ed = pathlib.Path(args.engine_dir)
    stream = stream_create()
    enc = Engine(ed / f"{args.enc}.engine", stream,
                 {"image": (1, 2, 3, 96, 96), "agent_pos": (1, 2, 2)})
    unet = Engine(ed / f"{args.unet}.engine", stream,
                  {"sample": (1, 16, 2), "timestep": (1,), "global_cond": (1, 132)})
    alphas_cp = make_alphas_cumprod(100)
    ts, step_ratio = ddim_timesteps(args.steps)
    init_noise = np.load(args.init_noise) if args.init_noise else None
    rec_root = pathlib.Path(args.record_dir) if args.record_dir else None

    scores = []
    episodes = []
    t0 = time.time()
    for i in range(args.n_test):
        seed = args.test_start_seed + i
        score, nsteps, rews = run_episode(
            seed, enc, unet, stream, ts, step_ratio, alphas_cp, args.noise_seed, init_noise,
            record_dir=(rec_root / f"seed{seed}") if rec_root else None)
        scores.append(score)
        episodes.append({"seed": seed, "score": score, "n_env_steps": nsteps,
                         "rewards": [round(float(r), 4) for r in rews]})
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_test}] seed={seed} score={score:.3f} "
                  f"running_mean={np.mean(scores):.4f} ({time.time()-t0:.0f}s)", flush=True)
    mean_score = float(np.mean(scores))
    print(f"=== closed-loop | unet={args.unet} enc={args.enc} steps={args.steps} "
          f"n_test={args.n_test} seeds {args.test_start_seed}..{args.test_start_seed+args.n_test-1} ===")
    print(f"  mean_score = {mean_score:.4f}")
    print(f"  per-episode: min={min(scores):.3f} max={max(scores):.3f} std={np.std(scores):.3f}")
    print(f"  wall: {time.time()-t0:.0f}s")
    if rec_root is not None:
        rec_root.mkdir(parents=True, exist_ok=True)
        (rec_root / "manifest.json").write_text(json.dumps({
            "unet": args.unet, "enc": args.enc, "steps": args.steps,
            "init_noise": args.init_noise, "noise_seed": args.noise_seed,
            "test_start_seed": args.test_start_seed, "n_test": args.n_test,
            "mean_score": mean_score, "episodes": episodes,
        }, indent=2))
        print(f"  recorded -> {rec_root}/manifest.json ({len(episodes)} episodes)")
    enc.free(); unet.free()


if __name__ == "__main__":
    main()
