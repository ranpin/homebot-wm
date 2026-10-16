"""Generate Push-T demo videos of the compressed (DDIM + INT8) policy.

Runs the INT8 ONNX policy over a set of candidate env seeds — one episode per
seed, recorded to H264 by the PushTImageRunner's built-in VideoRecorder — and
writes a manifest mapping seed -> coverage score -> video path. A follow-up
ffmpeg step (see repo docs) upscales + labels + concatenates the top episodes
into a showcase reel.

The runner couples render_size to the 96x96 observation the policy was trained
on, so videos come out at 96x96; upscale in post (do NOT raise render_size here
or the obs shape no longer matches the exported encoder).

Run on pc-3070 from /data/tmp/chenrunbin/diffusion_policy with PYTHONPATH=$PWD:
  ./dp_venv/bin/python make_demo.py \
      --checkpoint data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt \
      --int8_dir onnx_export_int8 --steps 50
"""
import argparse
import json
import pathlib

import hydra

from eval_onnx_pusht import OnnxPolicy
from export_onnx import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--int8_dir", default="onnx_export_int8")
    ap.add_argument("--steps", type=int, default=50,
                    help="DDIM inference steps (50 = recommended deploy config)")
    ap.add_argument("--seeds", default=",".join(str(10000 + i) for i in range(12)),
                    help="comma-separated env seeds to try (one episode each)")
    ap.add_argument("--out_dir", default="demo_out")
    args = ap.parse_args()

    cfg, ref_policy = load_policy(args.checkpoint, device="cpu")
    int8_dir = pathlib.Path(args.int8_dir)
    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # One shared policy object; the diffusion initial-noise seed is fixed (0) so
    # episode differences come only from the env seed (initial block/agent pose).
    policy = OnnxPolicy(int8_dir / "obs_encoder_int8.onnx", int8_dir / "unet_int8.onnx",
                        ref_policy, args.steps, seed=0, scheduler_type="ddim")

    manifest = []
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for k, seed in enumerate(seeds):
        out = out_root / f"seed{seed}"
        (out / "media").mkdir(parents=True, exist_ok=True)
        # n_test=1 + n_test_vis=1 -> exactly one recorded episode per run, so the
        # single mp4 in out/media unambiguously belongs to `seed`.
        runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=str(out),
            n_train=0, n_train_vis=0, n_test=1, n_test_vis=1, n_envs=1,
            test_start_seed=seed)
        log = runner.run(policy)
        score = float(log.get("test/mean_score", float("nan")))
        vids = sorted((out / "media").glob("*.mp4"))
        video = str(vids[0]) if vids else None
        manifest.append({"seed": seed, "score": score, "video": video})
        print(f"[{k+1}/{len(seeds)}] seed {seed}: score={score:.4f} video={video}",
              flush=True)

    # Merge with any existing manifest so incremental seed runs accumulate
    # (dedupe by seed, keep the latest score for a re-run seed).
    mpath = out_root / "manifest.json"
    merged = {}
    if mpath.exists():
        for m in json.loads(mpath.read_text()):
            merged[m["seed"]] = m
    for m in manifest:
        merged[m["seed"]] = m
    manifest = sorted(merged.values(), key=lambda m: -m["score"])
    mpath.write_text(json.dumps(manifest, indent=2))
    print("\n=== ranked episodes (best first) ===")
    for m in manifest:
        print(f"  seed {m['seed']}: {m['score']:.4f}  {m['video']}")


if __name__ == "__main__":
    main()
