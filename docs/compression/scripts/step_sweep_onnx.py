"""CPU step sweep over the FP32 ONNX policy (Phase 1, no GPU needed).

Reuses the OnnxPolicy + runner harness from eval_onnx_pusht.py. Loads the ONNX
sessions once, then varies num_inference_steps (DDIM) to trace the
accuracy-vs-steps curve and find the knee. Same env seeds + same initial-noise
seed across all step counts, so step count is the only variable.

Example:
  PYTHONPATH=$PWD python step_sweep_onnx.py \
      --checkpoint data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt \
      --steps 50,25,16,10,5 --n_test 10
"""
import argparse
import json
import pathlib

from eval_onnx_pusht import OnnxPolicy, run_eval
from export_onnx import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx_dir", default="onnx_export")
    ap.add_argument("--steps", default="50,25,16,10,5")
    ap.add_argument("--scheduler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="step_sweep_onnx.json")
    args = ap.parse_args()

    cfg, ref = load_policy(args.checkpoint, device="cpu")
    onnx_dir = pathlib.Path(args.onnx_dir)
    step_list = [int(s) for s in args.steps.split(",")]

    # one policy object; only num_inference_steps changes between runs
    pol = OnnxPolicy(onnx_dir / "obs_encoder.onnx", onnx_dir / "unet.onnx",
                     ref, step_list[0], seed=args.seed, scheduler_type=args.scheduler)

    results = []
    for n in step_list:
        pol.num_inference_steps = n
        score = run_eval(cfg, pol, args.n_test, f"FP32 {args.scheduler}{n}")
        results.append({"steps": n, "scheduler": args.scheduler, "mean_score": score})

    print("\n=== step sweep (FP32 ONNX, %s, n_test=%d) ===" % (args.scheduler.upper(), args.n_test))
    print(f"{'steps':>6}{'mean_score':>12}")
    for r in results:
        print(f"{r['steps']:>6d}{r['mean_score']:>12.4f}")
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
