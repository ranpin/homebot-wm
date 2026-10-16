"""Build TensorRT engines (FP16 + INT8) from the exported ONNX components.

*** RUN THIS ON THE ORIN (aarch64). *** TensorRT engines are architecture- and
version-specific: an engine built on x86 will NOT load on the Orin, and an engine
built with one TensorRT version will not load with another. Build on the exact
target (JetPack's TensorRT).

Builds two engines per precision -- obs_encoder + unet. The diffusion sampling
loop stays OUTSIDE the engines: at inference you call the unet engine once per
denoising step (DDIM, see the n50 sweep: DDIM-50 ~= DDPM-100 at half the calls,
DDIM-5 ~= 20x fewer calls for ~8 pts). INT8 consumes the QDQ ONNX produced by
quantize_int8.py (TensorRT reads the Q/DQ nodes directly -- no separate
calibration cache needed).

UNTESTED: written while the pc-3070 GPU was down and no TensorRT was available,
so it is syntax-checked only. Expect to adjust for the Orin's TensorRT version
(JetPack 6.x = TRT 10.x) and the int64 `timestep` input (TRT prefers int32; if
the parse fails on timestep, cast it to int32 in export_onnx.py and re-export).

Example (on Orin):
  python build_trt.py --onnx_dir onnx_export --int8_dir onnx_export_int8_pc \
      --out_dir trt_engines --precisions fp16,int8 --benchmark
"""
import argparse
import pathlib
import time

import numpy as np

try:
    import tensorrt as trt
except ImportError as e:
    raise SystemExit("TensorRT not importable -- this script must run on the Orin "
                     "with JetPack's TensorRT installed.") from e

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _explicit_batch_flag():
    """TRT 8 needs the EXPLICIT_BATCH flag; TRT 10 makes it the default."""
    flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    return (1 << int(flag)) if flag is not None else 0


def build_engine(onnx_path, engine_path, precision, workspace_gb=4.0,
                 max_batch=8, verbose=False):
    onnx_path = pathlib.Path(onnx_path)
    engine_path = pathlib.Path(engine_path)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(_explicit_batch_flag())
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise RuntimeError(f"failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("  WARN: platform lacks fast FP16; building FP16 anyway")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # QDQ ONNX carries the scales; no IInt8Calibrator needed.
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision}")

    # dynamic batch profile: min=opt=1 (deployment is single-obs), max=max_batch
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        if any(d == -1 for d in shape):
            mn = [1 if d == -1 else d for d in shape]
            op = [1 if d == -1 else d for d in shape]
            mx = [max_batch if d == -1 else d for d in shape]
            profile.set_shape(inp.name, mn, op, mx)
    config.add_optimization_profile(profile)

    if verbose:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  built {engine_path.name}  precision={precision}  "
          f"{len(serialized)/1e6:.1f} MB  ({time.time()-t0:.0f}s)")
    return engine_path


def benchmark_engine(engine_path, n_iter=50, n_warmup=10):
    """Latency of one engine forward at batch=1, using torch tensors as buffers."""
    import torch
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # allocate I/O buffers from the engine's tensor specs
    bindings, shapes = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        shape = tuple(1 if d == -1 else d for d in shape)
        shapes[name] = shape
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name == "timestep":
                bindings[name] = torch.zeros(shape, dtype=torch.int64, device="cuda")
            else:
                bindings[name] = torch.randn(shape, dtype=torch.float32, device="cuda")
        else:
            bindings[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        context.set_input_shape(name, shapes[name])
        context.set_tensor_address(name, bindings[name].data_ptr())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    dt = (time.time() - t0) / n_iter * 1000
    print(f"  {pathlib.Path(engine_path).name}: {dt:.2f} ms/forward (batch=1, {n_iter} iters)")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default="onnx_export")
    ap.add_argument("--int8_dir", default="onnx_export_int8_pc")
    ap.add_argument("--out_dir", default="trt_engines")
    ap.add_argument("--precisions", default="fp16,int8")
    ap.add_argument("--workspace_gb", type=float, default=4.0)
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"TensorRT {trt.__version__}")
    onnx_dir = pathlib.Path(args.onnx_dir)
    int8_dir = pathlib.Path(args.int8_dir)
    out = pathlib.Path(args.out_dir)
    built = []

    for prec in [p.strip() for p in args.precisions.split(",") if p.strip()]:
        print(f"=== precision: {prec} ===")
        if prec == "int8":
            src_dir = int8_dir
            enc_onnx, unet_onnx = src_dir / "obs_encoder_int8.onnx", src_dir / "unet_int8.onnx"
        else:
            src_dir = onnx_dir
            enc_onnx, unet_onnx = src_dir / "obs_encoder.onnx", src_dir / "unet.onnx"
        e1 = build_engine(enc_onnx, out / f"obs_encoder_{prec}.engine", prec,
                          args.workspace_gb, args.max_batch, args.verbose)
        e2 = build_engine(unet_onnx, out / f"unet_{prec}.engine", prec,
                          args.workspace_gb, args.max_batch, args.verbose)
        built += [(prec, "obs_encoder", e1), (prec, "unet", e2)]

    if args.benchmark:
        print("=== latency (batch=1) ===")
        for prec, name, eng in built:
            benchmark_engine(eng)
        # the deployed cost is unet_engine x num_ddim_steps per action prediction
        print("\nnote: per predict_action the unet engine runs num_ddim_steps times; "
              "obs_encoder runs once. Multiply unet latency by the step count.")

    print("done")


if __name__ == "__main__":
    main()
