"""Full predict_action WALL-CLOCK benchmark on the Orin (no torch).

Upgrades the trtexec engine-only numbers to a true end-to-end measurement:
loads the obs_encoder + unet TRT engines and runs the COMPLETE diffusion-policy
predict_action -- encode the observation once, then the DDIM denoising loop
(unet engine xN, with the numpy DDIM step + host<->device transfers between
calls) -- and times the wall clock.

Device memory via ctypes + libcudart (the Orin has no torch/pycuda/cuda-python).
DDIM math (squaredcos_cap_v2 alphas_cumprod, eta=0 step) is replicated in numpy
from the diffusers DDIMScheduler, so the loop is faithful (not engine spam).

Boundary: measures policy inference (encode + denoise loop + transfers + DDIM
math). Excludes upstream image preprocessing (center-crop/normalize) and the
trivial action slice. global_cond is uploaded ONCE (constant across steps);
sample ping-pongs host<->device each step because the DDIM update runs in numpy.

Example (on dog, /home/vision/homebot_wm):
  python3 bench_predict_action.py --unet unet_int8 --enc obs_encoder_fp16 --steps 50
"""
import argparse
import ctypes
import math
import pathlib
import time

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
cudart = ctypes.CDLL("libcudart.so")

TRT_TO_NP = {trt.float32: np.float32, trt.float16: np.float16,
             trt.int32: np.int32, trt.int64: np.int64, trt.bool: np.bool_}


def _ck(err, fn):
    if err != 0:
        raise RuntimeError(f"{fn} failed: cudaError={err}")


def cuda_malloc(n):
    p = ctypes.c_void_p()
    _ck(cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMalloc")
    return p.value


def cuda_free(a):
    _ck(cudart.cudaFree(ctypes.c_void_p(a)), "cudaFree")


def h2d(dst, src):
    src = np.ascontiguousarray(src)
    _ck(cudart.cudaMemcpy(ctypes.c_void_p(dst), src.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(src.nbytes), 1), "h2d")  # 1=HostToDevice


def d2h(src, shape, dtype):
    out = np.empty(shape, dtype=dtype)
    _ck(cudart.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src),
                          ctypes.c_size_t(out.nbytes), 2), "d2h")  # 2=DeviceToHost
    return out


def stream_create():
    s = ctypes.c_void_p()
    _ck(cudart.cudaStreamCreate(ctypes.byref(s)), "streamCreate")
    return s.value


def stream_sync(s):
    _ck(cudart.cudaStreamSynchronize(ctypes.c_void_p(s)), "streamSync")


# ---- DDIM schedule (diffusers DDIMScheduler, squaredcos_cap_v2, eta=0) ----
def make_alphas_cumprod(n=100, max_beta=0.999):
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = np.array([min(1 - alpha_bar((i + 1) / n) / alpha_bar(i / n), max_beta)
                      for i in range(n)], dtype=np.float64)
    return np.cumprod(1.0 - betas)


def ddim_timesteps(num_inference, num_train=100):
    step_ratio = num_train // num_inference
    ts = (np.arange(0, num_inference) * step_ratio).round()[::-1].astype(np.int64)
    return ts, step_ratio


class Engine:
    """One TRT engine with pre-allocated device buffers (batch=1, static shapes)."""

    def __init__(self, path, stream, input_shapes):
        rt = trt.Runtime(TRT_LOGGER)
        with open(path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"deserialize failed: {path}")
        self.context = self.engine.create_execution_context()
        self.stream = stream
        self.in_shapes = dict(input_shapes)
        self.dev, self.dtype = {}, {}
        for name, shape in input_shapes.items():
            self.context.set_input_shape(name, shape)
            dt = TRT_TO_NP.get(self.engine.get_tensor_dtype(name), np.float32)
            self.dtype[name] = dt
            self.dev[name] = cuda_malloc(int(np.prod(shape)) * np.dtype(dt).itemsize)
            self.context.set_tensor_address(name, self.dev[name])
        self.out_name, self.out_shape = None, None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                if self.out_name is not None:
                    print(f"  WARN: {path} has >1 output; using {self.out_name}")
                    continue
                self.out_name = name
                self.out_shape = tuple(self.context.get_tensor_shape(name))
                dt = TRT_TO_NP.get(self.engine.get_tensor_dtype(name), np.float32)
                self.dtype[name] = dt
                self.dev[name] = cuda_malloc(int(np.prod(self.out_shape)) * np.dtype(dt).itemsize)
                self.context.set_tensor_address(name, self.dev[name])
        if self.out_name is None:
            raise RuntimeError(f"no output tensor found in {path}")

    def set_input(self, name, arr):
        h2d(self.dev[name], np.asarray(arr, dtype=self.dtype[name]))

    def run(self):
        self.context.execute_async_v3(self.stream)

    def get_output(self):
        return d2h(self.dev[self.out_name], self.out_shape, self.dtype[self.out_name])

    def free(self):
        for a in self.dev.values():
            cuda_free(a)
        self.dev = {}


def predict_action(enc, unet, stream, image, agent_pos, ts, step_ratio, alphas_cp, rng,
                   phases=None):
    """One full policy forward: encode obs once, then the DDIM denoise loop.

    If `phases` (a dict) is given, accumulate per-phase wall-clock (seconds) into
    it so the orchestration overhead can be separated from the engine compute.
    """
    pc = time.perf_counter

    def acc(k, dt):
        if phases is not None:
            phases[k] = phases.get(k, 0.0) + dt

    sample_shape = unet.in_shapes["sample"]
    # 1. encode observation (runs ONCE per predict_action)
    t0 = pc()
    enc.set_input("image", image)
    enc.set_input("agent_pos", agent_pos)
    acc("enc_h2d", pc() - t0)
    t0 = pc(); enc.run(); stream_sync(stream); acc("enc_run", pc() - t0)
    t0 = pc()
    global_cond = enc.get_output()                 # [1,132] on host
    unet.set_input("global_cond", global_cond)     # constant across steps -> upload once
    acc("enc_d2h", pc() - t0)
    # 2. DDIM denoising loop (unet runs xN)
    sample = rng.standard_normal(sample_shape).astype(np.float32)
    for t in ts:
        t0 = pc()
        unet.set_input("sample", sample)
        unet.set_input("timestep", np.array([t], dtype=np.int64))
        acc("step_h2d", pc() - t0)
        t0 = pc(); unet.run(); stream_sync(stream); acc("step_run", pc() - t0)
        t0 = pc(); eps = unet.get_output().astype(np.float32); acc("step_d2h", pc() - t0)
        t0 = pc()
        t_prev = int(t) - step_ratio
        a_t = np.float32(alphas_cp[int(t)])
        a_prev = np.float32(alphas_cp[t_prev]) if t_prev >= 0 else np.float32(1.0)
        x0 = (sample - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
        sample = (np.sqrt(a_prev) * x0 + np.sqrt(1.0 - a_prev) * eps).astype(np.float32)
        acc("step_numpy", pc() - t0)
    return sample                                   # action [1,16,2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", default="engines")
    ap.add_argument("--unet", default="unet_int8")
    ap.add_argument("--enc", default="obs_encoder_fp16")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--calib_dir", default="calib_data")
    ap.add_argument("--diag", action="store_true",
                    help="run one instrumented predict_action and print the per-phase breakdown")
    ap.add_argument("--tight", action="store_true",
                    help="also run a back-to-back unet exec loop (no memcpy/numpy gaps) "
                         "to compare against trtexec plain on the same process")
    args = ap.parse_args()

    ed = pathlib.Path(args.engine_dir)
    stream = stream_create()
    enc = Engine(ed / f"{args.enc}.engine", stream,
                 {"image": (1, 2, 3, 96, 96), "agent_pos": (1, 2, 2)})
    unet = Engine(ed / f"{args.unet}.engine", stream,
                  {"sample": (1, 16, 2), "timestep": (1,), "global_cond": (1, 132)})
    print(f"loaded {args.enc} (out {enc.out_name}{enc.out_shape}) + "
          f"{args.unet} (out {unet.out_name}{unet.out_shape})")

    cd = pathlib.Path(args.calib_dir)
    image = np.load(cd / "enc_image.npy")[0:1]        # (1,2,3,96,96) real obs
    agent_pos = np.load(cd / "enc_agentpos.npy")[0:1]  # (1,2,2)
    alphas_cp = make_alphas_cumprod(100)
    ts, step_ratio = ddim_timesteps(args.steps)
    rng = np.random.default_rng(0)

    def one():
        return predict_action(enc, unet, stream, image, agent_pos,
                              ts, step_ratio, alphas_cp, rng)

    # warm up FIRST so --diag / --tight reflect steady state, not cold start
    for _ in range(args.warmup):
        one()
    stream_sync(stream)

    if args.tight:
        # back-to-back unet.run()+sync with NO memcpy/numpy gaps -- directly
        # comparable to trtexec plain. Isolates pure engine exec speed in this
        # process from the real loop's orchestration gaps. If tight << the real
        # loop's per-step, the gap is DVFS/contention during the inter-step idle.
        enc.set_input("image", image)
        enc.set_input("agent_pos", agent_pos)
        enc.run(); stream_sync(stream)
        unet.set_input("global_cond", enc.get_output())
        unet.set_input("sample", np.zeros(unet.in_shapes["sample"], np.float32))
        unet.set_input("timestep", np.array([int(ts[0])], dtype=np.int64))
        for _ in range(10):
            unet.run(); stream_sync(stream)
        N = max(args.iters, 1) * args.steps
        t0 = time.perf_counter()
        for _ in range(N):
            unet.run(); stream_sync(stream)
        dt = (time.perf_counter() - t0) / N * 1000.0
        print(f"--- tight unet loop (run+sync, no gaps): {dt:.3f} ms/exec "
              f"over {N} execs (trtexec plain was 2.43ms) ---")

    if args.diag:
        phases = {}
        predict_action(enc, unet, stream, image, agent_pos,
                       ts, step_ratio, alphas_cp, rng, phases)
        stream_sync(stream)
        total = sum(phases.values()) * 1000.0
        print(f"--- per-phase breakdown (steps={args.steps}, WARM 1 predict_action) ---")
        for k in ("enc_h2d", "enc_run", "enc_d2h",
                  "step_h2d", "step_run", "step_d2h", "step_numpy"):
            v = phases.get(k, 0.0) * 1000.0
            if k.startswith("step"):
                print(f"  {k:12s} total={v:8.2f}ms  per-step={v / args.steps:6.3f}ms")
            else:
                print(f"  {k:12s} total={v:8.2f}ms")
        print(f"  SUM={total:.2f}ms")

    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        act = one()
        stream_sync(stream)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    hz = 1000.0 / times.mean()
    print(f"=== predict_action wall-clock | unet={args.unet} enc={args.enc} steps={args.steps} ===")
    print(f"  mean={times.mean():.2f}ms  median={np.median(times):.2f}ms  "
          f"min={times.min():.2f}  max={times.max():.2f}  ->  {hz:.1f} Hz")
    print(f"  action: shape={act.shape} finite={bool(np.isfinite(act).all())} "
          f"min={act.min():.3f} max={act.max():.3f} mean={act.mean():.3f}")
    enc.free()
    unet.free()


if __name__ == "__main__":
    main()
