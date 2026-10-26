"""Compose the Orin TRT FP16-vs-INT8 closed-loop comparison reel.

Side-by-side (left=FP16, right=INT8) rollouts on the SAME env seed + SAME torch
initial noise, recorded on the Jetson AGX Orin via eval_trt_pusht.py --record_dir.
Frames are the env's native 512x512 canvas (captured without touching the 96x96
obs path), downscaled to 480 per panel here.

Three seeds are shown in BOTH directions of the per-episode chaotic swing, on
purpose: single-episode INT8-vs-FP16 diffs are dominated by chaos (+0.49 to
-1.00), and only the aggregate reveals the systematic -0.12 penalty (report
Table 5). The ending card says exactly this.

REQUIRES an ffmpeg built WITH drawtext (--enable-libfreetype). Verified
2026-10-26: pc-3070 ffmpeg 6.1.1 has drawtext/tpad/hstack/pad/overlay and
NotoSansCJK-Bold.ttc -> this is where the reel was composed. The Mac's homebrew
ffmpeg 8.0.1 has NO drawtext (no freetype/harfbuzz/fontconfig in the build) and
no PIL, so it cannot render the labels -- do not try to run it there.

Frames are produced on the Orin (dog) and shipped as a tarball, e.g.:
  tar czf orin_demo_frames.tgz rec_fp16_d50/seed{10002,10006,10011} \
      rec_int8_d50/seed{10002,10006,10011} rec_*_d50/manifest.json

  python3 compose_orin.py --root <extract_dir> --out orin_reel.mp4
"""
import argparse
import pathlib
import subprocess

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG = "0x101018"
# Orin full predict_action wall-clock at DDIM-50 (report Table 4)
FP16_LAT = "238ms · 4.2Hz"
INT8_LAT = "147ms · 6.8Hz"

SEEDS = [
    {"seed": 10006, "fp16": 1.000, "int8": 1.000, "note": "打平"},
    {"seed": 10011, "fp16": 0.974, "int8": 0.589, "note": "INT8 退化"},
    {"seed": 10002, "fp16": 0.510, "int8": 1.000, "note": "INT8 反而更好"},
]


def esc(s):
    return (s.replace("\\", "\\\\").replace(":", "\\:")
             .replace("'", "\\'").replace("%", "\\%"))


def encode(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)


def dt(text, size, x, y, color="white", box=False):
    b = ":box=1:boxcolor=black@0.6:boxborderw=5" if box else ""
    return (f"drawtext=fontfile={FONT}:text='{esc(text)}':fontcolor={color}:"
            f"fontsize={size}:x={x}:y={y}{b}")


def count_frames(d):
    return len(list(pathlib.Path(d).glob("frame_*.png")))


def card(w, h, fps, dur, lines, out):
    vf = ",".join(dt(t, s, x, y, c) for (t, s, x, y, c) in lines)
    encode(["-f", "lavfi", "-i", f"color=c={BG}:s={w}x{h}:d={dur}:r={fps}",
            "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out)])


def pair_clip(root, seed, fp16_score, int8_score, fps, speed, out):
    fp_dir = root / "rec_fp16_d50" / f"seed{seed}"
    it_dir = root / "rec_int8_d50" / f"seed{seed}"
    nfp, nit = count_frames(fp_dir), count_frames(it_dir)
    nmax = max(nfp, nit)
    fp_pad, it_pad = nmax - nfp, nmax - nit
    dur = nmax / fps
    left = f"FP16  {fp16_score:.2f}  ·  {FP16_LAT}"
    right = f"INT8  {int8_score:.2f}  ·  {INT8_LAT}"
    banner = f"seed {seed} · Orin TensorRT · DDIM-50"
    footer = "同一 seed · 同一初始噪声（torch randn）· 左右同步对照"
    fc = (
        f"[0:v]scale=480:480:flags=lanczos,fps={fps},"
        f"tpad=stop={fp_pad}:stop_mode=clone[fp];"
        f"[1:v]scale=480:480:flags=lanczos,fps={fps},"
        f"tpad=stop={it_pad}:stop_mode=clone[it];"
        f"[fp][it]hstack=inputs=2,"
        f"pad=960:560:0:50:color={BG},"
        f"{dt(banner, 21, '(w-text_w)/2', 12)},"
        f"{dt(left, 17, 14, 60, '0x9fd3ff', box=True)},"
        f"{dt(right, 17, 494, 60, '0xffb74d', box=True)},"
        f"{dt(footer, 15, '(w-text_w)/2', 532, '0xc8c8d0')},"
        f"setpts=PTS/{speed}"
    )
    encode(["-framerate", str(fps), "-i", str(fp_dir / "frame_%04d.png"),
            "-framerate", str(fps), "-i", str(it_dir / "frame_%04d.png"),
            "-filter_complex", fc,
            "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
    return dur / speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/orin_demo")
    ap.add_argument("--out", default="/tmp/orin_demo/orin_reel.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--speed", type=float, default=2.0)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out)
    W, H, fps, speed = 960, 560, args.fps, args.speed
    common_card = (W, H, fps)
    clips = []
    n = len(SEEDS)

    title = root / "_orin_title.mp4"
    card(*common_card, 6.0, [
        ("Jetson AGX Orin · 实机闭环", 34, "(w-text_w)/2", 120, "white"),
        ("TensorRT 部署 · DDIM-50 · FP16 vs INT8", 22, "(w-text_w)/2", 190, "0x9fd3ff"),
        ("同一 seed、同一初始噪声，左右分屏对照", 19, "(w-text_w)/2", 250, "0xc8c8d0"),
        ("载体 Push-T（2D 抽象基准，无机械臂）", 17, "(w-text_w)/2", 310, "0xc8c8d0"),
        (f"左 = FP16（{FP16_LAT}）   右 = INT8（{INT8_LAT}）", 18, "(w-text_w)/2", 380, "0xffd54f"),
    ], title)
    clips.append(title)

    for i, s in enumerate(SEEDS):
        idx = i + 1
        intro = root / f"_orin_intro{i}.mp4"
        card(*common_card, 1.6, [
            (f"对照 {idx} / {n} · seed {s['seed']}", 40, "(w-text_w)/2", 200, "white"),
            (f"FP16 {s['fp16']:.2f}  vs  INT8 {s['int8']:.2f}   ——  {s['note']}",
             22, "(w-text_w)/2", 280, "0xc8c8d0"),
        ], intro)
        clips.append(intro)

        clip = root / f"_orin_pair{i}.mp4"
        d = pair_clip(root, s["seed"], s["fp16"], s["int8"], fps, speed, clip)
        clips.append(clip)
        print(f"pair {idx}/{n}: seed {s['seed']}  FP16={s['fp16']:.3f} "
              f"INT8={s['int8']:.3f}  clip {d:.1f}s", flush=True)

    end = root / "_orin_end.mp4"
    card(*common_card, 7.0, [
        ("单集差异由混沌主导", 30, "(w-text_w)/2", 130, "white"),
        ("INT8 可打平、可更好、也可更差（±0.5~1.0 双向摆动）", 19, "(w-text_w)/2", 200, "0xc8c8d0"),
        ("系统性代价只在多集聚合后显现", 22, "(w-text_w)/2", 265, "0xffd54f"),
        ("TRT 原生 INT8 闭环 −0.12 vs FP16", 22, "(w-text_w)/2", 330, "0x9fd3ff"),
        ("表 5：6 噪声种子 × n=50 配对 · p<0.001", 18, "(w-text_w)/2", 390, "0xc8c8d0"),
    ], end)
    clips.append(end)

    listfile = root / "_orin_concat.txt"
    listfile.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    encode(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)])
    for c in clips:
        c.unlink(missing_ok=True)
    listfile.unlink(missing_ok=True)
    print(f"reel written: {out}  ({n} pairs)", flush=True)


if __name__ == "__main__":
    main()
