"""Compose the top Push-T demo episodes into a labeled, separated, explained reel.

v2 (per user feedback that the old reel was incomprehensible and text was clipped):
  1. an informative Chinese title card — what Push-T is, an explicit
     "no robot arm / abstract 2D benchmark" clarification, and the config;
  2. a per-episode intro card ("片段 N/4" + seed + coverage) that acts as both
     a separator and a label, so the viewer always knows which segment is playing;
  3. a persistent top banner on each episode repeating that label.
Canvas is 480x480 (was 384) so the burned-in text has room and is not clipped.
CJK text is rendered with NotoSansCJK (DejaVu has no Chinese glyphs -> tofu).

The rich task explanation + color legend + per-segment breakdown live in the
showcase page HTML *below* the video (docs/index.html DEMO tab), not burned in.

The runner renders at 96x96 because render_size is coupled to the policy's
observation size; upscaling happens here in post (lanczos), not in the env.

  python3 compose_reel.py --manifest demo_out/manifest.json --top 4
"""
import argparse
import json
import pathlib
import subprocess

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def esc(s):
    # ffmpeg drawtext escapes backslash, colon, quote, percent
    return (s.replace("\\", "\\\\").replace(":", "\\:")
             .replace("'", "\\'").replace("%", "\\%"))


def encode(args_list):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args_list, check=True)


def dt(text, size, y, color="white"):
    """One centered drawtext line."""
    return (f"drawtext=fontfile={FONT}:text='{esc(text)}':fontcolor={color}:"
            f"fontsize={size}:x=(w-text_w)/2:y={y}")


def card(size, fps, dur, lines, common, out):
    """A dark card: solid color source + stacked centered drawtext lines."""
    vf = ",".join(dt(t, s, y, c) for (t, s, y, c) in lines)
    encode(["-f", "lavfi",
            "-i", f"color=c=0x14141e:s={size}x{size}:d={dur}:r={fps}",
            "-vf", vf, *common, str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="demo_out/manifest.json")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--speed", type=float, default=2.0,
                    help="episode playback speedup (setpts=PTS/speed); cards unaffected")
    ap.add_argument("--config", default="DDIM-50 + INT8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.manifest).parent
    out = pathlib.Path(args.out) if args.out else root / "reel.mp4"
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    top = [m for m in manifest if m.get("video")][: args.top]
    if not top:
        raise SystemExit("no episodes with a video in the manifest")
    n = len(top)
    common = ["-r", str(args.fps), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    clips = []

    # --- title card: what Push-T is + the "no robot arm" clarification + config ---
    title = root / "_title.mp4"
    card(args.size, args.fps, 5.5, [
        ("Push-T · 2D 基准任务", 32, 105, "white"),
        ("质点 agent 推 T 块去对齐目标轮廓", 19, 180, "0xc8c8d0"),
        ("无机械臂 —— 抽象 2D 基准", 23, 240, "0xffd54f"),
        ("演示任务无关的压缩方法链", 17, 284, "0xc8c8d0"),
        (f"策略 = {args.config} · {args.speed:g}× 速", 20, 355, "0x9fd3ff"),
    ], common, title)
    clips.append(title)

    # --- per episode: intro card (separator + label) then the labeled clip ---
    for i, m in enumerate(top):
        idx = i + 1
        seed, score = m["seed"], m["score"]

        intro = root / f"_intro{i}.mp4"
        card(args.size, args.fps, 1.5, [
            (f"片段 {idx} / {n}", 42, 195, "white"),
            (f"seed {seed} · coverage {score:.2f}", 22, 270, "0xc8c8d0"),
        ], common, intro)
        clips.append(intro)

        clip = root / f"_clip{i}.mp4"
        banner = f"片段 {idx}/{n} · seed {seed} · coverage {score:.2f}"
        vf = (
            f"scale={args.size}:{args.size}:flags=lanczos,"
            f"setpts=PTS/{args.speed},"
            f"drawtext=fontfile={FONT}:text='{esc(banner)}':fontcolor=white:fontsize=18:"
            f"x=(w-text_w)/2:y=10:box=1:boxcolor=black@0.6:boxborderw=6"
        )
        encode(["-i", m["video"], "-vf", vf, *common, str(clip)])
        clips.append(clip)
        print(f"episode {idx}/{n}: seed {seed}  coverage {score:.3f}", flush=True)

    listfile = root / "_concat.txt"
    listfile.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    encode(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)])

    for c in clips:
        c.unlink(missing_ok=True)
    listfile.unlink(missing_ok=True)
    print(f"reel written: {out}  ({n} episodes)", flush=True)


if __name__ == "__main__":
    main()
