"""Compose the top Push-T demo episodes into an upscaled, labeled H264 reel.

Reads the manifest written by make_demo.py, takes the top-K episodes by
coverage score, upscales each 96x96 clip to --size (lanczos), burns in a
seed/score caption, prepends a short title card, and concatenates everything
into one reel. Needs ffmpeg with libx264 + a ttf font (run on pc-3070).

The runner renders at 96x96 because render_size is coupled to the policy's
observation size; upscaling happens here in post, not in the env.

  python3 compose_reel.py --manifest demo_out/manifest.json --top 4
"""
import argparse
import json
import pathlib
import subprocess

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def esc(s):
    # ffmpeg drawtext escapes backslash, colon, quote, percent
    return (s.replace("\\", "\\\\").replace(":", "\\:")
             .replace("'", "\\'").replace("%", "\\%"))


def encode(args_list):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args_list, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="demo_out/manifest.json")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--config", default="DDIM-50 + INT8")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="episode playback speedup (setpts=PTS/speed); title card unaffected")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.manifest).parent
    out = pathlib.Path(args.out) if args.out else root / "reel.mp4"
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    top = [m for m in manifest if m.get("video")][: args.top]
    if not top:
        raise SystemExit("no episodes with a video in the manifest")

    common = ["-r", str(args.fps), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    clips = []

    if not args.no_title:
        title = root / "_title.mp4"
        line1 = esc("Diffusion Policy - Push-T")
        line2 = esc("compressed to " + args.config)
        vf = (
            f"drawtext=fontfile={FONT}:text='{line1}':fontcolor=white:fontsize=30:"
            f"x=(w-text_w)/2:y=(h-text_h)/2-24,"
            f"drawtext=fontfile={FONT}:text='{line2}':fontcolor=0xd0d0d0:fontsize=21:"
            f"x=(w-text_w)/2:y=h/2+22"
        )
        encode(["-f", "lavfi",
                "-i", f"color=c=0x14141e:s={args.size}x{args.size}:d=3:r={args.fps}",
                "-vf", vf, "-t", "3", *common, str(title)])
        clips.append(title)

    for i, m in enumerate(top):
        clip = root / f"_clip{i}.mp4"
        caption = esc(f"seed {m['seed']}    score {m['score']:.2f}")
        vf = (
            f"scale={args.size}:{args.size}:flags=lanczos,"
            f"setpts=PTS/{args.speed},"
            f"drawtext=fontfile={FONT}:text='{caption}':fontcolor=white:fontsize=19:"
            f"x=(w-text_w)/2:y=h-32:box=1:boxcolor=black@0.55:boxborderw=7"
        )
        encode(["-i", m["video"], "-vf", vf, *common, str(clip)])
        clips.append(clip)
        print(f"clip {i}: seed {m['seed']}  score {m['score']:.3f}", flush=True)

    listfile = root / "_concat.txt"
    listfile.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    encode(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)])

    for c in clips:
        c.unlink(missing_ok=True)
    listfile.unlink(missing_ok=True)
    print(f"reel written: {out}  ({len(top)} episodes)", flush=True)


if __name__ == "__main__":
    main()
