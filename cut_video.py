#!/usr/bin/env python3
"""
cut_video.py - Build a cleaned video from detection metadata using ffmpeg.

This is the *editing* half of the pipeline. It reads the JSON written by
detect_hands.py and uses ffmpeg to drop the hand frames, keeping only the
"keep_segments". Audio is carried along and re-synced to the kept video so the
result stays in sync.

Cutting is frame-accurate: the kept frame ranges are converted into ffmpeg
select/aselect expressions on frame number (n), then the timestamps are reset
so surviving frames play back contiguously.

Usage:
    python cut_video.py input.hands.json output.mp4
    python cut_video.py input.hands.json output.mp4 --no-audio
    python cut_video.py input.hands.json output.mp4 --crf 18 --preset slow
    python cut_video.py input.hands.json output.mp4 --pad 3   # trim hands harder
    python cut_video.py input.hands.json output.mp4 --pad -3  # keep more in/out

--pad grows (positive) or shrinks (negative) each cut-out segment at cut time,
so you can re-tune the edges without re-running detection.
"""

import argparse
import json
import shutil
import subprocess
import sys


def have_ffmpeg(name="ffmpeg"):
    return shutil.which(name) is not None


def has_audio_stream(path):
    """Return True if ffprobe reports an audio stream in `path`."""
    if not have_ffmpeg("ffprobe"):
        return False
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def build_select_expr(segments):
    """ffmpeg select expression matching frames inside any [start, end] run."""
    # between(n, start, end) is inclusive on both ends.
    return "+".join(f"between(n,{s},{e})" for s, e in segments)


def segments_to_flags(segments, n):
    """Boolean list of length n, True for frames inside any [start, end] run."""
    flags = [False] * n
    for s, e in segments:
        for i in range(max(0, s), min(n, e + 1)):
            flags[i] = True
    return flags


def adjust_flags(flags, pad):
    """Grow (pad>0) or shrink (pad<0) the True (cut-out) regions by |pad| frames.

    Positive `pad` dilates each cut-out segment outward: frames within `pad` of
    a hand frame are also dropped, so undetected "semi-frames" where a hand is
    entering or leaving get cut too.

    Negative `pad` erodes each cut-out segment inward: a frame stays dropped
    only if every frame within `|pad|` of it was also a hand frame, so the
    borderline in/out frames are kept instead.
    """
    n = len(flags)
    if pad == 0 or n == 0:
        return list(flags)

    r = abs(pad)
    out = [False] * n
    if pad > 0:
        for i, f in enumerate(flags):
            if f:
                for j in range(max(0, i - r), min(n, i + r + 1)):
                    out[j] = True
    else:
        for i, f in enumerate(flags):
            if f:
                lo, hi = max(0, i - r), min(n, i + r + 1)
                if all(flags[j] for j in range(lo, hi)):
                    out[i] = True
    return out


def flags_to_keep_segments(drop):
    """Inclusive [start, end] runs of frames where `drop` is False."""
    keep = []
    n = len(drop)
    i = 0
    while i < n:
        if not drop[i]:
            j = i
            while j < n and not drop[j]:
                j += 1
            keep.append([i, j - 1])
            i = j
        else:
            i += 1
    return keep


def apply_pad(keep, n, pad):
    """Return keep segments after growing/shrinking the cut-out gaps by `pad`.

    Works from the kept segments alone: the cut-out (hand) frames are their
    complement over [0, n), so this stays correct even if metadata is edited.
    """
    keep_flags = segments_to_flags(keep, n)
    drop = [not k for k in keep_flags]
    drop = adjust_flags(drop, pad)
    return flags_to_keep_segments(drop)


def cut(args):
    if not have_ffmpeg():
        sys.exit("error: ffmpeg not found on PATH. Install ffmpeg first.")

    with open(args.metadata) as fh:
        meta = json.load(fh)

    src = args.input or meta.get("input")
    if not src:
        sys.exit("error: metadata has no 'input' and --input not given.")

    keep = meta.get("keep_segments") or []
    if not keep:
        sys.exit("error: no keep_segments in metadata - nothing to write "
                 "(every frame had a hand?).")

    if args.pad:
        n = meta.get("frame_count") or 0
        if not n:
            for s, e in keep:
                n = max(n, e + 1)
        keep = apply_pad(keep, n, args.pad)
        if not keep:
            sys.exit("error: --pad removed every frame - nothing to write. "
                     "Try a smaller value.")

    frames_kept = sum(e - s + 1 for s, e in keep)

    vexpr = build_select_expr(keep)

    # Reset PTS so kept frames/samples play back contiguously.
    vf = f"select='{vexpr}',setpts=N/FRAME_RATE/TB"

    want_audio = (not args.no_audio) and has_audio_stream(src)

    cmd = ["ffmpeg", "-y", "-i", src]
    if want_audio:
        # Audio select must use time, not frame number. Convert each kept frame
        # range to a [start_t, end_t) time window using fps.
        fps = float(meta.get("fps") or 30.0)
        aparts = []
        for s, e in keep:
            aparts.append(f"between(t,{s / fps:.6f},{(e + 1) / fps:.6f})")
        aexpr = "+".join(aparts)
        af = f"aselect='{aexpr}',asetpts=N/SR/TB"
        cmd += ["-vf", vf, "-af", af]
    else:
        cmd += ["-vf", vf, "-an"]

    cmd += [
        "-c:v", args.vcodec,
        "-crf", str(args.crf),
        "-preset", args.preset,
        "-pix_fmt", "yuv420p",
    ]
    if want_audio:
        cmd += ["-c:a", "aac", "-b:a", args.abitrate]
    cmd.append(args.output)

    print("running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"error: ffmpeg exited with code {result.returncode}")

    print(f"wrote: {args.output} "
          f"({frames_kept} of "
          f"{meta.get('frame_count', '?')} frames kept)")


def main():
    p = argparse.ArgumentParser(
        description="Cut a video down to its hand-free frames using ffmpeg, "
                    "driven by detect_hands.py metadata.")
    p.add_argument("metadata", help="detection metadata JSON from detect_hands.py")
    p.add_argument("output", nargs="?", default="output.mp4",
                   help="output video file (default: output.mp4)")
    p.add_argument("--input", default=None,
                   help="source video (default: the path recorded in metadata)")
    p.add_argument("--pad", type=int, default=0,
                   help="grow (positive) or shrink (negative) each cut-out "
                        "(hand) segment by N frames on each side. Positive "
                        "hides undetected in/out frames near hands; negative "
                        "reveals more frames going in and out (default: 0)")
    p.add_argument("--no-audio", action="store_true",
                   help="drop audio entirely")
    p.add_argument("--vcodec", default="libx264", help="video codec (default: libx264)")
    p.add_argument("--crf", type=int, default=20,
                   help="x264 CRF quality, lower = better (default: 20)")
    p.add_argument("--preset", default="medium",
                   help="x264 preset (default: medium)")
    p.add_argument("--abitrate", default="192k",
                   help="audio bitrate (default: 192k)")
    args = p.parse_args()
    cut(args)


if __name__ == "__main__":
    main()
