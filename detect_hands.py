#!/usr/bin/env python3
"""
detect_hands.py - Find frames containing hands and write detection metadata.

This is the *detection* half of the pipeline. It scans a leather-stitching (or
any close-up handcraft) video with MediaPipe's HandLandmarker and writes a JSON
metadata file describing which frames contain hands and which contiguous
segments should be kept or dropped. It does not touch the video itself - feed
the metadata to cut_video.py to produce the cleaned output with ffmpeg.

Detection uses the MediaPipe Tasks HandLandmarker model, a purpose-built
hand-landmark model, so it keys on actual hand structure rather than skin color
and won't be fooled by warm-toned leather. The model file is downloaded
automatically on first run.

Usage:
    python detect_hands.py input.mp4
    python detect_hands.py input.mp4 -o input.hands.json --confidence 0.6 --pad 3

The output JSON looks like:
    {
      "input": "input.mp4",
      "fps": 30.0,
      "frame_count": 1234,
      "width": 1920,
      "height": 1080,
      "detector": "mediapipe-hand-landmarker",
      "confidence": 0.5,
      "pad": 3,
      "frames_kept": 800,
      "frames_dropped": 434,
      "keep_segments": [[0, 119], [200, 540], ...],   # inclusive frame ranges
      "drop_segments": [[120, 199], ...]
    }
"""

import argparse
import json
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


def ensure_model(model_path=None):
    """Return a path to the hand_landmarker.task model, downloading if needed."""
    if model_path:
        if not os.path.exists(model_path):
            sys.exit(f"error: model file not found: {model_path}")
        return model_path

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"downloading hand landmarker model -> {path}")
        try:
            urllib.request.urlretrieve(MODEL_URL, path)
        except Exception as exc:  # network / permissions
            sys.exit(f"error: could not download model: {exc}\n"
                     f"Download it manually from {MODEL_URL} and pass "
                     f"--model <path>.")
    return path


class HandDetector:
    """Hand detector backed by MediaPipe Tasks HandLandmarker (VIDEO mode)."""

    def __init__(self, confidence=0.5, max_hands=2, model_path=None):
        path = ensure_model(model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def has_hand(self, frame_bgr, timestamp_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return bool(result.hand_landmarks)

    def close(self):
        self._landmarker.close()


def frames_to_segments(flags):
    """Turn a per-frame boolean list into inclusive [start, end] runs.

    Returns (keep_segments, drop_segments) where each is a list of [start, end]
    frame indices (inclusive). A "keep" run is where flag is False.
    """
    keep, drop = [], []
    if not flags:
        return keep, drop

    run_start = 0
    run_val = flags[0]
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != run_val:
            seg = [run_start, i - 1]
            (drop if run_val else keep).append(seg)
            if i < len(flags):
                run_start = i
                run_val = flags[i]
    return keep, drop


def detect(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"error: could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    detector = HandDetector(confidence=args.confidence, model_path=args.model)

    flags = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # VIDEO mode wants a monotonically increasing timestamp in ms.
        timestamp_ms = int(idx * 1000 / fps)
        flags.append(detector.has_hand(frame, timestamp_ms))
        idx += 1
        if total and idx % 30 == 0:
            print(f"\rscanning {idx}/{total} "
                  f"({100.0 * idx / total:5.1f}%)", end="", flush=True)
    print()
    cap.release()
    detector.close()

    n = len(flags)
    if n == 0:
        sys.exit("error: no frames read from input.")

    # Expand each detection by --pad frames on either side so brief in/out
    # hand movements don't leave jittery single clean frames.
    if args.pad > 0:
        padded = [False] * n
        for i, f in enumerate(flags):
            if f:
                for j in range(max(0, i - args.pad), min(n, i + args.pad + 1)):
                    padded[j] = True
        flags = padded

    keep_segments, drop_segments = frames_to_segments(flags)
    frames_dropped = sum(1 for f in flags if f)
    frames_kept = n - frames_dropped

    meta = {
        "input": os.path.abspath(args.input),
        "fps": fps,
        "frame_count": n,
        "width": width,
        "height": height,
        "detector": "mediapipe-hand-landmarker",
        "confidence": args.confidence,
        "pad": args.pad,
        "frames_kept": frames_kept,
        "frames_dropped": frames_dropped,
        "keep_segments": keep_segments,
        "drop_segments": drop_segments,
    }

    out_path = args.output or (os.path.splitext(args.input)[0] + ".hands.json")
    with open(out_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"done: {frames_kept} frames kept, {frames_dropped} dropped "
          f"({n} total) across {len(keep_segments)} keep segment(s)")
    print(f"wrote metadata: {out_path}")
    print(f"next: python cut_video.py {out_path} output.mp4")


def main():
    p = argparse.ArgumentParser(
        description="Detect hands in a video and write keep/drop metadata.")
    p.add_argument("input", help="input video file")
    p.add_argument("-o", "--output", default=None,
                   help="metadata JSON path (default: <input>.hands.json)")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="MediaPipe min detection/tracking confidence "
                        "(default: 0.5)")
    p.add_argument("--pad", type=int, default=0,
                   help="also drop N frames on each side of every detection "
                        "(default: 0)")
    p.add_argument("--model", default=None,
                   help="path to hand_landmarker.task (default: auto-download "
                        "next to this script)")
    args = p.parse_args()
    detect(args)


if __name__ == "__main__":
    main()
