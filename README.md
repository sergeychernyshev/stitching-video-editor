# stitching-video-editor

A quick two-step pipeline for processing leather stitching (and other close-up
handcraft) videos: it automatically **removes frames that have hands in them**,
so the finished output focuses on the work, not your hands.

The work is split into a **detection** step and an **editing** step that
communicate through a small JSON metadata file:

1. **`detect_hands.py`** — scans the video with
   [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
   (a purpose-built hand-landmark model, so it isn't fooled by warm-toned
   leather) and writes a JSON file describing which frame ranges to keep/drop.
2. **`cut_video.py`** — reads that JSON and uses **ffmpeg** to produce the
   cleaned output video, frame-accurately, with audio re-synced to the kept
   frames.

Splitting it this way means you can detect once and re-cut at different quality
settings, hand-edit the keep/drop ranges in the JSON, or run the two steps on
different machines.

## Install

```bash
pip install -r requirements.txt   # opencv + mediapipe (detection)
# editing step needs ffmpeg + ffprobe on PATH:
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt install ffmpeg
```

## Usage

```bash
# 1. Detect hands -> writes input.hands.json
python detect_hands.py input.mp4

# 2. Cut the video with ffmpeg using that metadata
python cut_video.py input.hands.json output.mp4
```

### Detection options (`detect_hands.py`)

| Flag           | Description                                                       |
| -------------- | ---------------------------------------------------------------- |
| `-o/--output`  | Metadata JSON path (default `<input>.hands.json`)                |
| `--confidence` | MediaPipe min detection/tracking confidence (default `0.5`)      |
| `--pad`        | Also drop N frames on each side of every detection (default `0`) |

### Editing options (`cut_video.py`)

| Flag         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `--input`    | Source video (default: the path recorded in metadata)   |
| `--no-audio` | Drop audio entirely                                     |
| `--vcodec`   | Video codec (default `libx264`)                         |
| `--crf`      | x264 quality, lower = better (default `20`)             |
| `--preset`   | x264 preset (default `medium`)                          |
| `--abitrate` | Audio bitrate (default `192k`)                          |

## Metadata format

`detect_hands.py` writes something like:

```json
{
  "input": "/abs/path/input.mp4",
  "fps": 30.0,
  "frame_count": 1234,
  "width": 1920,
  "height": 1080,
  "detector": "mediapipe",
  "confidence": 0.5,
  "pad": 0,
  "frames_kept": 800,
  "frames_dropped": 434,
  "keep_segments": [[0, 119], [200, 540]],
  "drop_segments": [[120, 199]]
}
```

`keep_segments` / `drop_segments` are inclusive `[start, end]` frame ranges.
You can edit these by hand before running `cut_video.py` to fine-tune the cut.

## Notes

- Dropping frames mid-clip means the kept audio is stitched together too; it
  stays in sync with the kept video but skips along with it. Use `--no-audio`
  if you'd rather add a soundtrack later.
- `cut_video.py` re-encodes (frame-accurate cuts can't be done losslessly).
  Lower `--crf` for higher quality, raise it for smaller files.
