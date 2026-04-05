from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_timecode(value: str) -> float:
    raw = value.strip()
    if not raw:
        raise ValueError("empty timestamp")
    if ":" not in raw:
        return float(raw)

    parts = raw.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    raise ValueError(f"unsupported timestamp format: {value}")


def load_timestamps(path: Path | None) -> list[float]:
    if path is None:
        return []
    timestamps: list[float] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        timestamps.append(parse_timecode(stripped))
    return timestamps


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def remove_matching_files(output_dir: Path, pattern: str) -> None:
    for existing in output_dir.glob(pattern):
        if existing.is_file():
            existing.unlink()


def extract_scene_frames(video: Path, output_dir: Path, threshold: float) -> int:
    remove_matching_files(output_dir, "scene-*.jpg")
    pattern = output_dir / "scene-%04d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=gt(scene\\,{threshold})",
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        str(pattern),
    ]
    run(cmd)
    return len(list(output_dir.glob("scene-*.jpg")))


def extract_timestamp_frames(video: Path, output_dir: Path, timestamps: list[float]) -> int:
    remove_matching_files(output_dir, "cue-*.jpg")
    count = 0
    for idx, seconds in enumerate(timestamps, 1):
        target = output_dir / f"cue-{idx:04d}-{seconds:09.3f}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(target),
        ]
        run(cmd)
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-notes extract-keyframes",
        description="Extract candidate keyframes for note generation.",
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", default="output/keyframes", help="Directory to store screenshots.")
    parser.add_argument("--scene-threshold", type=float, default=0.28, help="FFmpeg scene-change threshold.")
    parser.add_argument("--timestamps-file", default=None, help="Optional text file with timestamps to extract.")
    parser.add_argument("--no-scene-detect", action="store_true", help="Skip scene-change extraction.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1
    if not video.is_file():
        print(
            f"Invalid --video path (expected a file, got non-file): {video}",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_path = Path(args.timestamps_file).expanduser().resolve() if args.timestamps_file else None
    if timestamp_path:
        if not timestamp_path.exists():
            print(f"Timestamps file not found: {timestamp_path}", file=sys.stderr)
            return 1
        if not timestamp_path.is_file():
            print(
                f"Invalid --timestamps-file path (expected a file, got non-file): {timestamp_path}",
                file=sys.stderr,
            )
            return 1

    try:
        timestamps = load_timestamps(timestamp_path)
    except OSError as exc:
        print(f"Failed to read timestamps file '{timestamp_path}': {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Invalid timestamp in '{timestamp_path}': {exc}", file=sys.stderr)
        return 1

    scene_count = 0
    cue_count = 0

    try:
        if not args.no_scene_detect:
            scene_count = extract_scene_frames(video, output_dir, args.scene_threshold)

        if timestamps:
            cue_count = extract_timestamp_frames(video, output_dir, timestamps)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg and ensure it is available on PATH.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed while extracting keyframes (exit code {exc.returncode}).", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to run ffmpeg for keyframe extraction: {exc}", file=sys.stderr)
        return 1

    print(f"Output directory: {output_dir}")
    print(f"Scene frames: {scene_count}")
    print(f"Timestamp frames: {cue_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
