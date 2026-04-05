from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mka", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-notes prepare-audio",
        description="Prepare a deterministic local audio file for transcription.",
    )
    parser.add_argument("--input", required=True, help="Path to a local audio or video file.")
    parser.add_argument("--output-dir", required=True, help="Directory where the prepared audio file is stored.")
    parser.add_argument(
        "--audio-name",
        default=None,
        help="Optional output filename stem. Defaults to the input stem.",
    )
    parser.add_argument(
        "--audio-format",
        default="wav",
        choices=["wav", "mp3", "m4a", "flac"],
        help="Audio format to create when the input is a video file.",
    )
    return parser


def output_path(output_dir: Path, input_path: Path, args: argparse.Namespace) -> Path:
    stem = args.audio_name or input_path.stem
    suffix = input_path.suffix.lower() if input_path.suffix.lower() in AUDIO_SUFFIXES else f".{args.audio_format.lower()}"
    return output_dir / f"{stem}{suffix}"


def copy_audio(input_path: Path, target: Path) -> None:
    if input_path.resolve() == target.resolve():
        return
    shutil.copy2(input_path, target)


def extract_audio(input_path: Path, target: Path, audio_format: str) -> None:
    codec_args = {
        "wav": ["-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1"],
        "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
        "m4a": ["-c:a", "aac", "-b:a", "192k"],
        "flac": ["-c:a", "flac"],
    }[audio_format]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        *codec_args,
        str(target),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_path(output_dir, input_path, args)

    try:
        if input_path.suffix.lower() in AUDIO_SUFFIXES:
            copy_audio(input_path, target)
            action = "Copied"
        else:
            extract_audio(input_path, target, args.audio_format)
            action = "Extracted"
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed while preparing audio: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg and ensure it is on your PATH.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to prepare audio: {exc}", file=sys.stderr)
        return 1

    print(f"{action} audio: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
