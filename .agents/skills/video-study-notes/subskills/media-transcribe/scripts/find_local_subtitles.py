#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SUBTITLE_SUFFIXES = (".srt", ".vtt", ".ass", ".ssa", ".sub")
TEXT_SUFFIXES = (".txt",)
SUBTITLE_HINTS = ("sub", "subs", "subtitle", "subtitles", "cc")
SUBTITLE_DIR_HINTS = ("sub", "subs", "subtitle", "subtitles", "caption", "captions", "cc")
LANGUAGE_HINT_RE = r"[a-z]{2,3}(?:[-_][a-z0-9]{2,8})*"


def subtitle_stem_match(candidate_stem: str, stem: str) -> bool:
    if candidate_stem == stem:
        return True

    escaped_stem = re.escape(stem)
    hint_group = "|".join(SUBTITLE_HINTS)
    pattern = (
        rf"^{escaped_stem}"
        rf"(?:"
        rf"[._ -](?:{hint_group})(?:[._ -]{LANGUAGE_HINT_RE})?"
        rf"|"
        rf"[._ -]{LANGUAGE_HINT_RE}(?:[._ -](?:{hint_group}))?"
        rf")$"
    )
    return re.fullmatch(pattern, candidate_stem, flags=re.IGNORECASE) is not None


def sort_key(path: Path, stem: str, video_parent: Path) -> tuple[int, int, int, int, str]:
    same_dir = 0 if path.parent == video_parent else 1
    exact_stem = 0 if path.stem == stem else 1
    suffixes = SUBTITLE_SUFFIXES + TEXT_SUFFIXES
    suffix_rank = suffixes.index(path.suffix.lower()) if path.suffix.lower() in suffixes else len(suffixes)
    explicit_hint = 0 if any(token in path.stem.lower() for token in SUBTITLE_HINTS) else 1
    return (same_dir, exact_stem, suffix_rank, explicit_hint, path.name.lower())


def is_candidate_subtitle(path: Path, stem: str) -> bool:
    suffix = path.suffix.lower()
    if suffix in SUBTITLE_SUFFIXES:
        return subtitle_stem_match(path.stem, stem)
    if suffix in TEXT_SUFFIXES:
        return path.stem != stem and subtitle_stem_match(path.stem, stem)
    return False


def candidate_directories(search_dir: Path) -> list[Path]:
    directories = [search_dir.resolve()]
    for child in search_dir.iterdir():
        if child.is_dir() and any(token in child.name.lower() for token in SUBTITLE_DIR_HINTS):
            directories.append(child.resolve())
    return directories


def find_matches(video: Path, search_dir: Path) -> list[Path]:
    stem = video.stem
    seen: set[Path] = set()
    matches: list[Path] = []
    for directory in candidate_directories(search_dir):
        for candidate in directory.iterdir():
            if not candidate.is_file():
                continue
            if candidate == video:
                continue
            if not is_candidate_subtitle(candidate, stem):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(resolved)
    return sorted(matches, key=lambda path: sort_key(path, stem, video.parent.resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find sidecar subtitle text files for a local video.")
    parser.add_argument("--video", required=True, help="Path to the local video file.")
    parser.add_argument("--search-dir", help="Directory to scan. Defaults to the video's parent directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    search_dir = Path(args.search_dir).expanduser().resolve() if args.search_dir else video.parent
    if not search_dir.is_dir():
        print(f"Search directory not found: {search_dir}", file=sys.stderr)
        return 1

    matches = find_matches(video, search_dir)
    payload = {
        "video": str(video),
        "search_dir": str(search_dir),
        "matches": [str(path) for path in matches],
        "preferred": str(matches[0]) if matches else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
