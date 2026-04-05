#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SERIES_KEYS = ("playlist_title", "series", "playlist", "album", "season", "course")
VIDEO_KEYS = ("title", "fulltitle", "episode", "id")
INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
WHITESPACE_RE = re.compile(r"\s+")
SKILL_HOME_MARKERS = (".agents", ".agent", ".codex")


def discover_skill_home() -> Path | None:
    seen: set[Path] = set()
    search_starts = [Path.cwd().resolve(), Path(__file__).resolve()]
    for start in search_starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.name in SKILL_HOME_MARKERS and candidate.is_dir():
                return candidate
            for marker in SKILL_HOME_MARKERS:
                nested = candidate / marker
                if nested.is_dir():
                    return nested.resolve()
    return None


def default_output_base() -> Path:
    skill_home = discover_skill_home()
    if skill_home is not None:
        return (skill_home.parent / "output").resolve()
    return (Path.cwd() / "output").resolve()


def sanitize_name(value: str | None) -> str:
    raw = (value or "").strip()
    raw = INVALID_CHARS_RE.sub(" - ", raw)
    raw = WHITESPACE_RE.sub(" ", raw).strip(" .")
    return raw or "untitled"


def first_nonempty(mapping: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def selected_entry(
    entries: list[object], entry_index: int | None
) -> dict[str, object] | None:
    if entry_index is None:
        return None
    if entry_index < 1 or entry_index > len(entries):
        raise ValueError(f"--entry-index must be between 1 and {len(entries)}.")

    entry = entries[entry_index - 1]
    if not isinstance(entry, dict):
        raise ValueError(f"Selected entry {entry_index} is not a metadata object.")
    return entry


def titles_from_metadata(
    metadata: dict[str, object],
    series_override: str | None,
    video_override: str | None,
    entry_index: int | None,
) -> tuple[bool, str | None, str]:
    series_title = series_override or first_nonempty(metadata, SERIES_KEYS)
    video_title = video_override

    entries = metadata.get("entries")
    if isinstance(entries, list) and entries:
        if not series_title:
            series_title = first_nonempty(
                metadata, ("title", "fulltitle", "playlist", "playlist_title")
            )
        if not video_title:
            chosen_entry = selected_entry(entries, entry_index)
            if chosen_entry is not None:
                video_title = first_nonempty(chosen_entry, VIDEO_KEYS)
            elif len(entries) == 1 and isinstance(entries[0], dict):
                video_title = first_nonempty(entries[0], VIDEO_KEYS)
            else:
                raise ValueError(
                    "Metadata looks like a series container with multiple entries. Pass --entry-index or --video-title for the selected item."
                )
    elif entry_index is not None:
        raise ValueError("--entry-index requires metadata JSON with an entries list.")

    if not video_title:
        video_title = first_nonempty(metadata, VIDEO_KEYS)

    if not video_title:
        raise ValueError(
            "Could not determine a video title. Pass --video-title explicitly."
        )

    sanitized_series = sanitize_name(series_title) if series_title else None
    sanitized_video = sanitize_name(video_title)
    if sanitized_series == sanitized_video:
        sanitized_series = None

    is_series = bool(sanitized_series)
    return is_series, sanitized_series, sanitized_video


def titles_from_local_video(
    local_video: Path,
    series_override: str | None,
    video_override: str | None,
) -> tuple[bool, str | None, str]:
    video_title = sanitize_name(video_override or local_video.stem)
    series_title = sanitize_name(series_override) if series_override else None
    return bool(series_title), series_title, video_title


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve the per-video project root for video-study-notes."
    )
    parser.add_argument(
        "--output-base",
        default=str(default_output_base()),
        help="Base output directory. Defaults to <workspace_root>/output where <workspace_root> is the parent of the nearest .agents, .agent, or .codex directory.",
    )
    parser.add_argument(
        "--metadata-json", help="Path to yt-dlp metadata JSON for a URL source."
    )
    parser.add_argument("--local-video", help="Path to a local video source.")
    parser.add_argument(
        "--series-title", help="Optional explicit series / playlist / course title."
    )
    parser.add_argument("--video-title", help="Optional explicit per-video title.")
    parser.add_argument(
        "--entry-index",
        type=int,
        help="1-based selected entry index when metadata JSON contains multiple entries.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if bool(args.metadata_json) == bool(args.local_video):
        print(
            "Provide exactly one of --metadata-json or --local-video.", file=sys.stderr
        )
        return 1

    output_base = Path(args.output_base).expanduser().resolve()

    try:
        if args.metadata_json:
            metadata_path = Path(args.metadata_json).expanduser().resolve()
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            input_kind = "url"
            is_series, series_title, video_title = titles_from_metadata(
                payload,
                args.series_title,
                args.video_title,
                args.entry_index,
            )
        else:
            local_video = Path(args.local_video).expanduser().resolve()
            if not local_video.exists():
                print(f"Local video not found: {local_video}", file=sys.stderr)
                return 1
            input_kind = "local_video"
            is_series, series_title, video_title = titles_from_local_video(
                local_video,
                args.series_title,
                args.video_title,
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to resolve project root: {exc}", file=sys.stderr)
        return 1

    project_root = (
        output_base / series_title / video_title
        if is_series and series_title
        else output_base / video_title
    )
    result = {
        "input_kind": input_kind,
        "is_series": is_series,
        "series_title": series_title,
        "video_title": video_title,
        "project_root": str(project_root),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
