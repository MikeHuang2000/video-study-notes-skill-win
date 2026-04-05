#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import NoReturn


SKILL_ROOT = Path(__file__).resolve().parents[3]
SKILL_VENV_PYTHON = SKILL_ROOT / ".venv" / "bin" / "python"
SKILL_COOKIES = SKILL_ROOT / "cookies.txt"
INFO_ONLY_FLAGS = {"--help", "-h", "--version", "-U", "--update", "--rm-cache-dir"}


def exec_program(executable: Path | str, argv: list[str]) -> NoReturn:
    os.execv(str(executable), [str(executable), *argv])


def exec_python_module(python_bin: Path | str, argv: list[str]) -> NoReturn:
    python_text = str(python_bin)
    os.execv(python_text, [python_text, "-m", "yt_dlp", *argv])


def is_info_only(argv: list[str]) -> bool:
    return bool(argv) and not any("://" in arg for arg in argv) and any(arg in INFO_ONLY_FLAGS for arg in argv)


def maybe_add_cookies(argv: list[str]) -> list[str]:
    if not SKILL_COOKIES.is_file():
        return argv

    cookie_flags = {"--cookies", "--cookies-from-browser", "--netrc", "--netrc-cmd"}
    if any(arg in cookie_flags for arg in argv):
        return argv

    if is_info_only(argv):
        return argv

    return ["--cookies", str(SKILL_COOKIES), *argv]


def main() -> int:
    argv = maybe_add_cookies(sys.argv[1:])

    override = os.environ.get("VIDEO_NOTES_YT_DLP")
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file():
            exec_program(override_path, argv)
        print(f"VIDEO_NOTES_YT_DLP points to a missing executable: {override_path}", file=sys.stderr)
        return 1

    try:
        __import__("yt_dlp")
    except ImportError:
        pass
    else:
        exec_python_module(sys.executable, argv)

    if SKILL_VENV_PYTHON.is_file() and Path(sys.executable) != SKILL_VENV_PYTHON:
        exec_python_module(SKILL_VENV_PYTHON, argv)

    path_ytdlp = shutil.which("yt-dlp")
    if path_ytdlp:
        exec_program(path_ytdlp, argv)

    print(
        "yt-dlp is not available. Run `bash scripts/bootstrap_linux.sh` from the skill root, "
        "or set VIDEO_NOTES_YT_DLP to the executable path.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
