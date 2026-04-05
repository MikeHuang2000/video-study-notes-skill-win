---
name: video-study-notes
description: Use this skill whenever the user wants to turn a video URL or a local video file into local study notes, including reusing existing subtitles when available, transcribing only when no usable subtitle text exists, extracting keyframes, and producing a concise Markdown learning note with embedded screenshots. This Windows-adapted version intentionally keeps only the core workflow and reads the default faster-whisper model path from `config.json`.
---

# video-study-notes

This skill is a minimal workspace-local adaptation of `video-study-notes-skill`.

It keeps the core workflow only:

- resolve a deterministic project root under `<workspace_root>/output/`
- download media or subtitles with `yt-dlp` when the source is a URL
- reuse local sidecar subtitles when the source is a local video
- fall back to local transcription only when no usable subtitle text exists
- extract candidate keyframes with `ffmpeg`
- produce one concise Markdown note using `assets/note-template.md`

This version is designed for the current Windows workspace and intentionally excludes Linux bootstrap, installer, and environment auto-configuration flows.

## Bundled helpers

- `scripts/prepare_audio.py`
- `scripts/extract_keyframes.py`
- `subskills/yt-dlp/scripts/run_yt_dlp.py`
- `subskills/yt-dlp/scripts/resolve_project_root.py`
- `subskills/media-transcribe/scripts/find_local_subtitles.py`
- `subskills/media-transcribe/scripts/transcribe_audio.py`

## Model policy

Transcription reads the default model path from:

`<skill_root>/config.json`

The current config points to:

`<workspace_root>/fast-whisper-model/faster-whisper-large-v3-turbo`

This keeps the model directory workspace-local while allowing the actual model name to change without editing the transcription script.

## Workflow

1. Determine whether the input is a URL or a local video path.
2. Resolve the project root with `subskills/yt-dlp/scripts/resolve_project_root.py` before creating output directories.
3. Create `video/`, `audio/`, `subtitles/`, `transcripts/`, and `keyframes/` under the project root.
4. Prefer existing subtitle text in this order:
   - downloaded subtitle tracks from `yt-dlp`
   - local sidecar subtitle files
   - generated `.srt` from `media-transcribe`
5. Only transcribe when no usable subtitle text exists.
6. Extract candidate screenshots into `keyframes/`.
7. Write `notes.md` in Chinese by default, using `assets/note-template.md` as the primary structure.

## Output contract

Always keep results under one per-video project root:

- `output/<Video Title>/...`
- or `output/<Series Title>/<Video Title>/...`

Expected files and directories:

- `video/`
- `audio/`
- `subtitles/`
- `transcripts/`
- `keyframes/`
- `notes.md`
- `keyframe_timestamps.txt`

## Notes writing rules

- Final note language defaults to Chinese.
- Keep technical terms, APIs, commands, and model names in English when that improves precision.
- Insert images near the paragraph they support using relative paths like `keyframes/scene-0007.jpg`.
- Do not reference missing images.
- End with a short `Takeaways` section.
