#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
SKILL_ROOT = WORKSPACE_ROOT / ".agents" / "skills" / "video-study-notes"
CONFIG_PATH = SKILL_ROOT / "config.json"


def load_skill_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Skill config not found: {CONFIG_PATH}. Create it before running transcription."
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def default_model_path() -> Path:
    config = load_skill_config()
    model_rel = config.get("default_model")
    if not isinstance(model_rel, str) or not model_rel.strip():
        raise ValueError(f"Invalid 'default_model' in skill config: {CONFIG_PATH}")
    return (WORKSPACE_ROOT / model_rel).resolve()


def transcribe_defaults() -> dict:
    config = load_skill_config()
    defaults = config.get("transcribe_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(
            f"Invalid 'transcribe_defaults' in skill config: {CONFIG_PATH}"
        )
    return defaults


def add_windows_nvidia_dll_dirs() -> None:
    if os.name != "nt":
        return

    candidates: list[Path] = []
    for root in (Path(sys.prefix), Path(sys.prefix) / "Lib" / "site-packages"):
        nvidia_root = root / "nvidia"
        candidates.extend(
            [
                nvidia_root / "cublas" / "bin",
                nvidia_root / "cuda_runtime" / "bin",
                nvidia_root / "cuda_nvrtc" / "bin",
                nvidia_root / "cudnn" / "bin",
            ]
        )

    seen: set[Path] = set()
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    new_entries: list[str] = []
    for directory in candidates:
        resolved = directory.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        os.add_dll_directory(str(resolved))
        resolved_str = str(resolved)
        if resolved_str not in path_entries and resolved_str not in new_entries:
            new_entries.append(resolved_str)

    if new_entries:
        os.environ["PATH"] = os.pathsep.join([*new_entries, *path_entries])


add_windows_nvidia_dll_dirs()


from faster_whisper import BatchedInferencePipeline, WhisperModel


def srt_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1_000
    millis = total_ms % 1_000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def choose_defaults() -> tuple[str, str, str, bool, int]:
    model_path = str(default_model_path())
    if shutil.which("nvidia-smi"):
        return (model_path, "cuda", "float16", True, 16)
    return (model_path, "cpu", "int8", False, 1)


def build_parser() -> argparse.ArgumentParser:
    defaults = transcribe_defaults()
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video to txt and srt with faster-whisper."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the local audio/video file."
    )
    parser.add_argument(
        "--output-dir",
        default="downloads/media-transcribe/transcripts",
        help="Directory for transcription outputs.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. zh or en. Omit to auto-detect.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Whisper model name or local model path. Defaults to the model declared in .agents/skills/video-study-notes/config.json.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type. Use auto for pragmatic defaults.",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size for decoding."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=int(defaults.get("best_of", 5)),
        help="Number of candidates when sampling with non-zero temperature.",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=float(defaults.get("patience", 1.0)),
        help="Beam search patience factor.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=float(defaults.get("length_penalty", 1.0)),
        help="Length penalty for decoding.",
    )
    parser.add_argument(
        "--temperature",
        default=",".join(
            str(x) for x in defaults.get("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ),
        help="Comma-separated temperatures, e.g. 0.0,0.2,0.4.",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=float(defaults.get("compression_ratio_threshold", 1.4)),
        help="Threshold for gzip compression ratio fallback.",
    )
    parser.add_argument(
        "--log-prob-threshold",
        type=float,
        default=float(defaults.get("log_prob_threshold", -10.0)),
        help="Threshold for average log probability fallback.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=float(defaults.get("no_speech_threshold", 0.9)),
        help="Threshold for no speech detection.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_true",
        default=bool(defaults.get("condition_on_previous_text", False)),
        help="Condition each segment on the previous decoded text.",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="Disable conditioning on previous decoded text.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=defaults.get("initial_prompt") or None,
        help="Optional initial prompt.",
    )
    parser.add_argument(
        "--prefix",
        default=defaults.get("prefix") or None,
        help="Optional decoding prefix.",
    )
    parser.add_argument(
        "--suppress-blank",
        dest="suppress_blank",
        action="store_true",
        default=bool(defaults.get("suppress_blank", True)),
        help="Suppress blank outputs.",
    )
    parser.add_argument(
        "--no-suppress-blank",
        dest="suppress_blank",
        action="store_false",
        help="Do not suppress blank outputs.",
    )
    parser.add_argument(
        "--suppress-tokens",
        default=str(defaults.get("suppress_tokens", "-1")),
        help="Comma-separated token ids to suppress.",
    )
    parser.add_argument(
        "--without-timestamps",
        action="store_true",
        default=bool(defaults.get("without_timestamps", False)),
        help="Disable timestamp generation.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        default=bool(defaults.get("word_timestamps", False)),
        help="Enable word-level timestamps.",
    )
    parser.add_argument(
        "--prepend-punctuations",
        default=str(defaults.get("prepend_punctuations", "\"'“¿([{-")),
        help="Punctuations to prepend during word merge.",
    )
    parser.add_argument(
        "--append-punctuations",
        default=str(defaults.get("append_punctuations", "\"'.。,，!！?？:：”)]}、")),
        help="Punctuations to append during word merge.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(defaults.get("repetition_penalty", 1.0)),
        help="Repetition penalty.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=int(defaults.get("no_repeat_ngram_size", 0)),
        help="No repeat ngram size.",
    )
    parser.add_argument(
        "--prompt-reset-on-temperature",
        type=float,
        default=float(defaults.get("prompt_reset_on_temperature", 0.5)),
        help="Prompt reset threshold on temperature fallback.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=defaults.get("max_new_tokens"),
        help="Optional maximum number of new tokens.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=int(defaults.get("chunk_length", 30)),
        help="Audio chunk length in seconds.",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=float,
        default=float(defaults.get("hallucination_silence_threshold", 0.5)),
        help="Silence threshold to skip hallucinations.",
    )
    parser.add_argument(
        "--language-detection-threshold",
        type=float,
        default=float(defaults.get("language_detection_threshold", 0.5)),
        help="Threshold for language detection.",
    )
    parser.add_argument(
        "--language-detection-segments",
        type=int,
        default=int(defaults.get("language_detection_segments", 1)),
        help="Number of segments used for language detection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size when batched inference is enabled.",
    )
    parser.add_argument(
        "--vad-filter", action="store_true", default=True, help="Enable VAD filtering."
    )
    parser.add_argument(
        "--no-vad-filter",
        dest="vad_filter",
        action="store_false",
        help="Disable VAD filtering.",
    )
    return parser


def parse_temperature(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_suppress_tokens(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def vad_parameters_from_config() -> dict | None:
    defaults = transcribe_defaults()
    vad = defaults.get("vad_parameters")
    if vad is None:
        return None
    if not isinstance(vad, dict):
        raise ValueError(f"Invalid 'vad_parameters' in skill config: {CONFIG_PATH}")
    return vad


def resolve_runtime(args: argparse.Namespace) -> tuple[str, str, str, bool, int]:
    auto_model, auto_device, auto_compute, auto_batched, auto_batch = choose_defaults()
    model = auto_model if args.model == "auto" else args.model
    device = auto_device if args.device == "auto" else args.device
    compute = auto_compute if args.compute_type == "auto" else args.compute_type
    batched = auto_batched if device == auto_device else device == "cuda"
    batch_size = (
        auto_batch
        if args.batch_size == 16 and args.device == "auto"
        else args.batch_size
    )
    return model, device, compute, batched, batch_size


def make_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    model_path = (
        Path(args.model).expanduser().resolve()
        if args.model != "auto"
        else default_model_path()
    )
    if not model_path.exists():
        print(f"Model path not found: {model_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name, device, compute_type, batched, batch_size = resolve_runtime(args)

    try:
        model = make_model(model_name, device, compute_type)
    except Exception as exc:
        if device != "cuda":
            raise
        print(
            f"CUDA path failed ({exc}). Falling back to CPU int8 local model.",
            file=sys.stderr,
        )
        model_name, device, compute_type, batched, batch_size = (
            str(default_model_path()),
            "cpu",
            "int8",
            False,
            1,
        )
        model = make_model(model_name, device, compute_type)

    transcriber = BatchedInferencePipeline(model=model) if batched else model

    kwargs = {
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "temperature": parse_temperature(args.temperature),
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "log_prob_threshold": args.log_prob_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "condition_on_previous_text": args.condition_on_previous_text,
        "initial_prompt": args.initial_prompt,
        "prefix": args.prefix,
        "suppress_blank": args.suppress_blank,
        "suppress_tokens": parse_suppress_tokens(args.suppress_tokens),
        "without_timestamps": args.without_timestamps,
        "word_timestamps": args.word_timestamps,
        "prepend_punctuations": args.prepend_punctuations,
        "append_punctuations": args.append_punctuations,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "prompt_reset_on_temperature": args.prompt_reset_on_temperature,
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "hallucination_silence_threshold": args.hallucination_silence_threshold,
        "language_detection_threshold": args.language_detection_threshold,
        "language_detection_segments": args.language_detection_segments,
        "language": args.language,
        "vad_filter": args.vad_filter,
    }
    vad_parameters = vad_parameters_from_config()
    if args.vad_filter and vad_parameters:
        kwargs["vad_parameters"] = vad_parameters
    if batched:
        kwargs["batch_size"] = batch_size

    try:
        segments, info = transcriber.transcribe(str(input_path), **kwargs)
        segments = list(segments)
    except Exception as exc:
        if device != "cuda":
            raise
        print(
            f"CUDA transcription failed ({exc}). Falling back to CPU int8 local model.",
            file=sys.stderr,
        )
        model_name, device, compute_type, batched, batch_size = (
            str(default_model_path()),
            "cpu",
            "int8",
            False,
            1,
        )
        model = make_model(model_name, device, compute_type)
        transcriber = model
        kwargs = {
            "beam_size": args.beam_size,
            "best_of": args.best_of,
            "patience": args.patience,
            "length_penalty": args.length_penalty,
            "temperature": parse_temperature(args.temperature),
            "compression_ratio_threshold": args.compression_ratio_threshold,
            "log_prob_threshold": args.log_prob_threshold,
            "no_speech_threshold": args.no_speech_threshold,
            "condition_on_previous_text": args.condition_on_previous_text,
            "initial_prompt": args.initial_prompt,
            "prefix": args.prefix,
            "suppress_blank": args.suppress_blank,
            "suppress_tokens": parse_suppress_tokens(args.suppress_tokens),
            "without_timestamps": args.without_timestamps,
            "word_timestamps": args.word_timestamps,
            "prepend_punctuations": args.prepend_punctuations,
            "append_punctuations": args.append_punctuations,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "prompt_reset_on_temperature": args.prompt_reset_on_temperature,
            "max_new_tokens": args.max_new_tokens,
            "chunk_length": args.chunk_length,
            "hallucination_silence_threshold": args.hallucination_silence_threshold,
            "language_detection_threshold": args.language_detection_threshold,
            "language_detection_segments": args.language_detection_segments,
            "language": args.language,
            "vad_filter": args.vad_filter,
        }
        if args.vad_filter and vad_parameters:
            kwargs["vad_parameters"] = vad_parameters
        segments, info = transcriber.transcribe(str(input_path), **kwargs)
        segments = list(segments)

    stem = input_path.stem
    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"
    json_path = output_dir / f"{stem}.transcription.json"

    full_text = "\n".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )
    txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")

    with srt_path.open("w", encoding="utf-8") as handle:
        for idx, segment in enumerate(segments, 1):
            text = segment.text.strip()
            if not text:
                continue
            handle.write(f"{idx}\n")
            handle.write(f"{srt_ts(segment.start)} --> {srt_ts(segment.end)}\n")
            handle.write(text + "\n\n")

    payload = {
        "input": str(input_path),
        "outputs": {
            "txt": str(txt_path),
            "srt": str(srt_path),
            "json": str(json_path),
        },
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": getattr(info, "duration_after_vad", None),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "batched": batched,
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "temperature": parse_temperature(args.temperature),
        "vad_filter": args.vad_filter,
        "vad_parameters": vad_parameters,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in segments
        ],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Detected language: {info.language} ({info.language_probability:.4f})")
    print(f"Model: {model_name}")
    print(f"TXT: {txt_path}")
    print(f"SRT: {srt_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(transcribe(build_parser().parse_args()))
