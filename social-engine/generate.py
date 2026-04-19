import asyncio

import base64
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from kokoro_onnx import Kokoro
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat
from pypdf import PdfReader, PdfWriter

try:
    import edge_tts
except Exception:
    edge_tts = None


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"
BIN_DIR = BASE_DIR / "bin"
FONTS_DIR = BASE_DIR / "fonts"
MODELS_DIR = BASE_DIR / "models" / "kokoro"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# Bind local FFmpeg to this runtime only.
os.environ["PATH"] = str(BIN_DIR) + os.pathsep + os.environ.get("PATH", "")

BRAND_BLACK = (0x0E, 0x0E, 0x17)
BRAND_RED = (0xF7, 0x50, 0x49)
BRAND_CYAN = (0x5E, 0xF6, 0xFF)
BRAND_WHITE = (0xFF, 0xFF, 0xFF)
MAX_LINKEDIN_PDF_MB = 25
SQUARE_SIZE = 1080
TOP_IMAGE_AREA_HEIGHT = 620
LINKEDIN_LOGO_PATH = BASE_DIR / "logo.png"
LINKEDIN_LOGO_MAX_WIDTH = 132
LINKEDIN_LOGO_MAX_HEIGHT = 72
LINKEDIN_LOGO_MARGIN = 26

MARKETING_TOKENS = {
    "unlock",
    "discover",
    "magic",
    "journey",
    "vibes",
    "cosmos",
    "galaxy",
    "surprise",
    "explore",
    "future",
    "welcome",
    "amazing",
    "awesome",
    "viral",
    "wow",
    "experience",
    "yourself",
    "check",
}

DIDASCALIC_PATTERNS = [
    r"scene focus:\s*",
    r"implementation note:\s*",
    r"visual complexity:\s*",
    r"complexity:\s*",
    r"text-safe(?:\s+overlay)? zone:\s*",
]

_KOKORO_ENGINE: Kokoro | None = None


def ensure_environment() -> None:
    os.chdir(BASE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    required_binaries = [BIN_DIR / "ffmpeg.exe", BIN_DIR / "ffprobe.exe"]
    missing = [str(path) for path in required_binaries if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required binaries in social-engine/bin: " + ", ".join(missing)
        )


def find_assets() -> tuple[Path, list[Path], str]:
    if not ASSETS_DIR.exists():
        raise FileNotFoundError(f"Missing assets folder: {ASSETS_DIR}")

    video_files = sorted(ASSETS_DIR.glob("*.mp4"), key=lambda p: p.name.lower())
    if len(video_files) != 1:
        raise ValueError(
            f"Expected exactly 1 mp4 in assets/, found {len(video_files)}."
        )

    image_paths = sorted(
        [
            *ASSETS_DIR.glob("*.png"),
            *ASSETS_DIR.glob("*.jpg"),
            *ASSETS_DIR.glob("*.jpeg"),
        ],
        key=lambda p: p.name.lower(),
    )
    if not image_paths:
        raise ValueError("No PNG/JPG images found in assets/.")

    prompt_file = ASSETS_DIR / "prompt.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_file}")

    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError("assets/prompt.md is empty.")

    return video_files[0], image_paths, prompt_text


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("Model response did not contain a JSON object.")

    parsed = json.loads(raw_text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON root must be an object.")
    return parsed


def _extract_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    return str(content or "")


def _request_json(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )

    raw = _extract_message_content(response.choices[0].message.content)
    return _extract_json_object(raw)


def _build_slider_variation_hint() -> tuple[str, str]:
    configured_seed = os.getenv("SLIDER_VARIATION_SEED", "").strip()
    if configured_seed:
        seed = configured_seed
    else:
        seed = str(random.SystemRandom().randint(100_000, 999_999))

    rng = random.Random(seed)
    angles = [
        "interaction-first framing with concise technical language",
        "performance-first framing focused on runtime stability",
        "narrative framing from hero intro to slider transition",
        "design-system framing focused on shader consistency",
        "engineering framing centered on SEO and usability trade-offs",
    ]
    rhythms = [
        "short direct lines",
        "compact analytical lines",
        "clear progression between slides",
        "balanced visual and technical emphasis",
    ]

    hint = f"{rng.choice(angles)}; {rng.choice(rhythms)}"
    return seed, hint


def parse_prompt_sections(prompt_text: str) -> dict[str, Any]:
    normalized_prompt = prompt_text.replace("\r\n", "\n")
    normalized_prompt = re.sub(
        r"\s+#\s*(context|video\s+script(?:\s+description)?|video\s+description|slides)\s*:",
        lambda match: f"\n# {match.group(1)}:",
        normalized_prompt,
        flags=re.IGNORECASE,
    )

    sections = {
        "context": [],
        "video_description": [],
        "slides": [],
    }

    current = "context"
    for raw_line in normalized_prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized_line = re.sub(r"^[#>*\-\s]+", "", line).strip()
        if not normalized_line:
            continue

        lower = normalized_line.lower()
        if lower.startswith("context:"):
            current = "context"
            remainder = normalized_line.split(":", 1)[1].strip()
            if remainder:
                sections[current].append(remainder)
            continue

        if re.match(r"^video\s+(script\s+)?description\s*:", lower) or lower.startswith("video script:"):
            current = "video_description"
            remainder = normalized_line.split(":", 1)[1].strip()
            if remainder:
                sections[current].append(remainder)
            continue

        if lower.startswith("script description:"):
            current = "video_description"
            remainder = normalized_line.split(":", 1)[1].strip()
            if remainder:
                sections[current].append(remainder)
            continue

        if lower.startswith("slides:"):
            current = "slides"
            remainder = normalized_line.split(":", 1)[1].strip()
            if remainder:
                sections[current].append(remainder)
            continue

        sections[current].append(normalized_line)

    context_text = " ".join(sections["context"]).strip()
    video_description = " ".join(sections["video_description"]).strip()
    slide_lines = sections["slides"]

    if not context_text:
        context_text = prompt_text.strip()

    if not video_description:
        video_description = context_text

    return {
        "context": context_text,
        "video_description": video_description,
        "slides_lines": slide_lines,
    }


def _clean_video_script_source(video_script: str) -> str:
    cleaned = (video_script or "").strip()
    cleaned = re.sub(
        r"^\s*(video\s+(script|description)|script\s+description)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("infoes", "infos")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _prepare_tts_script_source(text: str) -> str:
    clean = re.sub(r"https?://", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"\bwww\.", "", clean, flags=re.IGNORECASE)

    clean = re.sub(r"\bthree\s*\.\s*js\b", "threejs", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bthree\s+js\b", "threejs", clean, flags=re.IGNORECASE)

    clean = re.sub(
        r"\b[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b",
        lambda match: match.group(0).replace(".", " dot "),
        clean,
    )

    clean = re.sub(r"[/\\|]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _pause_seconds_for_dot_run(dot_count: int) -> float:
    return max(0.18, min(1.50, 0.22 * dot_count))


def _split_tts_script_parts(text: str) -> list[dict[str, Any]]:
    prepared = _prepare_tts_script_source(text)
    if not prepared:
        return []

    parts: list[dict[str, Any]] = []
    for chunk in re.split(r"(\.{2,})", prepared):
        if not chunk:
            continue

        if re.fullmatch(r"\.+", chunk):
            parts.append({"kind": "pause", "duration": _pause_seconds_for_dot_run(len(chunk))})
            continue

        normalized = _normalize_speech_text_for_tts(chunk, preserve_pauses=True)
        if normalized:
            parts.append({"kind": "text", "text": normalized})

    return parts


def _segments_from_video_script(video_script: str, target_words: int) -> list[str]:
    source = _clean_video_script_source(video_script)
    if not source:
        return ["Showing the complete portfolio flow from hero to slider and back"]

    chunks = re.split(r"\bthen\b|(?<=[.!?])\s+", source, flags=re.IGNORECASE)
    segments: list[str] = []

    for chunk in chunks:
        text = chunk.strip(" .,!?:;-")
        if not text:
            continue

        words = text.split()
        if len(words) > 16:
            text = " ".join(words[:16]).strip()

        if text and text not in segments:
            segments.append(text + ".")

    if not segments:
        segments = [source]

    target_segment_count = max(4, min(8, int(round(target_words / 13))))
    return segments[:target_segment_count]


def _build_voiceover_script_from_video_script(video_script: str, target_words: int) -> str:
    source = _clean_video_script_source(video_script)
    if not source:
        return "portfolio walkthrough from hero to slider and back"

    return _prepare_tts_script_source(source)


def _is_plain_slide_hint(image_hint: str) -> bool:
    lower_hint = image_hint.strip().lower()
    if not lower_hint:
        return True

    if any(token in lower_hint for token in ["background", "plain", "color"]):
        return True

    if re.fullmatch(r"#?[0-9a-f]{3,8}", lower_hint):
        return True

    if re.fullmatch(r"(?:rgb|rgba)\([^)]+\)", lower_hint):
        return True

    return False


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _match_image_hint(image_hint: str, image_paths: list[Path]) -> str | None:
    hint = image_hint.strip()
    if not hint:
        return None

    lower_hint = hint.lower()
    if "background" in lower_hint or "plain" in lower_hint:
        return None

    candidates = {path.name.lower(): path.name for path in image_paths}
    if lower_hint in candidates:
        return candidates[lower_hint]

    hint_stem = _normalize_token(Path(hint).stem)
    if hint_stem:
        for path in image_paths:
            if _normalize_token(path.stem) == hint_stem:
                return path.name

    hint_no_digits = re.sub(r"\d+", "", hint_stem)
    matches: list[str] = []
    if hint_no_digits:
        for path in image_paths:
            path_stem = _normalize_token(path.stem)
            if hint_no_digits in path_stem or path_stem in hint_no_digits:
                matches.append(path.name)

    if len(matches) == 1:
        return matches[0]

    for path in image_paths:
        stem = _normalize_token(path.stem)
        if hint_stem and (hint_stem in stem or stem in hint_stem):
            return path.name

    return None


def parse_slide_specs(slide_lines: list[str], image_paths: list[Path]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for line in slide_lines:
        cleaned = line.strip().lstrip("-*").strip()
        if not cleaned:
            continue

        match = re.match(r"^slide\s*\d+\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE)
        payload = match.group(1).strip() if match else cleaned

        parts = [part.strip() for part in payload.split(",") if part.strip()]
        if not parts:
            continue

        if len(parts) < 2:
            continue

        image_hint = parts[0]
        style = "plain" if _is_plain_slide_hint(image_hint) else "image"
        image_filename = _match_image_hint(image_hint, image_paths) if style == "image" else None

        if len(parts) >= 3:
            title_hint = parts[1]
            body_hint = ", ".join(parts[2:]).strip()
        else:
            title_hint = ""
            body_hint = parts[1]

        if style == "plain" and not title_hint:
            title_hint = "Implementation Review"

        specs.append(
            {
                "style": style,
                "image_hint": image_hint,
                "image_filename": image_filename,
                "title_hint": title_hint,
                "body_hint": body_hint,
                "raw": cleaned,
            }
        )

    if not specs:
        for path in image_paths:
            specs.append(
                {
                    "style": "image",
                    "image_hint": path.name,
                    "image_filename": path.name,
                    "title_hint": f"Slide {path.stem}",
                    "body_hint": "Technical implementation highlight from the project.",
                    "raw": path.name,
                }
            )

    return specs


def _kokoro_tts_to_wav(
    script_text: str,
    output_path: Path,
    target_duration: float | None = None,
) -> tuple[float, float, str]:
    engine = _get_kokoro_engine()

    voice = os.getenv("TTS_KOKORO_VOICE", "am_fenrir")
    lang = os.getenv("TTS_KOKORO_LANG", "en-us")

    try:
        speed = float(os.getenv("TTS_KOKORO_SPEED", "1.02"))
    except ValueError:
        speed = 1.02
    speed = max(0.78, min(1.20, speed))

    available_voices = set(engine.get_voices())
    if voice not in available_voices:
        if "am_fenrir" in available_voices:
            voice = "am_fenrir"
        elif "am_michael" in available_voices:
            voice = "am_michael"
        elif available_voices:
            voice = sorted(available_voices)[0]

    cleaned_script = re.sub(r"\s+", " ", script_text).strip()

    best_samples = None
    best_sr = 24000
    best_duration = 0.0

    for _ in range(3):
        samples, sample_rate = engine.create(
            cleaned_script,
            voice=voice,
            speed=speed,
            lang=lang,
            trim=True,
        )
        duration = len(samples) / sample_rate

        best_samples = samples
        best_sr = sample_rate
        best_duration = duration

        if not target_duration or target_duration <= 0:
            break

        ratio = duration / target_duration
        if 0.92 <= ratio <= 1.08:
            break

        speed = max(0.78, min(1.20, speed * ratio))

    if best_samples is None:
        raise RuntimeError("Kokoro TTS failed to generate audio.")

    sf.write(str(output_path), best_samples, best_sr)
    return best_duration, speed, voice


def _synthesize_tts_chunk(
    backend: str,
    script_text: str,
    output_path: Path,
) -> tuple[float, float, str, list[dict[str, Any]] | None]:
    if backend == "elevenlabs":
        return _elevenlabs_tts_to_wav(script_text, output_path, target_duration=None)

    if backend == "edge":
        return _edge_tts_to_wav(script_text, output_path, target_duration=None)

    if backend == "kokoro":
        duration, speed, voice = _kokoro_tts_to_wav(script_text, output_path, target_duration=None)
        return duration, speed, voice, None

    raise ValueError(f"Unsupported TTS backend: {backend}")


def _run_tts_backend_with_pauses(
    backend: str,
    script_text: str,
    output_path: Path,
    target_duration: float,
) -> tuple[float, float, str, str, list[dict[str, Any]] | None]:
    parts = _split_tts_script_parts(script_text)
    if not parts:
        raise RuntimeError("Empty script passed to TTS.")

    audio_parts: list[np.ndarray] = []
    word_boundaries: list[dict[str, Any]] = []
    temp_paths: list[Path] = []
    sample_rate = 24000
    offset = 0.0
    used_speed = 1.0
    used_voice = backend

    try:
        for index, part in enumerate(parts):
            if part["kind"] == "pause":
                pause_seconds = float(part["duration"])
                pause_samples = int(round(sample_rate * pause_seconds))
                if pause_samples > 0:
                    audio_parts.append(np.zeros(pause_samples, dtype=np.float32))
                    offset += pause_seconds
                continue

            chunk_text = str(part["text"])
            chunk_path = output_path.with_name(f".{output_path.stem}_{backend}_{index}.wav")
            temp_paths.append(chunk_path)

            duration, speed, voice, chunk_boundaries = _synthesize_tts_chunk(
                backend,
                chunk_text,
                chunk_path,
            )

            chunk_samples, chunk_sample_rate = sf.read(str(chunk_path), dtype="float32")
            if chunk_sample_rate != sample_rate:
                if not audio_parts and sample_rate == 24000:
                    sample_rate = int(chunk_sample_rate)
                elif chunk_sample_rate != sample_rate:
                    raise RuntimeError(
                        f"Unexpected TTS sample rate {chunk_sample_rate}, expected {sample_rate}."
                    )

            chunk_samples = np.asarray(chunk_samples, dtype=np.float32)
            if chunk_samples.ndim > 1:
                chunk_samples = chunk_samples[:, 0]
            audio_parts.append(chunk_samples)

            if chunk_boundaries is None:
                chunk_boundaries = _build_word_boundaries_from_script(chunk_text, duration)

            for item in chunk_boundaries:
                word_boundaries.append(
                    {
                        "start": offset + float(item["start"]),
                        "end": offset + float(item["end"]),
                        "text": item["text"],
                    }
                )

            offset += duration
            used_speed = speed
            used_voice = voice

        combined_samples = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        original_duration = float(len(combined_samples)) / float(sample_rate) if sample_rate > 0 else 0.0
        sf.write(str(output_path), combined_samples, sample_rate)

        duration = original_duration
        speed_hint = used_speed

        if target_duration and target_duration > 0 and duration > target_duration * 1.12:
            desired_duration = target_duration * 0.98
            atempo = max(0.80, min(1.35, duration / desired_duration))
            adjusted_path = output_path.with_name(f".{output_path.stem}_{backend}_adjusted.wav")
            try:
                ffmpeg.run(
                    ffmpeg.output(
                        ffmpeg.input(str(output_path)).audio.filter("atempo", atempo),
                        str(adjusted_path),
                        ac=1,
                        ar=sample_rate,
                        acodec="pcm_s16le",
                    ),
                    overwrite_output=True,
                    quiet=True,
                )
                shutil.move(str(adjusted_path), str(output_path))
                duration = _probe_duration_seconds(output_path)
                if original_duration > 0:
                    scale = duration / original_duration
                    for item in word_boundaries:
                        item["start"] = float(item["start"]) * scale
                        item["end"] = float(item["end"]) * scale
                speed_hint *= atempo
            finally:
                if adjusted_path.exists():
                    adjusted_path.unlink()

        return duration, speed_hint, used_voice, backend, word_boundaries
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()


def _tts_to_wav(
    script_text: str,
    output_path: Path,
    target_duration: float,
) -> tuple[float, float, str, str, list[dict[str, Any]] | None]:
    backend = os.getenv("TTS_BACKEND", "auto").strip().lower()
    errors: list[str] = []

    if backend == "elevenlabs":
        return _run_tts_backend_with_pauses("elevenlabs", script_text, output_path, target_duration)

    if backend == "edge":
        return _run_tts_backend_with_pauses("edge", script_text, output_path, target_duration)

    if backend == "kokoro":
        return _run_tts_backend_with_pauses("kokoro", script_text, output_path, target_duration)

    for candidate_backend in ("elevenlabs", "edge", "kokoro"):
        try:
            return _run_tts_backend_with_pauses(candidate_backend, script_text, output_path, target_duration)
        except Exception as exc:
            errors.append(f"{candidate_backend} failed: {exc}")

    raise RuntimeError("TTS generation failed. " + "; ".join(errors))


def _tts_to_wav(
    script_text: str,
    output_path: Path,
    target_duration: float,
) -> tuple[float, float, str, str, list[dict[str, Any]] | None]:
    backend = os.getenv("TTS_BACKEND", "auto").strip().lower()
    errors: list[str] = []

    if backend == "elevenlabs":
        return _run_tts_backend_with_pauses("elevenlabs", script_text, output_path, target_duration)

    if backend == "edge":
        return _run_tts_backend_with_pauses("edge", script_text, output_path, target_duration)

    if backend == "kokoro":
        return _run_tts_backend_with_pauses("kokoro", script_text, output_path, target_duration)

    for candidate_backend in ("elevenlabs", "edge", "kokoro"):
        try:
            return _run_tts_backend_with_pauses(candidate_backend, script_text, output_path, target_duration)
        except Exception as exc:
            errors.append(f"{candidate_backend} failed: {exc}")

    raise RuntimeError("TTS generation failed. " + "; ".join(errors))


def _image_to_data_url(image_path: Path, max_side: int = 768) -> str:
    with Image.open(image_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")

    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _heuristic_image_analysis(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")

    gray = image.convert("L")
    stat = ImageStat.Stat(gray)

    width, height = image.size
    orientation = "landscape"
    if height > width:
        orientation = "portrait"
    elif height == width:
        orientation = "square"

    return {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "orientation": orientation,
        "brightness": round(stat.mean[0], 1),
        "contrast": round(stat.stddev[0], 1),
    }


def analyze_images(client: OpenAI, image_paths: list[Path]) -> list[dict[str, Any]]:
    vision_model = os.getenv(
        "NVIDIA_NIM_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct"
    )

    analyses: list[dict[str, Any]] = []

    for image_path in image_paths:
        fallback = _heuristic_image_analysis(image_path)

        try:
            data_url = _image_to_data_url(image_path)
            response = client.chat.completions.create(
                model=vision_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise visual analyst. Return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Return JSON keys: visual_summary, likely_subject, best_use. "
                                    "Keep values short and technical."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            raw = _extract_message_content(response.choices[0].message.content)
            vision_data = _extract_json_object(raw)
        except Exception:
            vision_data = {
                "visual_summary": "Project visual for technical showcase",
                "likely_subject": "frontend module",
                "best_use": "implementation evidence",
            }

        analyses.append({**fallback, **vision_data})

    return analyses


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _extract_first_url(text: str) -> str | None:
    match = re.search(r"https?://\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", text)
    if not match:
        return None

    value = match.group(0).rstrip(").,;:")
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://{value}"


def _normalize_x_post(text: str, prompt_text: str) -> str:
    cleaned = " ".join(text.split())
    existing_tags = re.findall(r"#\w+", cleaned)
    existing_low = {tag.lower() for tag in existing_tags}

    preferred_tags = ["#Threejs", "#WebGL", "#WebPerformance", "#SemanticHTML"]
    selected_tags: list[str] = []
    for tag in preferred_tags:
        if tag.lower() in existing_low:
            selected_tags.append(tag)

    for tag in ["#Threejs", "#WebGL", "#WebPerformance"]:
        if tag not in selected_tags:
            selected_tags.append(tag)

    selected_tags = selected_tags[:4]
    non_tags = re.sub(r"#\w+", "", cleaned).strip()

    quality_keywords = ["semantic", "performance", "3d", "shader", "webgl"]
    present_keywords = sum(1 for key in quality_keywords if key in non_tags.lower())
    if present_keywords < 2:
        non_tags = (
            "Launching a high-performance 3D portfolio with custom GLSL interactions, "
            "semantic HTML, and production-grade optimization."
        )

    url = _extract_first_url(prompt_text)
    payload = non_tags
    if url:
        payload = f"{payload} {url}"

    payload = f"{payload} {' '.join(selected_tags)}".strip()

    if len(payload) > 280:
        hash_tags = " ".join(selected_tags)
        url_text = f" {url}" if url else ""
        limit = max(0, 280 - len(hash_tags) - len(url_text) - 2)
        short_non_tags = non_tags[:limit].rstrip(" ,.;:")
        payload = f"{short_non_tags}{url_text} {hash_tags}".strip()

    return payload


def _sanitize_technical_text(text: str, max_words: int) -> str:
    clean = re.sub(r"[#]+\w+", "", text)

    clean = re.sub(r"\bempasize\b", "emphasize", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\btheejs\b", "three.js", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bthreejs\b", "three.js", clean, flags=re.IGNORECASE)

    clean = re.sub(r"\bcheck\s+it\s+out\b", "review", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bexperience\s+it\s+for\s+yourself\b", "review the implementation", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bfor\s+yourself\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bencourage\s+the\s+reader\b", "summarize the technical outcome", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bwow\s+factor\b", "visual impact", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bfactor\b", "visual impact", clean, flags=re.IGNORECASE)

    for pattern in DIDASCALIC_PATTERNS:
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)

    clean = clean.replace("!", ".")
    clean = re.sub(r"\s+", " ", clean).strip()

    tokens: list[str] = []
    for token in clean.split():
        low = re.sub(r"[^a-z0-9-]", "", token.lower())
        if low in MARKETING_TOKENS:
            continue
        tokens.append(token)

    clean = " ".join(tokens).strip(" .")
    if not clean:
        clean = "Technical implementation details and performance considerations from the build."

    return _truncate_words(clean, max_words)


def _normalize_speech_text_for_tts(text: str, preserve_pauses: bool = False) -> str:
    clean = re.sub(r"https?://", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"\bwww\.", "", clean, flags=re.IGNORECASE)

    clean = re.sub(r"\bthree\s*\.\s*js\b", "threejs", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bthree\s+js\b", "threejs", clean, flags=re.IGNORECASE)

    clean = re.sub(
        r"\b[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b",
        lambda match: match.group(0).replace(".", " dot "),
        clean,
    )

    clean = re.sub(r"[/\\|]", " ", clean)

    if preserve_pauses:
        clean = re.sub(r"[\"'()\[\]{}]", " ", clean)
        # Dot runs are pause units: 1 dot = short pause, N dots = N short pauses.
        clean = re.sub(
            r"\.{2,}",
            lambda match: (".\n" * len(match.group(0))).rstrip("\n"),
            clean,
        )
        clean = re.sub(r"(?:\s*,\s*){2,}", ", ", clean)
        clean = re.sub(r"\s*([,;:!?])\s*", r"\1\n", clean)

        # Single dots still create a short pause; keep multi-dot newline stacks intact.
        clean = re.sub(r"(?<!\.)\.(?!\.)\s*", ".\n", clean)

        # Normalize spaces inside each line without collapsing pause line breaks.
        lines = [re.sub(r"\s+", " ", line).strip() for line in clean.split("\n")]
        clean = "\n".join(line for line in lines if line)
        clean = re.sub(r"\n{7,}", "\n\n\n\n\n\n", clean).strip()
        return clean

    clean = re.sub(r"[.,!?;:()\[\]\{\}\"']", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean


def _build_word_boundaries_from_script(script_text: str, total_duration: float) -> list[dict[str, Any]]:
    if total_duration <= 0:
        return []

    words = [
        token
        for token in re.findall(r"[A-Za-z0-9-]+", script_text)
        if token and token.strip()
    ]
    if not words:
        return []

    weights = [max(2, len(word)) for word in words]
    total_weight = sum(weights)
    if total_weight <= 0:
        return []

    boundaries: list[dict[str, Any]] = []
    cursor = 0.0
    for word, weight in zip(words, weights):
        duration = total_duration * (weight / total_weight)
        start = cursor
        end = cursor + duration
        boundaries.append(
            {
                "start": max(0.0, start),
                "end": max(start + 0.08, end),
                "text": word,
            }
        )
        cursor = end

    return boundaries


def _video_timeline_notes(video_duration: float, scene_cuts: list[float]) -> str:
    if not scene_cuts:
        return (
            f"Video duration is {video_duration:.2f}s. No hard scene cuts detected. "
            "Use progression based on described actions and keep spoken flow continuous."
        )

    markers = ", ".join(f"{point:.2f}s" for point in scene_cuts[:10])
    return (
        f"Video duration is {video_duration:.2f}s. Scene cut markers: {markers}. "
        "Align narration beats to these transitions and keep pacing consistent."
    )


def _normalize_x_image_plan(raw_plan: Any, image_paths: list[Path]) -> list[dict[str, str]]:
    valid_names = {path.name for path in image_paths}
    normalized: list[dict[str, str]] = []

    if isinstance(raw_plan, list):
        for item in raw_plan:
            if not isinstance(item, dict):
                continue

            filename = str(item.get("filename", "")).strip()
            reason = str(item.get("reason", "")).strip()

            if filename not in valid_names:
                continue
            if not reason:
                reason = "Strong visual support for the post narrative."

            normalized.append(
                {
                    "filename": filename,
                    "reason": _sanitize_technical_text(reason, max_words=18),
                }
            )

    if not normalized:
        for path in image_paths[:3]:
            normalized.append(
                {
                    "filename": path.name,
                    "reason": "Use as supporting visual for technical credibility.",
                }
            )

    return normalized[:3]


def _extract_stack_terms(prompt_text: str) -> list[str]:
    terms = []
    known = [
        "three.js",
        "gsap",
        "webgl",
        "glsl",
        "semantic html",
        "shader",
        "performance",
        "seo",
    ]
    lower_prompt = prompt_text.lower()

    for item in known:
        if item in lower_prompt:
            terms.append(item)

    return terms


def _fallback_video_beats(video_description: str, context_text: str) -> list[str]:
    normalized_desc = re.sub(r"\((.*?)\)", r". \1", video_description)
    chunks = re.split(r"\bthen\b|;|\.|,", normalized_desc, flags=re.IGNORECASE)
    beats: list[str] = []

    for chunk in chunks:
        lowered = chunk.lower()
        if not lowered.strip():
            continue

        if "hero" in lowered or "logo" in lowered:
            beats.append(
                "Hero opens with fluid logo simulation and stable frame pacing."
            )
            continue

        if "slider" in lowered:
            beats.append(
                "Transition into slider keeps shader continuity across sections."
            )
            continue

        if "card" in lowered or "material" in lowered or "shader" in lowered:
            beats.append(
                "Card opening highlights material response, shader depth, and interaction precision."
            )
            continue

        if "scroll" in lowered:
            beats.append(
                "Scroll pass confirms motion smoothing and stable interaction state."
            )
            continue

        if "back" in lowered or "start" in lowered or "return" in lowered:
            beats.append(
                "Return to hero confirms state restoration and visual continuity."
            )
            continue

        normalized = _sanitize_technical_text(chunk, max_words=11)
        if normalized and len(normalized.split()) >= 4:
            beats.append(normalized.rstrip(".") + ".")

    if len(beats) < 4:
        terms = ", ".join(_extract_stack_terms(context_text)[:3])
        stack_suffix = f" with {terms}" if terms else ""
        beats = [
            f"Hero interaction validates fluid simulation{stack_suffix}.",
            "Slider transition demonstrates controlled shader pipeline behavior.",
            "Card interaction validates material response and touch precision.",
            "Return sequence confirms restored state and visual consistency.",
        ]

    deduped: list[str] = []
    for beat in beats:
        if beat not in deduped:
            deduped.append(beat)

    return deduped[:6]


def _keyword_overlap(left: str, right: str) -> int:
    left_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", left.lower())
        if len(token) > 3
    }
    right_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", right.lower())
        if len(token) > 3
    }
    return len(left_tokens.intersection(right_tokens))


def _normalize_tiktok_segments(
    raw_segments: Any,
    video_description: str,
    context_text: str,
    target_words: int,
) -> list[str]:
    base_beats = [
        _sanitize_technical_text(segment, max_words=13).rstrip(" .") + "."
        for segment in _fallback_video_beats(video_description, context_text)
    ]

    raw_candidates: list[str] = []
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, str):
                continue
            candidate = _sanitize_technical_text(item, max_words=13).rstrip(" .")
            if not candidate:
                continue
            if any(term in candidate.lower() for term in ["visit", "check", "discover", "magic", "surprising difference"]):
                continue
            raw_candidates.append(candidate + ".")

    selected: list[str] = []
    for base in base_beats:
        best = ""
        best_score = 0
        for candidate in raw_candidates:
            score = _keyword_overlap(base, candidate)
            if score > best_score:
                best = candidate
                best_score = score

        chosen = best if (best and best_score >= 1) else base
        word_count = len(chosen.split())
        if word_count < 6 or word_count > 13:
            chosen = _sanitize_technical_text(chosen, max_words=13).rstrip(" .") + "."
        selected.append(chosen)

    deduped: list[str] = []
    for beat in selected:
        normalized = _sanitize_technical_text(beat, max_words=13).rstrip(" .") + "."
        if normalized not in deduped:
            deduped.append(normalized)

    if len(deduped) < 4:
        for beat in base_beats:
            if beat not in deduped:
                deduped.append(beat)
            if len(deduped) >= 4:
                break

    target_beat_count = max(5, min(7, int(round(target_words / 14))))
    if len(deduped) < target_beat_count:
        for beat in base_beats:
            normalized = _sanitize_technical_text(beat, max_words=13).rstrip(" .") + "."
            if normalized not in deduped:
                deduped.append(normalized)
            if len(deduped) >= target_beat_count:
                break

    return deduped[:target_beat_count]


def _extend_tiktok_segments_for_duration(
    segments: list[str],
    context_text: str,
    target_words: int,
) -> list[str]:
    target_segment_count = max(6, min(8, int(round(target_words / 14))))

    normalized: list[str] = []
    for segment in segments:
        clean = _sanitize_technical_text(segment, max_words=13).rstrip(" .") + "."
        if clean and clean not in normalized:
            normalized.append(clean)

    if len(normalized) >= target_segment_count:
        return normalized[:target_segment_count]

    stack_terms = _extract_stack_terms(context_text)
    stack_text = " ".join(stack_terms[:4]) if stack_terms else "threejs gsap glsl"

    extras = [
        f"{stack_text} stack keeps transitions responsive and rendering stable under interaction.",
        "Hero interaction uses fluid simulation tuned for smooth pointer response and frame pacing.",
        "Slider navigation keeps shader states coherent while moving across portfolio sections.",
        "Card open and close sequence keeps material depth and touch response predictable.",
        "Semantic structure and seo constraints remain intact while visuals stay fully immersive.",
        "Return to hero confirms state restoration and continuity across the full interaction loop.",
    ]

    for extra in extras:
        clean_extra = _sanitize_technical_text(extra, max_words=13).rstrip(" .") + "."
        if clean_extra not in normalized:
            normalized.append(clean_extra)
        if len(normalized) >= target_segment_count:
            break

    total_words = sum(len(item.split()) for item in normalized)
    extra_index = 0
    while total_words < int(target_words * 0.82) and extra_index < len(extras):
        clean_extra = _sanitize_technical_text(extras[extra_index], max_words=13).rstrip(" .") + "."
        if clean_extra not in normalized:
            normalized.append(clean_extra)
            total_words = sum(len(item.split()) for item in normalized)
        extra_index += 1

    return normalized[:target_segment_count]


def _reread_tiktok_segments(
    client: OpenAI,
    model: str,
    video_description: str,
    source_video_draft: str,
    video_study_notes: list[str],
    video_duration: float,
    scene_cuts: list[float],
    draft_segments: list[str],
    target_words: int,
) -> list[str]:
    if not draft_segments:
        return draft_segments

    review_prompt = f"""
You are quality control for TikTok narration.
Reread and rewrite the draft narration beats before production.

Return strict JSON only:
{{
  "segments": ["string"]
}}

Rules:
- Keep exact chronology from video description.
- 5 to 7 segments total.
- Each segment 7 to 13 words.
- Spoken and technical, not salesy.
- No clickbait, no CTA, no hashtags.
- Avoid punctuation pauses (no periods in output text).
- Keep meaning clear and production-ready.

Video timeline notes:
{_video_timeline_notes(video_duration, scene_cuts)}

Video content study notes:
{json.dumps(video_study_notes, indent=2)}

Video description:
{video_description}

Source video script draft:
{source_video_draft}

Draft segments:
{json.dumps(draft_segments, indent=2)}
""".strip()

    try:
        payload = _request_json(
            client=client,
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You edit scripts for concise technical narration. Return JSON only.",
                },
                {"role": "user", "content": review_prompt},
            ],
        )
        reviewed = _normalize_tiktok_segments(
            payload.get("segments"),
            video_description,
            source_video_draft,
            target_words,
        )
        if len(reviewed) >= 4:
            return reviewed
    except Exception:
        pass

    return draft_segments


def _build_voiceover_script(
    beats: list[str],
    target_words: int,
) -> str:
    normalized_beats = [
        _sanitize_technical_text(beat, max_words=13).rstrip(" .")
        for beat in beats
        if beat and beat.strip()
    ]

    if not normalized_beats:
        return "Technical walkthrough of interaction flow shader behavior and performance checks"

    script = " then ".join(normalized_beats).strip()
    words = script.split()

    if len(words) > target_words:
        script = " ".join(words[:target_words]).rstrip(" ,.;:")

    technical_terms = ["shader", "semantic", "render", "performance", "frame"]
    if not any(term in script.lower() for term in technical_terms):
        script = f"{script} shader pipeline and performance stability stay in focus"

    return _normalize_speech_text_for_tts(script)


def _extend_voiceover_script_for_duration(
    script_text: str,
    script_beats: list[str],
    min_words: int,
) -> str:
    base_words = script_text.split()
    if len(base_words) >= min_words:
        return script_text

    extras: list[str] = []
    for beat in script_beats:
        clean = _sanitize_technical_text(beat, max_words=14).rstrip(" .")
        if not clean:
            continue
        extras.append(f"{clean} with consistent frame timing and precise interaction response")

    extras.extend(
        [
            "Shader transitions remain coherent while moving across sections and opening cards",
            "Semantic structure stays intact while visual complexity scales during interactions",
            "Scroll pacing and pointer input remain stable across the full navigation loop",
            "Final return to hero confirms state restoration and visual continuity",
        ]
    )

    for extra in extras:
        extra_words = extra.split()
        if len(base_words) >= min_words:
            break

        overlap = " ".join(extra_words[:5]).lower()
        if overlap and overlap in " ".join(base_words).lower():
            continue

        base_words.extend(extra_words)

    return _normalize_speech_text_for_tts(" ".join(base_words))


def _compose_linkedin_slides(
    slide_specs: list[dict[str, Any]],
    rewrites: Any,
    context_text: str,
    image_paths: list[Path],
    variation_seed: str,
) -> list[dict[str, Any]]:
    rewrites_by_index: dict[int, dict[str, str]] = {}

    if isinstance(rewrites, list):
        for idx, item in enumerate(rewrites, start=1):
            if not isinstance(item, dict):
                continue

            slide_number = item.get("slide_number")
            if isinstance(slide_number, int) and slide_number > 0:
                key = slide_number
            else:
                key = idx

            rewrites_by_index[key] = {
                "title": str(item.get("title", "")).strip(),
                "body": str(item.get("body", "")).strip(),
            }

    image_lookup = {path.name: path for path in image_paths}
    used_images: set[str] = set()
    slides: list[dict[str, Any]] = []

    for idx, spec in enumerate(slide_specs, start=1):
        rewrite = rewrites_by_index.get(idx, {})
        style = str(spec.get("style", "plain"))

        if style == "plain":
            title_source = spec.get("title_hint") or rewrite.get("title") or "Technical Summary"
            body_source = spec.get("body_hint") or rewrite.get("body") or context_text
        else:
            body_source = spec.get("body_hint") or rewrite.get("body") or ""
            if spec.get("title_hint"):
                title_source = spec.get("title_hint")
            elif spec.get("body_hint"):
                title_source = _truncate_words(str(spec.get("body_hint")), 6).rstrip(".")
            else:
                title_source = rewrite.get("title") or ""

        if not title_source:
            title_source = "Implementation Review" if style == "plain" else f"Slide {idx}"

        if not body_source:
            body_source = context_text

        title = " ".join(str(title_source).split()).strip()
        body = " ".join(str(body_source).split()).strip()

        if style == "plain":
            if not title:
                title = "Technical Summary"

        image_filename = spec.get("image_filename")

        if style == "image" and isinstance(image_filename, str) and image_filename in image_lookup:
            if image_filename in used_images:
                style = "plain"
                image_filename = None
            else:
                used_images.add(image_filename)
        else:
            style = "plain"
            image_filename = None

        if not body:
            body = "Implementation summary covering architecture, shader behavior, and optimization outcomes."

        slides.append(
            {
                "title": title or f"Slide {idx}",
                "body": body,
                "image_filename": image_filename,
                "style": style,
            }
        )

    if not slides:
        for path in image_paths:
            slides.append(
                {
                    "title": _sanitize_technical_text(path.stem, max_words=6),
                    "body": "Implementation details and performance evidence from this section.",
                    "image_filename": path.name,
                    "style": "image",
                }
            )
        slides.append(
            {
                "title": "System Summary",
                "body": "Pipeline integrates semantic structure, shader workflow, and production performance checks.",
                "image_filename": None,
                "style": "plain",
            }
        )

    return slides


def _reread_linkedin_slides(
    client: OpenAI,
    model: str,
    draft_slides: list[dict[str, Any]],
    slide_specs: list[dict[str, Any]],
    context_text: str,
    variation_seed: str,
    variation_hint: str,
) -> list[dict[str, Any]]:
    if not draft_slides:
        return draft_slides

    if os.getenv("LINKEDIN_REREAD_ENABLED", "false").strip().lower() not in {"1", "true", "yes"}:
        return draft_slides

    review_prompt = f"""
You are final quality control for LinkedIn carousel text.
Reread each slide text and rewrite for clarity and technical precision.

Return strict JSON only:
{{
  "slides": [
    {{"slide_number": 1, "title": "string", "body": "string"}}
  ]
}}

Rules:
- Keep same slide count and order.
- Keep each slide intent based on slide specs.
- Title max 7 words.
- Body max 16 words.
- Technical, professional, and creatively engaging tone.
- No clickbait words, no vague hype.
- Final plain slide must stay clean and readable.

Variation directive:
- Seed: {variation_seed}
- Use this angle: {variation_hint}
- Keep facts unchanged, but wording should differ from previous runs.

Context:
{context_text}

Slide specs:
{json.dumps(slide_specs, indent=2)}

Draft slides:
{json.dumps(draft_slides, indent=2)}
""".strip()

    rng = random.Random(f"reread-{variation_seed}")
    plain_title_options = [
        "Technical Summary",
        "Build Snapshot",
        "Engineering Notes",
        "Implementation Recap",
    ]
    plain_body_options = [
        "Three.js, GSAP, and GLSL pipeline balanced visual impact, SEO structure, and runtime interaction performance.",
        "Shader-driven interactions stayed responsive while preserving semantic structure, accessibility flow, and search indexing quality.",
        "Architecture combined visual complexity with stable performance budgets, predictable navigation flow, and strong usability outcomes.",
        "Interactive rendering, SEO structure, and motion timing were tuned together to keep the experience smooth and reliable.",
    ]

    try:
        payload = _request_json(
            client=client,
            model=model,
            temperature=0.35,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior technical copy editor for creative professional posts. Return JSON only.",
                },
                {"role": "user", "content": review_prompt},
            ],
        )
    except Exception:
        return draft_slides

    raw_slides = payload.get("slides")
    if not isinstance(raw_slides, list):
        return draft_slides

    replacements: dict[int, dict[str, str]] = {}
    for entry in raw_slides:
        if not isinstance(entry, dict):
            continue
        index = entry.get("slide_number")
        if not isinstance(index, int) or index <= 0:
            continue

        title = _sanitize_technical_text(str(entry.get("title", "")), max_words=7)
        body = _sanitize_technical_text(str(entry.get("body", "")), max_words=16)
        if not title or not body:
            continue

        replacements[index] = {"title": title, "body": body}

    revised: list[dict[str, Any]] = []
    for idx, slide in enumerate(draft_slides, start=1):
        style = str(slide.get("style", "plain")).strip().lower()
        if style == "plain":
            body = _sanitize_technical_text(str(slide.get("body", "")), max_words=16)
            if any(term in body.lower() for term in ["visit", "check", "discover", "website", "contact", "feel free", "go "]):
                body = rng.choice(plain_body_options)

            revised.append(
                {
                    **slide,
                    "title": rng.choice(plain_title_options),
                    "body": body,
                }
            )
            continue

        replacement = replacements.get(idx)
        if not replacement:
            revised.append(slide)
            continue

        revised.append(
            {
                **slide,
                "title": replacement["title"],
                "body": replacement["body"],
            }
        )

    return revised


def _probe_duration_seconds(media_path: Path) -> float:
    probe = ffmpeg.probe(str(media_path))
    duration = probe.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"Unable to read duration for {media_path}")
    return float(duration)


def _detect_scene_cuts(video_path: Path) -> list[float]:
    ffmpeg_exe = BIN_DIR / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        return []

    command = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-i",
        str(video_path),
        "-filter:v",
        "select='gt(scene,0.28)',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )

    output = (result.stderr or "") + "\n" + (result.stdout or "")
    matches = re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", output)

    cuts: list[float] = []
    for value in matches:
        try:
            cuts.append(float(value))
        except ValueError:
            continue

    return sorted({round(item, 3) for item in cuts})


def _select_video_study_timestamps(video_duration: float, scene_cuts: list[float]) -> list[float]:
    anchors = [
        max(0.5, video_duration * 0.08),
        max(0.8, video_duration * 0.32),
        max(1.1, video_duration * 0.56),
        max(1.4, video_duration * 0.82),
    ]

    for cut in scene_cuts[:6]:
        if 0.5 < cut < video_duration - 0.6:
            anchors.append(cut)

    unique: list[float] = []
    for ts in sorted(anchors):
        clamped = max(0.1, min(video_duration - 0.1, ts))
        if not unique or abs(clamped - unique[-1]) > 1.2:
            unique.append(clamped)

    return unique[:5]


def _extract_video_frame(video_path: Path, timestamp: float, frame_path: Path) -> bool:
    try:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_stream = ffmpeg.output(
            ffmpeg.input(str(video_path), ss=max(0.0, timestamp)),
            str(frame_path),
            vframes=1,
            format="image2",
            q=2,
        )
        ffmpeg.run(frame_stream, overwrite_output=True, quiet=True)
        return frame_path.exists() and frame_path.stat().st_size > 0
    except Exception:
        return False


def _study_video_content(
    client: OpenAI,
    video_path: Path,
    video_description: str,
    video_duration: float,
    scene_cuts: list[float],
) -> list[str]:
    timestamps = _select_video_study_timestamps(video_duration, scene_cuts)
    if not timestamps:
        return _fallback_video_beats(video_description, video_description)

    vision_model = os.getenv(
        "NVIDIA_NIM_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct"
    )

    study_dir = OUTPUT_DIR / ".video_study"
    notes: list[str] = []

    for index, timestamp in enumerate(timestamps, start=1):
        frame_path = study_dir / f"frame_{index:02}.jpg"
        if not _extract_video_frame(video_path, timestamp, frame_path):
            continue

        try:
            data_url = _image_to_data_url(frame_path)
            response = client.chat.completions.create(
                model=vision_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze video moments for voiceover planning. Return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Return JSON keys: action, technical_focus. "
                                    "Describe what user is doing and what technical effect is visible."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            raw = _extract_message_content(response.choices[0].message.content)
            payload = _extract_json_object(raw)
            action = _sanitize_technical_text(str(payload.get("action", "")), max_words=11)
            technical_focus = _sanitize_technical_text(str(payload.get("technical_focus", "")), max_words=10)

            if action and technical_focus:
                notes.append(f"{action} with {technical_focus}")
            elif action:
                notes.append(action)
        except Exception:
            continue

    if study_dir.exists():
        shutil.rmtree(study_dir, ignore_errors=True)

    if not notes:
        return _fallback_video_beats(video_description, video_description)

    deduped: list[str] = []
    for note in notes:
        clean = _sanitize_technical_text(note, max_words=14)
        if clean and clean not in deduped:
            deduped.append(clean)

    return deduped[:6]


def generate_copy(prompt_text: str, image_paths: list[Path], video_path: Path) -> dict[str, Any]:
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NVIDIA_NIM_API_KEY is missing in .env")

    model_name = os.getenv("NVIDIA_NIM_MODEL", "meta/llama-3.1-405b-instruct")
    client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")

    sections = parse_prompt_sections(prompt_text)
    slide_specs = parse_slide_specs(sections["slides_lines"], image_paths)
    video_script_source = _clean_video_script_source(sections["video_description"])

    image_analysis = analyze_images(client, image_paths)
    video_duration = _probe_duration_seconds(video_path)
    scene_cuts = _detect_scene_cuts(video_path)
    target_words = max(72, min(120, int(video_duration * 2.35)))
    slider_variation_seed, slider_variation_hint = _build_slider_variation_hint()

    image_filenames = [path.name for path in image_paths]

    posts_prompt = f"""
Create X and LinkedIn copy from project context and slide inputs.

Return JSON schema:
{{
  "x_post": "string",
  "x_image_plan": [{{"filename":"string","reason":"string"}}],
  "linkedin_rewrites": [{{"slide_number": 1, "title": "string", "body": "string"}}]
}}

Rules:
- Use ONLY the context and slide specs.
- Do not use video description for this request.
- Tone must be technical, professional, and a bit creative.
- No clickbait phrasing.
- linkedin_rewrites length must equal number of slide specs provided.
- Each LinkedIn title max 7 words.
- Each LinkedIn body max 20 words.
- x_post <= 280 chars with 2-4 relevant hashtags.
- Available image filenames: {json.dumps(image_filenames)}
- LinkedIn slider wording must vary by run seed while keeping factual consistency.
- Slider variation seed: {slider_variation_seed}
- Slider wording angle: {slider_variation_hint}

Context section:
{sections['context']}

Slide specs (ordered):
{json.dumps(slide_specs, indent=2)}

Additional image context (for X image selection only):
{json.dumps(image_analysis, indent=2)}
""".strip()

    posts_payload = _request_json(
        client=client,
        model=model_name,
        temperature=0.65,
        messages=[
            {
                "role": "system",
                "content": "You are a senior content strategist for professional tech posts. Output strict JSON only.",
            },
            {"role": "user", "content": posts_prompt},
        ],
    )

    x_post = _normalize_x_post(str(posts_payload.get("x_post", "")).strip(), sections["context"])
    if not x_post:
        raise ValueError("x_post is missing or empty in model response.")

    x_image_plan = _normalize_x_image_plan(posts_payload.get("x_image_plan"), image_paths)

    tiktok_segments = _segments_from_video_script(video_script_source, target_words)
    tiktok_script = _build_voiceover_script_from_video_script(video_script_source, target_words)

    linkedin_slides = _compose_linkedin_slides(
        slide_specs,
        posts_payload.get("linkedin_rewrites"),
        sections["context"],
        image_paths,
        variation_seed=slider_variation_seed,
    )
    linkedin_slides = _reread_linkedin_slides(
        client=client,
        model=model_name,
        draft_slides=linkedin_slides,
        slide_specs=slide_specs,
        context_text=sections["context"],
        variation_seed=slider_variation_seed,
        variation_hint=slider_variation_hint,
    )

    return {
        "x_post": x_post,
        "x_image_plan": x_image_plan,
        "tiktok_script": tiktok_script,
        "tiktok_segments": tiktok_segments,
        "linkedin_slides": linkedin_slides,
        "image_analysis": image_analysis,
        "video_duration": video_duration,
        "scene_cuts": scene_cuts,
        "slider_variation_seed": slider_variation_seed,
        "prompt_sections": sections,
    }


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)


def _ensure_kokoro_assets() -> None:
    model_url = os.getenv("TTS_KOKORO_MODEL_URL", KOKORO_MODEL_URL)
    voices_url = os.getenv("TTS_KOKORO_VOICES_URL", KOKORO_VOICES_URL)

    if not KOKORO_MODEL_PATH.exists():
        print("Downloading Kokoro model...")
        _download_file(model_url, KOKORO_MODEL_PATH)

    if not KOKORO_VOICES_PATH.exists():
        print("Downloading Kokoro voices...")
        _download_file(voices_url, KOKORO_VOICES_PATH)


def _get_kokoro_engine() -> Kokoro:
    global _KOKORO_ENGINE

    if _KOKORO_ENGINE is None:
        _ensure_kokoro_assets()
        _KOKORO_ENGINE = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))

    return _KOKORO_ENGINE


async def _edge_tts_synthesize(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
    volume: str,
    output_mp3: Path,
) -> list[dict[str, Any]]:
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed.")

    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    audio_chunks: list[bytes] = []
    word_boundaries: list[dict[str, Any]] = []
    sentence_boundaries: list[dict[str, Any]] = []

    async for chunk in communicator.stream():
        chunk_type = chunk.get("type")
        if chunk_type == "audio":
            data = chunk.get("data")
            if isinstance(data, bytes):
                audio_chunks.append(data)
            continue

        if chunk_type == "WordBoundary":
            text_token = str(chunk.get("text", "")).strip()
            if not text_token:
                continue

            offset = float(chunk.get("offset", 0)) / 10_000_000.0
            duration = float(chunk.get("duration", 0)) / 10_000_000.0
            word_boundaries.append(
                {
                    "start": max(0.0, offset),
                    "end": max(offset + 0.08, offset + duration),
                    "text": text_token,
                }
            )
            continue

        if chunk_type == "SentenceBoundary":
            text_token = str(chunk.get("text", "")).strip()
            if not text_token:
                continue

            offset = float(chunk.get("offset", 0)) / 10_000_000.0
            duration = float(chunk.get("duration", 0)) / 10_000_000.0
            sentence_boundaries.append(
                {
                    "start": max(0.0, offset),
                    "end": max(offset + 0.20, offset + duration),
                    "text": text_token,
                }
            )

    if not audio_chunks:
        raise RuntimeError("edge-tts returned no audio data.")

    if not word_boundaries and sentence_boundaries:
        try:
            chunk_size = int(os.getenv("TTS_SUBTITLE_WORDS_PER_CHUNK", "4"))
        except ValueError:
            chunk_size = 4
        chunk_size = max(2, min(6, chunk_size))

        for sentence in sentence_boundaries:
            phrase = re.sub(r"\s+", " ", str(sentence["text"])).strip()
            if not phrase:
                continue

            chunks = _chunk_words(phrase, chunk_size=chunk_size)
            if not chunks:
                continue

            sentence_start = float(sentence["start"])
            sentence_end = float(sentence["end"])
            sentence_duration = max(0.30, sentence_end - sentence_start)
            chunk_duration = sentence_duration / len(chunks)

            for index, chunk_text in enumerate(chunks):
                start = sentence_start + (index * chunk_duration)
                end = sentence_start + ((index + 1) * chunk_duration)
                word_boundaries.append(
                    {
                        "start": start,
                        "end": max(start + 0.08, end),
                        "text": chunk_text,
                    }
                )

    output_mp3.write_bytes(b"".join(audio_chunks))
    return word_boundaries


def _edge_tts_to_wav(
    script_text: str,
    output_path: Path,
    target_duration: float | None = None,
) -> tuple[float, float, str, list[dict[str, Any]]]:
    if edge_tts is None:
        raise RuntimeError("edge-tts backend unavailable.")

    voice = os.getenv("TTS_EDGE_VOICE", "en-US-ChristopherNeural").strip() or "en-US-ChristopherNeural"
    rate = os.getenv("TTS_EDGE_RATE", "+0%").strip() or "+0%"
    pitch = os.getenv("TTS_EDGE_PITCH", "+4Hz").strip() or "+4Hz"
    volume = os.getenv("TTS_EDGE_VOLUME", "+0%").strip() or "+0%"

    cleaned_script = _normalize_speech_text_for_tts(script_text, preserve_pauses=True)
    if not cleaned_script:
        raise RuntimeError("Empty script passed to edge-tts.")

    temp_mp3 = OUTPUT_DIR / ".voiceover_edge.mp3"
    if temp_mp3.exists():
        temp_mp3.unlink()

    word_boundaries = asyncio.run(
        _edge_tts_synthesize(cleaned_script, voice, rate, pitch, volume, temp_mp3)
    )

    try:
        ffmpeg.run(
            ffmpeg.output(
                ffmpeg.input(str(temp_mp3)),
                str(output_path),
                ac=1,
                ar=24000,
                acodec="pcm_s16le",
            ),
            overwrite_output=True,
            quiet=True,
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"FFmpeg failed converting edge-tts output to wav:\n{stderr}") from exc
    finally:
        if temp_mp3.exists():
            temp_mp3.unlink()

    duration = _probe_duration_seconds(output_path)

    if word_boundaries:
        max_boundary_end = max(float(item["end"]) for item in word_boundaries)
        if max_boundary_end > 0:
            time_scale = duration / max_boundary_end
            scaled_boundaries: list[dict[str, Any]] = []
            for item in word_boundaries:
                start = float(item["start"]) * time_scale
                end = float(item["end"]) * time_scale
                scaled_boundaries.append(
                    {
                        "start": max(0.0, start),
                        "end": max(start + 0.08, end),
                        "text": str(item["text"]),
                    }
                )
            word_boundaries = scaled_boundaries

    speed_hint = 1.0
    match = re.match(r"([+-]?\d+)%", rate)
    if match:
        speed_hint = max(0.7, min(1.3, 1.0 + (int(match.group(1)) / 100.0)))

    atempo: float | None = None
    if target_duration and target_duration > 0:
        if duration > target_duration * 1.01:
            desired_duration = target_duration * 0.995
            atempo = max(0.85, min(2.0, duration / desired_duration))

    if atempo is not None and abs(atempo - 1.0) > 0.01:
        adjusted_path = OUTPUT_DIR / ".voiceover_edge_adjusted.wav"
        try:
            ffmpeg.run(
                ffmpeg.output(
                    ffmpeg.input(str(output_path)).audio.filter("atempo", atempo),
                    str(adjusted_path),
                    ac=1,
                    ar=24000,
                    acodec="pcm_s16le",
                ),
                overwrite_output=True,
                quiet=True,
            )
            shutil.move(str(adjusted_path), str(output_path))
            duration = _probe_duration_seconds(output_path)
            speed_hint *= atempo
            adjusted_boundaries: list[dict[str, Any]] = []
            for item in word_boundaries:
                start = float(item["start"]) / atempo
                end = float(item["end"]) / atempo
                adjusted_boundaries.append(
                    {
                        "start": start,
                        "end": max(start + 0.08, end),
                        "text": str(item["text"]),
                    }
                )
            word_boundaries = adjusted_boundaries
        finally:
            if adjusted_path.exists():
                adjusted_path.unlink()

    return duration, speed_hint, voice, word_boundaries


def _elevenlabs_tts_to_wav(
    script_text: str,
    output_path: Path,
    target_duration: float | None = None,
) -> tuple[float, float, str, list[dict[str, Any]]]:
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not configured")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "ErXwobaYiN019PkySvjV").strip() or "ErXwobaYiN019PkySvjV"
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3").strip() or "eleven_v3"

    def _parse_float_env(name: str, default: float, low: float, high: float) -> float:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = float(raw)
        except ValueError:
            value = default
        return max(low, min(high, value))

    stability = _parse_float_env("ELEVENLABS_STABILITY", 0.35, 0.0, 1.0)
    similarity_boost = _parse_float_env("ELEVENLABS_SIMILARITY_BOOST", 0.78, 0.0, 1.0)
    style = _parse_float_env("ELEVENLABS_STYLE", 0.32, 0.0, 1.0)
    speaker_boost = os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").strip().lower() != "false"

    cleaned_script = _normalize_speech_text_for_tts(script_text)
    if not cleaned_script:
        raise RuntimeError("Empty script passed to elevenlabs backend")

    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=mp3_44100_128"

    temp_mp3 = OUTPUT_DIR / ".voiceover_elevenlabs.mp3"
    if temp_mp3.exists():
        temp_mp3.unlink()

    model_candidates = [model_id, "eleven_multilingual_v2", "eleven_flash_v2_5"]
    model_candidates = [candidate for i, candidate in enumerate(model_candidates) if candidate and candidate not in model_candidates[:i]]

    request_error: str | None = None
    for candidate_model in model_candidates:
        payload = {
            "text": cleaned_script,
            "model_id": candidate_model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": speaker_boost,
            },
        }
        request = urllib.request.Request(
            endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                temp_mp3.write_bytes(response.read())
            model_id = candidate_model
            request_error = None
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            request_error = f"HTTP {exc.code} {body}"
            continue
        except Exception as exc:
            request_error = str(exc)
            continue

    if request_error:
        raise RuntimeError(f"ElevenLabs request failed: {request_error}")

    try:
        ffmpeg.run(
            ffmpeg.output(
                ffmpeg.input(str(temp_mp3)),
                str(output_path),
                ac=1,
                ar=24000,
                acodec="pcm_s16le",
            ),
            overwrite_output=True,
            quiet=True,
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"FFmpeg failed converting elevenlabs output to wav:\n{stderr}") from exc
    finally:
        if temp_mp3.exists():
            temp_mp3.unlink()

    duration = _probe_duration_seconds(output_path)
    speed_hint = 1.0

    atempo: float | None = None
    if target_duration and target_duration > 0 and duration > target_duration * 1.12:
        desired_duration = target_duration * 0.98
        atempo = max(0.80, min(1.35, duration / desired_duration))

    if atempo is not None and abs(atempo - 1.0) > 0.01:
        adjusted_path = OUTPUT_DIR / ".voiceover_elevenlabs_adjusted.wav"
        try:
            ffmpeg.run(
                ffmpeg.output(
                    ffmpeg.input(str(output_path)).audio.filter("atempo", atempo),
                    str(adjusted_path),
                    ac=1,
                    ar=24000,
                    acodec="pcm_s16le",
                ),
                overwrite_output=True,
                quiet=True,
            )
            shutil.move(str(adjusted_path), str(output_path))
            duration = _probe_duration_seconds(output_path)
            speed_hint *= atempo
        finally:
            if adjusted_path.exists():
                adjusted_path.unlink()

    boundaries = _build_word_boundaries_from_script(cleaned_script, duration)
    return duration, speed_hint, f"elevenlabs:{voice_id}", boundaries


def _kokoro_tts_to_wav(
    script_text: str,
    output_path: Path,
    target_duration: float | None = None,
) -> tuple[float, float, str]:
    engine = _get_kokoro_engine()

    voice = os.getenv("TTS_KOKORO_VOICE", "am_fenrir")
    lang = os.getenv("TTS_KOKORO_LANG", "en-us")

    try:
        speed = float(os.getenv("TTS_KOKORO_SPEED", "1.02"))
    except ValueError:
        speed = 1.02
    speed = max(0.78, min(1.20, speed))

    available_voices = set(engine.get_voices())
    if voice not in available_voices:
        if "am_fenrir" in available_voices:
            voice = "am_fenrir"
        elif "am_michael" in available_voices:
            voice = "am_michael"
        elif available_voices:
            voice = sorted(available_voices)[0]

    cleaned_script = re.sub(r"\s+", " ", script_text).strip()


    best_samples = None
    best_sr = 24000
    best_duration = 0.0

    for _ in range(3):
        samples, sample_rate = engine.create(
            cleaned_script,
            voice=voice,
            speed=speed,
            lang=lang,
            trim=True,
        )
        duration = len(samples) / sample_rate

        best_samples = samples
        best_sr = sample_rate
        best_duration = duration

        if not target_duration or target_duration <= 0:
            break

        ratio = duration / target_duration
        if 0.92 <= ratio <= 1.08:
            break

        speed = max(0.78, min(1.20, speed * ratio))

    if best_samples is None:
        raise RuntimeError("Kokoro TTS failed to generate audio.")

    sf.write(str(output_path), best_samples, best_sr)
    return best_duration, speed, voice


def _tts_to_wav(
    script_text: str,
    output_path: Path,
    target_duration: float,
) -> tuple[float, float, str, str, list[dict[str, Any]] | None]:
    backend = os.getenv("TTS_BACKEND", "auto").strip().lower()
    errors: list[str] = []

    normalized_script = _normalize_speech_text_for_tts(script_text, preserve_pauses=True)

    if backend in {"auto", "elevenlabs"}:
        try:
            duration, speed, voice, word_boundaries = _elevenlabs_tts_to_wav(
                normalized_script,
                output_path,
                target_duration=target_duration,
            )
            return duration, speed, voice, "elevenlabs", word_boundaries

            if backend == "elevenlabs":
                duration, speed, voice, used_backend, word_boundaries = _run_tts_backend_with_pauses(
                    "elevenlabs",
                    script_text,
                    output_path,
                    target_duration,
                )
                return duration, speed, voice, used_backend, word_boundaries

            if backend == "edge":
                duration, speed, voice, used_backend, word_boundaries = _run_tts_backend_with_pauses(
                    "edge",
                    script_text,
                    output_path,
                    target_duration,
                )
                return duration, speed, voice, used_backend, word_boundaries

            if backend == "kokoro":
                duration, speed, voice, used_backend, word_boundaries = _run_tts_backend_with_pauses(
                    "kokoro",
                    script_text,
                    output_path,
                    target_duration,
                )
                return duration, speed, voice, used_backend, word_boundaries

            for candidate_backend in ("elevenlabs", "edge", "kokoro"):
                try:
                    duration, speed, voice, used_backend, word_boundaries = _run_tts_backend_with_pauses(
                        candidate_backend,
                        script_text,
                        output_path,
                        target_duration,
                    )
                    return duration, speed, voice, used_backend, word_boundaries
                except Exception as exc:
                    errors.append(f"{candidate_backend} failed: {exc}")

            raise RuntimeError("TTS generation failed. " + "; ".join(errors))
        except Exception as exc:
            errors.append(f"elevenlabs failed: {exc}")
            if backend == "elevenlabs":
                raise RuntimeError("; ".join(errors)) from exc

    if backend in {"auto", "edge"}:
        try:
            duration, speed, voice, word_boundaries = _edge_tts_to_wav(
                normalized_script,
                output_path,
                target_duration=target_duration,
            )
            return duration, speed, voice, "edge", word_boundaries
        except Exception as exc:
            errors.append(f"edge-tts failed: {exc}")
            if backend == "edge":
                raise RuntimeError("; ".join(errors)) from exc

    if backend in {"auto", "kokoro"}:
        try:
            duration, speed, voice = _kokoro_tts_to_wav(
                normalized_script,
                output_path,
                target_duration=target_duration,
            )
            boundaries = _build_word_boundaries_from_script(normalized_script, duration)
            return duration, speed, voice, "kokoro", boundaries
        except Exception as exc:
            errors.append(f"kokoro failed: {exc}")

    raise RuntimeError("TTS generation failed. " + "; ".join(errors))


def _seconds_to_srt(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    sec, ms = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"


def _ffmpeg_filter_path(path: Path) -> str:
    absolute_path = path.resolve()
    try:
        relative_path = absolute_path.relative_to(BASE_DIR)
        return relative_path.as_posix()
    except ValueError:
        return Path(os.path.relpath(absolute_path, BASE_DIR)).as_posix()


def _chunk_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    return [
        " ".join(words[index : index + chunk_size])
        for index in range(0, len(words), chunk_size)
    ]


def _build_beat_boundaries(
    beat_count: int,
    total_duration: float,
    scene_cuts: list[float],
    beat_texts: list[str],
) -> list[float]:
    if beat_count <= 0:
        return [0.0, total_duration]

    valid_cuts = [cut for cut in scene_cuts if 0.3 < cut < total_duration - 0.3]
    if len(valid_cuts) >= beat_count - 1:
        boundaries = [0.0]
        boundaries.extend(valid_cuts[: beat_count - 1])
        boundaries.append(total_duration)
        return boundaries

    weights = [max(3, len(text.split())) for text in beat_texts]
    total_weight = sum(weights) if weights else beat_count

    boundaries = [0.0]
    cumulative = 0.0
    for index in range(beat_count - 1):
        cumulative += weights[index] / total_weight
        boundaries.append(total_duration * cumulative)
    boundaries.append(total_duration)

    return boundaries


def _build_timed_subtitles_from_beats(
    beats: list[str],
    total_duration: float,
    scene_cuts: list[float],
) -> list[dict[str, Any]]:
    if not beats:
        return []

    cleaned_beats = [
        _sanitize_technical_text(beat, max_words=16).rstrip(" .")
        for beat in beats
        if beat and beat.strip()
    ]
    cleaned_beats = [beat for beat in cleaned_beats if beat]
    if not cleaned_beats:
        return []

    boundaries = _build_beat_boundaries(len(cleaned_beats), total_duration, scene_cuts, cleaned_beats)

    subtitle_segments: list[dict[str, Any]] = []
    for index, beat in enumerate(cleaned_beats):
        beat_start = float(boundaries[index])
        beat_end = float(boundaries[index + 1])
        beat_duration = max(0.4, beat_end - beat_start)

        chunks = _chunk_words(beat, chunk_size=3)
        if not chunks:
            continue

        chunk_duration = beat_duration / len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            start = beat_start + chunk_index * chunk_duration
            end = beat_start + (chunk_index + 1) * chunk_duration
            if end - start < 0.30:
                continue

            subtitle_segments.append(
                {
                    "start": max(0.0, start),
                    "end": min(total_duration, end),
                    "text": chunk,
                }
            )

    return subtitle_segments


def _build_subtitles_from_voiceover_audio(
    word_boundaries: list[dict[str, Any]] | None,
    total_duration: float,
    fallback_beats: list[str],
    scene_cuts: list[float],
) -> list[dict[str, Any]]:
    if not word_boundaries:
        return _build_timed_subtitles_from_beats(
            fallback_beats,
            total_duration=total_duration,
            scene_cuts=scene_cuts,
        )

    subtitle_segments: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []

    try:
        words_per_chunk = int(os.getenv("TTS_SUBTITLE_WORDS_PER_CHUNK", "4"))
    except ValueError:
        words_per_chunk = 4
    words_per_chunk = max(2, min(6, words_per_chunk))

    def flush_bucket() -> None:
        if not bucket:
            return

        start = float(bucket[0]["start"])
        end = float(bucket[-1]["end"])
        text = " ".join(str(item["text"]) for item in bucket)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            bucket.clear()
            return

        if end - start < 0.25:
            end = start + 0.25

        subtitle_segments.append(
            {
                "start": max(0.0, start),
                "end": min(total_duration, end),
                "text": text,
            }
        )
        bucket.clear()

    for word in word_boundaries:
        bucket.append(word)
        duration = float(bucket[-1]["end"]) - float(bucket[0]["start"])
        token = str(word["text"])

        should_split = (
            len(bucket) >= words_per_chunk
            or duration >= 1.35
            or token.endswith((".", "?", "!", ";", ",", ":"))
        )
        if should_split:
            flush_bucket()

    flush_bucket()

    cleaned_segments: list[dict[str, Any]] = []
    previous_end = 0.0
    for segment in subtitle_segments:
        start = max(previous_end, float(segment["start"]))
        end = max(start + 0.25, float(segment["end"]))
        if start >= total_duration:
            break
        cleaned_segments.append(
            {
                "start": start,
                "end": min(total_duration, end),
                "text": segment["text"],
            }
        )
        previous_end = cleaned_segments[-1]["end"]

    if cleaned_segments:
        return cleaned_segments

    return _build_timed_subtitles_from_beats(
        fallback_beats,
        total_duration=total_duration,
        scene_cuts=scene_cuts,
    )


def _write_srt(segments: list[dict[str, Any]], subtitles_path: Path) -> None:
    with subtitles_path.open("w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments, start=1):
            text = re.sub(r"\s+", " ", str(segment["text"])).strip()
            if not text:
                continue

            start_time = _seconds_to_srt(float(segment["start"]))
            end_time = _seconds_to_srt(float(segment["end"]))
            srt_file.write(f"{index}\n{start_time} --> {end_time}\n{text}\n\n")


def build_tiktok_video(
    video_path: Path,
    script_text: str,
    script_beats: list[str],
    scene_cuts: list[float],
) -> None:
    if not script_text.strip():
        raise ValueError("TikTok script is empty.")

    voiceover_path = OUTPUT_DIR / "voiceover.wav"
    subtitles_path = OUTPUT_DIR / "subtitles.srt"
    final_video_path = OUTPUT_DIR / "tiktok_final.mp4"

    video_duration = _probe_duration_seconds(video_path)
    tts_script = _normalize_speech_text_for_tts(script_text, preserve_pauses=True)
    voice_duration, used_speed, used_voice, used_backend, word_boundaries = _tts_to_wav(
        tts_script,
        voiceover_path,
        target_duration=video_duration,
    )

    subtitle_segments = _build_subtitles_from_voiceover_audio(
        word_boundaries=word_boundaries,
        total_duration=video_duration,
        fallback_beats=script_beats,
        scene_cuts=scene_cuts,
    )
    if not subtitle_segments:
        fallback_boundaries = _build_word_boundaries_from_script(tts_script, video_duration)
        subtitle_segments = _build_subtitles_from_voiceover_audio(
            word_boundaries=fallback_boundaries,
            total_duration=video_duration,
            fallback_beats=script_beats,
            scene_cuts=scene_cuts,
        )
    if not subtitle_segments:
        raise RuntimeError("Subtitle generation failed: no subtitle segments were produced.")

    _write_srt(subtitle_segments, subtitles_path)

    video_input = ffmpeg.input(str(video_path))
    processed_video = (
        video_input.video
        .filter("scale", 1080, 1920, force_original_aspect_ratio="increase")
        .filter("crop", 1080, 1920)
    )

    voice_audio = (
        ffmpeg.input(str(voiceover_path))
        .audio.filter("highpass", f=70)
        .filter("lowpass", f=13200)
        .filter("acompressor", threshold=0.10, ratio=1.8, attack=12, release=110)
        .filter("equalizer", f=3200, width_type="q", width=1.0, g=2.2)
        .filter("loudnorm", i=-16, lra=7, tp=-1.5)
    )

    processed_audio = (
        voice_audio.filter("apad")
        .filter("atrim", end=video_duration)
        .filter("asetpts", "N/SR/TB")
    )

    try:
        subtitle_margin_v = int(os.getenv("TTS_SUBTITLE_MARGIN_V", "180"))
    except ValueError:
        subtitle_margin_v = 180

    try:
        subtitle_margin_l = int(os.getenv("TTS_SUBTITLE_MARGIN_L", "54"))
    except ValueError:
        subtitle_margin_l = 54

    try:
        subtitle_font_size = int(os.getenv("TTS_SUBTITLE_FONT_SIZE", "34"))
    except ValueError:
        subtitle_font_size = 34

    subtitle_font_name = os.getenv("TTS_SUBTITLE_FONT_NAME", "Syncopate").strip() or "Syncopate"

    try:
        subtitle_alignment = int(os.getenv("TTS_SUBTITLE_ALIGNMENT", "2"))
    except ValueError:
        subtitle_alignment = 2
    subtitle_alignment = max(1, min(9, subtitle_alignment))

    force_style = (
        f"Alignment={subtitle_alignment},FontName={subtitle_font_name},FontSize={subtitle_font_size},Bold=1,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H88000000,"
        f"BorderStyle=3,Outline=3,Shadow=0,MarginL={subtitle_margin_l},MarginR={subtitle_margin_l},MarginV={subtitle_margin_v}"
    )

    fontsdir = FONTS_DIR if FONTS_DIR.exists() else BASE_DIR
    subtitled_video = processed_video.filter(
        "subtitles",
        _ffmpeg_filter_path(subtitles_path),
        fontsdir=_ffmpeg_filter_path(fontsdir),
        force_style=force_style,
    )

    output_stream = ffmpeg.output(
        subtitled_video,
        processed_audio,
        str(final_video_path),
        t=video_duration,
        vcodec="libx264",
        acodec="aac",
        audio_bitrate="192k",
        pix_fmt="yuv420p",
        movflags="+faststart",
    )

    try:
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"FFmpeg failed while building TikTok video:\n{stderr}") from exc

    print(
        "TikTok TTS settings "
        f"backend={used_backend}, voice={used_voice}, speed={used_speed:.2f}, voiceover_duration={voice_duration:.2f}s, "
        f"video_duration={video_duration:.2f}s"
    )


def build_tiktok_voiceover_audio(video_path: Path, script_text: str) -> None:
    if not script_text.strip():
        raise ValueError("TikTok script is empty.")

    voiceover_path = OUTPUT_DIR / "voiceover.wav"
    video_duration = _probe_duration_seconds(video_path)
    tts_script = _normalize_speech_text_for_tts(script_text, preserve_pauses=True)
    voice_duration, used_speed, used_voice, used_backend, _ = _tts_to_wav(
        tts_script,
        voiceover_path,
        target_duration=video_duration,
    )

    print(
        "TikTok audio-only TTS settings "
        f"backend={used_backend}, voice={used_voice}, speed={used_speed:.2f}, voiceover_duration={voice_duration:.2f}s, "
        f"video_duration={video_duration:.2f}s"
    )


def _load_font(candidates: list[Path], font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def _title_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load_font([FONTS_DIR / "Syncopate-Bold.ttf"], font_size)


def _body_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load_font(
        [FONTS_DIR / "Syncopate-Regular.ttf", FONTS_DIR / "Syncopate-Bold.ttf"],
        font_size,
    )


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])
        left, _, right, _ = draw.textbbox((0, 0), candidate, font=font)
        if right - left <= max_width or not current:
            current.append(word)
            continue

        lines.append(" ".join(current))
        current = [word]

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def _draw_linkedin_text_block(
    image: Image.Image,
    title: str,
    body: str,
    y_start: int,
    y_end: int | None = None,
    single_line_title: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)

    if y_end is None:
        y_end = SQUARE_SIZE - 40

    max_width = 920

    best_layout: tuple[ImageFont.ImageFont, ImageFont.ImageFont, str, str, int] | None = None
    for shrink_step in range(0, 10):
        title_size = max(28, 42 - (shrink_step * 2))
        body_size = max(20, 26 - shrink_step)

        title_font = _title_font(title_size)
        body_font = _body_font(body_size)

        if single_line_title:
            title_wrapped = _sanitize_technical_text(title, max_words=4)
        else:
            title_wrapped = _wrap_text(draw, title, title_font, max_width)
        body_wrapped = _wrap_text(draw, body, body_font, max_width)

        title_bbox = draw.multiline_textbbox((0, 0), title_wrapped, font=title_font, spacing=8)
        body_bbox = draw.multiline_textbbox((0, 0), body_wrapped, font=body_font, spacing=7)
        total_height = (title_bbox[3] - title_bbox[1]) + 18 + (body_bbox[3] - body_bbox[1])

        if y_start + total_height <= y_end:
            best_layout = (title_font, body_font, title_wrapped, body_wrapped, total_height)
            break

    if best_layout is None:
        title_font = _title_font(28)
        body_font = _body_font(20)
        if single_line_title:
            title_wrapped = _sanitize_technical_text(title, max_words=4)
        else:
            title_wrapped = _wrap_text(draw, title, title_font, max_width)

        trimmed_body = body
        while trimmed_body:
            body_wrapped = _wrap_text(draw, trimmed_body, body_font, max_width)
            title_bbox = draw.multiline_textbbox((0, 0), title_wrapped, font=title_font, spacing=8)
            body_bbox = draw.multiline_textbbox((0, 0), body_wrapped, font=body_font, spacing=7)
            total_height = (title_bbox[3] - title_bbox[1]) + 18 + (body_bbox[3] - body_bbox[1])
            if y_start + total_height <= y_end:
                best_layout = (title_font, body_font, title_wrapped, body_wrapped, total_height)
                break

            words = trimmed_body.split()
            if len(words) <= 6:
                break
            trimmed_body = " ".join(words[:-2]).rstrip(" ,.;:") + "..."

    if best_layout is None:
        title_font = _title_font(28)
        body_font = _body_font(20)
        if single_line_title:
            title_wrapped = _sanitize_technical_text(_truncate_words(title, 4), max_words=4)
        else:
            title_wrapped = _wrap_text(draw, _truncate_words(title, 6), title_font, max_width)
        body_wrapped = _wrap_text(draw, _truncate_words(body, 12), body_font, max_width)
    else:
        title_font, body_font, title_wrapped, body_wrapped, _ = best_layout

    left_margin = 74
    draw.rectangle([left_margin, y_start - 14, left_margin + 140, y_start - 8], fill=BRAND_RED)

    draw.multiline_text(
        (left_margin, y_start),
        title_wrapped,
        font=title_font,
        fill=BRAND_WHITE,
        spacing=8,
    )

    title_bbox = draw.multiline_textbbox((left_margin, y_start), title_wrapped, font=title_font, spacing=8)
    body_y = title_bbox[3] + 18

    draw.multiline_text(
        (left_margin, body_y),
        body_wrapped,
        font=body_font,
        fill=BRAND_CYAN,
        spacing=7,
    )


def _stamp_linkedin_logo(canvas: Image.Image) -> None:
    if not LINKEDIN_LOGO_PATH.exists():
        return

    try:
        with Image.open(LINKEDIN_LOGO_PATH) as raw_logo:
            logo = ImageOps.exif_transpose(raw_logo).convert("RGBA")
    except Exception:
        return

    if logo.width <= 0 or logo.height <= 0:
        return

    scale = min(
        LINKEDIN_LOGO_MAX_WIDTH / logo.width,
        LINKEDIN_LOGO_MAX_HEIGHT / logo.height,
        1.0,
    )
    if scale < 1.0:
        logo = logo.resize(
            (
                max(1, int(logo.width * scale)),
                max(1, int(logo.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    x = max(0, canvas.width - logo.width - LINKEDIN_LOGO_MARGIN)
    y = max(0, canvas.height - logo.height - LINKEDIN_LOGO_MARGIN)
    canvas.paste(logo, (x, y), logo)


def _render_linkedin_image_slide(
    source_image: Path,
    title: str,
    body: str,
    output_image: Path,
) -> None:
    with Image.open(source_image) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")

    canvas = Image.new("RGB", (SQUARE_SIZE, SQUARE_SIZE), BRAND_BLACK)

    scale = min(
        SQUARE_SIZE / image.width,
        TOP_IMAGE_AREA_HEIGHT / image.height,
        1.0,
    )
    if scale < 1.0:
        resized = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    else:
        resized = image

    x = (SQUARE_SIZE - resized.width) // 2
    canvas.paste(resized, (x, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, TOP_IMAGE_AREA_HEIGHT, SQUARE_SIZE, SQUARE_SIZE), fill=BRAND_BLACK)

    _draw_linkedin_text_block(
        canvas,
        title,
        body,
        y_start=TOP_IMAGE_AREA_HEIGHT + 36,
        y_end=SQUARE_SIZE - 36,
    )

    _stamp_linkedin_logo(canvas)

    canvas.save(
        output_image,
        format="JPEG",
        quality=84,
        optimize=True,
        progressive=True,
    )


def _render_linkedin_plain_slide(title: str, body: str, output_image: Path) -> None:
    image = Image.new("RGB", (SQUARE_SIZE, SQUARE_SIZE), BRAND_BLACK)
    draw = ImageDraw.Draw(image)

    title = _sanitize_technical_text(title, max_words=4) or "Technical Summary"

    draw.rectangle((0, 0, SQUARE_SIZE, 96), fill=(12, 12, 20))
    draw.rectangle((0, SQUARE_SIZE - 96, SQUARE_SIZE, SQUARE_SIZE), fill=(12, 12, 20))

    _draw_linkedin_text_block(
        image,
        title,
        body,
        y_start=220,
        y_end=SQUARE_SIZE - 120,
        single_line_title=True,
    )

    _stamp_linkedin_logo(image)

    image.save(
        output_image,
        format="JPEG",
        quality=86,
        optimize=True,
        progressive=True,
    )


def _jpg_to_pdf(jpg_path: Path, pdf_path: Path) -> None:
    with Image.open(jpg_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=100.0)


def build_linkedin_carousel(image_paths: list[Path], slides: list[dict[str, Any]]) -> None:
    image_lookup = {path.name: path for path in image_paths}

    temp_images: list[Path] = []
    temp_pdfs: list[Path] = []

    for index, slide in enumerate(slides, start=1):
        temp_image = OUTPUT_DIR / f".linkedin_slide_{index:02}.jpg"
        temp_pdf = OUTPUT_DIR / f".linkedin_slide_{index:02}.pdf"

        title = str(slide.get("title", "Technical Focus")).strip() or "Technical Focus"
        body = str(slide.get("body", "")).strip() or "Technical implementation detail."
        style = str(slide.get("style", "plain")).strip().lower()
        image_filename = slide.get("image_filename")

        if style == "image" and isinstance(image_filename, str) and image_filename in image_lookup:
            _render_linkedin_image_slide(image_lookup[image_filename], title, body, temp_image)
        else:
            _render_linkedin_plain_slide(title, body, temp_image)

        _jpg_to_pdf(temp_image, temp_pdf)
        temp_images.append(temp_image)
        temp_pdfs.append(temp_pdf)

    writer = PdfWriter()
    for pdf_path in temp_pdfs:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)

    final_pdf = OUTPUT_DIR / "linkedin_carousel.pdf"
    with final_pdf.open("wb") as output_file:
        writer.write(output_file)

    for temp_file in [*temp_images, *temp_pdfs]:
        if temp_file.exists():
            temp_file.unlink()

    final_size_mb = final_pdf.stat().st_size / (1024 * 1024)
    if final_size_mb > MAX_LINKEDIN_PDF_MB:
        print(
            f"Warning: linkedin_carousel.pdf is {final_size_mb:.2f} MB "
            f"(target < {MAX_LINKEDIN_PDF_MB} MB)."
        )


def write_x_outputs(x_post: str, x_image_plan: list[dict[str, str]]) -> None:
    x_post_path = OUTPUT_DIR / "x_post.txt"

    lines = [x_post.strip(), "", "Recommended images for this post:"]
    for index, item in enumerate(x_image_plan, start=1):
        lines.append(f"{index}. {item['filename']} - {item['reason']}")

    x_post_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_environment()

    print("[1/5] Discovering assets...")
    video_path, image_paths, prompt_text = find_assets()
    print(f"Found video: {video_path.name}")
    print(f"Found images: {len(image_paths)}")

    print("[2/5] Generating prompt-structured copy with NVIDIA NIM...")
    content = generate_copy(prompt_text, image_paths=image_paths, video_path=video_path)

    generated_copy_path = OUTPUT_DIR / "generated_copy.json"
    generated_copy_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Saved generated copy JSON: {generated_copy_path.name}")

    print("[3/5] Writing X post and image plan...")
    write_x_outputs(content["x_post"], content["x_image_plan"])
    print("Saved X post: x_post.txt")

    print("[4/5] Building TikTok video (premium/edge/kokoro TTS + sync subtitles)...")
    build_tiktok_video(
        video_path,
        content["tiktok_script"],
        content["tiktok_segments"],
        content["scene_cuts"],
    )
    print("Saved TikTok video: tiktok_final.mp4")

    print("[5/5] Building LinkedIn square carousel...")
    build_linkedin_carousel(image_paths, content["linkedin_slides"])
    print("Saved LinkedIn carousel: linkedin_carousel.pdf")

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        sys.exit(1)
