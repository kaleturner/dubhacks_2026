"""Speech-to-text for POST /transcribe.

Phones that have no speech recognizer upload a short clip. Mk1.5 accepts
WAV, MP3, and FLAC, so ffmpeg converts the browser recording first.
The response stays {"text": "..."} from shared.schemas.TranscribeResult.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

MODEL = "perceptron-mk1.5"
TIMEOUT_S = 20
MAX_BYTES = 8 * 1024 * 1024

_PROMPT = (
    "Transcribe the speech verbatim. Return only the words spoken, "
    "with no quotes and no commentary. "
    "Mark a word you cannot hear as [unclear]. "
    "If there is no speech, return an empty string."
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

_EMPTY = {"[unclear]", "unclear", "(unclear)", "no speech", "silence"}


def transcribe_bytes(data: bytes, content_type: str = "", filename: str = "") -> str:
    """Return the spoken words, or an empty string when the clip has none."""
    if not data:
        raise RuntimeError("empty recording")
    if len(data) > MAX_BYTES:
        raise RuntimeError("recording is too long")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        wav_path = _to_wav(data, _suffix(content_type, filename), folder)
        return _clean(_call(wav_path))


def _suffix(content_type: str, filename: str) -> str:
    label = f"{content_type} {filename}".lower()
    if "webm" in label:
        return ".webm"
    if "ogg" in label:
        return ".ogg"
    if "mpeg" in label or "mp3" in label:
        return ".mp3"
    if "flac" in label:
        return ".flac"
    if "mp4" in label or "m4a" in label or "aac" in label:
        return ".m4a"
    if "wav" in label or "wave" in label:
        return ".wav"
    return ".webm"


def _to_wav(data: bytes, suffix: str, folder: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")
    source = folder / f"in{suffix}"
    destination = folder / "speech.wav"
    source.write_bytes(data)
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-ac", "1", "-ar", "16000", str(destination)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not destination.is_file():
        detail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "could not read that recording")
    return destination


def _call(wav_path: Path) -> str:
    try:
        from perceptron import audio
    except ImportError as exc:
        raise RuntimeError("perceptron is not installed") from exc

    from ml.perception import _answer, _sdk

    response = _sdk().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    audio(str(wav_path)),
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
        max_completion_tokens=200,
        timeout=TIMEOUT_S,
        reasoning_effort="none",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "transcript", "strict": True, "schema": _SCHEMA},
        },
    )
    payload = json.loads(_answer(response))
    if not isinstance(payload, dict):
        raise RuntimeError("Mk1.5 returned JSON that is not an object")
    return str(payload.get("text", ""))


def _clean(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'").strip()
    if cleaned.lower() in _EMPTY:
        return ""
    return cleaned
