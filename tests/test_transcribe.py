import math
import shutil
import struct
import wave
from io import BytesIO
from pathlib import Path

import pytest

from ml.transcribe import transcribe_bytes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def tone_wav() -> bytes:
    buffer = BytesIO()
    rate = 16000
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(rate // 2)
        )
        handle.writeframes(frames)
    return buffer.getvalue()


def test_transcribe_bytes_returns_cleaned_words(monkeypatch):
    def fake_call(path: Path) -> str:
        assert path.suffix == ".wav"
        assert path.stat().st_size > 44
        return '"record a 15 second video of this apple"'

    monkeypatch.setattr("ml.transcribe._call", fake_call)
    assert transcribe_bytes(tone_wav(), "audio/wav", "speech.wav") == (
        "record a 15 second video of this apple"
    )


def test_transcribe_bytes_drops_an_unclear_clip(monkeypatch):
    monkeypatch.setattr("ml.transcribe._call", lambda path: "[unclear]")
    assert transcribe_bytes(tone_wav(), "audio/wav") == ""


def test_transcribe_bytes_rejects_empty_audio():
    with pytest.raises(RuntimeError, match="empty recording"):
        transcribe_bytes(b"", "audio/webm")
