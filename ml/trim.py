"""ffmpeg trim helper. ML owns this. Backend calls trim_clip during POST."""

import json
import shutil
import subprocess
from pathlib import Path


def trim_clip(video_path: str, start_s: float, duration_s: float, out_path: str) -> str:
    """Cut a clip of exactly duration_s, with a short fade in and out.

    Input is the wrist camera's mp4v file from robot.cameras (640x480, 30 fps).
    Re-encode to H.264 + yuv420p + faststart: phones will not play mp4v, and the
    fades need a re-encode anyway. Return out_path.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(video_path)

    start, duration, pad = _window(source, start_s, duration_s)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if pad >= 0.01:
        filters.append(f"tpad=stop_mode=clone:stop_duration={pad:.3f}")
    fade = min(0.4, duration / 4)
    if fade >= 0.05:
        fade_out = max(0.0, duration - fade)
        filters.append(f"fade=t=in:st=0:d={fade:.3f},fade=t=out:st={fade_out:.3f}:d={fade:.3f}")
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    if filters:
        command[8:8] = ["-vf", ",".join(filters)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.is_file():
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "ffmpeg failed")
    return str(destination)


def _window(source: Path, start_s: float, duration_s: float) -> tuple[float, float, float]:
    """Keep a full-length cut when the suggested start runs past the file.

    The third value is how long to hold the last frame when the take is short.
    """
    start = max(0.0, float(start_s))
    duration = max(0.1, float(duration_s))
    length = _duration(source)
    if length is None:
        return start, duration, 0.0
    if length + 0.04 < duration:
        return 0.0, duration, duration - length
    if start + duration > length:
        start = max(0.0, length - duration)
    return start, duration, 0.0


def _duration(source: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
