"""ffmpeg trim helper. ML owns this. Backend calls trim_clip during POST."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

# Light stabilization, then a gentle grade: the wrist cam is flat and a little soft.
LOOK = "deshake=rx=32:ry=32:edge=clamp,eq=contrast=1.08:saturation=1.3:gamma=1.03,unsharp=5:5:0.5"
CROSSFADE_S = 0.4


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
    filters = [LOOK]
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


def concat_clips(clip_paths: list[str], out_path: str, crossfade_s: float = 0.0) -> str:
    """Join clips made by trim_clip into one file. They share a codec, so a plain join
    needs no re-encode. With crossfade_s each cut dissolves into the next; every clip
    but the last must then be crossfade_s longer than its share of the total."""
    if not clip_paths:
        raise ValueError("no clips to join")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], destination)
        return str(destination)
    if crossfade_s > 0:
        return _crossfade(ffmpeg, clip_paths, destination, crossfade_s)
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "clips.txt"
        lines = []
        for path in clip_paths:
            quoted = str(Path(path).resolve()).replace("'", "'\\''")
            lines.append(f"file '{quoted}'")
        listing.write_text("\n".join(lines) + "\n")
        command = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", "-movflags", "+faststart", str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.is_file():
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "ffmpeg concat failed")
    return str(destination)


def _crossfade(ffmpeg: str, clip_paths: list[str], destination: Path, fade: float) -> str:
    inputs, chain, offset, last = [], [], 0.0, "[0:v]"
    for path in clip_paths:
        inputs += ["-i", str(path)]
    for i in range(1, len(clip_paths)):
        offset += (_duration(Path(clip_paths[i - 1])) or 0.0) - fade
        out = f"[x{i}]"
        chain.append(f"{last}[{i}:v]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f}{out}")
        last = out
    command = [
        ffmpeg, "-y", *inputs, "-filter_complex", ";".join(chain), "-map", last, "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.is_file():
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "ffmpeg crossfade failed")
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
