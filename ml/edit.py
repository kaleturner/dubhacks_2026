"""Cut a graded music edit next to the raw clip.

The track is City Loop (wipics, CC0), in ml/music/. The raw file is left
alone. Phones play both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

MUSIC = Path(__file__).resolve().parent / "music" / "city-loop.mp3"

_FONTS = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/FiraSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    # macOS, where the server runs next to the arm; without these the title was skipped.
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)


def cut_edit(video_path: str, out_path: str, duration_s: float, title: str) -> str:
    """Regrade video_path, lay the CC0 track under it, and write out_path."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(video_path)
    if not MUSIC.is_file():
        raise FileNotFoundError(str(MUSIC))

    span = min(max(0.1, float(duration_s)), _duration(source))
    fade_out = max(0.0, span - 0.45)
    audio_out = max(0.0, span - 0.6)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        text_file = Path(tmp) / "title.txt"
        text_file.write_text(_title(title) + "\n")
        video = "[0:v]" + ",".join(_video_filters(span, fade_out, text_file))
        audio = (
            f"[1:a]atrim=0:{span:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d=0.3,afade=t=out:st={audio_out:.3f}:d=0.55,"
            "volume=0.9[a]"
        )
        command = [
            ffmpeg, "-y",
            "-i", str(source),
            "-stream_loop", "-1", "-i", str(MUSIC),
            "-filter_complex", f"{video};{audio}",
            "-map", "[v]", "-map", "[a]",
            "-t", f"{span:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.is_file():
        detail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "edit failed")
    return str(destination)


def _video_filters(span: float, fade_out: float, text_file: Path) -> list[str]:
    filters = [
        "crop=w=iw*0.94:h=ih*0.94:"
        "x='(in_w-out_w)/2+8*sin(t*1.3)':"
        "y='(in_h-out_h)/2+5*cos(t*0.9)'",
        "scale=640:480:flags=lanczos",
        "eq=contrast=1.1:saturation=1.5:brightness=0.02",
        "unsharp=5:5:0.7",
        "vignette=PI/4",
        "fade=t=in:st=0:d=0.18:color=0x11130F",
        f"fade=t=out:st={fade_out:.3f}:d=0.4:color=0x11130F",
    ]
    font = _font()
    if font is not None:
        filters.append(
            "drawtext="
            f"fontfile={_filter_path(font)}:"
            f"textfile={_filter_path(text_file)}:"
            "fontsize=34:fontcolor=0xF4F1EA:"
            "borderw=3:bordercolor=0x11130F@0.7:"
            "x=(w-text_w)/2:y=h-th-26:"
            "enable='lt(t,2.8)'"
        )
    filters.append("setsar=1[v]")
    return filters


def _title(title: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == " " else " " for ch in title)
    words = " ".join(cleaned.split()).upper()
    return (words[:28] or "CUT")


def _font() -> Path | None:
    for candidate in _FONTS:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def _filter_path(path: Path) -> str:
    text = path.resolve().as_posix()
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _duration(source: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is not installed")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(source)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("could not read the clip length")
    return float(json.loads(result.stdout)["format"]["duration"])
