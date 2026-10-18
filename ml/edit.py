"""Cut a graded music edit next to the raw clip.

Three CC0 electronic beds live in ml/music/, listed in shared/tracks.json.
The prompt, including the model's style, picks which one plays. The raw file
is left alone. The edit crops the white gripper off the bottom, drops stretches
where the camera is sitting still, and cuts those pieces so each one lands on
that bed's own pulse. Phones play the edit first.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

MUSIC_DIR = Path(__file__).resolve().parent / "music"
MUSIC = MUSIC_DIR / "city-loop.mp3"
CATALOG = Path(__file__).resolve().parents[1] / "shared" / "tracks.json"
FPS = 30

_FONTS = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/FiraSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    # macOS, where the server runs next to the arm; without these the title was skipped.
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)

# Warm the flat wrist picture and lift contrast, on top of the saturation the
# earlier grade already used. The vignette and title stay.
_GRADE = (
    "eq=contrast=1.16:brightness=0.02:saturation=1.5:gamma=0.96,"
    "colorbalance=rs=0.05:gs=0.015:bs=-0.055:rm=0.04:gm=0.0:bm=-0.03:"
    "rh=0.05:gh=0.02:bh=-0.04,"
    "unsharp=5:5:0.7,"
    "vignette=PI/4"
)

_GRIDS: dict[str, tuple[float, float] | None] = {}


def beds() -> list[dict]:
    rows = json.loads(CATALOG.read_text())
    for row in rows:
        row["path"] = str(MUSIC_DIR / row["file"])
    return rows


def _ranked(prompt: str) -> list[dict]:
    text = (prompt or "").lower()
    tokens = set(re.findall(r"[a-z0-9']+", text))
    scored = []
    for index, bed in enumerate(beds()):
        total = 0
        for word in bed["words"]:
            if " " in word and word in text:
                total += 2
            elif word in tokens:
                total += 1
        scored.append((total, index, bed))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [bed for _, _, bed in scored]


def choose_track(prompt: str) -> dict:
    """The bed whose words best match the prompt and the model's style."""
    return _ranked(prompt)[0]


def cut_edit(video_path: str, out_path: str, duration_s: float, title: str, prompt: str = "") -> str:
    """Regrade video_path, lay the chosen CC0 track under it, and write out_path."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(video_path)
    music = Path(choose_track(prompt or title)["path"])
    if not music.is_file():
        raise FileNotFoundError(str(music))

    width, height, length = _geometry(source)
    span = min(max(0.1, float(duration_s)), length)
    ranges = _fit_beats(_interesting(source, span), span, _beat_period(music))
    if not ranges:
        ranges = [(0.0, span)]
    claw = _claw_amount(source, width, height, span)
    crop = _crop_window(width, height, claw)
    out_span = sum(end - start for start, end in ranges)

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        text_file = Path(tmp) / "title.txt"
        text_file.write_text(_title(title) + "\n")
        graph = _graph(ranges, crop, out_span, text_file, _beat_offset(music))
        command = [
            ffmpeg, "-y",
            "-i", str(source),
            "-stream_loop", "-1", "-i", str(music),
            "-filter_complex", graph,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        detail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "edit failed")
    return str(destination)


def _interesting(source: Path, span: float) -> list[tuple[float, float]]:
    """Keep the moving parts. A take that never really sits still is kept whole."""
    sampled = _motion(source, span)
    if sampled is None:
        return [(0.0, span)]
    times, scores = sampled
    ranges = _interesting_ranges(times, scores, span)
    kept = sum(end - start for start, end in ranges)
    if kept < 1.0 and kept < span * 0.45:
        return [(0.0, span)]
    return ranges


def _interesting_ranges(times: np.ndarray, scores: np.ndarray, duration: float) -> list[tuple[float, float]]:
    if len(scores) < 6:
        return [(0.0, duration)]
    low, high = np.percentile(scores, [25, 75])
    # No split worth making: the shot is quiet throughout, or it moves the whole time.
    if high < 0.55 or (low > 0 and high < low * 2.2):
        return [(0.0, duration)]
    thresh = max(0.4, float(high) * 0.22)
    active = scores >= thresh
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 12
    filled = _fill_gaps(active, max(1, int(round(0.35 / step))))
    min_run = max(1, int(round(0.5 / step)))
    ranges: list[tuple[float, float]] = []
    index = 0
    while index < len(filled):
        if not filled[index]:
            index += 1
            continue
        end = index
        while end < len(filled) and filled[end]:
            end += 1
        if end - index >= min_run:
            start_s = max(0.0, float(times[index]))
            end_s = min(duration, float(times[end - 1]) + step)
            if end_s - start_s >= 0.45:
                ranges.append((start_s, end_s))
        index = end
    if not ranges:
        return [(0.0, duration)]
    kept = sum(end - start for start, end in ranges)
    if kept > duration * 0.92:
        return [(0.0, duration)]
    return ranges


def _fill_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
    filled = active.copy()
    index = 0
    count = len(filled)
    while index < count:
        if active[index]:
            index += 1
            continue
        end = index
        while end < count and not active[end]:
            end += 1
        if index > 0 and end < count and (end - index) <= max_gap:
            filled[index:end] = True
        index = end
    return filled


def _fit_beats(
    ranges: list[tuple[float, float]], span: float, period: float | None
) -> list[tuple[float, float]]:
    """Shrink each piece to a whole number of beats so the joins hit the pulse."""
    if not period or period <= 0:
        return [(max(0.0, start), min(span, end)) for start, end in ranges]
    fitted: list[tuple[float, float]] = []
    for start, end in ranges:
        start, end = max(0.0, start), min(span, end)
        frames = _beat_frames(end - start, period)
        if frames < 1:
            continue
        window = frames / FPS
        extra = (end - start) - window
        origin = start + max(0.0, extra) / 2
        fitted.append(_place(origin, frames, span))
    if fitted:
        return fitted
    frames = _beat_frames(span, period)
    if frames < 1:
        return [(0.0, span)]
    return [_place((span - frames / FPS) / 2, frames, span)]


def _beat_frames(length: float, period: float) -> int:
    beats = int(length / period + 1e-6)
    while beats > 0 and beats * period > length + 1 / FPS:
        beats -= 1
    if beats < 1:
        return 0
    return max(1, int(round(beats * period * FPS)))


def _place(origin: float, frames: int, span: float) -> tuple[float, float]:
    start = max(0.0, origin)
    end = start + frames / FPS
    if end > span:
        end = int(span * FPS) / FPS
        start = max(0.0, end - frames / FPS)
    return (start, end)


def _crop_window(width: int, height: int, claw: int) -> tuple[int, int, float, float, float, float]:
    """A 4:3 window above the claw, with room for the small drift to stay inside it.

    Returns width, height, x, y, and how far x and y may drift.
    """
    claw = int(min(max(0, claw), int(height * 0.32)))
    safe_h = height - claw
    aspect = width / height
    win_h = max(2, int(safe_h * 0.94) // 2 * 2)
    win_w = max(2, int(round(win_h * aspect)) // 2 * 2)
    if win_w > width:
        win_w = width - width % 2
        win_h = max(2, int(round(win_w / aspect)) // 2 * 2)
    win_h = min(win_h, safe_h - safe_h % 2)
    x0 = (width - win_w) / 2
    y0 = (safe_h - win_h) / 2
    return win_w, win_h, x0, y0, min(8.0, x0), min(5.0, y0)


def _claw_amount(source: Path, width: int, height: int, span: float) -> int:
    frames = _sample_frames(source, width, height, span, 8)
    if not frames:
        return 0
    measured = [_claw_rows(frame) for frame in frames]
    # One bright frame should not decide the crop. Most of the take has to agree.
    return int(np.percentile(measured, 70))


def _claw_rows(frame: np.ndarray) -> int:
    """How many bottom rows the white gripper occupies, including a small margin.

    The wrist camera looks out over the white SO-101 claw, so the jaws sit on
    the bottom edge, usually in from both sides. A pale object in the middle
    of the shot is not connected to that edge in the same way and is left alone.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 160)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    count, labels, _, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    seed = max(4, height // 40)
    touching = set(np.unique(labels[height - seed :, :]).tolist()) - {0}
    if not touching:
        return 0
    mask = np.isin(labels, list(touching))
    band_top = int(height * 0.58)
    ys, xs = np.where(mask)
    keep = ys >= band_top
    if int(keep.sum()) < 0.008 * height * width:
        return 0
    ys, xs = ys[keep], xs[keep]
    on_left = bool(np.any(xs < width * 0.42))
    on_right = bool(np.any(xs > width * 0.58))
    row_counts = np.bincount(ys, minlength=height)
    # Both jaws, or a bright band that actually fills the bottom. A pale product
    # sitting in the middle fails both and stays in frame.
    bar = int(np.sum(row_counts >= width * 0.35)) >= 8
    if not ((on_left and on_right) or bar):
        return 0
    solid = np.flatnonzero(row_counts >= width * 0.06)
    solid = solid[solid >= band_top]
    if solid.size == 0:
        return 0
    margin = max(18, int(round(height * 0.045)))
    rows = height - int(solid.min()) + margin
    if rows < int(height * 0.03):
        return 0
    return int(min(rows, height * 0.32))


def _graph(
    ranges: list[tuple[float, float]],
    crop: tuple[int, int, float, float, float, float],
    span: float,
    text_file: Path,
    music_offset: float,
) -> str:
    win_w, win_h, x0, y0, dx, dy = crop
    parts: list[str] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(ranges):
        label = f"v{index}"
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,fps={FPS}[{label}]"
        )
        labels.append(f"[{label}]")
    if len(labels) == 1:
        head = labels[0]
    else:
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[cat]")
        head = "[cat]"
    fade_in = min(0.18, span * 0.08)
    fade_dur = min(0.4, max(0.05, span * 0.1))
    fade_out = max(0.0, span - fade_dur)
    filters = [
        f"crop=w={win_w}:h={win_h}:x='{x0:.2f}+{dx:.2f}*sin(t*1.3)':y='{y0:.2f}+{dy:.2f}*cos(t*0.9)'",
        "scale=640:480:flags=lanczos",
        _GRADE,
        f"fade=t=in:st=0:d={fade_in:.3f}:color=0x11130F",
        f"fade=t=out:st={fade_out:.3f}:d={fade_dur:.3f}:color=0x11130F",
    ]
    font = _font()
    if font is not None and span >= 1.5:
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
    parts.append(head + ",".join(filters))
    audio_end = music_offset + span
    audio_out = max(0.0, span - min(0.55, span * 0.12))
    audio_fade = min(0.55, span * 0.12)
    parts.append(
        f"[1:a]atrim=start={music_offset:.3f}:end={audio_end:.3f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={min(0.3, span * 0.08):.3f},"
        f"afade=t=out:st={audio_out:.3f}:d={audio_fade:.3f},"
        "volume=0.9[a]"
    )
    return ";".join(parts)


def _beat_period(music: Path) -> float | None:
    grid = _beat_grid(music)
    return None if grid is None else grid[0]


def _beat_offset(music: Path) -> float:
    grid = _beat_grid(music)
    return 0.0 if grid is None else grid[1]


def _beat_grid(music: Path) -> tuple[float, float] | None:
    key = str(music.resolve())
    if key not in _GRIDS:
        _GRIDS[key] = _measure_beat(music)
    return _GRIDS[key]


def _measure_beat(music: Path) -> tuple[float, float] | None:
    """Period and first-beat offset of the percussion, or None if it has no pulse.

    City Loop carries a steady high hat around 110 BPM and a faster low layer.
    The edit follows the slower of the clocks that actually lock, so a cut
    lands on a hit without chopping the shot into hat ticks.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    sample_rate = 22050
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(music), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True,
    )
    if raw.returncode != 0 or len(raw.stdout) < sample_rate:
        return None
    pcm = np.frombuffer(raw.stdout, dtype=np.float32)
    hop, window = 256, 2048
    count = 1 + (pcm.size - window) // hop
    if count < 32:
        return None
    index = np.arange(window)[None, :] + hop * np.arange(count)[:, None]
    spectrum = np.abs(np.fft.rfft(pcm[index] * np.hanning(window), axis=1))
    freqs = np.fft.rfftfreq(window, 1 / sample_rate)
    envelopes = (
        _flux(spectrum[:, freqs <= 180]),
        _flux(spectrum[:, freqs >= 2500]),
        _flux(spectrum),
    )
    rate = sample_rate / hop
    found: list[tuple[float, float, float]] = []
    for env in envelopes:
        period, strength, offset = _clock(env, rate)
        if period is None or strength < 1.3:
            continue
        found.append((strength, period, offset))
    if not found:
        return None
    best = max(item[0] for item in found)
    locked = [item for item in found if item[0] >= best * 0.85]
    _, period, offset = max(locked, key=lambda item: item[1])
    if min(offset, period - offset) <= 0.04:
        offset = 0.0
    return (period, offset)


def _flux(band: np.ndarray) -> np.ndarray:
    delta = np.diff(band, axis=0)
    env = np.maximum(delta, 0).sum(axis=1)
    return np.convolve(env, np.ones(3) / 3, mode="same")


def _clock(env: np.ndarray, rate: float) -> tuple[float | None, float, float]:
    centered = env - np.median(env)
    centered = np.maximum(centered, 0)
    size = len(centered)
    spec = np.fft.rfft(centered, n=size * 2)
    corr = np.fft.irfft(spec * np.conj(spec))[:size]
    corr /= corr[0] + 1e-9

    def lag_at(bpm: float) -> int:
        return int(round(60 / bpm * rate))

    lags = np.arange(lag_at(170), lag_at(80) + 1)
    lags = lags[(lags > 1) & (lags < size)]
    if lags.size == 0:
        return None, 0.0, 0.0
    scores = corr[lags]
    peak = int(np.argmax(scores))
    shift = 0.0
    if 0 < peak < len(scores) - 1:
        left, mid, right = scores[peak - 1], scores[peak], scores[peak + 1]
        denom = left - 2 * mid + right
        if abs(denom) > 1e-9:
            shift = float(np.clip(0.5 * (left - right) / denom, -1, 1))
    period = (float(lags[peak]) + shift) / rate
    lag = max(2, int(round(period * rate)))
    best_ratio, best_off = 0.0, 0
    half = lag // 2
    for offset in range(lag):
        on_beat = centered[offset::lag]
        off_beat = centered[(offset + half) % lag :: lag]
        if on_beat.size < 4 or off_beat.size < 4:
            continue
        ratio = float(on_beat.mean() / (off_beat.mean() + 1e-9))
        if ratio > best_ratio:
            best_ratio, best_off = ratio, offset
    return period, best_ratio, best_off / rate


def _motion(source: Path, span: float) -> tuple[np.ndarray, np.ndarray] | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    raw = subprocess.run(
        [
            ffmpeg, "-v", "error", "-t", f"{span:.3f}", "-i", str(source),
            "-vf", "scale=160:120,fps=12", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True,
    )
    frame = 160 * 120
    count = len(raw.stdout) // frame
    if raw.returncode != 0 or count < 4:
        return None
    frames = np.frombuffer(raw.stdout[: count * frame], dtype=np.uint8).reshape(count, 120, 160)
    delta = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).mean(axis=(1, 2))
    scores = np.convolve(delta, np.ones(5) / 5, mode="same")
    times = np.arange(1, count) / 12
    return times, scores


def _sample_frames(source: Path, width: int, height: int, span: float, count: int) -> list[np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or span <= 0:
        return []
    rate = count / span
    raw = subprocess.run(
        [
            ffmpeg, "-v", "error", "-t", f"{span:.3f}", "-i", str(source),
            "-vf", f"fps={rate:.6f},scale={width}:{height}",
            "-frames:v", str(count),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        capture_output=True,
    )
    frame = width * height * 3
    got = len(raw.stdout) // frame
    if raw.returncode != 0 or got < 1:
        return []
    view = np.frombuffer(raw.stdout[: got * frame], dtype=np.uint8).reshape(got, height, width, 3)
    return [view[index].copy() for index in range(got)]


def _geometry(source: Path) -> tuple[int, int, float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is not installed")
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "json", str(source),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("could not read the clip")
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(payload["format"]["duration"])


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
