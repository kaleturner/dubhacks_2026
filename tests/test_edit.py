"""The music edit crops the claw, drops still stretches, and lands on the beat."""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.edit import CLAW_MAX, MUSIC, _beat_grid, _claw_rows, _crop_window, _interesting_ranges, choose_track, cut_edit

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def write_clip(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    height, width = frames[0].shape[:2]
    raw = b"".join(frame.tobytes() for frame in frames)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        input=raw, capture_output=True,
    )
    assert result.returncode == 0, result.stderr[-200:]


def probe(path: Path, entries: str) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def duration(path: Path) -> float:
    return float(probe(path, "format=duration"))


def frame_at(path: Path, second: float) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{second:.3f}", "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout
    width, height = probe(path, "stream=width,height").split()
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(int(height), int(width), 3)


def paint(shift: int, claw: bool) -> np.ndarray:
    yy, xx = np.mgrid[0:480, 0:640]
    frame = np.full((480, 640, 3), (70, 90, 60), np.uint8)
    frame[(yy + xx + shift) % 28 < 8] = (40, 55, 35)
    cx, cy = 200 + shift % 220, 210
    cv2.circle(frame, (cx, cy), 46, (30, 180, 220), -1)
    if claw:
        frame[390:, :150] = (235, 238, 240)
        frame[390:, 490:] = (235, 238, 240)
        frame[372:392, 40:130] = (20, 20, 20)
        frame[372:392, 510:600] = (20, 20, 20)
    return frame


def test_claw_measurement_ignores_a_centered_pale_object():
    frame = paint(0, claw=False)
    cv2.rectangle(frame, (280, 300), (360, 470), (230, 230, 230), -1)
    assert _claw_rows(frame) == 0


def test_claw_measurement_is_capped_to_keep_the_product():
    # The painted claw reaches ~100+ rows; the crop stops at CLAW_MAX so a product
    # sitting just above the jaws stays in the edit.
    rows = _claw_rows(paint(0, claw=True))
    assert rows == int(480 * CLAW_MAX)


def test_crop_window_cannot_drift_back_onto_the_claw():
    claw = int(480 * CLAW_MAX)
    win_w, win_h, x0, y0, dx, dy = _crop_window(640, 480, claw)
    assert y0 - dy >= -1e-6
    assert y0 + win_h + dy <= 480 - claw + 1e-6
    assert x0 - dx >= -1e-6
    assert x0 + win_w + dx <= 640 + 1e-6
    assert abs(win_w / win_h - 640 / 480) < 0.03


def test_a_steady_shot_is_not_chopped_up():
    times = np.linspace(0, 10, 120, endpoint=False)
    assert _interesting_ranges(times, np.full(120, 4.0), 10) == [(0.0, 10.0)]
    assert _interesting_ranges(times, np.full(120, 0.2), 10) == [(0.0, 10.0)]


def test_a_still_stretch_between_moves_is_dropped():
    times = np.linspace(0, 8, 96, endpoint=False)
    scores = np.where((times > 2.2) & (times < 5.0), 0.05, 9.0)
    ranges = _interesting_ranges(times, scores, 8)
    assert ranges
    assert all(not (start > 2.5 and end < 4.7) for start, end in ranges)
    assert ranges[0][1] < 2.9
    assert ranges[-1][0] > 4.3


def test_the_prompt_and_its_style_pick_the_bed():
    assert choose_track("record a 15 second video of this apple cinematic")["id"] == "city-loop"
    assert choose_track("a bright upbeat ad for this apple")["id"] == "trance-city"
    assert choose_track("a fast hype party clip of this duck")["id"] == "trance-battle"
    assert choose_track("record this duck cinematic")["id"] == "city-loop"


def _pcm(path, start, seconds=0.4):
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds}",
            "-i", str(path), "-ac", "1", "-ar", "22050", "-f", "f32le", "-",
        ],
        capture_output=True,
    )
    return np.frombuffer(raw.stdout, dtype=np.float32)


def _likeness(left, right):
    count = min(left.size, right.size)
    if count < 1000:
        return 0.0
    left = left[:count] - left[:count].mean()
    right = right[:count] - right[:count].mean()
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))


def test_the_chosen_bed_is_the_one_under_the_edit_and_on_its_beat(tmp_path):
    frames = [paint(index * 7, claw=False) for index in range(90)]
    source = tmp_path / "take.mp4"
    write_clip(source, frames)
    prompts = {
        "trance-battle": "a fast hype party clip",
        "trance-city": "a bright upbeat ad",
        "city-loop": "a calm cinematic shot",
    }
    for bed_id, prompt in prompts.items():
        bed = choose_track(prompt)
        assert bed["id"] == bed_id
        edited = tmp_path / f"{bed_id}.mp4"
        cut_edit(str(source), str(edited), 30, "duck", prompt)
        period, offset = _beat_grid(Path(bed["path"]))
        span = duration(edited)
        slipped = abs(span / period - round(span / period)) * period
        assert slipped < 0.05, f"{bed_id} is {slipped:.3f}s off its beat"
        heard = _pcm(edited, 0.2)
        own = _likeness(heard, _pcm(bed["path"], offset + 0.2))
        off = _likeness(heard, _pcm(bed["path"], offset + 0.2 + period / 2))
        assert own > 0.6, bed_id
        assert own > off + 0.15, f"{bed_id} on-beat {own:.2f} vs off-beat {off:.2f}"


def test_city_loop_locks_to_a_pulse():
    grid = _beat_grid(MUSIC)
    assert grid is not None
    period, offset = grid
    assert 0.4 <= period <= 0.7
    assert offset == 0.0 or 0 <= offset < period


def test_edit_drops_the_hold_crops_the_claw_and_hits_the_beat(tmp_path):
    frames = []
    for index in range(int(7.5 * 30)):
        second = index / 30
        moving = second < 2.2 or second >= 5.0
        if moving:
            frames.append(paint(index * 7, claw=True))
        else:
            held = np.full((480, 640, 3), (40, 170, 40), np.uint8)
            held[390:, :150] = (235, 238, 240)
            held[390:, 490:] = (235, 238, 240)
            frames.append(held)
    source = tmp_path / "take.mp4"
    edited = tmp_path / "edit.mp4"
    write_clip(source, frames)
    cut_edit(str(source), str(edited), 30, "duck")

    span = duration(edited)
    assert 2.5 < span < 5.6
    period, _ = _beat_grid(MUSIC)
    assert abs(span / period - round(span / period)) * period < 0.08
    assert "audio" in probe(edited, "stream=codec_type").split()

    picture = frame_at(edited, min(3.2, span * 0.72))
    corner = picture[460:478, 8:70]
    assert float(corner.mean()) < 145
    green = (picture[:, :, 1] > picture[:, :, 2] + 70) & (picture[:, :, 1] > picture[:, :, 0] + 70)
    assert float(green.mean()) < 0.08


def test_grade_pushes_a_flat_color(tmp_path):
    color = np.full((480, 640, 3), (90, 100, 130), np.uint8)
    source = tmp_path / "flat.mp4"
    edited = tmp_path / "flat-edit.mp4"
    write_clip(source, [color] * 45)
    cut_edit(str(source), str(edited), 30, "chip")
    picture = frame_at(edited, min(0.35, duration(edited) * 0.5))
    before = cv2.cvtColor(color[:1, :1], cv2.COLOR_BGR2HSV)[0, 0, 1]
    after = cv2.cvtColor(picture[200:280, 250:390], cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    assert after > before + 12
