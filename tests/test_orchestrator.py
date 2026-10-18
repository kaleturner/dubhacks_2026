import asyncio
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import ml.mock_perception as perception
from robot.arm import MockArm
from robot.cameras import MockCamera
from server.orchestrator import JobGate, Orchestrator
from shared.constants import SHOTS
from shared.schemas import Intent
from shared.shots import fallback_shots, guess_object

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

TWO = {name: {**spec, "enabled": name in ("overhead", "zoom"), "min_s": 1} for name, spec in SHOTS.items()}


def run_job(text, registry=None):
    sent = []
    plan = perception
    if registry is not None:
        plan = SimpleNamespace(
            parse_intent=lambda t: Intent(
                object=guess_object(t), style="x", shots=fallback_shots(t, registry=registry)
            ).model_dump(),
            locate=perception.locate,
            score_angles=perception.score_angles,
            best_window=perception.best_window,
        )

    async def send(msg):
        sent.append(msg)

    async def go():
        side, wrist = MockCamera(0, "side"), MockCamera(1, "wrist")
        side.open()
        wrist.open()
        arm = MockArm()
        arm.connect()
        try:
            await Orchestrator(send, arm, side, wrist, plan, JobGate(), registry=registry).run(text)
        finally:
            side.close()
            wrist.close()

    asyncio.run(go())
    return sent


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
    )
    return float(out.stdout)


def test_zoom_only_job_reaches_done():
    sent = run_job("record a 4 second video of this apple")
    assert sent[-1]["stage"] == "DONE"
    result = next(m for m in sent if m["type"] == "result")
    assert [s["task"] for s in result["shots"]] == ["zoom"]
    assert abs(duration("server/media/j1_final.mp4") - 4) < 0.3


def test_two_shot_job_is_stitched_to_requested_length():
    sent = run_job("record a 4 second overhead then zoom of this apple", registry=TWO)
    assert sent[-1]["stage"] == "DONE", sent[-1]
    result = next(m for m in sent if m["type"] == "result")
    assert [s["task"] for s in result["shots"]] == ["overhead", "zoom"]
    assert result["duration_s"] == 4
    rec = [m["detail"] for m in sent if m.get("stage") == "RECORD"]
    assert rec[0].startswith("shot 1/2: overhead") and rec[1].startswith("shot 2/2: zoom")
    assert abs(duration("server/media/j1_final.mp4") - 4) < 0.3
