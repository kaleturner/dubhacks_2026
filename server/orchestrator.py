"""Orchestrator. One prompt runs PARSE → DONE.

The wrist camera films. The phone only sends the prompt and shows status and
the result. Cameras and the arm are opened once at startup and shared here.
Nothing else may open the webcams.

Order, and the call that belongs there:

  PARSE   ml.perception.parse_intent(text)
  LOCATE  side.save_still(...) → ml.perception.locate(path, obj)
  SCOUT   for each pose in robot.poses.SCOUT_POSES:
            arm.move_to(pose) → wrist.save_still(media/{job}_pose_{i}.jpg)
  PLAN    ml.perception.score_angles(stills, obj) → choose orbit | push_in | hero_rise
  RECORD  wrist.start_recording(media/{job}_raw.mp4)
          → arm.run_vla or arm.run_scripted → wrist.stop_recording()
  POST    ml.perception.best_window (optional) + ml.trim.trim_clip → {job}_final.mp4
  DONE    send result {job_id, url, duration_s}
  ERROR   wrist.stop_recording(), arm.stop(), send status ERROR, and return
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ml.trim import trim_clip
from robot.poses import SCOUT_POSES
from server.config import MEDIA_DIR
from shared.constants import VLA_TASK_PROMPT
from shared.schemas import Intent, LocateResult, ResultMessage, ScoreResult, WindowResult

Send = Callable[[dict], Awaitable[None]]

PARSE_TIMEOUT_S = 20
LOCATE_TIMEOUT_S = 20
SCOUT_MOVE_S = 1.5
SCOUT_TIMEOUT_S = 15
SCORE_TIMEOUT_S = 30
RECORD_SLACK_S = 10
POST_TIMEOUT_S = 30


class JobGate:
    """Rejects a second prompt instead of queueing it. One arm, one job."""

    def __init__(self) -> None:
        self.busy = False
        self.job_n = 0

    def claim(self) -> str | None:
        if self.busy:
            return None
        self.busy = True
        self.job_n += 1
        return f"j{self.job_n}"

    def release(self) -> None:
        self.busy = False


class Orchestrator:
    def __init__(
        self,
        send: Send,
        arm: Any,
        side: Any,
        wrist: Any,
        perception: Any,
        gate: JobGate,
    ) -> None:
        self.send = send
        self.arm = arm
        self.side = side
        self.wrist = wrist
        self.perception = perception
        self.gate = gate

    async def status(self, stage: str, detail: str, **extra: Any) -> None:
        payload: dict[str, Any] = {"type": "status", "stage": stage, "detail": detail}
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        try:
            await self.send(payload)
        except Exception:
            return

    async def run(self, text: str) -> None:
        job_id = self.gate.claim()
        if job_id is None:
            await self.status("ERROR", "a job is already running")
            return
        try:
            await self._run(text, job_id)
        except Exception as exc:
            await self._stop_hardware()
            await self.status("ERROR", str(exc).strip() or exc.__class__.__name__)
        finally:
            self.gate.release()

    async def _run(self, text: str, job_id: str) -> None:
        intent = await self._parse(text)
        if intent is None:
            return

        if not await self._locate(job_id, intent):
            return

        stills, still_urls = await self._scout(job_id)
        await self._plan(stills, still_urls, intent)
        raw_path = await self._record(job_id, intent)
        await self._post(raw_path, job_id, intent)

        result = ResultMessage(
            type="result",
            job_id=job_id,
            url=f"/media/{job_id}_final.mp4",
            duration_s=intent.duration_s,
        )
        await self.send(result.model_dump())
        await self.status("DONE", "clip ready")

    async def _parse(self, text: str) -> Intent | None:
        await self.status("PARSE", "reading the prompt")
        raw = await self._call(PARSE_TIMEOUT_S, "PARSE", self.perception.parse_intent, text)
        try:
            intent = Intent.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("could not parse the prompt") from exc
        await self.status("PARSE", f"{intent.object}, {_seconds(intent.duration_s)}s, {intent.move}")
        return intent

    async def _locate(self, job_id: str, intent: Intent) -> bool:
        await self.status("LOCATE", f"looking for {intent.object}")
        side_path = MEDIA_DIR / f"{job_id}_side.jpg"

        async def shoot_and_locate() -> dict:
            await asyncio.to_thread(self.side.save_still, side_path)
            return await asyncio.to_thread(self.perception.locate, str(side_path), intent.object)

        raw = await self._await("LOCATE", LOCATE_TIMEOUT_S, shoot_and_locate())
        try:
            located = LocateResult.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("could not locate the object") from exc
        if not located.found:
            await self.status("ERROR", f"{intent.object} not in frame")
            return False
        await self.status("LOCATE", f"{intent.object} found")
        return True

    async def _scout(self, job_id: str) -> tuple[list[str], list[str]]:
        stills: list[str] = []
        urls: list[str] = []
        for index, pose in enumerate(SCOUT_POSES):
            await self.status("SCOUT", pose)
            await self._call(SCOUT_TIMEOUT_S, "SCOUT", self.arm.move_to, pose, SCOUT_MOVE_S)
            path = MEDIA_DIR / f"{job_id}_pose_{index}.jpg"
            await self._call(SCOUT_TIMEOUT_S, "SCOUT", self.wrist.save_still, path)
            stills.append(str(path))
            urls.append(f"/media/{job_id}_pose_{index}.jpg")
            await self.status("SCOUT", pose, stills=urls)
        return stills, urls

    async def _plan(self, stills: list[str], still_urls: list[str], intent: Intent) -> None:
        await self.status("PLAN", "scoring angles")
        raw = await self._call(
            SCORE_TIMEOUT_S,
            "PLAN",
            self.perception.score_angles,
            stills,
            intent.object,
        )
        try:
            scored = ScoreResult.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("could not score angles") from exc
        if not SCOUT_POSES:
            raise RuntimeError("no scout poses")
        best = min(max(scored.best_idx, 0), len(SCOUT_POSES) - 1)
        await self.status(
            "PLAN",
            f"best angle {best}, move {intent.move}",
            stills=still_urls,
            best_idx=best,
        )
        await self._call(SCOUT_TIMEOUT_S, "PLAN", self.arm.move_to, SCOUT_POSES[best], SCOUT_MOVE_S)

    async def _record(self, job_id: str, intent: Intent) -> Path:
        raw_path = MEDIA_DIR / f"{job_id}_raw.mp4"
        await self.status("RECORD", f"wrist cam rolling, {intent.move}")
        await asyncio.to_thread(self.wrist.start_recording, raw_path)
        started = time.monotonic()
        try:
            timeout = intent.duration_s + RECORD_SLACK_S
            if intent.move == "orbit":
                await self._call(
                    timeout,
                    "RECORD",
                    self.arm.run_vla,
                    VLA_TASK_PROMPT,
                    intent.duration_s,
                )
            else:
                await self._call(
                    timeout,
                    "RECORD",
                    self.arm.run_scripted,
                    intent.move,
                    intent.duration_s,
                )
            remain = intent.duration_s - (time.monotonic() - started)
            if remain > 0:
                await asyncio.sleep(remain)
        finally:
            await asyncio.to_thread(self.wrist.stop_recording)
        return raw_path

    async def _post(self, raw_path: Path, job_id: str, intent: Intent) -> None:
        final_path = MEDIA_DIR / f"{job_id}_final.mp4"
        await self.status("POST", f"trimming to {_seconds(intent.duration_s)}s")

        async def trim() -> None:
            start_s = 0.5
            try:
                raw_window = await asyncio.to_thread(
                    self.perception.best_window,
                    str(raw_path),
                    intent.duration_s,
                )
                start_s = WindowResult.model_validate(raw_window).start_s
            except Exception:
                start_s = 0.5
            await asyncio.to_thread(
                trim_clip,
                str(raw_path),
                start_s,
                intent.duration_s,
                str(final_path),
            )

        await self._await("POST", POST_TIMEOUT_S, trim())

    async def _call(self, timeout: float, stage: str, fn: Callable, *args: Any) -> Any:
        return await self._await(stage, timeout, asyncio.to_thread(fn, *args))

    async def _await(self, stage: str, timeout: float, awaitable: Awaitable[Any]) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"{stage} timed out") from exc

    async def _stop_hardware(self) -> None:
        try:
            await asyncio.to_thread(self.wrist.stop_recording)
        except Exception:
            pass
        try:
            await asyncio.to_thread(self.arm.stop)
        except Exception:
            pass


def _seconds(value: float) -> str:
    return f"{value:g}"
