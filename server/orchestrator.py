"""Orchestrator. One prompt runs PARSE → DONE.

The wrist camera films. The phone only sends the prompt and shows status and
the result. Cameras and the arm are opened once at startup and shared here.
Nothing else may open the webcams.

Order, and the call that belongs there:

  PARSE   ml.perception.parse_intent(text)
  LOCATE  side.save_still(...) → ml.perception.locate(path, obj)
  SCOUT   for each pose in robot.poses.SCOUT_POSES:
            arm.move_to(pose) → wrist.save_still(media/{job}_pose_{i}.jpg)
  PLAN    ml.perception.score_angles(stills, obj) → best scout angle
  RECORD  for each shot in intent.shots (the planner's task list, see shared.constants.SHOTS):
            arm.move_to(start pose) → wrist.start_recording(media/{job}_s{i}_raw.mp4)
            → arm.run_vla or arm.run_scripted → wrist.stop_recording()
            and start trimming that shot in the background
  POST    join the shot clips (ml.trim.concat_clips) → {job}_final.mp4
          then cut {job}_edit.mp4 (graded, CC0 electronic track) beside it
  DONE    send result {job_id, url, duration_s, shots, edit_url}
  ERROR   wrist.stop_recording(), arm.stop(), send status ERROR, and return
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import cv2
from pydantic import ValidationError

from ml.edit import cut_edit
from ml.trim import CROSSFADE_S, concat_clips, trim_clip
from robot.poses import SCOUT_POSES
from server.config import MEDIA_DIR
from shared.constants import SHOTS
from shared.schemas import Intent, LocateResult, ResultMessage, ScoreResult, Shot, WindowResult

Send = Callable[[dict], Awaitable[None]]
log = logging.getLogger(__name__)

PARSE_TIMEOUT_S = 20
LOCATE_TIMEOUT_S = 20
SCOUT_MOVE_S = 1.5
SCOUT_TIMEOUT_S = 15
SCORE_TIMEOUT_S = 30
RECORD_SLACK_S = 10
# Each shot is filmed for duration / KEEP and the middle is kept: the arm moves at about
# the recorded speed, and the unfold from rest and the fold back are cut.
KEEP = 0.7
CLAUDE_TIMEOUT_S = 300  # overview + up to two planned, filmed, reviewed takes
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
        use_vla: bool = False,
        registry: dict[str, dict] | None = None,
        use_claude: bool = False,
    ) -> None:
        self.send = send
        self.arm = arm
        self.side = side
        self.wrist = wrist
        self.perception = perception
        self.gate = gate
        self.use_vla = use_vla
        self.use_claude = use_claude
        self.registry = SHOTS if registry is None else registry

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

        if self.use_claude:
            trims = await self._claude(job_id, intent, text)
        else:
            stills, still_urls = await self._scout(job_id)
            await self._plan(stills, still_urls, intent)
            trims = await self._record(job_id, intent)
        edit_url = await self._post(trims, job_id, intent.object)

        result = ResultMessage(
            type="result",
            job_id=job_id,
            url=f"/media/{job_id}_final.mp4",
            duration_s=intent.duration_s,
            shots=intent.shots,
            edit_url=edit_url or None,
        )
        await self.send(result.model_dump(exclude_none=True))
        await self.status("DONE", "clip ready")

    async def _parse(self, text: str) -> Intent | None:
        await self.status("PARSE", "reading the prompt")
        raw = await self._call(PARSE_TIMEOUT_S, "PARSE", self.perception.parse_intent, text)
        try:
            intent = Intent.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("could not parse the prompt") from exc
        detail = f"{intent.object}, {_seconds(intent.duration_s)}s, {_describe(intent.shots)}"
        if intent.note:
            detail = f"{detail} ({intent.note})"
        await self.status("PARSE", detail)
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
        if hasattr(self.arm, "target_side_xy"):
            self.arm.target_side_xy = located.center if located.found else None
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
            f"best angle {best}, {_describe(intent.shots)}",
            stills=still_urls,
            best_idx=best,
        )
        await self._call(SCOUT_TIMEOUT_S, "PLAN", self.arm.move_to, SCOUT_POSES[best], SCOUT_MOVE_S)

    async def _record(self, job_id: str, intent: Intent) -> list[asyncio.Task]:
        """Film every shot in order. Each shot's trim runs while the next one films."""
        trims: list[asyncio.Task] = []
        if hasattr(self.arm, "center_on"):
            self.arm.center_on = intent.object  # replays steer to keep it in the wrist view
        try:
            for index, shot in enumerate(intent.shots):
                spec = self.registry[shot.task]
                raw_path = MEDIA_DIR / f"{job_id}_s{index}_raw.mp4"
                clip_path = MEDIA_DIR / f"{job_id}_s{index}.mp4"
                await self.status(
                    "RECORD",
                    f"shot {index + 1}/{len(intent.shots)}: {shot.task}, "
                    f"{_seconds(shot.duration_s)}s, wrist cam rolling",
                )
                await self._call(SCOUT_TIMEOUT_S, "RECORD", self.arm.move_to, spec["start_pose"], SCOUT_MOVE_S)
                if not (self.use_vla and spec.get("vla_prompt")) and hasattr(self.arm, "prepare_shot"):
                    # Pick the recorded take that matches where the object is, before the camera rolls.
                    await self._call(SCOUT_TIMEOUT_S, "RECORD", self.arm.prepare_shot, spec["scripted"])
                film_s = shot.duration_s / KEEP
                await self._film(raw_path, shot, spec, film_s)
                # Every clip but the last carries extra for the crossfade into the next one.
                keep_s = shot.duration_s + (CROSSFADE_S if index < len(intent.shots) - 1 else 0.0)
                trims.append(asyncio.create_task(self._trim(raw_path, clip_path, keep_s, film_s)))
        except BaseException:
            for task in trims:
                task.cancel()
            await asyncio.gather(*trims, return_exceptions=True)
            raise
        return trims

    async def _film(self, raw_path: Path, shot: Shot, spec: dict, film_s: float) -> None:
        await asyncio.to_thread(self.wrist.start_recording, raw_path)
        started = time.monotonic()
        try:
            timeout = film_s + RECORD_SLACK_S
            if self.use_vla and spec.get("vla_prompt"):
                await self._call(timeout, "RECORD", self.arm.run_vla, spec["vla_prompt"], film_s)
            else:
                await self._call(timeout, "RECORD", self.arm.run_scripted, spec["scripted"], film_s)
            remain = film_s - (time.monotonic() - started)
            if remain > 0:
                await asyncio.sleep(remain)
        finally:
            await asyncio.to_thread(self.wrist.stop_recording)

    async def _claude(self, job_id: str, intent: Intent, text: str) -> list[asyncio.Task]:
        """Claude looks, plans, films, reviews, and reshoots; its own words go to the phone."""
        from robot.claude_planner import quick_shoot

        loop = asyncio.get_running_loop()

        def say(line: str) -> None:
            line = line.removeprefix("[planner]").strip()
            if line and not line[0].isdigit():  # skip the per-waypoint table
                asyncio.run_coroutine_threadsafe(self.status("RECORD", f"Claude: {line}"[:400]), loop)

        await self.status("RECORD", "Claude is looking at the scene")
        brief = f"{text}. The product is the {intent.object}."
        out = MEDIA_DIR / f"{job_id}_claude.mp4"
        best = await self._await(
            "RECORD",
            CLAUDE_TIMEOUT_S,
            asyncio.to_thread(quick_shoot, self.arm, self.wrist, self.side, brief, intent.duration_s, str(out), 2, say, intent.object),
        )
        take = Path(best["path"])
        frames = cv2.VideoCapture(str(take))
        film_s = frames.get(cv2.CAP_PROP_FRAME_COUNT) / (frames.get(cv2.CAP_PROP_FPS) or 30.0)
        frames.release()
        await self.status("RECORD", f"Claude kept take {take.stem[-1]} ({best['review']['score']}/10)")
        clip = MEDIA_DIR / f"{job_id}_s0.mp4"
        return [asyncio.create_task(self._trim(take, clip, intent.duration_s, film_s))]

    async def _trim(self, raw_path: Path, clip_path: Path, duration_s: float, film_s: float) -> str:
        start_s = max(0.0, (film_s - duration_s) / 2)  # the middle of the take
        try:
            raw_window = await asyncio.to_thread(self.perception.best_window, str(raw_path), duration_s)
            start_s = WindowResult.model_validate(raw_window).start_s
        except Exception:
            pass
        return await asyncio.to_thread(trim_clip, str(raw_path), start_s, duration_s, str(clip_path))

    async def _post(self, trims: list[asyncio.Task], job_id: str, title: str) -> str:
        final_path = MEDIA_DIR / f"{job_id}_final.mp4"
        edit_path = MEDIA_DIR / f"{job_id}_edit.mp4"
        await self.status("POST", f"joining {len(trims)} shot{'s' if len(trims) != 1 else ''}")

        async def join() -> None:
            clips = await asyncio.gather(*trims)
            await asyncio.to_thread(concat_clips, list(clips), str(final_path), CROSSFADE_S)

        await self._await("POST", POST_TIMEOUT_S, join())
        await self.status("POST", "cutting the edit")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(cut_edit, str(final_path), str(edit_path), 30, title),
                timeout=POST_TIMEOUT_S,
            )
        except Exception as exc:
            log.warning("edit failed: %s", exc)
            return ""
        return f"/media/{job_id}_edit.mp4"

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


def _describe(shots: list[Shot]) -> str:
    names = " + ".join(shot.task for shot in shots)
    return f"{len(shots)} shot{'s' if len(shots) != 1 else ''}: {names}"
