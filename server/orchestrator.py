"""Orchestrator. Backend fills in each stage.

Order, and the call that belongs there:

  PARSE   ml.perception.parse_intent(text)
  LOCATE  grab a side-camera frame, ml.perception.locate(path, obj)
  SCOUT   for each pose in robot.poses.SCOUT_POSES:
            arm.move_to(pose) → send capture_still → wait for POST /upload/still
  PLAN    ml.perception.score_angles(stills, obj) → choose orbit | push_in | hero_rise
  RECORD  send start_recording → arm.run_vla or arm.run_scripted → send stop_recording
          → wait for POST /upload/video
  POST    ml.perception.best_window (optional) + ml.trim.trim_clip
  DONE    send result {job_id, url, duration_s}
  ERROR   send status ERROR, arm.stop(), and return
"""

from collections.abc import Awaitable, Callable

Send = Callable[[dict], Awaitable[None]]


class Orchestrator:
    def __init__(self, send: Send) -> None:
        self.send = send

    async def status(self, stage: str, detail: str) -> None:
        await self.send({"type": "status", "stage": stage, "detail": detail})

    async def run(self, text: str) -> None:
        del text
        await self.status("PARSE", "orchestrator not wired yet")
        raise NotImplementedError("Backend: implement PARSE → DONE in server/orchestrator.py")
