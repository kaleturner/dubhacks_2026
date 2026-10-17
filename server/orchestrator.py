"""Orchestrator. Backend fills in each stage.

The wrist camera films. The phone only sends the prompt and shows status and
the result. Open robot.cameras once at startup (MockCamera while USE_MOCKS=1)
and share it with the arm; nothing else may open the webcams.

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
  ERROR   send status ERROR, wrist.stop_recording(), arm.stop(), and return
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
