"""FastAPI orchestrator.

  WS   /ws
  GET  /media/{file}
  POST /transcribe                        audio → {"text": "..."}
  GET  /dashboard, /debug/*               live debug view (server/dashboard.py)

Run from the repo root:

  pip install -r requirements.txt
  python -m uvicorn server.main:app --reload --port 8000
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.config import (
    MEDIA_DIR,
    SIDE_CAM_INDEX,
    USE_MOCK_ARM,
    USE_MOCK_CAMERAS,
    USE_MOCK_PERCEPTION,
    USE_MOCKS,
    USE_VLA,
    WRIST_CAM_INDEX,
)
from server.dashboard import message_log, remember
from server.dashboard import router as dashboard_router
from server.orchestrator import JobGate, Orchestrator
from shared.schemas import PromptMessage

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if USE_MOCK_CAMERAS:
        from robot.cameras import MockCamera as CameraCls
    else:
        from robot.cameras import Camera as CameraCls
    if USE_MOCK_ARM:
        from robot.arm import MockArm as ArmCls
    else:
        from robot.arm import Arm as ArmCls
    if USE_MOCK_PERCEPTION:
        import ml.mock_perception as perception
    else:
        import ml.perception as perception

    side = CameraCls(SIDE_CAM_INDEX, "side")
    wrist = CameraCls(WRIST_CAM_INDEX, "wrist")
    arm = ArmCls(side, wrist)
    try:
        side.open()
        try:
            wrist.open()
            arm.connect()
        except Exception:
            wrist.close()
            raise
    except Exception:
        side.close()
        raise

    app.state.arm = arm
    app.state.side = side
    app.state.wrist = wrist
    app.state.perception = perception
    app.state.gate = JobGate()
    app.state.messages = message_log()
    try:
        yield
    finally:
        if USE_VLA:
            from robot.vla_client import close_warm_sessions

            close_warm_sessions()
        try:
            wrist.stop_recording()
        except Exception:
            pass
        try:
            arm.stop()
        except Exception:
            pass
        wrist.close()
        side.close()


app = FastAPI(title="Cinematic recorder", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(dashboard_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "use_mocks": USE_MOCKS}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """Audio file field `file` → {"text": "..."}. Mk1.5, after ffmpeg makes a wav."""
    from ml.transcribe import transcribe_bytes
    from shared.schemas import TranscribeResult

    data = await file.read()
    try:
        text = await asyncio.to_thread(
            transcribe_bytes,
            data,
            file.content_type or "",
            file.filename or "",
        )
    except RuntimeError as exc:
        log.warning("transcribe failed: %s", exc)
        status = 400 if str(exc) in {"empty recording", "recording is too long"} else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return TranscribeResult(text=text).model_dump()


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    messages = websocket.app.state.messages

    async def send(message: dict) -> None:
        remember(messages, message)
        await websocket.send_json(message)

    orchestrator = Orchestrator(
        send,
        websocket.app.state.arm,
        websocket.app.state.side,
        websocket.app.state.wrist,
        websocket.app.state.perception,
        websocket.app.state.gate,
        use_vla=USE_VLA,
    )
    try:
        while True:
            message = await websocket.receive_json()
            try:
                prompt = PromptMessage.model_validate(message)
            except ValidationError:
                await orchestrator.status("ERROR", "expected a prompt message")
                continue
            await orchestrator.run(prompt.text)
    except WebSocketDisconnect:
        return


app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

DIST = ROOT / "web" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="app")
