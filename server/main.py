"""FastAPI orchestrator.

  WS   /ws
  GET  /media/{file}
  POST /transcribe                        audio → {"text": "..."}

Run from the repo root:

  pip install -r requirements.txt
  python -m uvicorn server.main:app --reload --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
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
from server.orchestrator import JobGate, Orchestrator


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
    arm = ArmCls()
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
    try:
        yield
    finally:
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


@app.get("/health")
def health() -> dict:
    return {"ok": True, "use_mocks": USE_MOCKS}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    # Frontend fallback when webkitSpeechRecognition is missing.
    # Replace this with speech-to-text. Shape is always {"text": "..."}.
    _ = await file.read()
    return {"text": ""}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator = Orchestrator(
        websocket.send_json,
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
            if message.get("type") != "prompt":
                await orchestrator.status("ERROR", "expected a prompt message")
                continue
            await orchestrator.run(str(message.get("text", "")))
    except WebSocketDisconnect:
        return


app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

DIST = ROOT / "web" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="app")
