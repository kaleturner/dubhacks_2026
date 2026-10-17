"""FastAPI orchestrator.

  WS   /ws
  POST /upload/still?job_id=&pose_id=     jpeg
  POST /upload/video?job_id=              mp4 or webm
  GET  /media/{file}
  POST /transcribe                        audio → {"text": "..."}

Run from the repo root:

  pip install -r requirements.txt
  python -m uvicorn server.main:app --reload --port 8000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.config import MEDIA_DIR, USE_MOCKS
from server.orchestrator import Orchestrator

app = FastAPI(title="Cinematic recorder")
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


@app.post("/upload/still")
async def upload_still(job_id: str, pose_id: int, file: UploadFile = File(...)) -> dict:
    dest = MEDIA_DIR / f"{job_id}_pose_{pose_id}.jpg"
    dest.write_bytes(await file.read())
    return {"ok": True, "path": f"/media/{dest.name}"}


@app.post("/upload/video")
async def upload_video(job_id: str, file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov"}:
        suffix = ".webm"
    dest = MEDIA_DIR / f"{job_id}_raw{suffix}"
    dest.write_bytes(await file.read())
    return {"ok": True, "path": f"/media/{dest.name}"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    # Frontend fallback when webkitSpeechRecognition is missing.
    # Replace this with speech-to-text. Shape is always {"text": "..."}.
    _ = await file.read()
    return {"text": ""}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator = Orchestrator(websocket.send_json)
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") != "prompt":
                await orchestrator.status("ERROR", "expected a prompt message")
                continue
            try:
                await orchestrator.run(str(message.get("text", "")))
            except NotImplementedError as exc:
                await orchestrator.status("ERROR", str(exc))
    except WebSocketDisconnect:
        return


app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
