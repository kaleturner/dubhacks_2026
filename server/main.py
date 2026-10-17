"""FastAPI orchestrator.

  WS   /ws
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
