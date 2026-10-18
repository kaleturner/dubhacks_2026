import os
import time

os.environ.setdefault("USE_MOCKS", "1")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from robot import centering
from robot.arm import MockArm
from robot.cameras import MockCamera
from server.dashboard import message_log, remember, router


def make_app():
    """Dashboard routes on mock hardware, without server.main's lifespan (.env may ask for real cameras)."""
    app = FastAPI()
    app.include_router(router)
    app.state.side, app.state.wrist = MockCamera(0, "side"), MockCamera(1, "wrist")
    app.state.arm = MockArm()
    app.state.messages = message_log()
    return app


def test_state_and_wrist_stream(monkeypatch):
    app = make_app()
    app.state.side.open()
    app.state.wrist.open()
    try:
        app.state.arm._last_sent = [1.0, 2.0, 3.0, 4.0, 5.0, 97.1]
        remember(app.state.messages, {"type": "status", "stage": "PARSE", "detail": "reading the prompt"})
        remember(app.state.messages, {"type": "status", "stage": "RECORD", "detail": "shot 1/1: zoom"})
        live = {"object": "duck", "state": "tracking", "box": [100, 100, 80, 60], "score": 0.9,
                "offset": [1.5, -0.5], "updated": time.time()}
        monkeypatch.setattr(centering, "LIVE", live)
        client = TestClient(app)

        state = client.get("/debug/state").json()
        assert state["centering"]["state"] == "tracking"
        assert state["messages"][0]["stage"] == "RECORD"
        assert state["joints"]["gripper"] == 97.1

        r = client.get("/debug/wrist.mjpg?limit=1")
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        body = r.content
        assert body.startswith(b"--frame\r\n")
        assert body.split(b"\r\n\r\n", 1)[1][:2] == b"\xff\xd8"

        assert client.get("/debug/side.mjpg?limit=1").status_code == 200
        assert "/debug/state" in client.get("/dashboard").text
    finally:
        app.state.side.close()
        app.state.wrist.close()


def test_main_serves_dashboard_ahead_of_static_app():
    from server.main import app

    client = TestClient(app)  # no lifespan, so nothing opens hardware
    assert "/debug/wrist.mjpg" in client.get("/dashboard").text
    assert "centering" in client.get("/debug/state").json()
