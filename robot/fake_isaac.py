"""A local stand-in for the Isaac service, to run the VLA path end to end before a deployment exists.

  python -m robot.fake_isaac [--port 8020] [--latency-ms 150 300]
  ROBOTICS_BASE_URL=http://localhost:8020 DEPLOYMENT_ID=fake python -m robot.vla_client --task zoom --real

It speaks the calls robot/vla_client.py makes (capabilities, contract, session
grant, open, predict, close) and answers predict with the next 30 rows of the
task's recorded replay, timed by the observation timestamp. RTC is honoured: the
committed rows are echoed and the switch index is the requested delay. Replies
are delayed like the real network so deadlines and dropped chunks get exercised.
Every request's images are decoded and checked. It moves nothing and needs no key.
"""

import argparse
import json
import random
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request

from robot.poses import load_replay
from shared.constants import SHOTS

PROFILE = {"fps": 30, "action_horizon": 30, "execution_stride": 15, "max_prefix": 12, "deadline_ms": 400}
TASK_BY_PROMPT = {spec["vla_prompt"]: name for name, spec in SHOTS.items() if spec.get("vla_prompt")}

app = FastAPI(title="fake isaac")
SESSIONS: dict[str, dict] = {}
STATE = {"instruction": None, "latency": (0.15, 0.30), "base": ""}


@app.get("/v1/capabilities")
def capabilities() -> dict:
    return {"control_profiles": [PROFILE], "steps": [1000], "rtc": [False, True]}


@app.get("/v1/deployments/{deployment}/contract")
def contract(deployment: str) -> dict:
    return {"contract_id": f"fake-contract-{deployment}"}


@app.post("/v1/deployments/{deployment}/sessions")
async def grant(deployment: str, request: Request) -> dict:
    body = await request.json()
    SESSIONS[body["session_id"]] = {"opened": False, "log": []}
    return {"application_session_token": "fake-app", "modal_session_token": "fake-modal", "endpoint_url": STATE["base"]}


@app.post("/deployments/{deployment}/sessions")
async def open_session(deployment: str, request: Request) -> dict:
    body = await request.json()
    task = TASK_BY_PROMPT.get(body.get("instruction"))
    if task is None:
        return {"error": f"unknown instruction {body.get('instruction')!r}"}
    # The grant came first; attach the instruction to the newest unopened session.
    sid = next(s for s, v in reversed(SESSIONS.items()) if not v["opened"])
    SESSIONS[sid].update(opened=True, task=task, rows=np.array(load_replay(task)["actions"], dtype=float))
    print(f"[fake] session {sid}: {task}")
    return {"session_id": sid, "status": "ready"}


@app.post("/sessions/{sid}/predict")
async def predict(sid: str, request: Request) -> dict:
    received = time.monotonic()
    form = await request.form()
    meta = json.loads(form["metadata"])
    session = SESSIONS[sid]
    sizes = {}
    for name in ("side", "wrist"):
        raw = await form[name].read()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (480, 640):
            return {"error": f"{name} image is not a 640x480 PNG"}
        sizes[name] = len(raw) // 1024
    rows = session["rows"]
    start = int(round(meta["timestamp_seconds"] * PROFILE["fps"]))
    chunk = [rows[min(start + i, len(rows) - 1)].tolist() for i in range(PROFILE["action_horizon"])]
    rtc = meta.get("rtc")
    prefix = 0
    if rtc:
        prefix = int(rtc["delay"])
        chunk[:prefix] = rtc["committed_actions"][:prefix]
    delay = random.uniform(*STATE["latency"])
    time.sleep(max(0.0, delay - (time.monotonic() - received)))
    session["log"].append((meta["sequence"], prefix, sizes, round(delay * 1000)))
    print(f"[fake] {sid} seq {meta['sequence']} t={meta['timestamp_seconds']:.2f}s delay {prefix} rows, "
          f"images {sizes} KB, reply after {delay * 1000:.0f} ms")
    return {
        "chunk_id": f"{sid}-{meta['sequence']}",
        "actions": chunk,
        "prefix_length": prefix,
        "execute_from_index": prefix,
    }


@app.delete("/sessions/{sid}")
def close(sid: str) -> dict:
    log = SESSIONS.pop(sid, {}).get("log", [])
    print(f"[fake] closed {sid} after {len(log)} predicts")
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--latency-ms", type=float, nargs=2, default=(150, 300))
    args = parser.parse_args()
    STATE["latency"] = (args.latency_ms[0] / 1000, args.latency_ms[1] / 1000)
    STATE["base"] = f"http://localhost:{args.port}"
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
