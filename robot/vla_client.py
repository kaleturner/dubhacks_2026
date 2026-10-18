"""Isaac 0.5 client for the shot policies.

Arm.run_vla calls run_task. The call sequence and the real-time chunk rules are
in WORKSHOP.md; the endpoints are in https://robotics.perceptron.inc/docs.

  grant   POST /v1/deployments/{id}/sessions          team key, ttl 21600
  open    POST {endpoint}/deployments/{id}/sessions   session tokens, never the team key
  loop    POST {endpoint}/sessions/{sid}/predict      multipart: metadata + side + wrist PNG
  end     DELETE {endpoint}/sessions/{sid}

The docs show two path styles (the curl walkthrough and a /v1/inference/ prefix),
so each call tries the walkthrough style and falls back to the other on a 404.
Reply field names are not documented; parse_chunk reads them tolerantly.

Timing comes from GET /v1/capabilities control_profiles[0] (fps, stride,
max_prefix, deadline_ms). The control loop runs at fps on its own clock and
never waits for the network:

  - a request goes out every `stride` rows, one in flight at a time
  - a chunk is anchored at the tick its observation was captured
  - a reply later than the deadline, or whose switch point has already passed,
    is dropped; the old chunk's reserve rows play, then the arm holds
  - the new chunk takes over at its execute_from_index, not when it arrives

Actions are in robot units. Arm.send_joints replaces only the gripper with
GRIPPER_HOLD. Cameras come from robot.cameras and are never opened here.

  python -m robot.vla_client --task zoom --duration 6 [--real] [--live]

Without --live the arm is frozen (VLA_DRY_RUN=1) and the chunks are only printed.
"""

import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

import requests

from shared.constants import SHOTS
from shared.robotics_api import BASE_URL, USER_AGENT, Api, ApiError, read_key

DEFAULT_PROFILE = {"fps": 30, "execution_stride": 15, "max_prefix": 12, "deadline_ms": 400}
MAX_CONSECUTIVE_FAILURES = 5


class InferenceError(ApiError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Chunk:
    chunk_id: str | None
    rows: list[list[float]]
    execute_from: int = 0
    anchor: int = 0  # control tick that rows[0] belongs to


def parse_chunk(reply) -> Chunk:
    """Find the action rows, the chunk id, and the switch index in a predict reply."""

    def rows_in(node):
        if isinstance(node, list) and node and all(isinstance(r, list) and len(r) == 6 for r in node):
            return node
        if isinstance(node, dict):
            for key in ("actions", "action_chunk", "chunk", "rows"):
                if key in node and (found := rows_in(node[key])):
                    return found
            for value in node.values():
                if found := rows_in(value):
                    return found
        return None

    rows = rows_in(reply)
    if rows is None:
        keys = sorted(reply) if isinstance(reply, dict) else type(reply).__name__
        raise InferenceError(0, f"no 6-wide action rows in the predict reply (keys: {keys})")
    clean = [[float(v) for v in row] for row in rows]
    if not all(math.isfinite(v) for row in clean for v in row):
        raise InferenceError(0, "predict reply has non-finite actions")
    prefix = int(reply.get("prefix_length", 0) or 0) if isinstance(reply, dict) else 0
    execute_from = int(reply.get("execute_from_index", prefix) or 0) if isinstance(reply, dict) else 0
    chunk_id = None
    if isinstance(reply, dict):
        chunk_id = reply.get("chunk_id") or reply.get("id")
    return Chunk(chunk_id=None if chunk_id is None else str(chunk_id), rows=clean, execute_from=execute_from)


class Player:
    """Chunk bookkeeping on the control clock. No threads, no network: easy to test."""

    def __init__(self, profile: dict) -> None:
        self.fps = int(profile["fps"])
        self.stride = int(profile["execution_stride"])
        self.max_prefix = int(profile["max_prefix"])
        self.deadline_ms = float(profile["deadline_ms"])
        self.active: Chunk | None = None
        self.queued: Chunk | None = None
        self.latency_ticks = 5.0  # training used delays 1-12 with mean 5
        self.dropped = 0

    def should_request(self, tick: int) -> bool:
        newest = self.queued or self.active  # a chunk waiting to take over counts as the newest
        if newest is None:
            return True
        return tick - newest.anchor >= self.stride

    def delay(self) -> int:
        return int(min(self.max_prefix, max(1, math.ceil(self.latency_ticks))))

    def make_rtc(self, tick: int) -> dict | None:
        if self.active is None:
            return None
        start = tick - self.active.anchor
        d = self.delay()
        if start < 0 or start + d > len(self.active.rows):
            return None  # the old chunk ran out: start fresh
        return {
            "prior_chunk_id": self.active.chunk_id,
            "prior_start_index": start,
            "delay": d,
            "committed_actions": self.active.rows[start : start + d],
        }

    def accept(self, chunk: Chunk, request_tick: int, arrival_tick: int) -> bool:
        age_ms = (arrival_tick - request_tick) * 1000.0 / self.fps
        self.latency_ticks = 0.7 * self.latency_ticks + 0.3 * (arrival_tick - request_tick)
        if age_ms > self.deadline_ms:
            self.dropped += 1
            return False
        chunk.anchor = request_tick
        if self.active is not None and request_tick + chunk.execute_from < arrival_tick:
            self.dropped += 1  # the switch point is already behind us
            return False
        if self.active is None:
            chunk.execute_from = 0
        self.queued = chunk
        return True

    def row_for(self, tick: int) -> list[float] | None:
        if self.queued is not None and tick >= self.queued.anchor + self.queued.execute_from:
            self.active, self.queued = self.queued, None
        if self.active is None:
            return None
        index = tick - self.active.anchor
        if 0 <= index < len(self.active.rows):
            return self.active.rows[index]
        return None  # reserve rows used up: hold


class SessionClient:
    """Grant, open, predict, and close one robot session."""

    def __init__(self, deployment_id: str, instruction: str, *, api: Api | None = None, http=None) -> None:
        self.deployment_id = deployment_id
        self.instruction = instruction
        self.api = api
        self.http = http or requests
        self.session_id = f"robot-{uuid.uuid4().hex[:10]}"
        self.endpoint = ""
        self.headers: dict[str, str] = {}
        self.contract_id = ""
        self.profile = dict(DEFAULT_PROFILE)
        self.sequence = 0
        self._style = 0
        self.logged_keys = False

    def open(self) -> None:
        api = self.api or Api(read_key())
        caps = api.call("GET", "/v1/capabilities")
        profiles = caps.get("control_profiles") or []
        if profiles:
            self.profile = {**DEFAULT_PROFILE, **profiles[0]}
        self.contract_id = str(api.call("GET", f"/v1/deployments/{self.deployment_id}/contract")["contract_id"])
        grant = api.call(
            "POST",
            f"/v1/deployments/{self.deployment_id}/sessions",
            body={"session_id": self.session_id, "ttl_seconds": 21600},
        )
        self.endpoint = str(grant.get("endpoint_url") or BASE_URL).rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {grant['application_session_token']}",
            "Modal-Authorization": f"Bearer {grant['modal_session_token']}",
            "User-Agent": USER_AGENT,
        }
        reply = self._send(
            "POST",
            ["/deployments/{d}/sessions", "/v1/inference/deployments/{d}/sessions"],
            json={"instruction": self.instruction, "contract_id": self.contract_id, "rtc_enabled": True},
        )
        print(f"[vla] session {self.session_id}, reply keys {sorted(reply)}, paths style {self._style}")

    def predict(self, *, state, side_png, wrist_png, tick, rtc=None) -> Chunk:
        meta = {
            "contract_id": self.contract_id,
            "observation_id": f"{self.session_id}-{self.sequence}",
            "sequence": self.sequence,
            "timestamp_seconds": tick / self.profile["fps"],
            "state": [float(v) for v in state],
            "image_encoding": "png",
            "deadline_ms": self.profile["deadline_ms"],
        }
        if rtc is not None:
            meta["rtc"] = rtc
        self.sequence += 1
        files = {
            "metadata": (None, json.dumps(meta), "application/json"),
            "side": ("side.png", side_png, "image/png"),
            "wrist": ("wrist.png", wrist_png, "image/png"),
        }
        reply = self._send(
            "POST",
            ["/sessions/{s}/predict", "/v1/inference/sessions/{s}/predict"],
            files=files,
            timeout=self.profile["deadline_ms"] / 1000.0 + 1.0,
        )
        if not self.logged_keys:
            self.logged_keys = True
            print(f"[vla] first predict reply keys: {sorted(reply) if isinstance(reply, dict) else type(reply)}")
        return parse_chunk(reply)

    def close(self) -> None:
        try:
            self._send("DELETE", ["/sessions/{s}", "/v1/inference/sessions/{s}"])
        except Exception as exc:
            print(f"[vla] could not delete session: {exc}")

    def _send(self, method: str, forms: list[str], timeout: float = 30.0, **kw):
        order = forms[self._style :] + forms[: self._style]
        last = None
        for form in order:
            url = self.endpoint + form.format(d=self.deployment_id, s=self.session_id)
            resp = self.http.request(method, url, headers=self.headers, timeout=timeout, **kw)
            if resp.status_code == 404 and form != order[-1]:
                last = resp
                continue
            if resp.status_code >= 400:
                raise InferenceError(resp.status_code, f"{method} {url} -> {resp.status_code}: {resp.text[:300]}")
            self._style = forms.index(form)
            return resp.json() if resp.content else {}
        raise InferenceError(404, f"{method} not found in either path style: {last.text[:200] if last else ''}")


@dataclass
class _Flight:
    request_tick: int
    chunk: Chunk | None = None
    error: Exception | None = None
    arrival_tick: int = 0
    done: threading.Event = field(default_factory=threading.Event)


def run_task(
    instruction: str,
    duration_s: float,
    *,
    arm,
    side,
    wrist,
    stop: threading.Event | None = None,
    session: SessionClient | None = None,
    clock=time.monotonic,
    sleep=time.sleep,
) -> None:
    """Blocking. Runs the policy for duration_s on one instruction, then deletes the session."""
    if session is None:
        deployment_id = os.getenv("DEPLOYMENT_ID", "").strip()
        if not deployment_id:
            raise RuntimeError("DEPLOYMENT_ID is empty. Run: python -m ml.upload_and_train deployments")
        session = SessionClient(deployment_id, instruction)
    session.open()
    player = Player(session.profile)
    fps = player.fps
    t0 = clock()
    flight: _Flight | None = None
    failures = 0

    def fetch(f: _Flight, rtc, state) -> None:
        try:
            f.chunk = session.predict(
                state=state,
                side_png=side.png_bytes(),
                wrist_png=wrist.png_bytes(),
                tick=f.request_tick,
                rtc=rtc,
            )
        except Exception as exc:
            f.error = exc
        f.arrival_tick = int((clock() - t0) * fps)
        f.done.set()

    try:
        for tick in range(int(round(duration_s * fps))):
            if stop is not None and stop.is_set():
                break
            if flight is not None and flight.done.is_set():
                if flight.error is not None:
                    err = flight.error
                    failures += 1
                    if (isinstance(err, InferenceError) and err.status == 401) or failures >= MAX_CONSECUTIVE_FAILURES:
                        raise err
                    print(f"[vla] predict failed, holding: {err}")
                elif player.accept(flight.chunk, flight.request_tick, flight.arrival_tick):
                    failures = 0
                else:
                    print(f"[vla] dropped a late chunk (latency {flight.arrival_tick - flight.request_tick} rows)")
                flight = None
            if flight is None and player.should_request(tick):
                flight = _Flight(request_tick=tick)
                # The motor bus is read here, on the control thread, never from the worker.
                state = arm.get_state()
                threading.Thread(target=fetch, args=(flight, player.make_rtc(tick), state), daemon=True).start()
            row = player.row_for(tick)
            if row is not None:
                arm.send_joints(row)
            else:
                arm.hold()
            delay = t0 + (tick + 1) / fps - clock()
            if delay > 0:
                sleep(delay)
    finally:
        try:
            arm.hold()
        except Exception:
            pass
        session.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="zoom", choices=sorted(SHOTS))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--real", action="store_true", help="real arm and cameras (default: mocks)")
    parser.add_argument("--live", action="store_true", help="actually move the arm (default: frozen)")
    args = parser.parse_args()
    if not args.live:
        os.environ["VLA_DRY_RUN"] = "1"
    prompt = SHOTS[args.task]["vla_prompt"]
    if not prompt:
        raise SystemExit(f"{args.task} has no vla_prompt yet. Read it from the dataset into SHOTS first.")

    from robot.arm import Arm, MockArm
    from robot.cameras import Camera, MockCamera

    cam = Camera if args.real else MockCamera
    side = cam(int(os.getenv("SIDE_CAM_INDEX", "0")), "side")
    wrist = cam(int(os.getenv("WRIST_CAM_INDEX", "1")), "wrist")
    side.open()
    wrist.open()
    arm = (Arm if args.real else MockArm)(side, wrist)
    arm.connect()
    try:
        run_task(prompt, args.duration, arm=arm, side=side, wrist=wrist, stop=arm._stop)
    finally:
        side.close()
        wrist.close()


if __name__ == "__main__":
    main()
