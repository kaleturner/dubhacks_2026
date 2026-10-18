import json
import threading
import time

import pytest

from robot.arm import MockArm
from robot.vla_client import (
    Chunk,
    InferenceError,
    Player,
    SessionClient,
    parse_chunk,
    run_task,
)

PROFILE = {"fps": 30, "execution_stride": 15, "max_prefix": 12, "deadline_ms": 400}


def rows(n, base=0.0):
    return [[base + i, 0, 0, 0, 0, 0] for i in range(n)]


def chunk(base=0.0, execute_from=0, n=30, cid="c"):
    return Chunk(chunk_id=cid, rows=rows(n, base), execute_from=execute_from)


# --- parse_chunk --------------------------------------------------------

def test_parse_chunk_finds_rows_and_ids():
    c = parse_chunk({"chunk_id": "abc", "actions": rows(30), "prefix_length": 12, "execute_from_index": 12})
    assert c.chunk_id == "abc" and len(c.rows) == 30 and c.execute_from == 12


def test_parse_chunk_nested_and_missing():
    assert len(parse_chunk({"result": {"action_chunk": rows(5)}}).rows) == 5
    with pytest.raises(InferenceError):
        parse_chunk({"nothing": 1})
    with pytest.raises(InferenceError):
        parse_chunk({"actions": [[float("nan")] * 6]})


# --- Player: the real-time rules ---------------------------------------

def test_first_chunk_plays_from_the_tick_it_belongs_to():
    p = Player(PROFILE)
    assert p.should_request(0) and p.make_rtc(0) is None
    assert p.accept(chunk(), request_tick=0, arrival_tick=4)
    assert p.row_for(4)[0] == 4  # rows 0-3 already elapsed
    assert p.row_for(29)[0] == 29 and p.row_for(30) is None


def test_next_request_after_stride_carries_rtc():
    p = Player(PROFILE)
    p.accept(chunk(), 0, 3)
    p.row_for(3)
    assert not p.should_request(14) and p.should_request(15)
    rtc = p.make_rtc(15)
    assert rtc["prior_start_index"] == 15 and rtc["prior_chunk_id"] == "c"
    assert 1 <= rtc["delay"] <= 12 and len(rtc["committed_actions"]) == rtc["delay"]
    assert rtc["committed_actions"][0] == rows(30)[15]


def test_new_chunk_takes_over_at_execute_from_not_at_arrival():
    p = Player(PROFILE)
    p.accept(chunk(base=0), 0, 3)
    p.row_for(3)
    assert p.accept(chunk(base=100, execute_from=12, cid="d"), request_tick=15, arrival_tick=20)
    assert p.row_for(26)[0] == 26      # still the old chunk before the boundary
    assert p.row_for(27)[0] == 100 + 12  # 15 + 12 = 27: switched, row 12 of the new chunk


def test_late_reply_is_dropped_and_old_rows_keep_playing():
    p = Player(PROFILE)
    p.accept(chunk(), 0, 3)
    p.row_for(3)
    assert not p.accept(chunk(base=100, execute_from=12, cid="d"), request_tick=15, arrival_tick=15 + 13)  # 433 ms
    assert p.dropped == 1
    assert p.row_for(20)[0] == 20


def test_reply_whose_switch_point_has_passed_is_dropped():
    p = Player(PROFILE)
    p.accept(chunk(), 0, 3)
    p.row_for(3)
    assert not p.accept(chunk(base=100, execute_from=2, cid="d"), request_tick=15, arrival_tick=19)


def test_holds_when_no_chunk_covers_the_tick():
    p = Player(PROFILE)
    assert p.row_for(0) is None
    p.accept(chunk(n=10), 0, 1)
    assert p.row_for(9) is not None and p.row_for(10) is None


def test_no_new_request_while_a_chunk_waits_to_take_over():
    p = Player(PROFILE)
    p.accept(chunk(), 0, 3)
    p.row_for(3)
    p.accept(chunk(base=100, execute_from=12, cid="d"), request_tick=15, arrival_tick=20)
    assert not any(p.should_request(t) for t in range(16, 30))
    p.row_for(27)
    assert p.should_request(30)


def test_rtc_omitted_when_old_chunk_ran_out():
    p = Player(PROFILE)
    p.accept(chunk(n=16), 0, 2)
    p.row_for(2)
    assert p.make_rtc(15) is None  # 15 + delay > 16 rows


def test_delay_follows_measured_latency_within_bounds():
    p = Player(PROFILE)
    for _ in range(30):
        p.accept(chunk(), 0, 0)
    assert p.delay() == 1
    p.latency_ticks = 40
    assert p.delay() == 12


# --- SessionClient: paths, headers, bodies -----------------------------

class Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.content = b"x"
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self, reject_walkthrough=False):
        self.calls = []
        self.reject = reject_walkthrough

    def request(self, method, url, headers=None, timeout=None, **kw):
        self.calls.append((method, url, headers, kw))
        if self.reject and "/v1/inference/" not in url:
            return Resp(404, {"detail": "Not Found"})
        if url.endswith("/predict"):
            return Resp(200, {"chunk_id": "k1", "actions": rows(30)})
        return Resp(200, {"ok": True})


class FakeApi:
    def __init__(self):
        self.calls = []

    def call(self, method, path, **kw):
        self.calls.append((method, path, kw.get("body")))
        if path == "/v1/capabilities":
            return {"control_profiles": [PROFILE]}
        if path.endswith("/contract"):
            return {"contract_id": "c" * 64}
        return {"endpoint_url": "https://ep.example", "application_session_token": "APP", "modal_session_token": "MOD"}


def opened(http):
    s = SessionClient("dep1", "do the thing", api=FakeApi(), http=http)
    s.open()
    return s


def test_open_uses_tokens_not_team_key_and_walkthrough_paths():
    http = FakeHttp()
    s = opened(http)
    method, url, headers, kw = http.calls[0]
    assert url == "https://ep.example/deployments/dep1/sessions"
    assert headers["Authorization"] == "Bearer APP" and headers["Modal-Authorization"] == "Bearer MOD"
    assert kw["json"] == {"instruction": "do the thing", "contract_id": "c" * 64, "rtc_enabled": True}
    assert s.api.calls[2][2] == {"session_id": s.session_id, "ttl_seconds": 21600}


def test_falls_back_to_v1_inference_paths_and_remembers():
    http = FakeHttp(reject_walkthrough=True)
    s = opened(http)
    assert http.calls[-1][1] == "https://ep.example/v1/inference/deployments/dep1/sessions"
    s.predict(state=[0] * 6, side_png=b"a", wrist_png=b"b", tick=0)
    assert http.calls[-1][1].endswith(f"/v1/inference/sessions/{s.session_id}/predict")
    assert sum("/v1/inference/" not in c[1] for c in http.calls) == 1  # only the first probe


def test_predict_metadata_is_monotonic_and_png():
    http = FakeHttp()
    s = opened(http)
    for tick in (0, 15, 30):
        s.predict(state=[1, 2, 3, 4, 5, 6], side_png=b"a", wrist_png=b"b", tick=tick,
                  rtc=None if tick == 0 else {"prior_chunk_id": "k1", "prior_start_index": 15, "delay": 5, "committed_actions": rows(5)})
    metas = [json.loads(c[3]["files"]["metadata"][1]) for c in http.calls if c[1].endswith("/predict")]
    assert [m["sequence"] for m in metas] == [0, 1, 2]
    assert [m["timestamp_seconds"] for m in metas] == [0.0, 0.5, 1.0]
    assert "rtc" not in metas[0] and metas[1]["rtc"]["delay"] == 5
    assert metas[0]["image_encoding"] == "png" and metas[0]["state"] == [1, 2, 3, 4, 5, 6]
    files = http.calls[-1][3]["files"]
    assert set(files) == {"metadata", "side", "wrist"}


def test_http_error_carries_status():
    class Bad(FakeHttp):
        def request(self, *a, **k):
            return Resp(401, {"detail": "no"})

    s = SessionClient("dep1", "x", api=FakeApi(), http=FakeHttp())
    s.open()
    s.http = Bad()
    with pytest.raises(InferenceError) as err:
        s.predict(state=[0] * 6, side_png=b"", wrist_png=b"", tick=0)
    assert err.value.status == 401


# --- run_task: the whole loop on fakes ---------------------------------

class FakeCam:
    def png_bytes(self):
        return b"png"


class FakeSession:
    profile = PROFILE

    def __init__(self, latency_s=0.05, fail=None):
        self.latency_s = latency_s
        self.fail = fail
        self.requests = []
        self.closed = False
        self.opened = False
        self.n = 0

    def open(self):
        self.opened = True

    def predict(self, *, state, side_png, wrist_png, tick, rtc=None):
        self.requests.append((tick, rtc, state))
        time.sleep(self.latency_s)
        if self.fail:
            raise self.fail
        self.n += 1
        return chunk(base=1000 * self.n, execute_from=0 if rtc is None else 12, cid=f"c{self.n}")

    def close(self):
        self.closed = True


def drive(session, duration=1.5, stop=None):
    arm = MockArm()
    run_task("x", duration, arm=arm, side=FakeCam(), wrist=FakeCam(), session=session, stop=stop)
    return arm


def test_run_task_requests_on_the_stride_and_moves_the_arm():
    session = FakeSession()
    arm = drive(session)
    ticks = [r[0] for r in session.requests]
    assert ticks[0] == 0
    assert all(b - a >= 15 for a, b in zip(ticks, ticks[1:]))
    assert session.requests[0][1] is None and session.requests[1][1] is not None
    assert len(arm.sent) > 20
    assert session.closed


def test_run_task_never_sends_a_late_chunk():
    session = FakeSession(latency_s=0.6)  # past the 400 ms deadline
    arm = drive(session, duration=1.2)
    assert arm.sent == []
    assert session.closed


def test_run_task_stops_on_event_and_cleans_up():
    stop = threading.Event()
    stop.set()
    session = FakeSession()
    arm = drive(session, stop=stop)
    assert arm.sent == [] and session.closed


def test_run_task_aborts_after_repeated_failures_and_closes():
    session = FakeSession(latency_s=0.0, fail=InferenceError(503, "busy"))
    with pytest.raises(InferenceError):
        drive(session, duration=4.0)
    assert session.closed


def test_run_task_aborts_on_401_immediately():
    session = FakeSession(latency_s=0.0, fail=InferenceError(401, "expired"))
    with pytest.raises(InferenceError):
        drive(session, duration=4.0)
    assert len(session.requests) == 1


def test_run_task_needs_a_deployment(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_ID", raising=False)
    with pytest.raises(RuntimeError, match="DEPLOYMENT_ID"):
        run_task("x", 1, arm=MockArm(), side=FakeCam(), wrist=FakeCam())
