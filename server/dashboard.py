"""Live debug dashboard. Reads the server's cameras; never opens one.

  GET /dashboard            both feeds plus job, centering, and joint state
  GET /debug/state          JSON behind the status panel
  GET /debug/side.mjpg      MJPEG of the side camera
  GET /debug/wrist.mjpg     MJPEG of the wrist camera with the tracking box
"""

import asyncio
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from robot import centering
from shared.constants import JOINT_NAMES

STREAM_FPS = 12
JPEG_QUALITY = 70
BOX_FRESH_S = 1.0
KEEP_MESSAGES = 10
BOUNDARY = "frame"

router = APIRouter()


def message_log() -> deque:
    """Most recent job messages first. The /ws handler appends to the left."""
    return deque(maxlen=KEEP_MESSAGES)


def remember(log: deque, message: dict) -> None:
    log.appendleft({"t": time.time(), **message})


@router.get("/debug/state")
def state(request: Request) -> dict:
    app_state = request.app.state
    arm = getattr(app_state, "arm", None)
    joints = getattr(arm, "_last_sent", None)
    return {
        "time": time.time(),
        "centering": dict(centering.LIVE),
        "messages": list(getattr(app_state, "messages", ())),
        "joints": None if joints is None else dict(zip(JOINT_NAMES, (float(v) for v in joints))),
    }


def _overlay(frame: np.ndarray, live: dict) -> np.ndarray:
    """Tracking box, centre crosshair, and a state line on the wrist frame."""
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)

    fresh = time.time() - float(live.get("updated") or 0.0) < BOX_FRESH_S
    box = live.get("box")
    if live.get("state") == "tracking" and fresh and box:
        x, y, w, h = (int(round(v)) for v in box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        score = live.get("score")
        label = live.get("object") or "object"
        if score is not None:
            label = f"{label} {score:.2f}"
        cv2.putText(frame, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    pan, flex = (live.get("offset") or [0.0, 0.0])[:2]
    line = f"{live.get('state', '?')}{'' if fresh else ' (stale)'}  pan {pan:+.1f}  flex {flex:+.1f}"
    cv2.rectangle(frame, (0, height - 26), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, line, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _encode(frame: np.ndarray, wrist: bool) -> bytes | None:
    if wrist:
        frame = _overlay(frame, centering.LIVE)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


async def _mjpeg(request: Request, camera: Any, wrist: bool, limit: int):
    sent = 0
    while not await request.is_disconnected():
        started = time.monotonic()
        frame = camera.latest()
        if frame is not None:
            jpg = await asyncio.to_thread(_encode, frame, wrist)
            if jpg is not None:
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
                    + jpg
                    + b"\r\n"
                )
                sent += 1
                if limit and sent >= limit:
                    return
        await asyncio.sleep(max(0.0, 1 / STREAM_FPS - (time.monotonic() - started)))


@router.get("/debug/{name}.mjpg")
def stream(name: str, request: Request, limit: int = 0) -> StreamingResponse:
    """`limit` stops after that many frames; 0 streams until the client leaves."""
    if name not in ("side", "wrist"):
        raise HTTPException(404, "unknown camera")
    camera = getattr(request.app.state, name, None)
    if camera is None:
        raise HTTPException(503, f"{name} camera is not open")
    return StreamingResponse(
        _mjpeg(request, camera, name == "wrist", limit),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return PAGE


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Debug</title>
<style>
  :root { --bg:#0e1013; --panel:#171a1f; --line:#262b33; --text:#e6e8eb; --dim:#8a919c;
          --ok:#3fd07a; --warn:#f0b43c; --bad:#ef5b5b; --info:#5aa9f5; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:10px 16px; border-bottom:1px solid var(--line); }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  #conn { font-size:12px; color:var(--dim); margin-left:auto; }
  main { padding:16px; display:grid; gap:16px; }
  .feeds { display:flex; flex-wrap:wrap; gap:16px; }
  figure { margin:0; flex:1 1 420px; min-width:0; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; overflow:hidden; }
  figure img { display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#000; }
  figcaption { padding:6px 10px; font-size:12px; color:var(--dim); text-transform:uppercase; letter-spacing:.06em; }
  .panels { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:16px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
  h2 { margin:0 0 8px; font-size:12px; color:var(--dim); font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
  .big { font-size:22px; font-weight:700; }
  .detail { color:var(--text); margin-top:2px; word-break:break-word; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:700; font-size:13px;
          background:var(--line); }
  .ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); } .info { color:var(--info); }
  dl { display:grid; grid-template-columns:auto 1fr; gap:3px 12px; margin:8px 0 0; }
  dt { color:var(--dim); } dd { margin:0; font-variant-numeric:tabular-nums; font-family:ui-monospace, Menlo, monospace; }
  ol { list-style:none; margin:0; padding:0; max-height:260px; overflow:auto; }
  li { padding:4px 0; border-top:1px solid var(--line); font-size:13px; }
  li:first-child { border-top:0; }
  li .when { color:var(--dim); font-family:ui-monospace, Menlo, monospace; margin-right:6px; }
  li .stage { font-weight:600; margin-right:6px; }
</style>
</head>
<body>
<header><h1>Robot camera debug</h1><span id="conn">connecting...</span></header>
<main>
  <div class="feeds">
    <figure><img src="/debug/wrist.mjpg" alt="wrist camera"><figcaption>Wrist (films the clip)</figcaption></figure>
    <figure><img src="/debug/side.mjpg" alt="side camera"><figcaption>Side (C920)</figcaption></figure>
  </div>
  <div class="panels">
    <section>
      <h2>Job</h2>
      <div id="stage" class="big">idle</div>
      <div id="detail" class="detail"></div>
    </section>
    <section>
      <h2>Centering</h2>
      <span id="cstate" class="pill">-</span>
      <dl>
        <dt>object</dt><dd id="cobj">-</dd>
        <dt>score</dt><dd id="cscore">-</dd>
        <dt>pan</dt><dd id="cpan">-</dd>
        <dt>flex</dt><dd id="cflex">-</dd>
        <dt>box</dt><dd id="cbox">-</dd>
        <dt>age</dt><dd id="cage">-</dd>
      </dl>
    </section>
    <section>
      <h2>Joints (last sent)</h2>
      <dl id="joints"><dt>-</dt><dd>no command yet</dd></dl>
    </section>
    <section>
      <h2>Messages</h2>
      <ol id="log"></ol>
    </section>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const STAGE = { DONE: "ok", ERROR: "bad", RECORD: "info" };
const CSTATE = { tracking: "ok", searching: "warn", gone: "bad" };
const num = (v, d = 1) => v == null ? "-" : (v >= 0 ? "+" : "") + Number(v).toFixed(d);
const clock = t => new Date(t * 1000).toLocaleTimeString([], { hour12: false });

function render(s) {
  const status = s.messages.find(m => m.type === "status");
  $("stage").textContent = status ? status.stage : "idle";
  $("stage").className = "big " + (status ? STAGE[status.stage] || "" : "");
  $("detail").textContent = status ? status.detail : "";

  const c = s.centering, age = c.updated ? s.time - c.updated : null;
  $("cstate").textContent = c.state + (age != null && age > 1 ? " (stale)" : "");
  $("cstate").className = "pill " + (age != null && age <= 1 ? CSTATE[c.state] || "" : "");
  $("cobj").textContent = c.object || "-";
  $("cscore").textContent = c.score == null ? "-" : Number(c.score).toFixed(2);
  $("cpan").textContent = num(c.offset && c.offset[0]) + " deg";
  $("cflex").textContent = num(c.offset && c.offset[1]) + " deg";
  $("cbox").textContent = c.box ? c.box.map(v => Math.round(v)).join(", ") : "-";
  $("cage").textContent = age == null ? "never" : age.toFixed(1) + " s";

  const j = $("joints");
  j.replaceChildren();
  for (const [k, v] of Object.entries(s.joints || { "-": null })) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = v == null ? "no command yet" : v.toFixed(1);
    j.append(dt, dd);
  }

  const log = $("log");
  log.replaceChildren();
  for (const m of s.messages) {
    const li = document.createElement("li");
    const when = document.createElement("span"), stage = document.createElement("span");
    when.className = "when"; when.textContent = clock(m.t);
    stage.className = "stage " + (STAGE[m.stage] || ""); stage.textContent = m.type === "result" ? "RESULT" : m.stage;
    li.append(when, stage, document.createTextNode(m.type === "result" ? m.url : m.detail || ""));
    log.append(li);
  }
}

async function poll() {
  try {
    const r = await fetch("/debug/state", { cache: "no-store" });
    render(await r.json());
    $("conn").textContent = "live " + new Date().toLocaleTimeString([], { hour12: false });
    $("conn").className = "";
  } catch (e) {
    $("conn").textContent = "server unreachable";
    $("conn").className = "bad";
  }
  setTimeout(poll, 500);
}
poll();
</script>
</body>
</html>
"""
