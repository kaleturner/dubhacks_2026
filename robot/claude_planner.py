"""Claude plans the whole camera move, the arm plays it, Claude checks in midway.

  python -m robot.claude_planner --brief "cinematic reveal of the duck" --seconds 20

1. Survey: the arm visits a dozen safe poses and photographs the scene from each.
   Claude gets them as a labelled contact sheet, so it sees what every pose shows
   with the object where it is now. No camera model needed.
2. Plan: one call with the full briefing (what the rig is, its limits, shot
   language) returns a waypoint program: blends of surveyed poses, small offsets,
   seconds per segment, and checkpoints.
3. Play: one smooth curve through the waypoints at 30 Hz. At each checkpoint the
   live wrist frame goes back to Claude, which may rewrite the rest of the plan
   while the arm keeps moving.
4. Review: Claude looks at frames from the take and says keep or reshoot.

Every command still passes the joint envelope and the table guard in Arm.send_joints.
"""

import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path

import anthropic
import cv2
import numpy as np
from dotenv import load_dotenv

from robot.claude_director import ENVELOPE
from robot.poses import POSES, load_replay

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = os.getenv("CLAUDE_DIRECTOR_MODEL", "claude-opus-5-5")
HZ = 30
MAX_OFFSET = np.array([20.0, 10.0])  # pan, tilt nudges Claude may add to a waypoint
SEGMENT_S = (1.5, 8.0)


def _variant(q, pan=0.0, lift=0.0, tilt=0.0) -> list[float]:
    out = [float(v) for v in q]
    out[0] += pan
    out[1] += lift
    out[3] += tilt
    return out


def survey_poses() -> dict[str, list[float]]:
    """Label -> joints. Each passes the recorded envelope and the table guard, and keeps the
    gripper tip no further out than 20 cm unless it is 10 cm or more above the table, so
    the arm stays back from where the product sits (scout_2 touched the duck)."""
    s0, s1, s3, s4 = POSES["scout_0"], POSES["scout_1"], POSES["scout_3"], POSES["scout_4"]
    top = load_replay("sweep")["actions"][int(2.5 * 30)]  # high, looking straight down
    poses = {
        "A": _variant(POSES["home"]),                   # rest: low, close
        "B": _variant(s0),                              # front, medium
        "C": _variant(s0, pan=-25),                     # front-left
        "D": _variant(s0, pan=25),                      # front-right
        "E": _variant(s0, lift=-10, tilt=-20),          # raised, camera more level
        "F": _variant(s1, pan=-25, lift=-10, tilt=-20), # raised, level, from the left
        "G": _variant(s1, pan=25, lift=-10, tilt=-20),  # raised, level, from the right
        "H": _variant(s3, pan=-25, tilt=-10),           # low near the base, left
        "I": _variant(s3, pan=25, tilt=-10),            # low near the base, right
        "J": _variant(s4, pan=-25),                     # low, angled, left
        "K": _variant(s4, pan=25),                      # low, angled, right
        "L": _variant(top),                             # high top-down
        "M": _variant(top, pan=-25, lift=-10, tilt=-20),  # high wide, left
        "N": _variant(top, pan=25, lift=-10, tilt=-20),   # high wide, right
        "O": _variant(s1, pan=-25),                     # front-left, slightly closer
    }
    return poses


SYSTEM = """You are the director and camera operator of a robotic product-video rig.

The rig: an SO-101 robot arm (5 joints + an unused gripper) with a small camera on its wrist. That wrist camera films the shot; a fixed side camera watches the scene. A product sits on the table in front of the arm. The app turns a spoken request into a short product video.

How you control it: you do not steer joints directly. The arm has just photographed the scene from a set of safe survey poses, each with a letter. You see every pose's wrist view. You write a waypoint program built from those poses:
- pose: a survey letter. Optionally `toward` another letter with `mix` 0..1 to land between two poses (joint-space blend; 0.5 is halfway).
- pan_offset (degrees, -20..20; + turns the arm to its left, which moves the object toward the right of the wrist frame) and tilt_offset (-10..10; + tilts the camera down, which moves the object up in the frame) to fine-tune framing at that waypoint.
- seconds: time to travel from the previous waypoint to this one (1.5..8). The arm moves on one smooth curve through all waypoints, so steady durations give steady motion.
- checkpoint: true where you want to see the live wrist frame and possibly revise the rest of the plan. A revision takes you 3-5 s while the arm keeps moving, so put checkpoints at least 6 s before the end and no more than 2-3 per shot.

Limits: joints stay inside the range the arm was recorded in, and a table guard lifts anything that would come within 1.5 cm of the table, so plan clear of the table rather than relying on it. Never plan a path that drives the camera or the gripper into the product: if a survey view shows the product very large or touching the bottom edge, that pose is close to it. Blends between two poses pass through joint space, not a straight camera line, so keep consecutive waypoints similar unless you want a big move. The wrist camera is 640x480 and soft; motion faster than about 6 degrees per second blurs.

What reads as cinematic: one clear move per shot (a slow arc, a push-in, a rise, a reveal), the product well framed (center or rule of thirds), steady speed, a deliberate start and a held final frame. What reads as amateur: tiny nudges that look static, reversals, the product leaving the frame, the claw dominating the frame.

Use the whole duration the brief gives you."""


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "description": "one sentence: the shot you are going for"},
        "waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pose": {"type": "string"},
                    "toward": {"type": "string", "description": "second survey letter to blend toward, or empty"},
                    "mix": {"type": "number"},
                    "pan_offset": {"type": "number"},
                    "tilt_offset": {"type": "number"},
                    "seconds": {"type": "number"},
                    "checkpoint": {"type": "boolean"},
                    "why": {"type": "string"},
                },
                "required": ["pose", "toward", "mix", "pan_offset", "tilt_offset", "seconds", "checkpoint", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["intent", "waypoints"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "boolean"},
        "score": {"type": "integer", "description": "1-10"},
        "problems": {"type": "string"},
        "next_time": {"type": "string", "description": "what to change on a reshoot"},
    },
    "required": ["keep", "score", "problems", "next_time"],
    "additionalProperties": False,
}


def _b64(frame: np.ndarray, width: int = 640, quality: int = 80) -> str:
    h, w = frame.shape[:2]
    if w != width:
        frame = cv2.resize(frame, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.standard_b64encode(buf.tobytes()).decode()


def _image(frame: np.ndarray, width: int = 640) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(frame, width)}}


def contact_sheet(views: dict[str, np.ndarray]) -> np.ndarray:
    tiles = []
    for label, frame in views.items():
        tile = cv2.resize(frame, (320, 240))
        cv2.rectangle(tile, (0, 0), (44, 34), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])


class Planner:
    def __init__(self) -> None:
        workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        self.client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": workspace} if workspace else None)
        self.usage = {"input": 0, "output": 0, "calls": 0, "seconds": 0.0}

    def _call(self, content: list, schema: dict, effort: str) -> dict:
        started = time.monotonic()
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM,
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined: {response.stop_details}")
        self.usage["input"] += response.usage.input_tokens
        self.usage["output"] += response.usage.output_tokens
        self.usage["calls"] += 1
        self.usage["seconds"] += time.monotonic() - started
        return json.loads(next(b.text for b in response.content if b.type == "text"))

    def plan(self, brief: str, seconds: float, sheet: np.ndarray, poses: dict, side: np.ndarray, notes: str = "") -> dict:
        joints = "\n".join(f"{k}: {np.round(v[:5]).astype(int).tolist()}" for k, v in poses.items())
        return self._call(
            [
                {"type": "text", "text": "Survey contact sheet (wrist view from each labelled pose):"},
                _image(sheet, 1280),
                {"type": "text", "text": "Side view of the scene now:"},
                _image(side, 480),
                {"type": "text", "text": (
                    f"Survey pose joints [pan, lift, elbow, tilt, roll]:\n{joints}\n\n"
                    f"Brief: {brief}\nDuration: {seconds:.0f} s. Write the waypoint program. "
                    "The first waypoint is where the shot starts (the arm moves there before recording)."
                    + (f"\n\nYour previous take of this shot, and your review of it:\n{notes}" if notes else "")
                )},
            ],
            PLAN_SCHEMA,
            "medium",
        )

    def revise(self, brief: str, plan: dict, done: int, wrist: np.ndarray, time_left: float) -> dict:
        remaining = plan["waypoints"][done + 1 :]
        return self._call(
            [
                {"type": "text", "text": "Live wrist frame at your checkpoint:"},
                _image(wrist),
                {"type": "text", "text": (
                    f"Brief: {brief}\nYour intent: {plan['intent']}\n"
                    f"You just reached waypoint {done} of your plan: {json.dumps(plan['waypoints'][done])}\n"
                    f"Remaining waypoints: {json.dumps(remaining)}\nTime left: {time_left:.1f} s.\n"
                    "Return the waypoints that should follow from here (replacing the remaining ones). "
                    "The arm keeps moving toward the next one while you answer, so keep the first one close "
                    "to the current next waypoint unless the shot is going wrong. Return the remaining "
                    "waypoints unchanged if they still work."
                )},
            ],
            PLAN_SCHEMA,
            "low",
        )

    def review(self, brief: str, plan: dict, frames: list[np.ndarray]) -> dict:
        content = [{"type": "text", "text": f"Frames from the finished take, in order. Brief: {brief}. Intent: {plan['intent']}."}]
        for frame in frames:
            content.append(_image(frame, 480))
        content.append({"type": "text", "text": "Is this take good enough for a product ad? Be a strict editor."})
        return self._call(content, REVIEW_SCHEMA, "low")


def resolve(waypoint: dict, poses: dict) -> np.ndarray:
    q = np.array(poses[waypoint["pose"]], dtype=float)
    other = waypoint.get("toward") or ""
    if other in poses:
        mix = float(np.clip(waypoint.get("mix", 0.5), 0.0, 1.0))
        q = (1 - mix) * q + mix * np.array(poses[other], dtype=float)
    q[0] += float(np.clip(waypoint.get("pan_offset", 0.0), -MAX_OFFSET[0], MAX_OFFSET[0]))
    q[3] += float(np.clip(waypoint.get("tilt_offset", 0.0), -MAX_OFFSET[1], MAX_OFFSET[1]))
    q[:5] = np.clip(q[:5], ENVELOPE[:, 0], ENVELOPE[:, 1])
    return q


def segment(p0, p1, p2, p3, seconds: float) -> np.ndarray:
    """30 Hz rows from p1 to p2 on a Catmull-Rom spline (p0 and p3 set the tangents)."""
    n = max(2, int(round(seconds * HZ)))
    rows = []
    for k in range(1, n + 1):
        t = k / n
        t2, t3 = t * t, t * t * t
        rows.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out = np.array(rows)
    out[:, :5] = np.clip(out[:, :5], ENVELOPE[:, 0], ENVELOPE[:, 1])
    return out


GOOD_ENOUGH = 7  # review score at which a take is kept without a reshoot


def shoot(arm, wrist_cam, side_cam, brief: str, seconds: float, out_path: str, takes: int = 2, log=print) -> dict:
    """Survey once, then plan -> play -> review, reshooting with the review as notes."""
    planner = Planner()
    poses = survey_poses()
    views = {}
    arm.move_to("home", 2.0)
    for label, q in poses.items():
        arm.move_to(q, 1.8)
        time.sleep(0.8)  # let the arm settle; 0.35 s left blurred frames
        views[label] = wrist_cam.latest()
    sheet = contact_sheet(views)
    cv2.imwrite(str(Path(out_path).with_suffix(".survey.jpg")), sheet)
    side_view = side_cam.latest()
    notes, best = "", None
    for n in range(1, takes + 1):
        path = str(Path(out_path).with_name(f"{Path(out_path).stem}_take{n}.mp4"))
        log(f"[planner] take {n}")
        result = take(arm, wrist_cam, planner, brief, seconds, sheet, poses, side_view, notes, path, log)
        if best is None or result["review"]["score"] > best["review"]["score"]:
            best = result
        if result["review"]["keep"] or result["review"]["score"] >= GOOD_ENOUGH:
            break
        notes = (
            f"Intent: {result['plan']['intent']}\nWaypoints: {json.dumps(result['plan']['waypoints'])}\n"
            f"Review: score {result['review']['score']}/10. Problems: {result['review']['problems']} "
            f"Change next time: {result['review']['next_time']}"
        )
    usage = planner.usage
    cost = usage["input"] * 4e-6 + usage["output"] * 20e-6
    log(f"[planner] best take {best['path']} scored {best['review']['score']}; {usage['calls']} calls, "
        f"{usage['seconds']:.1f}s in calls, {usage['input']}+{usage['output']} tokens, ~${cost:.3f}")
    return {**best, "usage": usage}


def take(arm, wrist_cam, planner, brief, seconds, sheet, poses, side_view, notes, out_path, log) -> dict:
    plan = planner.plan(brief, seconds, sheet, poses, side_view, notes)
    log(f"[planner] intent: {plan['intent']}")
    for i, w in enumerate(plan["waypoints"]):
        blend = f"->{w['toward']} {w['mix']:.2f}" if w["toward"] else ""
        check = " [check]" if w["checkpoint"] else ""
        log(f"[planner]  {i}: {w['pose']}{blend} pan{w['pan_offset']:+.0f} tilt{w['tilt_offset']:+.0f} "
            f"{w['seconds']:.1f}s{check} - {w['why']}")

    waypoints = list(plan["waypoints"])
    start = resolve(waypoints[0], poses)
    arm.move_to(list(start), 2.5)
    wrist_cam.start_recording(out_path)
    frames, t0, done = [], time.monotonic(), 0
    pending: dict = {}
    previous, current = start, start
    next_tick = t0
    try:
        while done < len(waypoints) - 1:
            nxt = waypoints[done + 1]
            seg = float(np.clip(nxt["seconds"], *SEGMENT_S))
            target = resolve(nxt, poses)
            ahead = resolve(waypoints[done + 2], poses) if done + 2 < len(waypoints) else target
            rows = segment(previous, current, target, ahead, seg)
            for k, row in enumerate(rows):
                if "plan" in pending:  # a revision landed: splice it in after the waypoint we are heading to
                    revised = pending.pop("plan")
                    waypoints = waypoints[: done + 2] + revised["waypoints"]
                    log(f"[planner] revised at {time.monotonic() - t0:.1f}s: {len(revised['waypoints'])} waypoints, {revised['intent']}")
                arm.send_joints(list(row))
                if k % 15 == 0:
                    frames.append(wrist_cam.latest())
                next_tick += 1 / HZ
                time.sleep(max(0.0, next_tick - time.monotonic()))
            previous, current = current, target
            done += 1
            if waypoints[done]["checkpoint"] and done < len(waypoints) - 1:
                snapshot = wrist_cam.latest()
                left = seconds - (time.monotonic() - t0)

                def ask(plan_now=dict(plan, waypoints=list(waypoints)), at=done, img=snapshot, left=left):
                    try:
                        pending["plan"] = planner.revise(brief, plan_now, at, img, left)
                    except Exception as exc:
                        log(f"[planner] revision failed, keeping the plan: {exc}")

                threading.Thread(target=ask, daemon=True).start()
        time.sleep(1.0)  # hold the final frame
    finally:
        wrist_cam.stop_recording()
    review = planner.review(brief, plan, frames[:: max(1, len(frames) // 6)][:6])
    log(f"[planner] review: keep={review['keep']} score={review['score']} problems: {review['problems']} | next: {review['next_time']}")
    return {"plan": plan, "review": review, "path": out_path}


def main() -> int:
    from robot.arm import Arm
    from robot.cameras import Camera

    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", default="cinematic product reveal of the duck: start wide, travel around it, end on a close hero angle")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--takes", type=int, default=2)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "server" / "media" / "claude_plan.mp4"))
    args = parser.parse_args()
    wrist = Camera(int(os.getenv("WRIST_CAM_INDEX", "0")), "wrist")
    side = Camera(int(os.getenv("SIDE_CAM_INDEX", "1")), "side")
    wrist.open()
    side.open()
    arm = Arm(side, wrist)
    arm.connect()
    try:
        shoot(arm, wrist, side, args.brief, args.seconds, args.out, takes=args.takes)
    finally:
        arm.move_to("home", 2.5)
        wrist.close()
        side.close()
    print(f"clips: {Path(args.out).parent}/{Path(args.out).stem}_take*.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
