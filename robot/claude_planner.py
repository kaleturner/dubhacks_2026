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


def _variant(q, pan=0.0, lift=0.0, tilt=0.0, roll=0.0) -> list[float]:
    out = [float(v) for v in q]
    out[0] += pan
    out[1] += lift
    out[3] += tilt
    out[4] += roll
    return out


def survey_poses() -> dict[str, list[float]]:
    """Label -> starting joints for the survey; each is then aimed at the product.

    The arm's base turns and its other joints bend in one vertical plane that the wrist
    camera looks along, so aimed at the product the camera always sees it from the base's
    direction. What varies is distance and height along that line, plus roll. These
    starting poses spread over that range; all were reached safely before.
    """
    s0, s1, s3, s4 = POSES["scout_0"], POSES["scout_1"], POSES["scout_3"], POSES["scout_4"]
    top = load_replay("sweep")["actions"][int(2.5 * 30)]  # high, looking straight down
    return {
        "A": _variant(POSES["home"]),                 # low and close
        "B": _variant(s0),                            # low-medium
        "D": _variant(s0, lift=-10, tilt=-20),        # raised, camera more level
        "E": _variant(s1, lift=-10, tilt=-20),        # raised further
        "F": _variant(s3, tilt=-10),                  # near the base, looking out
        "G": _variant(s4, tilt=-5),                   # near the base, higher
        "H": _variant(top),                           # high, top-down
        "I": _variant(top, lift=-10, tilt=-20),       # high and wide
        "J": _variant(s0, roll=25),                   # low-medium, rolled left (dutch)
        "K": _variant(s0, roll=-25),                  # low-medium, rolled right (dutch)
    }


# The claw sits in the lower-left of the wrist image. Final clips (and what Claude sees)
# crop the left and bottom 20%; the aim point is that crop's center.
CROP = (0.2, 0.0, 0.8, 0.8)  # x, y, w, h as fractions of the frame
AIM = (CROP[0] + CROP[2] / 2, CROP[1] + CROP[3] / 2)


def crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, cw, ch = int(CROP[0] * w), int(CROP[1] * h), int(CROP[2] * w), int(CROP[3] * h)
    return cv2.resize(frame[y : y + ch, x : x + cw], (w, h))


def crop_video(src: str, dst: str) -> None:
    import subprocess

    vf = f"crop=iw*{CROP[2]}:ih*{CROP[3]}:iw*{CROP[0]}:ih*{CROP[1]},scale=640:480"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vf", vf, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-preset", "veryfast", dst], check=True)


def _safe(q) -> bool:
    from robot.camera_model import tool_pose
    from robot.safety import FLOOR_Z, lowest_z

    q = np.asarray(q, dtype=float)
    inside = bool(np.all((q[:5] >= ENVELOPE[:, 0]) & (q[:5] <= ENVELOPE[:, 1])))
    tip = tool_pose(q[:5])[:3, 3]
    return inside and lowest_z(list(q)) >= FLOOR_Z and (tip[0] <= 0.20 or tip[2] >= 0.10)


def aim(arm, wrist_cam, q, obj: str, log=print, rounds: int = 3):
    """Turn and tilt until the product sits at AIM. Returns (joints, frame) or (None, frame)."""
    from ml.perception import locate
    from robot.centering import CALIBRATION

    inverse = np.linalg.pinv(np.array(json.loads(CALIBRATION.read_text())["jacobian"]))
    q = np.array(q, dtype=float)
    arm.move_to(list(q), 1.8)
    best = None
    for _ in range(rounds):
        time.sleep(0.7)
        frame = wrist_cam.latest()
        path = Path("/tmp") / "aim.jpg"
        cv2.imwrite(str(path), frame)
        found = locate(str(path), obj)
        time.sleep(0.6)  # one Mk1.5 request at a time
        if not found["found"]:
            break
        h, w = frame.shape[:2]
        err = np.array([found["center"][0] / w - AIM[0], found["center"][1] / h - AIM[1]])
        if best is None or np.abs(err).max() < np.abs(best[2]).max():
            best = (q.copy(), frame, err)
        if np.abs(err).max() < 0.06:
            break
        step = np.clip(-(inverse @ err), -15, 15)  # degrees of pan, flex
        trial = q.copy()
        trial[0] += step[0]
        trial[3] += step[1]
        if not _safe(trial):
            break
        q = trial
        arm.move_to(list(q), 1.0)
    if best is None or np.abs(best[2]).max() > 0.2:
        return None, (best[1] if best else wrist_cam.latest())
    return best[0].tolist(), best[1]


SYSTEM = """You are the director and camera operator of a robotic product-video rig.

The rig: an SO-101 robot arm (5 joints + an unused gripper) with a small camera on its wrist. That wrist camera films the shot; a fixed side camera watches the scene. A product sits on the table in front of the arm. The app turns a spoken request into a short product video.

What the arm can and cannot do: its base turns and its other joints bend in one vertical plane that the wrist camera looks along. So whenever the camera is aimed at the product it sees it from the direction of the arm's base. It can NOT orbit or reach the product's sides. What it can do: move closer or farther along that line (push-in, pull-out), higher or lower (a crane move that arcs from a low, level view up to top-down), roll the image (dutch angle or a spin), and drift the product across the frame with small pans. Plan within that.

How you control it: you do not steer joints directly. The arm has just photographed the scene from a set of safe survey poses, each with a letter, each already aimed so the product sits near the center of the frame you see (the frames are cropped exactly as the final video will be). You see every pose's wrist view. You write a waypoint program built from those poses:
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

QUICK_SEGMENT_S = (1.0, 5.0)
QUICK_POSES_TEXT = """Pose menu (every pose is already turned to face the product; while the shot plays, an automatic tracker fine-tunes pan and tilt so the product stays framed, so you design the shape of the move, not the aim):
  LOW   camera near table height, nearly level: a low, intimate, heroic view
  LOW2  like LOW, camera tilted further up: shows more background, more depth
  MED   medium height, looking down at about 45 degrees
  RAISE raised and pulled back, camera level-ish: a wider establishing view
  HIGH  high, looking steeply down
  TOP   high, straight down: a graphic top-down view
  DUTCHL / DUTCHR  MED rolled 25 degrees left / right (tilted horizon)"""


def quick_poses(pan: float) -> dict[str, list[float]]:
    """The pose menu, each turned to `pan` (the base angle that faces the product); unsafe ones dropped."""
    s0, s1 = POSES["scout_0"], POSES["scout_1"]
    top = load_replay("sweep")["actions"][int(2.5 * 30)]
    menu = {
        "LOW": _variant(s0, lift=-10, tilt=-20),
        "LOW2": _variant(s0, lift=-10, tilt=-30),
        "MED": _variant(s0),
        "RAISE": _variant(s1, lift=-10, tilt=-20),
        "HIGH": _variant(top, lift=-10, tilt=-20),
        "TOP": _variant(top),
        "DUTCHL": _variant(s0, roll=25),
        "DUTCHR": _variant(s0, roll=-25),
    }
    out = {}
    for name, q in menu.items():
        q[0] = pan
        if _safe(q):
            out[name] = q
    return out


def find_pan(arm, wrist_cam, obj: str, log=print) -> tuple[float, np.ndarray]:
    """Look once from high and wide; return the base angle that faces the product."""
    from ml.perception import locate
    from robot.centering import CALIBRATION

    top = load_replay("sweep")["actions"][int(2.5 * 30)]
    look = _variant(top, lift=-10, tilt=-20)
    arm.move_to(look, 2.0)
    time.sleep(0.5)
    frame = wrist_cam.latest()
    path = Path("/tmp") / "overview.jpg"
    cv2.imwrite(str(path), frame)
    found = locate(str(path), obj)
    if not found["found"]:
        log("[planner] overview: product not seen, keeping the base straight")
        return float(look[0]), frame
    h, w = frame.shape[:2]
    x = found["center"][0] / w
    # Only the pan column: image x moves -0.0128 frame widths per degree of pan.
    per_deg = json.loads(CALIBRATION.read_text())["jacobian"][0][0]
    pan = float(np.clip(look[0] + (AIM[0] - x) / per_deg, ENVELOPE[0, 0], ENVELOPE[0, 1]))
    log(f"[planner] overview: product at x={x:.2f}; base turned to {pan:.0f} deg to face it")
    return pan, frame


def quick_shoot(arm, wrist_cam, side_cam, brief: str, seconds: float, out_path: str, takes: int = 2, log=print, obj: str | None = None) -> dict:
    """Look once from above, plan from a pose menu facing the product, play with live centering."""
    obj = obj or ("duck" if "duck" in brief.lower() else "product")
    planner = Planner()
    pan, overview = find_pan(arm, wrist_cam, obj, log)
    poses = quick_poses(pan)
    side_view = side_cam.latest()
    notes, best = "", None
    for n in range(1, takes + 1):
        path = str(Path(out_path).with_name(f"{Path(out_path).stem}_take{n}.mp4"))
        log(f"[planner] take {n}")
        plan = planner._call(
            [
                {"type": "text", "text": "Overview from high and wide (wrist camera, cropped as the final video):"},
                _image(crop(overview)),
                {"type": "text", "text": "Side view of the scene:"},
                _image(side_view, 480),
                {"type": "text", "text": (
                    f"{QUICK_POSES_TEXT}\nAvailable now: {', '.join(poses)}.\n\n"
                    f"Brief: {brief}\nDuration: {seconds:.0f} s. Segments may be 1 to 5 s; brisk, confident "
                    "moves are fine (up to about 15 degrees per second) as long as each move goes one way. "
                    "Write the waypoint program using the pose names; offsets are rarely needed."
                    + (f"\n\nYour previous take and your review of it:\n{notes}" if notes else "")
                )},
            ],
            PLAN_SCHEMA,
            "low",
        )
        result = play(arm, wrist_cam, planner, brief, seconds, poses, plan, path, obj, log)
        if best is None or result["review"]["score"] > best["review"]["score"]:
            best = result
        if result["review"]["keep"] or result["review"]["score"] >= GOOD_ENOUGH:
            break
        notes = (f"Intent: {plan['intent']}\nWaypoints: {json.dumps(plan['waypoints'])}\n"
                 f"Review: {result['review']['score']}/10. {result['review']['problems']} Next: {result['review']['next_time']}")
    usage = planner.usage
    cost = usage["input"] * 4e-6 + usage["output"] * 20e-6
    log(f"[planner] best take {best['path']} scored {best['review']['score']}; {usage['calls']} calls, "
        f"{usage['seconds']:.1f}s in calls, ~${cost:.3f}")
    return best


def play(arm, wrist_cam, planner, brief, seconds, poses, plan, out_path, obj, log) -> dict:
    """Play a plan on one curve with live centering, then crop and review it."""
    from robot.centering import Centerer, ObjectTracker

    log(f"[planner] intent: {plan['intent']}")
    for i, w in enumerate(plan["waypoints"]):
        log(f"[planner]  {i}: {w['pose']}{'->' + w['toward'] if w['toward'] else ''} {w['seconds']:.1f}s - {w['why']}")
    waypoints = [w for w in plan["waypoints"] if w["pose"] in poses]
    start = resolve(waypoints[0], poses)
    arm.move_to(list(start), 2.0)
    centerer = Centerer(ObjectTracker(wrist_cam, obj), target=AIM, max_flex_from_calib_deg=40.0, max_step_deg=2.0)
    # Settle the aim on the start pose before rolling; the first take opened on empty table.
    settle_until = time.monotonic() + 4.0
    while time.monotonic() < settle_until:
        arm.send_joints(list(centerer.apply(start)))
        time.sleep(1 / HZ)
    raw_path = str(Path(out_path).with_suffix(".raw.mp4"))
    wrist_cam.start_recording(raw_path)
    frames, next_tick = [], time.monotonic()
    previous = current = start
    try:
        for i in range(len(waypoints) - 1):
            target = resolve(waypoints[i + 1], poses)
            ahead = resolve(waypoints[i + 2], poses) if i + 2 < len(waypoints) else target
            seg = float(np.clip(waypoints[i + 1]["seconds"], *QUICK_SEGMENT_S))
            for k, row in enumerate(segment(previous, current, target, ahead, seg)):
                arm.send_joints(list(centerer.apply(row)))
                if k % 10 == 0:
                    frames.append(crop(wrist_cam.latest()))
                next_tick += 1 / HZ
                time.sleep(max(0.0, next_tick - time.monotonic()))
            previous, current = current, target
        time.sleep(1.0)  # hold the final frame
    finally:
        wrist_cam.stop_recording()
    crop_video(raw_path, out_path)
    log(f"[planner] tracked the {obj} {centerer.seen}/{centerer.ticks} ticks; final aim offset "
        f"pan {centerer.offset[0]:+.1f} tilt {centerer.offset[1]:+.1f}")
    review = planner.review(brief, plan, frames[:: max(1, len(frames) // 6)][:6])
    log(f"[planner] review: keep={review['keep']} score={review['score']} problems: {review['problems']} | next: {review['next_time']}")
    return {"plan": plan, "review": review, "path": out_path}


def shoot(arm, wrist_cam, side_cam, brief: str, seconds: float, out_path: str, takes: int = 2, log=print) -> dict:
    """Survey once, then plan -> play -> review, reshooting with the review as notes."""
    planner = Planner()
    views, poses = {}, {}
    arm.move_to("home", 2.0)
    for label, q in survey_poses().items():
        aimed, frame = aim(arm, wrist_cam, q, "duck" if "duck" in brief.lower() else "product", log)
        if aimed is None:
            log(f"[planner] survey {label}: product not centrable from here, left out")
            continue
        poses[label] = aimed
        views[label] = crop(frame)
        log(f"[planner] survey {label}: aimed at {np.round(aimed[:5]).astype(int).tolist()}")
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
    raw_path = str(Path(out_path).with_suffix(".raw.mp4"))
    wrist_cam.start_recording(raw_path)
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
                    frames.append(crop(wrist_cam.latest()))
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
    crop_video(raw_path, out_path)
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
    parser.add_argument("--survey", action="store_true", help="aimed survey of every pose (slow) instead of one overview")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "server" / "media" / "claude_plan.mp4"))
    args = parser.parse_args()
    wrist = Camera(int(os.getenv("WRIST_CAM_INDEX", "0")), "wrist")
    side = Camera(int(os.getenv("SIDE_CAM_INDEX", "1")), "side")
    wrist.open()
    side.open()
    arm = Arm(side, wrist)
    arm.connect()
    try:
        run = shoot if args.survey else quick_shoot
        run(arm, wrist, side, args.brief, args.seconds, args.out, takes=args.takes)
    finally:
        arm.move_to("home", 2.5)
        wrist.close()
        side.close()
    print(f"clips: {Path(args.out).parent}/{Path(args.out).stem}_take*.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
