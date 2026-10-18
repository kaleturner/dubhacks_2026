"""Claude flies the camera: one keyframe at a time, with the arm never waiting on it.

  python -m robot.claude_director --brief "slow dramatic reveal of the duck" --seconds 15

Every keyframe, Claude sees the current wrist and side frames and answers with a
small, bounded joint step and how long to take over it. While the arm eases to
that keyframe, the request for the next one is already in flight, so a 1-3 s
model call sets how often Claude can change its mind, not how smooth the motion
is. Steps are clamped per keyframe and to the joint envelope the recorded demos
covered; Arm.send_joints still lifts anything near the table.

Needs ANTHROPIC_API_KEY in .env, plus ANTHROPIC_WORKSPACE_ID when the key is not
scoped to a workspace. CLAUDE_DIRECTOR_MODEL overrides the model.
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

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = os.getenv("CLAUDE_DIRECTOR_MODEL", "claude-opus-5-5")
# Joint range of the 80 recorded takes (0.5-99.5 percentile), pan widened by 10 degrees.
ENVELOPE = np.array([[-36, 71], [-105, 24], [-9, 97], [-2, 103], [-169, 166]], dtype=float)
MAX_STEP = np.array([15, 12, 12, 12, 30], dtype=float)  # degrees per keyframe
SECONDS = (2.5, 4.0)  # a call takes ~3.3 s; keyframes at least that long keep the arm moving

KEYFRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "pan": {"type": "number", "description": "shoulder_pan change, degrees. + turns the arm to its left."},
        "lift": {"type": "number", "description": "shoulder_lift change, degrees. + leans the arm forward and down."},
        "elbow": {"type": "number", "description": "elbow_flex change, degrees. - straightens, reaching out."},
        "tilt": {"type": "number", "description": "wrist_flex change, degrees. + tilts the camera down."},
        "roll": {"type": "number", "description": "wrist_roll change, degrees. Rotates the image."},
        "seconds": {"type": "number", "description": "how long to take, 2.5 to 4.0"},
        "note": {"type": "string", "description": "a few words on what this keyframe is for"},
        "done": {"type": "boolean", "description": "true when the shot is complete"},
    },
    "required": ["pan", "lift", "elbow", "tilt", "roll", "seconds", "note", "done"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are the camera operator for a small robot arm (SO-101) that films products with a camera "
    "on its wrist. You direct one short keyframe at a time and see the result before the next one.\n"
    "Each turn you get the wrist camera frame (what is being filmed), a side view of the arm, the "
    "current joint angles in degrees, and the time left. Answer with joint changes for the next "
    "keyframe. Per keyframe, at most 15 degrees on pan, 12 on lift, elbow and tilt, 30 on roll; larger "
    "values are clamped. Each keyframe takes 2.5 to 4 seconds. Use the whole range: a shot should "
    "visibly travel (an arc around the product, a real push-in, a rise), moving steadily in one "
    "direction across keyframes. Tiny nudges look static; jerky reversals look amateur.\n"
    "Keep the product in frame and well composed. If it drifts toward an edge of the wrist frame, "
    "correct with pan (left/right) and tilt (up/down) before anything else. Never push the camera "
    "into the product or the table: keep it at least a hand's width away. Set done when the brief "
    "is achieved or time runs out."
)


def _jpeg(frame: np.ndarray, width: int) -> str:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.standard_b64encode(buf.tobytes()).decode()


class Director:
    def __init__(self, brief: str, seconds: float) -> None:
        # A key that is not scoped to a workspace must name one on every request.
        workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": workspace} if workspace else None
        self.client = anthropic.Anthropic(default_headers=headers)
        self.brief = brief
        self.seconds = seconds
        self.history: list[str] = []
        self.calls: list[dict] = []

    def next_keyframe(self, wrist: np.ndarray, side: np.ndarray, joints, time_left: float) -> dict:
        started = time.monotonic()
        text = (
            f"Brief: {self.brief}\n"
            f"Joints now [pan, lift, elbow, tilt, roll]: {[round(float(v)) for v in joints[:5]]}\n"
            f"Time left: {time_left:.1f} s of {self.seconds:.0f} s\n"
            f"Your last keyframes: {'; '.join(self.history[-4:]) or 'none yet'}"
        )
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": KEYFRAME_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Wrist camera:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg(wrist, 640)}},
                    {"type": "text", "text": "Side view:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg(side, 320)}},
                    {"type": "text", "text": text},
                ],
            }],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined: {response.stop_details}")
        frame = json.loads(next(b.text for b in response.content if b.type == "text"))
        self.history.append(frame["note"])
        self.calls.append({
            "latency_s": round(time.monotonic() - started, 2),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            **frame,
        })
        return frame


def apply_keyframe(joints, frame: dict) -> tuple[np.ndarray, float]:
    step = np.array([frame["pan"], frame["lift"], frame["elbow"], frame["tilt"], frame["roll"]], dtype=float)
    step = np.clip(step, -MAX_STEP, MAX_STEP)
    target = np.array(joints, dtype=float)
    target[:5] = np.clip(target[:5] + step, ENVELOPE[:, 0], ENVELOPE[:, 1])
    return target, float(np.clip(frame["seconds"], *SECONDS))


def fly(arm, wrist_cam, side_cam, brief: str, seconds: float) -> Director:
    """Blocking. Ease through Claude's keyframes for `seconds`, requesting the next one while moving."""
    director = Director(brief, seconds)
    start = time.monotonic()
    target = np.array(arm.get_state(), dtype=float)

    def request(planned: np.ndarray, out: dict) -> None:
        try:
            out["frame"] = director.next_keyframe(
                wrist_cam.latest(), side_cam.latest(), planned, seconds - (time.monotonic() - start)
            )
        except Exception as exc:  # a failed call means the arm holds and we ask again
            out["error"] = exc

    pending: dict = {}
    worker = threading.Thread(target=request, args=(target, pending), daemon=True)
    worker.start()
    while time.monotonic() - start < seconds:
        worker.join(timeout=max(0.0, seconds - (time.monotonic() - start)))
        if worker.is_alive():
            break
        if "error" in pending:
            print(f"[director] call failed, holding: {pending['error']}")
            frame = None
        else:
            frame = pending["frame"]
        if frame is not None and frame["done"]:
            print(f"[director] done: {frame['note']}")
            break
        if frame is not None:
            target, dur = apply_keyframe(target, frame)
            print(f"[director] {frame['note']} -> {np.round(target[:5]).astype(int).tolist()} over {dur:.1f}s "
                  f"(call {director.calls[-1]['latency_s']} s)")
        else:
            dur = 1.0
        # Ask for the next keyframe from where this one will end, then move there.
        pending = {}
        worker = threading.Thread(target=request, args=(target.copy(), pending), daemon=True)
        worker.start()
        arm.move_to(list(target), dur)
    return director


def main() -> int:
    from robot.arm import Arm
    from robot.cameras import Camera

    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", default="slow, dramatic reveal of the duck: start wide, drift around it, end on a close hero shot")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "server" / "media" / "claude_shot.mp4"))
    args = parser.parse_args()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in .env")

    wrist = Camera(int(os.getenv("WRIST_CAM_INDEX", "0")), "wrist")
    side = Camera(int(os.getenv("SIDE_CAM_INDEX", "1")), "side")
    wrist.open()
    side.open()
    arm = Arm(side, wrist)
    arm.connect()
    try:
        arm.move_to("scout_1", 2.5)
        wrist.start_recording(args.out)
        director = fly(arm, wrist, side, args.brief, args.seconds)
        wrist.stop_recording()
    finally:
        arm.move_to("home", 2.5)
        wrist.close()
        side.close()
    calls = director.calls
    if calls:
        lat = [c["latency_s"] for c in calls]
        tokens_in = sum(c["input_tokens"] for c in calls)
        tokens_out = sum(c["output_tokens"] for c in calls)
        print(f"{len(calls)} keyframes, call latency median {np.median(lat):.2f} s, max {max(lat):.2f} s; "
              f"{tokens_in} input + {tokens_out} output tokens")
    print(f"clip: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
