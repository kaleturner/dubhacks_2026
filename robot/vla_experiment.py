"""Run the deployed policy on any instruction and record what it does.

  python -m robot.vla_experiment "spin the camera above the object" --start sweep --seconds 12
  python -m robot.vla_experiment "orbit around the object" --start home

One instruction per run: the arm eases to a start pose, the policy drives it for
--seconds while the wrist camera records, then the arm goes home. Prints how far
each joint travelled, so runs on different objects and wordings can be compared.
For instructions the policy never saw, commands are also clamped to the joint
range the recorded demos covered, on top of the table guard every command passes.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from robot.arm import Arm
from robot.cameras import Camera
from robot.claude_director import ENVELOPE
from robot.poses import load_replay, resolve_pose
from robot.vla_client import close_warm_sessions, run_task

OUT_DIR = Path(__file__).resolve().parents[1] / "server" / "media" / "experiments"
NAMES = ("pan", "lift", "elbow", "tilt", "roll")


class EnvelopeArm(Arm):
    """Arm that keeps every command inside the recorded joint envelope and logs it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log: list[list[float]] = []
        self.clamped = 0

    def send_joints(self, joints):
        q = np.array(joints, dtype=float)
        inside = np.clip(q[:5], ENVELOPE[:, 0], ENVELOPE[:, 1])
        if not np.allclose(inside, q[:5]):
            self.clamped += 1
        q[:5] = inside
        out = super().send_joints(list(q))
        self.log.append(out)
        return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument("--start", default="home", help="home, or zoom / overhead / sweep for that shot's start pose")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in args.instruction.lower())[:40].strip("_")
    clip = OUT_DIR / f"{time.strftime('%H%M%S')}_{slug}{'_' + args.tag if args.tag else ''}.mp4"
    start = resolve_pose("home") if args.start == "home" else load_replay(args.start)["actions"][0]

    side = Camera(int(os.getenv("SIDE_CAM_INDEX", "1")), "side")
    wrist = Camera(int(os.getenv("WRIST_CAM_INDEX", "0")), "wrist")
    side.open()
    wrist.open()
    arm = EnvelopeArm(side, wrist)
    arm.connect()
    try:
        arm.move_to(list(start), 2.5)
        wrist.start_recording(str(clip))
        began = time.monotonic()
        run_task(args.instruction, args.seconds, arm=arm, side=side, wrist=wrist, stop=arm._stop)
        wrist.stop_recording()
        took = time.monotonic() - began
    finally:
        arm.move_to("home", 2.5)
        close_warm_sessions()
        wrist.close()
        side.close()

    path = np.array(arm.log)[:, :5]
    travel = path.max(axis=0) - path.min(axis=0)
    print(f"instruction: {args.instruction!r} from {args.start}, {took:.1f} s")
    print("joint travel (deg): " + ", ".join(f"{n} {t:.0f}" for n, t in zip(NAMES, travel)))
    print(f"end pose: {np.round(path[-1]).astype(int).tolist()}; envelope clamps: {arm.clamped}; "
          f"table lifts: {arm.floor_lifts}")
    print(f"clip: {clip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
