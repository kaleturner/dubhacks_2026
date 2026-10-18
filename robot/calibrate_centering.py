"""Measure how the wrist image moves per degree of pan and of wrist flex.

  python -m robot.calibrate_centering [--object duck]

Moves to the first scout pose that shows the object, then nudges shoulder_pan
and wrist_flex by +-4 degrees, slowly, tracking the object each time. Writes the
2x2 image Jacobian to robot/centering.json and returns to home. Keep the object
still while it runs (~30 s).
"""

import argparse
import json
import time

import numpy as np

from robot.arm import Arm
from robot.cameras import Camera
from robot.centering import CALIBRATION, FLEX, PAN, ObjectTracker
from robot.poses import SCOUT_POSES, resolve_pose

NUDGE_DEG = 4.0


def measure(tracker: ObjectTracker, settle_s: float = 0.6, samples: int = 12) -> np.ndarray | None:
    time.sleep(settle_s)
    seen = []
    for _ in range(samples * 3):
        hit = tracker.poll()
        if hit is not None:
            seen.append(hit[:2])
            if len(seen) == samples:
                break
        time.sleep(1 / 30)
    return np.mean(seen, axis=0) if len(seen) >= samples // 2 else None


def main() -> int:
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="duck")
    args = parser.parse_args()

    wrist = Camera(int(os.getenv("WRIST_CAM_INDEX", "0")), "wrist")
    wrist.open()
    arm = Arm(wrist=wrist)
    arm.connect()
    tracker = ObjectTracker(wrist, args.object)
    try:
        base = None
        for pose in SCOUT_POSES:
            arm.move_to(pose, 2.5)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and tracker.poll() is None:
                time.sleep(0.1)
            if measure(tracker) is not None:
                base = np.array(resolve_pose(pose), dtype=float)
                print(f"using {pose}")
                break
            print(f"{pose}: no {args.object} in the wrist view")
        if base is None:
            print(f"no scout pose shows the {args.object}; move it in front of the arm and rerun")
            return 1

        columns = []
        for joint in (PAN, FLEX):
            at = {}
            for sign in (+1, -1):
                target = base.copy()
                target[joint] += sign * NUDGE_DEG
                arm.move_to(list(target), 1.0)
                at[sign] = measure(tracker)
            arm.move_to(list(base), 1.0)
            if at[+1] is None or at[-1] is None:
                print(f"lost the {args.object} while nudging joint {joint}; rerun with it nearer the center")
                return 1
            columns.append((at[+1] - at[-1]) / (2 * NUDGE_DEG))
            print(f"joint {joint}: image moves {np.round(columns[-1], 4).tolist()} (frame fraction per degree)")

        jacobian = np.stack(columns, axis=1)
        if abs(np.linalg.det(jacobian)) < 1e-5:
            print("pan and flex move the image the same way; calibration is unusable")
            return 1
        CALIBRATION.write_text(
            json.dumps(
                {"jacobian": np.round(jacobian, 5).tolist(), "pose": np.round(base, 2).tolist(), "object": args.object},
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {CALIBRATION}")
        return 0
    finally:
        arm.move_to("home", 2.5)
        wrist.close()


if __name__ == "__main__":
    raise SystemExit(main())
