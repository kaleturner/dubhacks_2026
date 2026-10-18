"""Hand-guide the arm and save named poses to robot/poses.json.

  python -m robot.record_poses              home and scout_0..4
  python -m robot.record_poses scout_2      just one

Torque is switched off so the arm can be moved by hand. Hold the arm while it
goes limp. A shot's start pose comes from its replay, not from here.
"""

import sys

from robot.arm import Arm
from robot.poses import SCOUT_POSES, save_pose

DEFAULT = ["home", *SCOUT_POSES]


def main() -> int:
    names = sys.argv[1:] or DEFAULT
    arm = Arm()
    arm.connect()
    try:
        input("Support the arm, then press Enter to release the motors...")
        arm.release_torque()
        for name in names:
            input(f"Move the arm to {name!r}, then press Enter to save it...")
            state = arm.get_state()
            save_pose(name, state)
            print(f"saved {name}: {[round(v, 1) for v in state]}")
    finally:
        arm.hold_torque()
    return 0


if __name__ == "__main__":
    sys.exit(main())
