"""Find where the table is, in the URDF frame robot/safety.py uses.

  python -m robot.measure_table

Releases the follower's motors when you press Enter. Touch the gripper tip to the
table at a few spots (near, far, left, right) and once with the wrist as low as it
goes. The lowest point reached is the table; the floor for robot/safety.py is set
a margin above it and written to robot/table.json.
"""

import json
import time
from pathlib import Path

import numpy as np

from robot.arm import Arm
from robot.safety import lowest_z

SECONDS = 30
MARGIN = 0.015
OUT = Path(__file__).resolve().parent / "table.json"


def main() -> int:
    arm = Arm()
    arm.connect()
    input("Hold the white arm with one hand, then press Enter to switch its motors off...")
    arm.release_torque()
    print(f"Motors OFF. Touch the table with the gripper tip at 3-4 spots, then with the wrist. {SECONDS} s.")
    rows, start, shown = [], time.monotonic(), -1
    while (elapsed := time.monotonic() - start) < SECONDS:
        rows.append(arm.get_state())
        if int(elapsed) != shown and int(elapsed) % 5 == 0:
            shown = int(elapsed)
            print(f"  {SECONDS - shown} s left", flush=True)
        time.sleep(1 / 30)
    input("Time. Hold the arm where you want it to stay, then press Enter to switch the motors back on...")
    arm.hold_torque()
    heights = np.array([lowest_z(q) for q in rows])
    table = float(heights.min())
    OUT.write_text(json.dumps({"table_z": round(table, 4), "floor_z": round(table + MARGIN, 4)}, indent=2) + "\n")
    print(f"table at {table:+.3f} m in the URDF frame; floor set to {table + MARGIN:+.3f} in {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
