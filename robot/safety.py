"""Keep the gripper off the table.

Forward kinematics from the SO-101 URDF (robot/so101.urdf, TheRobotStudio's
so101_new_calib, whose joint zeros match LeRobot's degree calibration). Every
command passes through `above_floor` in Arm.send_joints; a command that would put
the gripper tip or the wrist below FLOOR_Z is lifted with shoulder_lift (then
wrist_flex) until it clears.

FLOOR_Z is in the URDF base frame. Over all 80 recorded takes the tip went no
lower than -0.022 (the model reads a couple of cm low against the real table),
and the demos never touched it, so the floor sits at that demonstrated minimum:
it only blocks what no demo ever did, such as centering tilting the wrist down.
"""

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parent / "so101.urdf"
TABLE = Path(__file__).resolve().parent / "table.json"  # written by python -m robot.measure_table
FLOOR_Z = json.loads(TABLE.read_text())["floor_z"] if TABLE.is_file() else -0.02
CHAIN = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LIFT, FLEX = 1, 3
STEP_DEG = 0.5
MAX_FIX_DEG = 40.0


def _origin(xyz: list[float], rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    m[:3, 3] = xyz
    return m


@lru_cache(maxsize=1)
def _frames() -> tuple[list[np.ndarray], np.ndarray]:
    joints = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}

    def origin(name: str) -> np.ndarray:
        o = joints[name].find("origin")
        return _origin([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()])

    return [origin(name) for name in CHAIN], origin("gripper_frame_joint")


def _rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def lowest_z(joints: list[float]) -> float:
    """Height of the lower of the wrist and the gripper tip, in the URDF base frame (m)."""
    origins, tip = _frames()
    m = np.eye(4)
    wrist_z = None
    for i, origin in enumerate(origins):
        m = m @ origin
        if i == FLEX + 1:  # wrist_roll's origin sits on the wrist, where the camera is mounted
            wrist_z = m[2, 3]
        m = m @ _rz(joints[i])
    return float(min(wrist_z, (m @ tip)[2, 3]))


def above_floor(joints: list[float], floor: float = FLOOR_Z) -> tuple[list[float], bool]:
    """The command, raised just enough to clear the floor. Returns (joints, changed)."""
    if lowest_z(joints) >= floor:
        return joints, False
    out = list(joints)
    for index in (LIFT, FLEX):
        start = out[index]
        up = 1.0 if lowest_z(_with(out, index, start + 1.0)) > lowest_z(_with(out, index, start - 1.0)) else -1.0
        moved = 0.0
        while lowest_z(out) < floor and moved < MAX_FIX_DEG:
            moved += STEP_DEG
            out[index] = start + up * moved
        if lowest_z(out) >= floor:
            break
    return out, True


def _with(joints: list[float], index: int, value: float) -> list[float]:
    out = list(joints)
    out[index] = value
    return out
