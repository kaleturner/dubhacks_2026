"""Named poses, joint clamps, and recorded replays.

Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg), gripper (0-100).

  robot/poses.json          hand-saved poses (python -m robot.record_poses)
  robot/replays/<task>.json one recorded demo per task (python -m robot.make_replays)

A shot's start pose "<task>_start" is the first row of that task's replay, so it
is never captured by hand.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POSES_FILE = HERE / "poses.json"
REPLAY_DIR = HERE / "replays"

# (min, max) per joint. Replace with the calibrated range before driving the real arm.
JOINT_LIMITS: list[tuple[float, float]] = [
    (-180.0, 180.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
    (0.0, 100.0),
]

# None means "not captured yet". poses.json fills these in.
POSES: dict[str, list[float] | None] = {
    "home": None,
    "scout_0": None,
    "scout_1": None,
    "scout_2": None,
    "scout_3": None,
    "scout_4": None,
}

SCOUT_POSES = ["scout_0", "scout_1", "scout_2", "scout_3", "scout_4"]


def _load_saved() -> None:
    if POSES_FILE.is_file():
        for name, joints in json.loads(POSES_FILE.read_text()).items():
            POSES[name] = [float(v) for v in joints]


def save_pose(name: str, joints: list[float]) -> None:
    saved = json.loads(POSES_FILE.read_text()) if POSES_FILE.is_file() else {}
    saved[name] = [round(float(v), 3) for v in joints]
    POSES_FILE.write_text(json.dumps(saved, indent=2) + "\n")
    POSES[name] = list(saved[name])


def load_replay(task: str) -> dict:
    path = REPLAY_DIR / f"{task}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no replay for {task!r}. Run: python -m robot.make_replays <dataset folder>"
        )
    return json.loads(path.read_text())


def resolve_pose(pose: str | list[float]) -> list[float]:
    if isinstance(pose, str):
        joints = POSES.get(pose)
        if joints is None and pose.endswith("_start"):
            joints = load_replay(pose[: -len("_start")])["actions"][0]
        if joints is None:
            raise KeyError(f"pose {pose!r} is not saved yet. Run: python -m robot.record_poses")
        return list(joints)
    if len(pose) != 6:
        raise ValueError(f"expected 6 joints, got {len(pose)}")
    return list(pose)


def clamp_joints(joints: list[float]) -> list[float]:
    clamped = []
    for value, (low, high) in zip(joints, JOINT_LIMITS, strict=True):
        clamped.append(min(high, max(low, value)))
    return clamped


_load_saved()
