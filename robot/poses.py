"""Named poses and joint clamps. Robotics fills the vectors in after calibration.

Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg), gripper (0-100).
"""

# (min, max) per joint. Replace with the calibrated range before driving the real arm.
JOINT_LIMITS: list[tuple[float, float]] = [
    (-180.0, 180.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
    (0.0, 100.0),
]

# None means "not captured yet". Record by hand, then save the 6-vector from get_state().
POSES: dict[str, list[float] | None] = {
    "home": None,
    "scout_0": None,
    "scout_1": None,
    "scout_2": None,
    "scout_3": None,
    "scout_4": None,
    "zoom_start": None,
    "zoom_in": None,
    "pan_start": None,
    "pan_end": None,
    "overhead_start": None,
}

SCOUT_POSES = ["scout_0", "scout_1", "scout_2", "scout_3", "scout_4"]


def resolve_pose(pose: str | list[float]) -> list[float]:
    if isinstance(pose, str):
        joints = POSES.get(pose)
        if joints is None:
            raise KeyError(f"pose {pose!r} is not in robot.poses.POSES yet")
        return list(joints)
    if len(pose) != 6:
        raise ValueError(f"expected 6 joints, got {len(pose)}")
    return list(pose)


def clamp_joints(joints: list[float]) -> list[float]:
    clamped = []
    for value, (low, high) in zip(joints, JOINT_LIMITS, strict=True):
        clamped.append(min(high, max(low, value)))
    return clamped
