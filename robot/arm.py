"""Arm interface used by the backend.

Ship MockArm on day 1 so the server never waits on the real arm.
Robotics replaces the NotImplemented methods on Arm; keep the method names.
"""

import time

from robot.poses import SCOUT_POSES, clamp_joints, resolve_pose
from shared.constants import SHOTS

__all__ = ["Arm", "MockArm", "SCOUT_POSES"]


class Arm:
    def connect(self) -> None:
        """Open the follower on ARM_PORT and load the LeRobot calibration."""
        raise NotImplementedError

    def get_state(self) -> list[float]:
        """6 values: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg), gripper (0-100)."""
        raise NotImplementedError

    def move_to(self, pose: str | list[float], duration_s: float) -> None:
        """Blocking eased move. Real arm: 30 Hz smoothstep, clamped to JOINT_LIMITS."""
        raise NotImplementedError

    def run_scripted(self, move: str, duration_s: float) -> None:
        """One of shared.constants.SCRIPTED_MOVES: "zoom", "pan_out", "overhead"."""
        raise NotImplementedError

    def run_vla(self, instruction: str, duration_s: float) -> None:
        """Run the fine-tuned policy on one task sentence (SHOTS[task]["vla_prompt"]). See robot/vla_client.py."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class MockArm(Arm):
    """Sleeps a short stand-in delay and prints. Same methods as Arm."""

    def __init__(self) -> None:
        self._state = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
        self.connected = False

    def connect(self) -> None:
        self.connected = True
        print("[MockArm] connect")

    def get_state(self) -> list[float]:
        return list(self._state)

    def move_to(self, pose: str | list[float], duration_s: float) -> None:
        # Named poses may still be None before calibration. Accept the name anyway.
        if not isinstance(pose, str):
            self._state = clamp_joints(resolve_pose(pose))
        print(f"[MockArm] move_to {pose!r} over {duration_s}s")
        time.sleep(min(duration_s, 0.25))

    def run_scripted(self, move: str, duration_s: float) -> None:
        print(f"[MockArm] run_scripted {move!r} for {duration_s}s")
        time.sleep(min(duration_s, 0.25))

    def run_vla(self, instruction: str, duration_s: float) -> None:
        print(f"[MockArm] run_vla {instruction!r} for {duration_s}s")
        time.sleep(min(duration_s, 0.25))

    def stop(self) -> None:
        print("[MockArm] stop")


if __name__ == "__main__":
    arm = MockArm()
    arm.connect()
    arm.move_to("scout_0", 1.0)
    arm.run_scripted("zoom", 2.0)
    arm.run_vla(SHOTS["zoom"]["vla_prompt"], 3.0)
    arm.stop()
    print(arm.get_state())
