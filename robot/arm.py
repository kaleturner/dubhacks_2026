"""Arm interface used by the backend.

MockArm sleeps and prints, so the server never waits on the real arm. Arm drives
the SO-101 follower through LeRobot. The method names are the contract in
CONTRACTS.md and PLAN.md section 2c; send_joints and the torque helpers are
extras for the VLA client and the pose recorder.

  python -m robot.arm                          MockArm smoke test
  python -m robot.arm --real                   connect, print the joint state, no motion
  python -m robot.arm --real --replay zoom     play one recorded demo (asks first)
"""

import os
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from robot.poses import SCOUT_POSES, clamp_joints, load_replay, resolve_pose
from shared.constants import GRIPPER_HOLD, JOINT_NAMES, SHOTS

__all__ = ["Arm", "MockArm", "SCOUT_POSES"]

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

HZ = 30
EASE_IN_DEG_PER_S = 30.0  # slowest sensible speed to reach a replay's first frame


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip() == "1"


class Arm:
    """SO-101 follower. Cameras are owned by the server; they are only handed in for the VLA."""

    def __init__(self, side=None, wrist=None, robot=None, *, hz: int = HZ) -> None:
        self.side = side
        self.wrist = wrist
        self._robot = robot
        self.hz = hz
        self.dry_run = _flag("VLA_DRY_RUN")
        self._stop = threading.Event()
        self._sleep = time.sleep
        self._last_sent: list[float] | None = None
        # Object name to keep centered in the wrist view during replays (robot/centering.py).
        self.center_on: str | None = None
        # Where locate found the object in the side view; replays pick the take recorded nearest to it.
        self.target_side_xy: list[float] | None = None

    # -- contract methods -------------------------------------------------

    def connect(self) -> None:
        """Open the follower on ARM_PORT with the calibration named by ARM_ID."""
        if self._robot is None:
            port = os.getenv("ARM_PORT", "").strip()
            arm_id = os.getenv("ARM_ID", "").strip()
            if not port or not arm_id:
                raise RuntimeError("set ARM_PORT (lerobot-find-port) and ARM_ID (the calibration id) in .env")
            try:
                from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            except ImportError:
                try:
                    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
                except ImportError as exc:
                    raise RuntimeError('lerobot is not installed. Run: pip install "lerobot[feetech]"') from exc
            step = float(os.getenv("ARM_MAX_STEP", "8"))
            config = SO101FollowerConfig(port=port, id=arm_id, use_degrees=True, max_relative_target=step)
            self._robot = SO101Follower(config)
        # calibrate=False: fail loudly instead of starting an interactive calibration.
        self._robot.connect(calibrate=False)
        print(f"[Arm] connected, dry_run={self.dry_run}")

    def get_state(self) -> list[float]:
        """6 values: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg), gripper (0-100)."""
        obs = self._robot.get_observation()
        return [float(obs[f"{name}.pos"]) for name in JOINT_NAMES]

    def move_to(self, pose: str | list[float], duration_s: float) -> None:
        """Blocking eased move. 30 Hz smoothstep, clamped to JOINT_LIMITS."""
        self._stop.clear()
        target = np.array(clamp_joints(resolve_pose(pose)), dtype=float)
        self._ease_to(target, duration_s)

    def run_scripted(self, move: str, duration_s: float) -> None:
        """Replay the recorded demo for a task ("zoom", "overhead", "sweep"), stretched to duration_s."""
        self._stop.clear()
        replay = load_replay(move, near=self.target_side_xy)
        if "object_side_xy" in replay:
            print(f"[Arm] {move}: take {replay['episode']}, its object was at {replay['object_side_xy']} vs now {self.target_side_xy}")
        rows = np.array(replay["actions"], dtype=float)
        centerer = self._centerer()  # starts finding the object during the ease-in
        first = rows[0]
        gap = float(np.abs(np.array(self.get_state()) - first)[:5].max())
        if gap > 3.0:
            self._ease_to(np.array(clamp_joints(list(first))), max(2.0, gap / EASE_IN_DEG_PER_S))
            if self._stop.is_set():
                return
        n = max(2, round(duration_s * self.hz))
        src = np.linspace(0.0, 1.0, len(rows))
        dst = np.linspace(0.0, 1.0, n)
        scaled = np.stack([np.interp(dst, src, rows[:, j]) for j in range(6)], axis=1)
        speed = (len(rows) / self.hz) / max(duration_s, 1e-6)
        if speed > 2.0 or speed < 0.5:
            print(f"[Arm] warning: {move} replay runs at {speed:.1f}x the recorded speed")
        self._play(scaled, centerer.apply if centerer else None)
        if centerer:
            from robot import centering

            centering._publish(object=None, state="searching", box=None, score=None)
            print(
                f"[Arm] {move}: {self.center_on} tracked {centerer.seen}/{centerer.ticks} ticks, "
                f"final offset pan {centerer.offset[0]:+.1f} flex {centerer.offset[1]:+.1f} deg"
            )

    def _centerer(self):
        if not self.center_on or self.wrist is None or _flag("NO_CENTERING"):
            return None
        from robot.centering import CALIBRATION, Centerer, ObjectTracker

        if not CALIBRATION.is_file():
            print("[Arm] no robot/centering.json; replaying without centering")
            return None
        centerer = Centerer(ObjectTracker(self.wrist, self.center_on))
        centerer.tracker.poll()
        return centerer

    def run_vla(self, instruction: str, duration_s: float) -> None:
        """Run the fine-tuned policy on one task sentence. See robot/vla_client.py."""
        if self.side is None or self.wrist is None:
            raise RuntimeError("Arm needs the side and wrist cameras for the VLA: Arm(side, wrist)")
        from robot.vla_client import run_task

        self._stop.clear()
        run_task(instruction, duration_s, arm=self, side=self.side, wrist=self.wrist, stop=self._stop)

    def stop(self) -> None:
        """Stop any move in progress and hold where it is. Torque stays on so the arm cannot drop."""
        self._stop.set()

    # -- helpers ----------------------------------------------------------

    def send_joints(self, joints: list[float]) -> list[float]:
        """The one place a command reaches the motors: clamp, force the gripper, then send."""
        out = clamp_joints([float(v) for v in joints])
        if GRIPPER_HOLD is not None:
            out[5] = float(GRIPPER_HOLD)
        self._last_sent = out
        if self.dry_run:
            return out
        self._robot.send_action({f"{name}.pos": out[i] for i, name in enumerate(JOINT_NAMES)})
        return out

    def hold(self) -> None:
        """Re-send the last command so the arm stays put."""
        if self._last_sent is not None and not self.dry_run:
            self._robot.send_action({f"{name}.pos": self._last_sent[i] for i, name in enumerate(JOINT_NAMES)})

    def release_torque(self) -> None:
        self._robot.bus.disable_torque()

    def hold_torque(self) -> None:
        self._robot.bus.enable_torque()

    def _ease_to(self, target: np.ndarray, duration_s: float) -> None:
        start = np.array(self.get_state(), dtype=float)
        n = max(1, round(duration_s * self.hz))
        rows = [start + (target - start) * smoothstep((i + 1) / n) for i in range(n)]
        self._play(np.array(rows))

    def _play(self, rows: np.ndarray, adjust=None) -> None:
        tick = 1.0 / self.hz
        next_t = time.monotonic()
        for row in rows:
            if self._stop.is_set():
                return
            self.send_joints(list(adjust(row) if adjust else row))
            next_t += tick
            delay = next_t - time.monotonic()
            if delay > 0:
                self._sleep(delay)


class MockArm(Arm):
    """Sleeps a short stand-in delay and prints. Same methods as Arm."""

    def __init__(self, side=None, wrist=None) -> None:
        super().__init__(side, wrist)
        self._state = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
        self.connected = False
        self.sent: list[list[float]] = []

    def connect(self) -> None:
        self.connected = True
        print("[MockArm] connect")

    def get_state(self) -> list[float]:
        return list(self._state)

    def move_to(self, pose: str | list[float], duration_s: float) -> None:
        # Named poses may still be unsaved. Accept the name anyway.
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

    def send_joints(self, joints: list[float]) -> list[float]:
        out = clamp_joints([float(v) for v in joints])
        if GRIPPER_HOLD is not None:
            out[5] = float(GRIPPER_HOLD)
        self._state = out
        self.sent.append(out)
        return out

    def release_torque(self) -> None:
        print("[MockArm] release_torque")

    def hold_torque(self) -> None:
        print("[MockArm] hold_torque")

    def stop(self) -> None:
        print("[MockArm] stop")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="connect to the follower and print its state")
    parser.add_argument("--replay", metavar="TASK", help="with --real: play one recorded demo")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    if not args.real:
        arm = MockArm()
        arm.connect()
        arm.move_to("scout_0", 1.0)
        arm.run_scripted("zoom", 2.0)
        arm.run_vla(SHOTS["zoom"]["vla_prompt"], 3.0)
        arm.stop()
        print(arm.get_state())
    else:
        arm = Arm()
        arm.connect()
        print("joint state:", [round(v, 1) for v in arm.get_state()])
        if args.replay:
            answer = input(
                f"Replay {args.replay!r} over {args.duration}s? Clear the table, keep a hand on the power switch. Type yes: "
            )
            if answer.strip().lower() == "yes":
                try:
                    arm.run_scripted(args.replay, args.duration)
                except KeyboardInterrupt:
                    arm.stop()
