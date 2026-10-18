import numpy as np
import pytest

from robot import arm as arm_mod
from robot.arm import Arm, smoothstep
from robot.poses import POSES, load_replay
from shared.constants import GRIPPER_HOLD, JOINT_NAMES


class FakeRobot:
    """Stands in for SO101Follower: remembers commands, reports the last one as the state."""

    def __init__(self, start=(0, 0, 0, 0, 0, 50)):
        self.state = list(map(float, start))
        self.sent = []
        self.connected_with = None

    def connect(self, calibrate=True):
        self.connected_with = calibrate

    def get_observation(self):
        return {f"{n}.pos": v for n, v in zip(JOINT_NAMES, self.state)}

    def send_action(self, action):
        row = [action[f"{n}.pos"] for n in JOINT_NAMES]
        self.sent.append(row)
        self.state = row


def make_arm(**kw):
    robot = FakeRobot(**kw)
    arm = Arm(robot=robot, hz=30)
    arm._sleep = lambda s: None
    arm.dry_run = False
    arm.connect()
    return arm, robot


def test_connect_never_starts_calibration():
    _, robot = make_arm()
    assert robot.connected_with is False


def test_smoothstep_endpoints():
    assert smoothstep(0) == 0 and smoothstep(1) == 1 and smoothstep(0.5) == 0.5


def test_move_to_reaches_target_with_eased_steps(monkeypatch):
    monkeypatch.setitem(POSES, "pose_a", [10, -20, 30, 5, 0, 60])
    arm, robot = make_arm()
    arm.move_to("pose_a", 1.0)
    assert len(robot.sent) == 30
    assert robot.sent[-1][:5] == pytest.approx([10, -20, 30, 5, 0])
    first, second = np.array(robot.sent[0][:5]), np.array(robot.sent[1][:5])
    assert np.abs(second - first).max() < np.abs(np.array(robot.sent[15][:5]) - np.array(robot.sent[14][:5])).max()


def test_commands_are_clamped_and_gripper_is_forced(monkeypatch):
    monkeypatch.setitem(POSES, "far", [500, 0, 0, 0, 0, 5])
    arm, robot = make_arm()
    arm.move_to("far", 0.1)
    assert max(r[0] for r in robot.sent) <= 180
    if GRIPPER_HOLD is not None:
        assert all(r[5] == GRIPPER_HOLD for r in robot.sent)


def test_unsaved_pose_names_the_fix():
    arm, _ = make_arm()
    with pytest.raises(KeyError, match="record_poses"):
        arm.move_to("scout_9", 1.0)


def test_replay_starts_at_first_frame_and_fits_duration():
    replay = load_replay("zoom")
    arm, robot = make_arm(start=replay["actions"][0])
    arm.run_scripted("zoom", 4.0)
    assert len(robot.sent) == 4 * 30
    assert robot.sent[0][:5] == pytest.approx(replay["actions"][0][:5], abs=0.5)
    assert robot.sent[-1][:5] == pytest.approx(replay["actions"][-1][:5], abs=0.5)


def test_replay_eases_in_from_far_away():
    replay = load_replay("zoom")
    arm, robot = make_arm(start=[0, 0, 0, 0, 0, 50])
    arm.run_scripted("zoom", 15.0)
    assert len(robot.sent) > 15 * 30  # ease-in rows plus the replay itself
    steps = np.abs(np.diff(np.array(robot.sent)[:, :5], axis=0))
    assert steps.max() < 8  # never a jump bigger than ARM_MAX_STEP-ish


def test_stop_ends_a_replay_early():
    arm, robot = make_arm(start=load_replay("zoom")["actions"][0])
    count = {"n": 0}
    real_send = robot.send_action

    def send(action):
        count["n"] += 1
        if count["n"] == 10:
            arm.stop()
        real_send(action)

    robot.send_action = send
    arm.run_scripted("zoom", 4.0)
    assert len(robot.sent) == 10


def test_dry_run_sends_nothing():
    arm, robot = make_arm(start=load_replay("zoom")["actions"][0])
    arm.dry_run = True
    arm.run_scripted("zoom", 1.0)
    assert robot.sent == []
    assert arm._last_sent is not None


def test_missing_replay_names_the_fix():
    arm, _ = make_arm()
    with pytest.raises(FileNotFoundError, match="make_replays"):
        arm.run_scripted("no_such_shot", 2.0)


def test_vla_needs_cameras():
    arm, _ = make_arm()
    with pytest.raises(RuntimeError, match="cameras"):
        arm.run_vla("x", 1.0)


def test_connect_without_env_explains(monkeypatch):
    monkeypatch.delenv("ARM_PORT", raising=False)
    monkeypatch.delenv("ARM_ID", raising=False)
    with pytest.raises(RuntimeError, match="ARM_PORT"):
        arm_mod.Arm().connect()


def test_replay_picks_take_nearest_the_object(tmp_path, monkeypatch):
    import json

    from robot import poses

    takes = [
        {"episode": 3, "object": "duck", "side_xy": [100, 300], "actions": [[1, 0, 0, 0, 0, 97]] * 2},
        {"episode": 9, "object": "duck", "side_xy": [400, 250], "actions": [[9, 0, 0, 0, 0, 97]] * 2},
    ]
    (tmp_path / "zoom_all.json").write_text(json.dumps({"task": "zoom", "takes": takes}))
    (tmp_path / "zoom.json").write_text(json.dumps({"task": "zoom", "actions": [[5, 0, 0, 0, 0, 97]] * 2}))
    monkeypatch.setattr(poses, "REPLAY_DIR", tmp_path)
    assert poses.load_replay("zoom", near=[380, 260])["episode"] == 9
    assert poses.load_replay("zoom", near=[90, 310])["episode"] == 3
    assert poses.load_replay("zoom")["actions"][0][0] == 5


def test_commands_are_kept_off_the_table():
    from robot.safety import FLOOR_Z, lowest_z

    arm, robot = make_arm()
    # Rest pose with the shoulder pushed down past where the tip meets the table.
    low = [-4.0, -70.0, 96.0, 60.0, -92.0, 50.0]
    assert lowest_z(low) < FLOOR_Z
    arm.send_joints(low)
    assert lowest_z(robot.sent[-1]) >= FLOOR_Z
    assert arm.floor_lifts == 1
