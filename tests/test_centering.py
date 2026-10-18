import numpy as np

from robot import centering
from robot.centering import FLEX, PAN, ROLL, Centerer


class FakeTracker:
    """Places the object where the current offsets would put it, using a known Jacobian."""

    def __init__(self, start, jacobian):
        self.start = np.array(start, dtype=float)
        self.jacobian = np.array(jacobian, dtype=float)
        self.centerer = None
        self.visible = True

    def poll(self):
        if not self.visible:
            return None
        cx, cy = self.start + self.jacobian @ self.centerer.offset
        return cx, cy, 0.2


def make(start, visible=True):
    jac = [[-0.0128, -0.002], [0.0094, -0.0174]]  # measured on the arm, see robot/centering.json
    tracker = FakeTracker(start, jac)
    tracker.visible = visible
    centerer = Centerer(tracker)
    centerer.jacobian = np.array(jac)
    centerer.inverse = np.linalg.pinv(centerer.jacobian)
    tracker.centerer = centerer
    return centerer


def at_calib(centerer):
    row = np.zeros(6)
    row[FLEX] = centerer.calib_flex
    return row


def test_offsets_bring_object_to_center():
    centerer = make((0.8, 0.25))
    row = at_calib(centerer)
    for _ in range(150):
        centerer.apply(row)
    cx, cy, _ = centerer.tracker.poll()
    assert abs(cx - 0.5) < 0.08 and abs(cy - 0.5) < 0.08
    assert np.abs(centerer.offset).max() <= 25.0


def test_offsets_apply_only_to_pan_and_flex():
    centerer = make((0.8, 0.25))
    row = at_calib(centerer)
    for _ in range(20):
        out = centerer.apply(row)
    assert out[PAN] != 0 and out[FLEX] != row[FLEX]
    assert out[1] == out[2] == out[ROLL] == out[5] == 0


def test_no_object_no_correction():
    centerer = make((0.9, 0.9), visible=False)
    for _ in range(30):
        out = centerer.apply(np.zeros(6))
    assert np.allclose(out, 0)


def test_rolling_freezes_offsets():
    centerer = make((0.8, 0.25))
    centerer.apply(at_calib(centerer))
    frozen = centerer.offset.copy()
    rolled = at_calib(centerer)
    rolled[ROLL] = 90
    for _ in range(20):
        centerer.apply(rolled)
    assert np.allclose(centerer.offset, frozen)


def test_looking_straight_down_freezes_offsets():
    centerer = make((0.8, 0.25))
    centerer.apply(at_calib(centerer))
    frozen = centerer.offset.copy()
    down = at_calib(centerer)
    down[FLEX] += 35  # overhead and sweep poses
    for _ in range(20):
        centerer.apply(down)
    assert np.allclose(centerer.offset, frozen)


def test_live_state_reports_offset():
    centerer = make((0.8, 0.25))
    for _ in range(5):
        centerer.apply(at_calib(centerer))
    assert centering.LIVE["offset"] == [round(float(v), 1) for v in centerer.offset]
