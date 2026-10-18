"""Keep the object centered in the wrist camera while a replay plays.

A replay is blind: it repeats the demo's joint path, so the camera only points
at the object when the object sits where it sat in that demo. This module
watches the wrist feed and nudges shoulder_pan (left/right) and wrist_flex
(up/down) so the object stays near the middle of the frame. The shot keeps its
shape; only the aim moves.

  Mk1.5 locate   finds the object once (and again whenever the track is lost),
                 0.5-2 s per call, in a background thread
  ViT tracker    follows it every frame, ~5 ms, handles the object growing
                 during a push-in (robot/models/vittrack.onnx, from opencv_zoo)
  Centerer       turns pixel error into joint offsets through the measured
                 image Jacobian in robot/centering.json

  python -m robot.calibrate_centering    measures that Jacobian on the arm
"""

import json
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "vittrack.onnx"
CALIBRATION = HERE / "centering.json"

PAN, FLEX, ROLL = 0, 3, 4
MIN_SCORE = 0.3  # ViT tracking score below this means the object is gone
MAX_COLOR_DISTANCE = 0.55  # Bhattacharyya distance from the seed's colors; the arm's shadow scores ~0.9
VERIFY_S = 1.5  # Mk1.5 re-finds the object this often and the tracker snaps to it

# What centering is doing right now, for the live dashboard. Written by the arm's
# control thread, read by the server; plain values only, replaced not mutated.
#   object   name being centered, or None when no shot is centering
#   state    "searching" (waiting on Mk1.5), "tracking", or "gone" (Mk1.5 says not in view)
#   box      [x, y, w, h] in wrist-frame pixels while tracking, else None
#   score    ViT tracking score while tracking, else None
#   offset   [pan, flex] correction in degrees currently added to the replay
#   updated  time.time() of the last write
LIVE: dict = {"object": None, "state": "searching", "box": None, "score": None, "offset": [0.0, 0.0], "updated": 0.0}


def _publish(**values) -> None:
    global LIVE
    LIVE = {**LIVE, **values, "updated": time.time()}


def _colors(frame: np.ndarray, box) -> np.ndarray | None:
    x, y, w, h = (int(round(v)) for v in box)
    height, width = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    crop = frame[y : min(height, y + h), x : min(width, x + w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


class ObjectTracker:
    """Where the object is in the latest wrist frame, or None."""

    def __init__(self, wrist, obj: str, locate=None) -> None:
        if locate is None:
            from ml.perception import locate
        self.wrist = wrist
        self.obj = obj
        self.locate = locate
        self._tracker = None
        self._lock = threading.Lock()
        self._seeding = False
        self._seed: tuple[np.ndarray, tuple[int, int, int, int]] | None = None
        self._colors: np.ndarray | None = None
        self._last_check = 0.0
        self._gone = False  # Mk1.5 looked and the object is not in view
        self._tmp = Path(tempfile.gettempdir()) / f"center_{id(self)}.jpg"

    def _new_tracker(self):
        params = cv2.TrackerVit_Params()
        params.net = str(MODEL)
        return cv2.TrackerVit_create(params)

    def _reseed(self, frame: np.ndarray) -> None:
        try:
            cv2.imwrite(str(self._tmp), frame)
            found = self.locate(str(self._tmp), self.obj)
            with self._lock:
                if found.get("found"):
                    x1, y1, x2, y2 = (int(round(v)) for v in found["bbox"])
                    self._seed = (frame, (x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
                    self._gone = False
                else:
                    self._gone = True
        except Exception as exc:  # a failed Mk1.5 call only means no correction this round
            print(f"[center] locate failed: {exc}")
        finally:
            self._seeding = False

    def _request_seed(self, frame: np.ndarray) -> None:
        if not self._seeding:
            self._seeding = True
            self._last_check = time.monotonic()
            threading.Thread(target=self._reseed, args=(frame,), daemon=True).start()

    def poll(self) -> tuple[float, float, float] | None:
        """(cx, cy, size) as fractions of the frame, or None while the object is unknown."""
        frame = self.wrist.latest()
        if frame is None:
            return None
        with self._lock:
            seed, self._seed = self._seed, None
            gone, self._gone = self._gone, False
        if seed is not None:
            tracker = self._new_tracker()
            tracker.init(seed[0], seed[1])
            self._tracker = tracker
            self._colors = _colors(seed[0], seed[1])
        elif gone:
            self._tracker = None  # Mk1.5 says it left the frame; do not chase whatever is left
            _publish(object=self.obj, state="gone", box=None, score=None)
        if self._tracker is None:
            self._request_seed(frame)
            if LIVE.get("state") != "gone":
                _publish(object=self.obj, state="searching", box=None, score=None)
            return None
        if time.monotonic() - self._last_check > VERIFY_S:
            self._request_seed(frame)  # keeps tracking meanwhile; the answer re-anchors the box
        ok, (x, y, w, h) = self._tracker.update(frame)
        colors = _colors(frame, (x, y, w, h)) if ok else None
        drifted = (
            colors is None
            or self._colors is None
            or cv2.compareHist(self._colors, colors, cv2.HISTCMP_BHATTACHARYYA) > MAX_COLOR_DISTANCE
        )
        score = float(self._tracker.getTrackingScore())
        if not ok or score < MIN_SCORE or drifted:
            self._tracker = None
            self._request_seed(frame)
            _publish(object=self.obj, state="searching", box=None, score=None)
            return None
        _publish(object=self.obj, state="tracking", box=[int(x), int(y), int(w), int(h)], score=round(score, 2))
        height, width = frame.shape[:2]
        return (x + w / 2) / width, (y + h / 2) / height, max(w / width, h / height)


class Centerer:
    """Accumulates pan/flex offsets that drive the tracked object to the frame center."""

    def __init__(
        self,
        tracker: ObjectTracker,
        *,
        gain: float = 0.12,
        max_step_deg: float = 1.5,
        max_offset_deg: float = 25.0,
        deadband: float = 0.06,
        max_roll_change_deg: float = 25.0,
    ) -> None:
        calib = json.loads(CALIBRATION.read_text())
        # Columns: image motion (fraction of frame) per degree of pan, per degree of flex.
        self.jacobian = np.array(calib["jacobian"], dtype=float)
        self.inverse = np.linalg.pinv(self.jacobian)
        self.tracker = tracker
        self.gain = gain
        self.max_step = max_step_deg
        self.max_offset = max_offset_deg
        self.deadband = deadband
        self.max_roll_change = max_roll_change_deg
        self.offset = np.zeros(2)  # pan, flex
        self._roll0: float | None = None
        self.seen = 0
        self.ticks = 0

    def apply(self, row: np.ndarray) -> np.ndarray:
        """The replay row with the current aim correction added. Call once per tick."""
        self.ticks += 1
        if self._roll0 is None:
            self._roll0 = float(row[ROLL])
        seen = self.tracker.poll()
        # The Jacobian was measured at one wrist roll. While a sweep rolls the camera,
        # image left/right no longer maps to pan, so the offsets are frozen.
        rolled = abs(float(row[ROLL]) - self._roll0) > self.max_roll_change
        if seen is not None:
            self.seen += 1
            if not rolled:
                error = np.array([seen[0] - 0.5, seen[1] - 0.5])
                if np.abs(error).max() > self.deadband:
                    step = -self.gain * (self.inverse @ error)
                    step = np.clip(step, -self.max_step, self.max_step)
                    self.offset = np.clip(self.offset + step, -self.max_offset, self.max_offset)
        _publish(offset=[round(float(v), 1) for v in self.offset])
        out = np.array(row, dtype=float)
        out[PAN] += self.offset[0]
        out[FLEX] += self.offset[1]
        return out
