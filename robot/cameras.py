"""Side and wrist USB cameras, opened once by the server process.

The wrist camera is the filming camera. It is also an Isaac input, so only this
module opens it: the VLA client, scouting, and the clip recorder all read the
latest frame from here. macOS will not let two processes share a webcam.

Both cameras run at 640x480, 30 fps, the size the Isaac dataset contract
expects. Indexes come from `lerobot-find-cameras opencv` and change after a
replug, so they live in .env (SIDE_CAM_INDEX, WRIST_CAM_INDEX).

  python -m robot.cameras          # MockCamera smoke test
  python -m robot.cameras --real   # open both, save one still from each
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 640, 480, 30

__all__ = ["Camera", "MockCamera", "WIDTH", "HEIGHT", "FPS"]


class Camera:
    """One OpenCV capture on a background thread. Keeps the latest frame and can write it to mp4."""

    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._writer: cv2.VideoWriter | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        if not cap.isOpened():
            raise RuntimeError(f"{self.name} camera {self.index} did not open; rerun lerobot-find-cameras")
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.name}", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 3.0
        while self.latest() is None:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.name} camera {self.index} opened but sent no frames")
            time.sleep(0.02)

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
            with self._lock:
                self._frame = frame
                if self._writer is not None:
                    self._writer.write(frame)

    def latest(self) -> np.ndarray | None:
        """BGR frame, 480x640x3. None until the first frame arrives."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def save_still(self, path: str | Path) -> str:
        frame = self.latest()
        if frame is None:
            raise RuntimeError(f"{self.name} camera has no frame yet")
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(path)

    def png_bytes(self) -> bytes:
        """The latest frame as PNG, the format Isaac predict expects."""
        frame = self.latest()
        if frame is None:
            raise RuntimeError(f"{self.name} camera has no frame yet")
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("png encode failed")
        return buf.tobytes()

    def start_recording(self, path: str | Path) -> None:
        """Write every captured frame to an mp4v file. ml.trim re-encodes it to H.264 for the phone."""
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise RuntimeError(f"cannot write {path}")
        with self._lock:
            self._writer = writer

    def stop_recording(self) -> None:
        with self._lock:
            writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()

    def close(self) -> None:
        self.stop_recording()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


class MockCamera(Camera):
    """Grey frames with a moving square. Same methods as Camera, no hardware."""

    def open(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"mockcam-{self.name}", daemon=True)
        self._thread.start()
        while self.latest() is None:
            time.sleep(0.01)

    def _loop(self) -> None:
        tick = 0
        while self._running:
            frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
            x = 40 + (tick * 4) % (WIDTH - 120)
            cv2.rectangle(frame, (x, 180), (x + 80, 260), (40, 40, 200), -1)
            cv2.putText(frame, self.name, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            with self._lock:
                self._frame = frame
                if self._writer is not None:
                    self._writer.write(frame)
            tick += 1
            time.sleep(1 / FPS)


if __name__ == "__main__":
    import argparse
    import os

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    cls = Camera if args.real else MockCamera
    side = cls(int(os.getenv("SIDE_CAM_INDEX", "0")), "side")
    wrist = cls(int(os.getenv("WRIST_CAM_INDEX", "1")), "wrist")
    for cam in (side, wrist):
        cam.open()
        print(cam.name, "->", cam.save_still(f"/tmp/{cam.name}_test.jpg"))
    wrist.start_recording("/tmp/wrist_test.mp4")
    time.sleep(1.0)
    wrist.stop_recording()
    print("wrist -> /tmp/wrist_test.mp4")
    side.close()
    wrist.close()
