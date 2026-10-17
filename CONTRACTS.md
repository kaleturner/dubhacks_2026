# Contracts

Frozen at kickoff. Build against these shapes. A change touches this file, `shared/schemas.py`, and `shared/examples/` in the same commit, and gets a line in the `AGENTS.md` log so the other tracks hear about it.

## Phone ↔ server

The phone is a remote. It sends the prompt and shows status and the result. It never films; the wrist camera does (see Cameras).

WebSocket: `wss://<PUBLIC_URL>/ws` (plain `ws://` on localhost). JSON text frames.

| Direction | Message |
|---|---|
| phone → server | `{"type":"prompt","text":"record a 15 second video of this apple"}` |
| server → phone | `{"type":"status","stage":"PARSE"\|"LOCATE"\|"SCOUT"\|"PLAN"\|"RECORD"\|"POST"\|"DONE"\|"ERROR","detail":"..."}` |
| server → phone | `{"type":"result","job_id":"j1","url":"/media/j1_final.mp4","duration_s":15}` |

REST:

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/media/{file}` | | the file |
| POST | `/transcribe` | audio file field `file` | `{"text":"..."}` |

Models: `shared/schemas.py`. Examples: `shared/examples/`.

## Cameras (`robot/cameras.py`)

```python
class Camera:                      # MockCamera has the same methods
    def __init__(self, index: int, name: str)   # name is "side" or "wrist"
    def open(self) -> None
    def latest(self) -> np.ndarray | None       # BGR 480x640x3
    def save_still(self, path) -> str           # jpeg
    def png_bytes(self) -> bytes                # for Isaac predict
    def start_recording(self, path) -> None     # mp4v, 640x480, 30 fps
    def stop_recording(self) -> None
    def close(self) -> None
```

- `side` is the Logitech C920, fixed, seeing the whole table. `wrist` is the arm camera and films the clip.
- Both are Isaac inputs at 640×480, 30 fps. The server opens each once (`SIDE_CAM_INDEX`, `WRIST_CAM_INDEX` in `.env`) and passes them to the arm and the VLA client. Nothing else opens the webcams; macOS will not share one between processes.
- Scout stills, the recorded clip, and Isaac frames all come from the same `wrist` object.

## Arm (`robot/arm.py`)

```python
class Arm:
    def connect(self) -> None
    def get_state(self) -> list[float]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg), gripper (0-100)
    def move_to(self, pose: str | list[float], duration_s: float) -> None  # blocking
    def run_scripted(self, move: str, duration_s: float) -> None  # "push_in" | "hero_rise" | "orbit_scripted"
    def run_vla(self, instruction: str, duration_s: float) -> None
    def stop(self) -> None
```

`MockArm` implements the same methods. Scout names are `robot.poses.SCOUT_POSES` (`scout_0` … `scout_4`).

Real `move_to` is 30 Hz smoothstep, clamped to `robot.poses.JOINT_LIMITS`.

## Perception (`ml/perception.py`)

```python
def parse_intent(text: str) -> dict
    # {"object":"apple","duration_s":15,"move":"orbit"|"push_in"|"hero_rise","style":"..."}

def locate(image_path: str, obj: str) -> dict
    # {"found":true,"bbox":[x1,y1,x2,y2],"center":[x,y]}

def score_angles(stills: list[str], obj: str) -> dict
    # {"best_idx":2,"scores":[{"idx":0,"lighting":7,"composition":6,"reason":"..."}]}

def best_window(video_path: str, duration_s: float) -> dict
    # {"start_s":1.2}   optional; fallback start_s=0.5
```

`ml.mock_perception` returns those shapes with hard-coded values. Backend imports that module while `USE_MOCKS=1`.

## Agreed constants

- VLA task prompt, verbatim, and the same string on every demonstration: `"orbit the camera around the object"` (`shared/constants.py`).
- Gripper hold: the gripper holds nothing. Robotics picks the value that keeps the jaws out of the wrist shot, records every demo with it, and writes it into `GRIPPER_HOLD`. After a chunk comes back in robot units, the client replaces only the gripper component with that value.
- Joint names and the Isaac call sequence are in [WORKSHOP.md](WORKSHOP.md).
- The team key lives in `team.key` (mode 600), never in git, commands, or chat. `.env` holds the other names from `.env.example`.
