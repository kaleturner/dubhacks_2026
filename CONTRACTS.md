# Contracts

Frozen at kickoff. Build against these shapes. A change touches this file, `shared/schemas.py`, and `shared/examples/` in the same commit, and gets a line in the `AGENTS.md` log so the other tracks hear about it.

## Phone ↔ server

The phone is a remote. It sends the prompt and shows status and the result. It never films; the wrist camera does (see Cameras).

WebSocket: `wss://<PUBLIC_URL>/ws` (plain `ws://` on localhost). JSON text frames.

| Direction | Message |
|---|---|
| phone → server | `{"type":"prompt","text":"record a 15 second video of this apple"}` |
| server → phone | `{"type":"status","stage":"PARSE"\|"LOCATE"\|"SCOUT"\|"PLAN"\|"RECORD"\|"POST"\|"DONE"\|"ERROR","detail":"...","stills":["/media/j1_pose_0.jpg"],"best_idx":2}` |
| server → phone | `{"type":"result","job_id":"j1","url":"/media/j1_final.mp4","duration_s":15,"shots":[{"task":"zoom","duration_s":15}],"edit_url":"/media/j1_edit.mp4"}` |

The edit's soundtrack is chosen on the server from the prompt and the parsed `style`, from the CC0 beds in `shared/tracks.json`. The phone does not pick it. `shots` on the result is optional: the planned tasks, in order. `duration_s` is their sum. `edit_url` is optional: the clip the phone shows first. It is the wrist take with the white gripper cropped off the bottom, still stretches removed, a warmer grade, and the CC0 electronic track cut so the joins land on its pulse. The raw file stays at `url`, behind a button. Shot progress rides in the RECORD `detail` ("shot 1/2: zoom, ..."). `stills` and `best_idx` are optional. `stills` is the scout JPEGs under `/media`, sent with each SCOUT status after that still is saved and again on PLAN. `best_idx` is sent on PLAN.

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
    def run_scripted(self, move: str, duration_s: float) -> None  # "zoom" | "overhead" | "sweep": replays that task's recorded demo from robot/replays/
    def run_vla(self, instruction: str, duration_s: float) -> None
    def stop(self) -> None
```

`MockArm` implements the same methods. `Arm(side, wrist)` takes the shared cameras for the VLA; `send_joints(six)` is the one place a command reaches the motors (clamps, forces the gripper to `GRIPPER_HOLD`, honors `VLA_DRY_RUN=1`). Scout names are `robot.poses.SCOUT_POSES` (`scout_0` … `scout_4`).

Real `move_to` is 30 Hz smoothstep, clamped to `robot.poses.JOINT_LIMITS`.

## Perception (`ml/perception.py`)

```python
def parse_intent(text: str) -> dict
    # {"object":"apple","style":"...","shots":[{"task":"zoom","duration_s":15}],"note":null}
    # tasks come from shared.constants.SHOTS and only enabled ones are offered.
    # 1-3 shots, durations sum to the requested total (1-30 s). note says when a
    # disabled move was asked for. shared.shots.normalize_shots makes any output safe.

def locate(image_path: str, obj: str) -> dict
    # {"found":true,"bbox":[x1,y1,x2,y2],"center":[x,y]}

def score_angles(stills: list[str], obj: str) -> dict
    # {"best_idx":2,"scores":[{"idx":0,"lighting":7,"composition":6,"reason":"..."}]}

def best_window(video_path: str, duration_s: float) -> dict
    # {"start_s":1.2}   optional; fallback start_s=0.5
```

`ml.mock_perception` returns those shapes with hard-coded values. Backend imports that module while `USE_MOCKS=1`.

## Agreed constants

- Tasks live in `SHOTS` in `shared/constants.py`. Each has one VLA sentence, verbatim, the same on every demonstration of that task and at inference. The tasks are `zoom` (in then out), `overhead`, and `sweep` (one full pass of the camera from above). Phase 1 trains `zoom`: `"push in toward the object and pull back"`. `overhead` and `sweep` have `vla_prompt: None` until their sentences are read from the recorded dataset, and stay `enabled: False` until trained (`shared/constants.py`).
- Gripper hold: the gripper holds nothing. Robotics picks the value that keeps the jaws out of the wrist shot, records every demo with it, and writes it into `GRIPPER_HOLD`. After a chunk comes back in robot units, the client replaces only the gripper component with that value.
- Joint names and the Isaac call sequence are in [WORKSHOP.md](WORKSHOP.md).
- The team key lives in `team.key` (mode 600), never in git, commands, or chat. `.env` holds the other names from `.env.example`.
