# Contracts

Frozen at kickoff. Build against these shapes. A change touches this file, `shared/schemas.py`, and `shared/examples/` in the same commit, and gets a line in the `AGENTS.md` log so the other tracks hear about it.

## Phone ↔ server

WebSocket: `wss://<PUBLIC_URL>/ws` (plain `ws://` on localhost). JSON text frames.

| Direction | Message |
|---|---|
| phone → server | `{"type":"prompt","text":"record a 15 second video of this apple"}` |
| server → phone | `{"type":"status","stage":"PARSE"\|"LOCATE"\|"SCOUT"\|"PLAN"\|"RECORD"\|"POST"\|"DONE"\|"ERROR","detail":"..."}` |
| server → phone | `{"type":"capture_still","job_id":"j1","pose_id":3}` then the phone POSTs the jpeg |
| server → phone | `{"type":"start_recording","job_id":"j1"}` |
| server → phone | `{"type":"stop_recording","job_id":"j1"}` then the phone POSTs the video |
| server → phone | `{"type":"result","job_id":"j1","url":"/media/j1_final.mp4","duration_s":15}` |

REST:

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/upload/still?job_id=&pose_id=` | jpeg file field `file` | saved under `server/media/` |
| POST | `/upload/video?job_id=` | mp4, webm, or mov file field `file` | saved under `server/media/` |
| GET | `/media/{file}` | | the file |
| POST | `/transcribe` | audio file field `file` | `{"text":"..."}` |

Models: `shared/schemas.py`. Examples: `shared/examples/`.

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
- Phone gripper hold: not measured yet. Robotics writes the number into `PHONE_GRIPPER_HOLD`. After a chunk comes back in robot units, the client replaces only the gripper component with that value.
- Joint names and the Isaac call sequence are in [WORKSHOP.md](WORKSHOP.md).
- The team key lives in `team.key` (mode 600), never in git, commands, or chat. `.env` holds the other names from `.env.example`.
