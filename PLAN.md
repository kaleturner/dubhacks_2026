# Robotic Cinematic Ad Recorder: Team Plan (DubHacks '26 · Perceptron Robotics Track)

Workshop corrections (Isaac vs Mk1.5, joint names, the robotics API) are in [WORKSHOP.md](WORKSHOP.md). Where this plan and that file disagree, follow WORKSHOP.md.

## 1. What we're building
A user says *"Record a 15 second video of this apple."* The phone, held in the SO-101's gripper, runs our web app. A laptop orchestrator:
1. Uses Perceptron Mk1.5 to parse the request and locate the object.
2. Moves the arm through a few scout angles, where the phone snaps a still at each, and has Mk1.5 pick the angle with the best light and composition.
3. Performs a cinematic move while the phone records: the orbit is a fine-tuned Isaac VLA; the push-in and hero rise are scripted.
4. Trims the clip to exactly N seconds (≤30 s processing) and sends it back to the phone.

**Decisions we've made:** phone held in the gripper (with pads) · hybrid motion · lighting chosen by angle, no lamps · single take auto-trimmed · PWA web app · voice input with text fallback · the VLA uses the provided USB side and wrist cameras, so the phone only records.

**Hard facts:**
- Workshop at 12:30–1:15 (required). Devpost draft due **10:30 PM Sat**. Final with demo video due **10:00 AM Sun**. Public repo, with at least 70% of the code written here.
- Up to 4 teams share one arm, so **arm time is the scarcest resource**.
- VLA training: about 50 demos; 1k steps for a quick check, 10k steps (~5 h) for the final run. **Orbit demos must be uploaded by about 6 PM.**
- API keys go in `.env` (gitignored). Never commit them or paste them into AI tools.

---

## 2. Kickoff (first 20 min, everyone together): freeze these contracts
Once these are agreed, all 4 people can build against mocks and merge later without rework. Commit them as `CONTRACTS.md` plus `shared/` stubs.

### 2a. Repo skeleton (one person pushes in 5 min)
```
/web        PWA (Vite + React)              → Frontend
/server     FastAPI orchestrator            → Backend
/robot      arm control + VLA client        → Robotics
/ml         Mk1.5 prompts + VLA data/train  → ML/Video
/shared     contracts (schemas, examples)
.env.example  PERCEPTRON_API_KEY= ROBOTICS_TEAM_KEY= DEPLOYMENT_ID= ARM_PORT= PUBLIC_URL=
```

### 2b. Phone ⇄ Server WebSocket (`wss://<PUBLIC_URL>/ws`, JSON)
| Direction | Message |
|---|---|
| phone→server | `{"type":"prompt","text":"record a 15 second video of this apple"}` |
| server→phone | `{"type":"status","stage":"PARSE"\|"LOCATE"\|"SCOUT"\|"PLAN"\|"RECORD"\|"POST"\|"DONE"\|"ERROR","detail":"..."}` |
| server→phone | `{"type":"capture_still","job_id":"j1","pose_id":3}` → phone POSTs the still |
| server→phone | `{"type":"start_recording","job_id":"j1"}` / `{"type":"stop_recording","job_id":"j1"}` → phone POSTs the video |
| server→phone | `{"type":"result","job_id":"j1","url":"/media/j1_final.mp4","duration_s":15}` |

**REST:** `POST /upload/still?job_id=&pose_id=` (jpeg) · `POST /upload/video?job_id=` (mp4/webm) · `GET /media/{file}` · `POST /transcribe` (audio blob → `{"text":...}`, fallback only).

### 2c. Robot interface (Python, `robot/arm.py`), used by the backend
```python
class Arm:
    def connect(self) -> None
    def get_state(self) -> list[float]          # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg), gripper (0-100)
    def move_to(self, pose: str | list[float], duration_s: float) -> None   # blocking, eased, clamped
    def run_scripted(self, move: str, duration_s: float) -> None           # "push_in" | "hero_rise" | "orbit_scripted"
    def run_vla(self, instruction: str, duration_s: float) -> None         # Isaac orbit via Robotics API
    def stop(self) -> None
SCOUT_POSES: list[str]   # e.g. ["scout_0", ... "scout_4"]
```
Ship a `MockArm` with the same methods (sleeps and prints) on day 1, so the backend never waits for the arm.

### 2d. Perception interface (Python, `ml/perception.py`), used by the backend
```python
def parse_intent(text: str) -> dict        # {"object":"apple","duration_s":15,"move":"orbit"|"push_in"|"hero_rise","style":"..."}
def locate(image_path: str, obj: str) -> dict   # {"found":true,"bbox":[x1,y1,x2,y2],"center":[x,y]}
def score_angles(stills: list[str], obj: str) -> dict  # {"best_idx":2,"scores":[{"idx":0,"lighting":7,"composition":6,"reason":"..."}]}
def best_window(video_path: str, duration_s: float) -> dict  # {"start_s":1.2}  (optional; fallback start_s=0.5)
```
Ship a `mock_perception.py` that returns hard-coded examples of each shape.

### 2e. Agree today
- The VLA task prompt, verbatim: `"orbit the camera around the object"`.
- The constant gripper value for holding the phone (measure it after calibration).
- The tunnel tool (cloudflared or ngrok) and who owns the `PUBLIC_URL`.

---

## 3. Individual tracks: start immediately after kickoff

### 🤖 Robotics (owns the arm and all arm time)
**Start now:**
1. Attend the workshop. Find the ports and calibrate the leader and follower with LeRobot; save the calibration file.
2. Mount the provided wrist and side USB cameras. Fit the phone in the gripper with foam or rubber pads, find the stable grip value, and test slow moves for slippage.
3. **Top priority: record 50–60 orbit demos** with ML/Video, using the agreed prompt. Use 5–6 apple positions × 10 demos, slow, always orbiting the same direction, with the phone in the gripper.
4. Write `robot/arm.py` plus `MockArm`. Build a pose library (`poses.py`) by moving the arm by hand and saving joint values: home, scout_0..4, push_in start/end, hero top. Implement `move_to` with 30 Hz smoothstep interpolation and joint clamps.
5. `vla_client.py`: Robotics API session grant → create session → predict loop (side + wrist frames, state) → execute 30-action chunks with RTC → override the gripper to the hold value → stop at `duration_s` → delete the session. First test on the 1k-step checkpoint.

**Mocks:** none needed. **Handoff:** the real `Arm` class to Backend by about 9 PM.

### 🧠 ML / Video (owns Mk1.5 and the VLA training pipeline)
**Start now (no arm needed):**
1. Get the Mk1.5 key; install `pip install "perceptron>=0.4.0"`; smoke-test using apple photos from your own phone.
2. Write `ml/perception.py` with JSON-schema structured outputs for `parse_intent`, `locate` and `score_angles`. Tune the prompts on 5–10 sample apple photos taken from different angles and lighting. Ship `mock_perception.py` first.
3. Write the dataset pipeline script `ml/upload_and_train.py`: register the dataset → reserve uploads → PUT the files → finalize → poll validation → `POST /v1/jobs` (standard, expert_only, horizon 30, RTC on). Accept steps as an argument and log job status and metrics.
4. Help Robotics record demos (operate the leader arm or reset the object), then run 1k steps. If the result looks sane, **start the 10k run by about 6–7 PM**.
5. `best_window` (optional) plus the ffmpeg trim helper `ml/trim.py` (stream copy, fade in/out, exact `duration_s`).
6. Own the final demo video for Devpost on Sunday morning.

**Mocks:** none. **Handoff:** `perception.py` to Backend by about 5 PM; deployment ID by Sunday morning.

### 🖥️ Backend (owns the orchestrator and the integration)
**Start now (all mocked):**
1. FastAPI skeleton: the `/ws` hub, the upload endpoints, `/media` static files, `.env` loading.
2. `server/orchestrator.py` state machine: PARSE → LOCATE (side-cam frame) → SCOUT (for each pose: `arm.move_to` → `capture_still` → wait for the upload) → PLAN (`score_angles` → pick the move) → RECORD (`start_recording` → run the move → `stop_recording`) → POST (trim) → DONE. Push a `status` message at every stage, with timeouts and an `ERROR` path.
3. Run everything against `MockArm` + `mock_perception` plus a tiny fake phone client (a Python WS script) so the whole loop works before anyone else is done.
4. Set up the HTTPS tunnel and publish the `PUBLIC_URL` to the team. Normalize phone video (webm/mov → mp4) with ffmpeg.
5. Swap in the real `perception.py` (about 5 PM) and the real `Arm` (about 9 PM).

**Mocks:** MockArm, mock_perception, fake phone client. **Integration owner:** runs every merge checkpoint.

### 📱 Frontend (owns the phone PWA)
**Start now (works against a local mock WS server):**
1. Vite + React app, mobile-first and big-button UI. Screens: **Prompt** (mic plus text box) → **Live status** (stage stepper) → **Result** (video player with a download/share button).
2. Camera: `getUserMedia` with the rear camera at 1080p/30, `MediaRecorder` recording (mp4 on iOS, webm on Android), canvas snapshot for stills, Wake Lock to keep the screen on, and exposure/focus lock where supported.
3. Voice: `webkitSpeechRecognition`. When it's missing, fall back to recording audio and calling `POST /transcribe`.
4. WS client implementing every message in §2b. Write a 30-line `mock_server.js` that replays a scripted job so you can develop alone.
5. **Test on the actual team phones early.** iOS Safari's HTTPS and MediaRecorder quirks are the main risk.
6. Own the Devpost draft text and screenshots by **10:30 PM**.

**Mocks:** mock WS server. **Handoff:** the PWA served from the tunnel URL by about 7 PM.

---

## 4. Merge checkpoints
| Time | Checkpoint | Pass criteria |
|---|---|---|
| ~1:45 PM | Contracts frozen, repo skeleton pushed | Everyone pulls; mocks exist |
| ~6:00 PM | **Demos uploaded**; real perception in the backend | Dataset validation passes; backend + mock arm + real Mk1.5 work on stills |
| ~7:30 PM | **Phone ↔ backend** | Real phone: prompt → status updates → mock-arm "recording" → trimmed clip returned |
| ~9:30 PM | **Real arm, scripted moves** | Full end-to-end: "Record a 10 second video of this apple" → scout → push-in → 10 s clip in ≤30 s |
| 10:30 PM | Devpost draft submitted | ✓ |
| Overnight | VLA orbit on the 1k checkpoint, then swap to 10k | Orbit works at 3 untried apple positions. If not, use `orbit_scripted` as the fallback |
| Sun 9 AM | Demo video recorded, repo public | Final Devpost submitted by 10 AM |

## 5. Risks and fallbacks
- **The phone slips or overloads the wrist:** pads, landscape grip, poses close to the base, slow moves, clamps.
- **The VLA orbit is unreliable:** use `orbit_scripted`; the demo still shows Mk1.5 shot planning.
- **Arm contention:** only Robotics and ML need the arm; everyone else works on mocks.
- **Venue WiFi:** keep a phone hotspot as backup; keep clips at 1080p and short.
- **iOS recording quirks:** the backend normalizes with ffmpeg; test on day 1.

## 6. Verification (end-to-end)
1. Unit: each `perception.py` function returns schema-valid JSON on sample photos; `Arm.move_to` reaches every named pose; `vla_client` gets a 30×6 chunk back from `/predict`.
2. Pipeline: Robotics API dataset validation passes, the job succeeds, and the deployment is `ready`.
3. E2E scripted: the spoken prompt yields a clip of exactly the requested length on the phone within 30 s of recording ending.
4. E2E VLA: the orbit style works at 3 untrained object positions.
