# AGENTS.md

Team board for humans and coding agents. Read this, `PLAN.md`, `CONTRACTS.md`, and `WORKSHOP.md` before editing. When you start, get blocked, or hand something off, update **Now** and append one line to the **Log**.

## Now

- Repo skeleton is in. Workshop notes are in `WORKSHOP.md`. Isaac 0.5 drives the arm. Mk1.5 does perception.
- Critical path: zoom demos on the arm (phase 1 is zoom in+out only; `sweep` and `overhead` come later). Same task sentence every episode: `push in toward the object and pull back`. Dataset must reach `ready` before the 1,000-update job. Quota is two successful jobs, so plan the 10k run as the second one.
- **Pivot: no phone in the gripper.** The wrist camera films (`robot/cameras.py`), the C920 is the side camera, the phone is only the remote. Phone capture messages and upload routes are gone.
- Backend orchestrator runs PARSE → DONE on MockArm and MockCamera when perception is mocked. `ml.trim.trim_clip` writes an H.264 `{job}_final.mp4`. Status messages may include `stills` and `best_idx`.
- Real Mk1.5 is on for this checkout: `USE_MOCK_PERCEPTION=0`, arm and cameras still mocked. `parse_intent`, `locate`, and `score_angles` are tuned on nine circuit-board photos. A grey mock frame returns “not in frame.” `best_window` still falls back to `start_s=0.5`.
- `PUBLIC_URL` is `https://behaviour-alphabetical-consecutive-wife.trycloudflare.com` (Cloudflare quick tunnel to port 8000). It serves the built phone app, `/ws`, and `/media` on that one origin, and stays up only while `cloudflared` is running. `npm run dev` talks to `ws://localhost:8000/ws`.
- `POST /transcribe` is still an empty string. Mic records and posts when speech recognition is missing.
- Open decisions: the measured `GRIPPER_HOLD`, and the camera indexes.
- Blockers: none for the mock loop or for Mk1.5 on stills. The real arm, the zoom demos, and speech-to-text are still open.

## Tracks

| Track | Folder | Start here | Leave alone |
|---|---|---|---|
| Robotics | `robot/` | `arm.py`, `poses.py`, `vla_client.py` | `web/`, `server/orchestrator.py` |
| ML / Video | `ml/` | `mock_perception.py` is done; implement `perception.py`, then `upload_and_train.py` | `web/`, `robot/arm.py` |
| Backend | `server/` | `orchestrator.py` against `MockArm` and `ml.mock_perception` | `robot/` real arm, `ml/perception.py` prompts |
| Frontend | `web/` | `npm install`, `npm run mock`, `npm run dev`. Prompt, status, result only; no phone camera | `server/`, `robot/`, `ml/` |

Work stays in your folder. Integration happens at the checkpoints in `PLAN.md` §4, with Backend running them.

## Rules

- Do not change a message, function signature, or return shape on your own. Edit `CONTRACTS.md`, `shared/schemas.py`, and `shared/examples/` together, then say so in the log.
- The robotics team key stays in `team.key`. Never commit it, put it in a command, or paste it into an agent chat. Inference uses the session tokens, not that key.
- `USE_MOCKS=1` until the real perception or arm handoff. The server must keep working on mocks if the arm is taken.
- Arm time is shared with other teams. Do not occupy the arm for frontend or backend work.
- Each task's VLA sentence in `shared/constants.py` `SHOTS` stays verbatim, in the demos and at inference. Phase 1 is `push in toward the object and pull back`. Enable a task only after its demos are trained.
- Joint order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Do not rescale actions. Replace only the gripper with `GRIPPER_HOLD`.
- Only the server opens the webcams, through `robot/cameras.py`. Close Rerun and `lerobot-teleoperate` first.
- This is a public repo. At least 70% of the code is written in this event.

## Run

From the repo root, after `python -m venv .venv && source .venv/bin/activate` and `pip install -r requirements.txt`:

```bash
python -m robot.arm                  # MockArm smoke test
python -m robot.cameras              # MockCamera smoke test (--real opens the USB cameras)
python -m uvicorn server.main:app --reload --port 8000
python server/fake_phone.py          # prompt in, status messages out
```

Frontend, in `web/`:

```bash
npm install
npm run mock          # scripted job on ws://localhost:8787
npm run dev           # http://localhost:5173
```

Copy `.env.example` to `.env` before calling Perceptron or the Robotics API. ML also installs `ml/requirements.txt`.

## Handoffs

Write the log line when the thing actually works, not when the file exists.

| What | From → to | Where it lands |
|---|---|---|
| `perception.py` | ML → Backend, ~5 PM | swap the import off `ml.mock_perception` |
| `PUBLIC_URL` | Backend → everyone | `.env` and the log |
| PWA on that URL | Frontend → Backend, ~7 PM | phone can open the tunnel |
| Real `Arm` | Robotics → Backend, ~9 PM | `USE_MOCKS=0` for the arm only |
| `DEPLOYMENT_ID` | ML → Robotics | `.env`, Sunday morning if the 10k run is the one we demo |
| Devpost draft | Frontend, 10:30 PM Sat | screenshots from the phone |
| Demo video | ML, Sunday morning | Devpost |

## Log

Append a row. Newest at the bottom.

| When | Who | Note |
|---|---|---|
| Sat kickoff | skeleton | Contracts frozen. MockArm, mock perception, FastAPI routes, and the phone mock WS are in. |
| Sat workshop | notes | Isaac 0.5 is the arm policy. Mk1.5 is perception. Joint names and the predict/RTC loop are in WORKSHOP.md. |
| Sat afternoon | pivot | Phone out of the gripper. Wrist cam films via `robot/cameras.py`; C920 is side. Removed capture_still/start_recording/stop_recording and /upload routes. `PHONE_GRIPPER_HOLD` → `GRIPPER_HOLD`. |
| Sat 4:30 PM | backend | Orchestrator runs PARSE → RECORD on mocks. POST calls `ml.trim.trim_clip` and stops at ERROR until ML lands it. `PUBLIC_URL` is `https://behaviour-alphabetical-consecutive-wife.trycloudflare.com`. |
| Sat 4:50 PM | frontend | Prompt, status, and result click through on the mock socket. Mic falls back to `POST /transcribe`. Share is hidden unless the browser can share. Error returns to the prompt. |
| Sat 5:15 PM | backend | Mock job reaches DONE. Trim writes H.264. Scout stills and `best_idx` ride on the status message. The tunnel origin serves the built phone app. |
| Sat 6:15 PM | backend | Real Mk1.5 parses, locates, and scores. Boxes convert from the 0–1000 grid to pixels. Tuned on nine circuit-board photos. A mock-arm job parses the prompt and rejects the grey side frame. |
| Sat 6:40 PM | backend | STATUS.md records the mock loop, the phone on one origin, and real Mk1.5 with the arm still mocked. |
| Sat evening | backend | Orchestrator plans shots. `parse_intent` now returns `{object, style, shots, note}`; `move` and top-level `duration_s` are gone, and the result gains optional `shots`. Tasks live in `SHOTS`; only `zoom` is enabled. Each shot films and trims in turn, then `concat_clips` joins them. Orbit and push-in/hero-rise are retired. CONTRACTS, schemas, examples, and `tests/` updated together. Robotics: capture `zoom_start` and `zoom_in` poses. ML: demos use the zoom sentence. |
| Sat night | ml | Zoom sentence is now the dataset's `push in toward the object and pull back` (registry, CONTRACTS, WORKSHOP). `ml/check_dataset.py` preflights a LeRobot folder. `ml/upload_and_train.py` has check, upload, train, deployments. The 22-episode `duck-shots-pushin-v1` passes preflight, but the gripper action is about 41 in episodes 0–7 and about 97 in 8–21, not a fixed 85. `GRIPPER_HOLD` is set to 97.1 (episodes 8–21); the 41 group is the odd one out. Not yet run against the live API: needs `team.key`. |
| Sat night | robotics | Real `Arm` (LeRobot SO-101), `robot/vla_client.py` (RTC loop, dry run), `robot/make_replays.py`, `robot/record_poses.py`, `robot/replays/zoom.json`. Tasks are now zoom, overhead, sweep (`pan_out` is gone). All tested on fakes, none on hardware. Bring-up steps are in STATUS.md. `Arm(side, wrist)` takes the cameras in the constructor; method list unchanged. |

| Sat 8:45 PM | robotics | Replays for zoom, overhead, and sweep built from the 80-episode set (`duck-shots-v3-final`: schema-unified parquet + fix_gripper, preflight OK). Real arm connects with ARM_ID=follower and replays zoom smoothly. |
| Sat 9:30 PM | robotics | Real-arm job on replays reaches DONE: "30 second ad for this duck" → zoom + overhead + sweep, 30 s clip. All three shots enabled, min_s 10 (≤1.5x replay speed), repeated tasks merge. locate trusts the box call (yes/no pre-check flipped on clear photos) plus a flat-crop guard. Scout poses come from replay frames. Tests write server/media/j1_* and overwrite real job files. |
