# AGENTS.md

Team board for humans and coding agents. Read this, `PLAN.md`, `CONTRACTS.md`, and `WORKSHOP.md` before editing. When you start, get blocked, or hand something off, update **Now** and append one line to the **Log**.

## Now

- Repo skeleton is in. Workshop notes are in `WORKSHOP.md`. Isaac 0.5 drives the arm. Mk1.5 does perception.
- Critical path: orbit demos on the arm. Same task sentence every episode. Dataset must reach `ready` before the 1,000-update job. Quota is two successful jobs, so plan the 10k run as the second one.
- **Pivot: no phone in the gripper.** The wrist camera films (`robot/cameras.py`), the C920 is the side camera, the phone is only the remote. Phone capture messages and upload routes are gone.
- Open decisions: tunnel tool (cloudflared or ngrok), who publishes `PUBLIC_URL`, the measured `GRIPPER_HOLD`, and the camera indexes.
- Blockers: none.

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
- VLA prompt stays verbatim, in the demos and at inference: `orbit the camera around the object`.
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
