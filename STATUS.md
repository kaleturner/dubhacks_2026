# Status

Where the app stands against [PLAN.md](PLAN.md) section 4. The phone is only the remote. The wrist camera films, and the C920 is the side camera.

## Current state

A prompt can run PARSE → DONE on MockArm and MockCamera, planned as a list of shots (phase 1 is `zoom` only), and the phone plays the trimmed clip. With `USE_MOCK_PERCEPTION=0` the same server calls real Mk1.5 for the prompt and the stills. The arm and the webcams stay mocked. `POST /transcribe` converts the recording to wav and returns the Mk1.5 transcript. `best_window` is unimplemented, so trim starts at 0.5 seconds unless the clip has to start earlier.

`npm run dev` talks to `ws://localhost:8000/ws`. A production build uses the page’s own origin. FastAPI serves `web/dist` plus `/ws` and `/media`. The last Cloudflare quick tunnel was `https://behaviour-alphabetical-consecutive-wife.trycloudflare.com`. It stays up only while `cloudflared` is running, and a restart prints a new host.

## What changed

- The orchestrator in [server/orchestrator.py](server/orchestrator.py) runs one job at a time from PARSE through DONE, with a timeout and an ERROR path on each stage. The planner returns 1–3 shots from the `SHOTS` registry in [shared/constants.py](shared/constants.py). Scout and PLAN run once. Each shot then moves to its start pose, films, and trims while the next one films. `concat_clips` joins them. A shot uses the VLA sentence when `USE_VLA` is on, otherwise its scripted fallback. Only `zoom` is enabled; a query for `sweep` or `overhead` says so in the PARSE detail and gets zoom. `tests/` covers the planner and a mock-arm job.
- [ml/trim.py](ml/trim.py) re-encodes the wrist clip to H.264, holds the last frame when the take is short, and returns a file of the requested length.
- Status messages can include `stills` and `best_idx`. The phone shows the scout strip and marks the chosen frame. That shape is in [CONTRACTS.md](CONTRACTS.md), [shared/schemas.py](shared/schemas.py), and [shared/examples/status.json](shared/examples/status.json).
- The phone app fills the prompt from the browser speech recognizer when that exists. Otherwise it records until a pause and posts the clip to `/transcribe`. Share appears only when the browser can share. An error or a dropped socket returns to the prompt.
- [ml/perception.py](ml/perception.py) calls Perceptron Mk1.5 for `parse_intent`, `locate`, and `score_angles`. Locate converts the 0–1000 boxes into pixels. A plain grey frame is not a match. `USE_MOCK_ARM`, `USE_MOCK_PERCEPTION`, and `USE_MOCK_CAMERAS` can override `USE_MOCKS` separately.
- Those prompts were checked on nine photos of a circuit board, kept out of git in `ml/samples/`. Locate boxed the board in every photo. Scoring picks the close, evenly lit photo and ranks the shadowed, cluttered, and far shots lower. A mock-arm job parsed “record a 15 second video of this circuit board” as a 15-second orbit, then stopped with “circuit board not in frame.”

## Still open

- ~6 PM, remaining: run [ml/upload_and_train.py](ml/upload_and_train.py) on `duck-shots-pushin-v1` (22 episodes, preflight passes) until the dataset is `ready`. It needs `team.key`. Gripper action is mixed (about 41 and about 97), so `GRIPPER_HOLD` is 97.1 (the 14-episode majority). The 1k training job comes after `ready`. Real perception on stills is in.
- ~7:30 PM: a real phone on the tunnel, from prompt to a trimmed clip, with the arm and cameras still mocked.
- ~9:30 PM: the real `Arm`, a measured `GRIPPER_HOLD`, calibrated poses, and a scripted push-in.
- 10:30 PM: the Devpost draft.
- Overnight and Sunday: the VLA zoom, or scripted `zoom` if the policy is unreliable, then the demo video. `sweep` and `overhead` wait for demos and a new training run.

## Real arm bring-up (for whoever has the arm)

Written and tested against fakes only. Nothing below has touched hardware yet. A person stays on the power switch for every motion step.

1. `pip install "lerobot[feetech]"` in the venv. `lerobot-find-port` gives the port.
2. In `.env`: `ARM_PORT`, `ARM_ID` (the id the arm was calibrated and recorded with), `ARM_MAX_STEP=8`.
3. `python -m robot.arm --real` connects and prints the six joint values. No motion. They should be sane degrees, gripper 0–100.
4. `python -m robot.record_poses` saves `home` and `scout_0..4` to `robot/poses.json`.
5. With the table clear: `python -m robot.arm --real --replay zoom --duration 15`. It eases to the demo's first frame, then plays a recorded demo (episode 12 of `duck-shots-pushin-v1`). A replay assumes the object sits where it did in that demo.
6. Full job: `USE_MOCK_ARM=0 USE_MOCK_CAMERAS=0 USE_VLA=0`, then a prompt from the phone. Each shot replays its recorded demo.
7. Once a deployment exists: `python -m ml.upload_and_train deployments`, put `DEPLOYMENT_ID` in `.env`, then `python -m robot.vla_client --task zoom --real` (arm frozen, prints chunks and latency). Only after that, add `--live`, then `USE_VLA=1`.

Open: the 22-episode upload failed ingestion with no reason (dataset `ds_429f8cf41742a98cc371de3c`; staff asked). The 75-episode merged folder is not on the laptop that built this, so the `overhead` and `sweep` sentences are unset in `SHOTS`. Run `python -m ml.check_dataset <folder> --print-tasks`, fill them in, then `python -m robot.make_replays <folder>`. The API allows only 1000 steps and 2 successful jobs.

