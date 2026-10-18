# Status

Where the app stands against [PLAN.md](PLAN.md) section 4. The phone is only the remote. The wrist camera films, and the C920 is the side camera.

## Current state

A prompt can run PARSE → DONE on MockArm and MockCamera, and the phone plays the trimmed clip. With `USE_MOCK_PERCEPTION=0` the same server calls real Mk1.5 for the prompt and the stills. The arm and the webcams stay mocked. `POST /transcribe` still returns an empty string. `best_window` is unimplemented, so trim starts at 0.5 seconds unless the clip has to start earlier.

`npm run dev` talks to `ws://localhost:8000/ws`. A production build uses the page’s own origin. FastAPI serves `web/dist` plus `/ws` and `/media`. The last Cloudflare quick tunnel was `https://behaviour-alphabetical-consecutive-wife.trycloudflare.com`. It stays up only while `cloudflared` is running, and a restart prints a new host.

## What changed

- The orchestrator in [server/orchestrator.py](server/orchestrator.py) runs one job at a time from PARSE through DONE, with a timeout and an ERROR path on each stage. Orbit calls the VLA prompt. Push-in and hero rise stay scripted.
- [ml/trim.py](ml/trim.py) re-encodes the wrist clip to H.264, holds the last frame when the take is short, and returns a file of the requested length.
- Status messages can include `stills` and `best_idx`. The phone shows the scout strip and marks the chosen frame. That shape is in [CONTRACTS.md](CONTRACTS.md), [shared/schemas.py](shared/schemas.py), and [shared/examples/status.json](shared/examples/status.json).
- The phone app records a prompt when speech recognition is missing and posts it to `/transcribe`. Share appears only when the browser can share. An error or a dropped socket returns to the prompt.
- [ml/perception.py](ml/perception.py) calls Perceptron Mk1.5 for `parse_intent`, `locate`, and `score_angles`. Locate converts the 0–1000 boxes into pixels. A plain grey frame is not a match. `USE_MOCK_ARM`, `USE_MOCK_PERCEPTION`, and `USE_MOCK_CAMERAS` can override `USE_MOCKS` separately.
- Those prompts were checked on nine photos of a circuit board, kept out of git in `ml/samples/`. Locate boxed the board in every photo. Scoring picks the close, evenly lit photo and ranks the shadowed, cluttered, and far shots lower. A mock-arm job parsed “record a 15 second video of this circuit board” as a 15-second orbit, then stopped with “circuit board not in frame.”

## Still open

- ~6 PM, remaining: the orbit demos and [ml/upload_and_train.py](ml/upload_and_train.py), until the dataset is `ready`. The 1k training job comes after `ready`. Real perception on stills is in.
- ~7:30 PM: a real phone on the tunnel, from prompt to a trimmed clip, with the arm and cameras still mocked.
- ~9:30 PM: the real `Arm`, a measured `GRIPPER_HOLD`, calibrated poses, and a scripted push-in.
- 10:30 PM: the Devpost draft. Speech-to-text stays empty until a Grok Voice fill-in.
- Overnight and Sunday: the VLA orbit, or `orbit_scripted` if the policy is unreliable, then the demo video.
