# Workshop notes

Official sources, condensed for this repo. If a slide and the live docs disagree, trust the docs.

- Hacker guide: https://dubhacks.notion.site/The-DubHacks-26-Hacker-Guide-15dff32c1d7e83748aac8154036aa658
- Devpost: https://dubhacks-2026.devpost.com/
- Isaac service: `https://robotics.perceptron.inc` — `/docs`, `/openapi.json`, `/v1/`, `/healthz`
- Arm setup: the Hugging Face LeRobot SO-101 guide. Commands match the SO-100 guide with `so101` in the type names.
- Participant API guide: press C on the workshop deck, or open the docs on the base URL.

## Two models

**Isaac 0.5** is the open model that drives the arm. It reads the side camera, the wrist camera, the task sentence, and the six joint values, and it writes the next second of joint targets. We fine-tune it on our own SO-101 demonstrations. We do not use it for “what is this object?” or “which angle looks best?”

**Perceptron Mk1.5** is the perception model. Use it for `parse_intent`, `locate`, and `score_angles`, and for audio transcription if the phone has no speech recognizer. The workshop called it the most capable model for receptive tasks. A structured call (one photo, a few references, JSON out) was described as a few seconds. The Q&A mentioned 300 requests per minute per perception key; confirm that in the docs before relying on it.

Mk1.5 boxes come back on a 0–1000 grid. `ml/perception.py` converts them into the pixel `bbox` / `center` in `CONTRACTS.md` before returning.

## Event rules that bind us

- Building that we present starts at 12:30 PM Saturday and stops at 10:00 AM Sunday.
- Devpost draft is mandatory at **10:30 PM Saturday**: https://dubhacks-2026.devpost.com/
- Final submission is **10:00 AM Sunday** and must include a video.
- Public GitHub. At least 70% of the code is written at the event.
- Exactly one title track, at most three sponsored side tracks. Confirm which Devpost boxes Perceptron occupies before the draft. The guide was still being updated.
- Judging is in person, 11:15 AM–2:30 PM Sunday. At least one of us has to be there.
- Perceptron workshop is 12:30–1:15 PM Saturday.
- Overnight workspace opens 11:00 PM at the CSE building (Gates Center, G01). Sleeping there is not allowed.
- Event Wi-Fi credentials are on the hacker guide. Venue Wi-Fi is a known risk; a phone hotspot is the backup.

## The arm

SO-101, six Feetech servos. Leader is the black arm with a handle; follower is the white arm with the claw. You move the leader, the follower mirrors it. While recording, follower joints are `observation.state` and leader joints are `action`.

This order is the contract. Every dataset row, request, and returned action uses it. Joints 0–4 are degrees when LeRobot records with `use_degrees=true`. The gripper is 0–100, not an angle.

| Index | Name |
|---|---|
| 0 | `shoulder_pan.pos` |
| 1 | `shoulder_lift.pos` |
| 2 | `elbow_flex.pos` |
| 3 | `wrist_flex.pos` |
| 4 | `wrist_roll.pos` |
| 5 | `gripper.pos` |

Cameras, named exactly this, 640×480, 30 fps:

- `observation.images.side` — fixed, sees the whole task
- `observation.images.wrist` — on the gripper

The phone in the gripper is our recording camera. It is not an Isaac camera. If a person cannot do the task from the side and wrist frames alone, the policy cannot either.

### Setup

1. `lerobot-find-port` for each arm. macOS ports look like `/dev/tty.usbmodem…`. Each arm has its own power and USB.
2. Calibrate both. Middle of every joint’s range, Enter, then sweep each joint through its full range. `--robot.id` and `--teleop.id` name the calibration files. Reuse those ids for teleop, record, and inference. Do not recalibrate in the middle of a dataset.
3. `lerobot-find-cameras` every session. Ids change after a replug. `lerobot-teleoperate --display_data=true` shows both feeds in Rerun before you record.

### Demonstrations

- About 50 episodes. About 10 per object position, from 5 or 6 positions.
- One task string for every episode, and the same string at inference: `orbit the camera around the object`.
- One way of holding the phone. Slow moves. Cameras stay put. The apple stays in frame.
- Start and stop on clean task boundaries.
- After recording, replay an episode on the follower. If the replay does not match the demo, fix that before training.
- LeRobot stamps frames as 30 Hz from the frame index. If the laptop only captured at 25 Hz, the dataset still says 30 and the motion is sped up. The validator will not catch it. Watch the cadence line.

## Isaac service

You hold a base URL, the team key, and a terminal. GPUs stay on their side. No SDK. A Hugging Face account is only needed if the dataset is published there.

### Keys

- Staff key goes in `team.key`, mode `600`. Build a header file from it. The key never appears in a command, URL, screenshot, log, git, or an agent chat.
- The team key is only for the control API (datasets, jobs, deployments, the session grant).
- Signed upload URLs and the inference API get their own credentials. Never send the team key to either.
- One key per team. Someone else’s id looks like a 404. If it leaks, tell staff; they revoke and reissue it.
- `GET` capabilities before assuming a mode, horizon, or step count exists. Then the OpenAPI schema.

### Dataset

Hugging Face repo is the recommended register path. Direct upload is the alternative (`meta/info.json` required).

Direct upload: reserve with an `Idempotency-Key`, `PUT` each signed URL (scoped to one file, 900 seconds, no team key), then finalize. Finalize pins hashes and starts ingest. Do not upload to that dataset again after finalize. A 303 is a poll: follow `Location` with no credentials and do not resend the body. While the state is still `uploading`, reserve the same files again for fresh URLs.

Ingest is not “uploaded”. Poll until the state is `ready`. Train only on `ready`. The worker checks fps, camera names, joint names, timestamps, and frame counts, converts to training shards, splits whole episodes about 90/10 (at least one train and one val episode), and computes q01/q99 normalization from train episodes only. The talk estimated 10–20 minutes. Their 50-episode test (609 MB, 16 files) became 41 train and 9 val episodes.

### Fine-tune

One `POST` once the dataset is ready. Workshop body: standard backend, `expert_only`, horizon 30, RTC on, 1000 updates. Send an `Idempotency-Key`. The same key and the same JSON returns the same job. A new job needs a new key.

`expert_only` trains the action expert only and is the hackathon default. `backbone_lora` and `full` exist on `standard`. `shared` is `expert_only` only. Do not assume a horizon or an RTC-off mode the capabilities list does not show.

The slides’ 1,000-update job is the workshop setting. The learning-rate schedule is a 10,000-update cycle, and the talk asked for a 10,000-step final run, about 5 hours. Default quota is 2 successful jobs and 2 queued jobs per team, 6 hours per job, 120 control requests per minute. A 1k check plus a 10k final uses both successes. Read the live limits before starting the second one.

Poll `GET /v1/jobs/{id}` every few seconds. Logs and metrics are paged (`cursor`, then `next_cursor`). Training complete, export complete, and inference ready are three different states. `GET /v1/deployments` after the checkpoint is `inference_ready`.

### Deploy contract

`GET /v1/deployments/{id}/contract` with the team key. Save `contract_id`. It pins joint order, units, camera names, horizon 30 at 30 Hz, RTC limits, and image limits. Put `contract_id` on the robot session and every predict. An old id is refused.

You send raw calibrated joints and frames up to 1920×1080. The deployment resizes and normalizes. Actions come back in robot units. Never normalize or rescale on our side. The phone-hold override replaces only `gripper.pos` with `PHONE_GRIPPER_HOLD` after the chunk arrives.

### Inference

1. `POST /v1/deployments/{id}/sessions` with the team key. `session_id` like `robot-01` (letters, digits, underscore, dash). `ttl_seconds` 21600, which is 6 hours and the maximum.
2. The grant returns `application_session_token` and `modal_session_token`. Write them to `inference.headers` as `Authorization` and `Modal-Authorization`.
3. `POST /v1/inference/deployments/{id}/sessions` with those headers, the `contract_id`, `rtc_enabled`, and the task sentence. One instruction per session, the sentence used while recording.
4. `POST /v1/inference/sessions/{session_id}/predict`.
5. `POST .../reset` between episodes. `DELETE` the robot session at the end. To continue, take a fresh grant with a new session id.

Predict is multipart. Metadata JSON carries `contract_id`, an observation id, a sequence starting at 0, a timestamp on the robot’s clock, and the six raw joints. Frames are binary parts named `side` and `wrist`, PNG, not base64. Two PNGs are about 0.5 MB. At 2 Hz that is about 8 Mbit/s of uplink. Sequence and timestamps strictly increase. Set a descriptive `User-Agent`; the default Python urllib agent gets rejected by some edges.

The first predict omits `rtc`. The reply is 30×6 absolute targets plus a `chunk_id`.

From the second predict on, the `rtc` block says what is already in motion: `prior_chunk_id`, `prior_start_index` (15 at the 2 Hz cadence), and `delay` (1–12 rows). Delay 12 is 400 ms, which is `deadline_ms`. The timestamp has to match the rows elapsed (15 rows later is +0.5 s). The reply echoes the committed rows, `prefix_length` 12, and `execute_from_index` 12. Switch at that boundary, not at the moment the reply arrives.

Control loop, two threads, one clock:

- Control ticks at 30 Hz and plays the active chunk. If no accepted chunk covers this tick, hold position.
- Inference fires every 15 rows. Keep one request in flight.
- A reply that arrives after capture + 400 ms is rejected. Never execute a late, stale, or malformed chunk. The old chunk still has 3 reserve rows, then hold.
- A delay over 12 rows is rejected. Training practised delays of 1–12, Poisson with mean 5.
- 503/504: do not resend the stale frame. Capture a new one and hold.
- 401 on inference usually means the session credentials expired, or the team key was sent there. Fresh grant, new session id, rewrite the headers, open a new robot session.

Our orbit client lives in `robot/vla_client.py`. It must follow this loop and then overwrite the gripper component with `PHONE_GRIPPER_HOLD`.

## Five rules from the deck

1. A late chunk is worse than no chunk. Hold.
2. The team key stays in a private file.
3. Contract order, degrees, gripper 0–100. Do not reorder or convert by guesswork.
4. Retry with the same idempotency key and the same JSON. Never blind re-POST.
5. Live capabilities and the pinned contract beat any example, including this file.
