"""Isaac 0.5 client for the shot policy (phase 1: zoom).

Backend calls Arm.run_vla, which should call run_task once this exists.
The call sequence is in WORKSHOP.md. Short version:

  grant  POST /v1/deployments/{id}/sessions          team key, ttl 21600
  open   POST /v1/inference/deployments/{id}/sessions
         Authorization + Modal-Authorization, never the team key
  loop   POST /v1/inference/sessions/{id}/predict
  reset  between episodes; DELETE when finished

Predict sends raw joints and PNG parts named side and wrist, from
robot.cameras (Camera.png_bytes). Do not open the webcams here; the wrist
camera is recording the clip at the same time. The first call
omits rtc. Later calls send prior_chunk_id, prior_start_index, and delay
(1–12). Drop any reply that arrives after capture + 400 ms, then hold.
Actions are already in robot units. Replace only the gripper with
GRIPPER_HOLD. Do not rescale the other joints.
"""

def run_task(instruction: str, duration_s: float) -> None:
    """Blocking run of one task. One instruction per session: open a session with the
    sentence from shared.constants.SHOTS[task]["vla_prompt"], run, then delete it."""
    del duration_s, instruction
    raise NotImplementedError
