"""Isaac 0.5 client for the orbit policy.

Backend calls Arm.run_vla, which should call run_orbit once this exists.
The call sequence is in WORKSHOP.md. Short version:

  grant  POST /v1/deployments/{id}/sessions          team key, ttl 21600
  open   POST /v1/inference/deployments/{id}/sessions
         Authorization + Modal-Authorization, never the team key
  loop   POST /v1/inference/sessions/{id}/predict
  reset  between episodes; DELETE when finished

Predict sends raw joints and PNG parts named side and wrist. The first call
omits rtc. Later calls send prior_chunk_id, prior_start_index, and delay
(1–12). Drop any reply that arrives after capture + 400 ms, then hold.
Actions are already in robot units. Replace only the gripper with
PHONE_GRIPPER_HOLD. Do not rescale the other joints.
"""

from shared.constants import VLA_TASK_PROMPT


def run_orbit(duration_s: float, instruction: str = VLA_TASK_PROMPT) -> None:
    """Blocking orbit. One instruction per session, the recorded sentence."""
    del duration_s, instruction
    raise NotImplementedError
