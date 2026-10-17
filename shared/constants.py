"""Values the whole team has agreed. Change them in CONTRACTS.md first."""

# Verbatim VLA task prompt. Do not paraphrase.
VLA_TASK_PROMPT = "orbit the camera around the object"

SCRIPTED_MOVES = ("push_in", "hero_rise", "orbit_scripted")
INTENT_MOVES = ("orbit", "push_in", "hero_rise")

STAGES = ("PARSE", "LOCATE", "SCOUT", "PLAN", "RECORD", "POST", "DONE", "ERROR")

# Isaac / LeRobot contract order. Joints 0–4 are degrees. Gripper is 0–100, not an angle.
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# The gripper holds nothing; the wrist camera films. Robotics picks the value
# that keeps the jaws out of the wrist shot and records every demo with it.
# The VLA loop overrides the gripper command to this value. None until measured.
GRIPPER_HOLD: float | None = None
