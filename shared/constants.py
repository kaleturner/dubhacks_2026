"""Values the whole team has agreed. Change them in CONTRACTS.md first."""

# One entry per recorded task. `vla_prompt` is the sentence used on every demo of
# that task and at inference: verbatim, never paraphrased. Only `enabled` tasks
# are offered to the planner. Phase 1 trains and runs `zoom` alone.
#   scripted    Arm.run_scripted name, the fallback when the policy is unreliable
#   start_pose  robot.poses.POSES key the arm moves to before the shot
#   order       position in a multi-shot story (establishing first, hero last)
SHOTS: dict[str, dict] = {
    "zoom": {
        "enabled": True,
        "vla_prompt": "zoom the camera in then out on the object",
        "scripted": "zoom",
        "start_pose": "zoom_start",
        "order": 2,
        "min_s": 4,
        "max_s": 30,
        "default_s": 10,
        "keywords": ("zoom", "push in", "push-in", "dolly", "close up", "close-up"),
        "look": "slow move toward the object, then back out",
    },
    "pan_out": {
        "enabled": False,
        "vla_prompt": "pan the camera out from the object",
        "scripted": "pan_out",
        "start_pose": "pan_start",
        "order": 0,
        "min_s": 4,
        "max_s": 20,
        "default_s": 8,
        "keywords": ("pan", "reveal", "pull back", "pull-back"),
        "look": "sideways sweep that opens up the scene around the object",
    },
    "overhead": {
        "enabled": False,
        "vla_prompt": "film the object from overhead",
        "scripted": "overhead",
        "start_pose": "overhead_start",
        "order": 1,
        "min_s": 4,
        "max_s": 20,
        "default_s": 8,
        "keywords": ("overhead", "top down", "top-down", "birds eye", "bird's eye", "from above"),
        "look": "top-down view looking straight at the object",
    },
}

MAX_SHOTS = 3
MAX_TOTAL_S = 30.0
DEFAULT_TOTAL_S = 15.0

SCRIPTED_MOVES = ("zoom", "pan_out", "overhead")

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
