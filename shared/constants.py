"""Values the whole team has agreed. Change them in CONTRACTS.md first."""

# One entry per recorded task. `vla_prompt` is the sentence used on every demo of
# that task and at inference: verbatim, never paraphrased. Only `enabled` tasks
# are offered to the planner. All three are recorded (80 episodes) and replayable.
#   scripted    Arm.run_scripted name, the fallback when the policy is unreliable
#   start_pose  robot.poses.POSES key the arm moves to before the shot
#   order       position in a multi-shot story (establishing first, hero last)
#   min_s       demos run ~15 s; 10 s keeps a replay at most 1.5x recorded speed
#   engine      "vla" runs the deployed Isaac policy when USE_VLA is on, "replay" always replays
#   prime_s     seconds of the task's own demo replayed before the policy takes over. From
#               the shared start pose the 1k-step policy did an overhead for every
#               instruction; primed into a state only its task reaches it spun (316 deg of
#               roll) and pushed in and back on the real arm
#   center      steer the replay to keep the object centered (robot/centering.py). Off for
#               sweep: its own fast pan into the top-down view wound the loop up and tilted
#               the camera off the table for the whole roll; picking the take aims it instead
SHOTS: dict[str, dict] = {
    "zoom": {
        "enabled": True,
        "engine": "vla",
        "prime_s": 5.0,
        "vla_prompt": "push in toward the object and pull back",
        "scripted": "zoom",
        "start_pose": "zoom_start",
        "order": 2,
        "min_s": 10,
        "max_s": 30,
        "default_s": 15,
        "keywords": ("zoom", "push in", "push-in", "dolly", "close up", "close-up"),
        "look": "slow move toward the object, then back out",
    },
    # vla_prompt is None until the sentence is read from the recorded dataset's
    # meta/tasks.parquet (python -m ml.check_dataset DIR --print-tasks). A task
    # with no sentence cannot be trained and stays disabled.
    "overhead": {
        "enabled": True,
        "engine": "vla",
        "prime_s": 0.0,
        "vla_prompt": "rise up over the object",
        "scripted": "overhead",
        "start_pose": "overhead_start",
        "order": 1,
        "min_s": 10,
        "max_s": 30,
        "default_s": 15,
        "keywords": ("overhead", "top down", "top-down", "birds eye", "bird's eye"),
        "look": "camera arcs up and over until it looks straight down on the object",
    },
    "sweep": {
        "enabled": True,
        "engine": "vla",
        "prime_s": 3.5,
        "center": False,
        "vla_prompt": "spin the camera above the object",
        "scripted": "sweep",
        "start_pose": "sweep_start",
        "order": 0,
        "min_s": 10,
        "max_s": 30,
        "default_s": 15,
        "keywords": ("sweep", "spin", "rotate", "turntable", "scan", "pan", "from above"),
        "look": "camera hovers above the object and rolls a full turn, so the object spins in frame",
    },
}

MAX_SHOTS = 3
MAX_TOTAL_S = 30.0
DEFAULT_TOTAL_S = 30.0  # room for all three shots at their 10 s minimum

SCRIPTED_MOVES = ("zoom", "overhead", "sweep")

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
GRIPPER_HOLD: float | None = 97.1  # duck-shots-pushin-v1: episodes 8-21 (14 of 22). Episodes 0-7 used ~41.
