import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MEDIA_DIR = Path(__file__).resolve().parent / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

PERCEPTRON_API_KEY = os.getenv("PERCEPTRON_API_KEY", "")
ROBOTICS_TEAM_KEY = os.getenv("ROBOTICS_TEAM_KEY", "")
DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID", "")
ARM_PORT = os.getenv("ARM_PORT", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
# From `lerobot-find-cameras opencv`. They change after a replug.
SIDE_CAM_INDEX = int(os.getenv("SIDE_CAM_INDEX", "0"))
WRIST_CAM_INDEX = int(os.getenv("WRIST_CAM_INDEX", "1"))
USE_MOCKS = os.getenv("USE_MOCKS", "1") == "1"


def _flag(name: str, default: bool) -> bool:
    """Unset or blank follows the default. Anything else is on only when the value is 1."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip() == "1"


# Optional overrides. Unset means follow USE_MOCKS, so a real-perception /
# mock-arm checkpoint does not turn the cameras or the arm on.
USE_MOCK_ARM = _flag("USE_MOCK_ARM", USE_MOCKS)
USE_MOCK_PERCEPTION = _flag("USE_MOCK_PERCEPTION", USE_MOCKS)
USE_MOCK_CAMERAS = _flag("USE_MOCK_CAMERAS", USE_MOCKS)

# Run the fine-tuned policy for each shot. Off means every shot uses its scripted
# fallback. Defaults on only when the arm is real and a deployment exists.
USE_VLA = _flag("USE_VLA", not USE_MOCK_ARM and bool(DEPLOYMENT_ID))
