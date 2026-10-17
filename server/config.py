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
