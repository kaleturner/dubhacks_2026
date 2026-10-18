"""Control-API access shared by ml/upload_and_train.py and robot/vla_client.py.

The team key comes from team.key (mode 600) or ROBOTICS_TEAM_KEY in .env. It is
never an argument and never printed. It goes only to the control API, never to a
signed upload URL and never to the inference endpoint (that uses session tokens).
"""

import os
import stat
from pathlib import Path
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("ROBOTICS_BASE_URL", "https://robotics.perceptron.inc").rstrip("/")
KEY_FILE = Path(os.getenv("TEAM_KEY_FILE", ROOT / "team.key"))
USER_AGENT = "dubhacks-cinematic-recorder/0.1"


class ApiError(RuntimeError):
    pass


def read_key() -> str:
    if not KEY_FILE.is_file():
        env_key = os.getenv("ROBOTICS_TEAM_KEY", "").strip()
        if env_key:
            return env_key
        raise SystemExit(
            f"{KEY_FILE} not found and ROBOTICS_TEAM_KEY is empty. "
            "Put the team key in one of them. It is never passed on the command line."
        )
    if KEY_FILE.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(f"{KEY_FILE} is readable by others. Run: chmod 600 {KEY_FILE}")
    key = KEY_FILE.read_text().strip()
    if not key:
        raise SystemExit(f"{KEY_FILE} is empty")
    return key


class Api:
    """Control API calls. The key lives only in this object's Authorization header."""

    def __init__(self, key: str) -> None:
        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})

    def call(self, method: str, path: str, *, idem: str | None = None, body: dict | None = None, ok=(200, 201, 202)):
        headers = {"Idempotency-Key": idem} if idem else {}
        resp = self.http.request(method, BASE_URL + path, json=body, headers=headers, timeout=60, allow_redirects=False)
        if resp.status_code == 303:
            return self.poll_redirect(resp.headers.get("Location", ""))
        if resp.status_code not in ok:
            raise ApiError(f"{method} {path} -> {resp.status_code}: {resp.text[:400]}")
        return resp.json() if resp.content else {}

    @staticmethod
    def poll_redirect(location: str):
        """A 303 is a poll. Follow Location with no credentials and no body."""
        resp = requests.get(urljoin(BASE_URL + "/", location), headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code != 200:
            raise ApiError(f"poll {location} -> {resp.status_code}: {resp.text[:400]}")
        return resp.json()
