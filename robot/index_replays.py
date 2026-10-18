"""Index every recorded take by where its object sat, so a shot can replay the closest one.

  python -m robot.index_replays ~/dubhacks26/exports/duck-shots-v3-final

One replay per task only fits objects placed where that demo's object was. This
writes robot/replays/<task>_all.json with every take of the task and the object's
center in the side (C920) view at the start of the take, found by Mk1.5. At run
time Arm.run_scripted picks the take whose object was nearest to where locate
found the current one. The C920 must not have moved since recording.
"""

import argparse
import glob
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ml.check_dataset import _tasks, registry_prompts
from robot.poses import REPLAY_DIR

FPS = 30
SEEK_S = 0.3  # side frame used to place the object; the arm is still at rest
PAUSE_S = 1.0

# Which object each recorded take filmed (from the recording log for duck-shots-v3).
OBJECTS = {
    "duck": [*range(0, 8), *range(22, 38), *range(54, 64)],
    "circuit board": [*range(8, 14), *range(38, 46), *range(64, 72)],
    "white charger": [*range(14, 22), *range(46, 54), *range(72, 80)],
}


def object_of(episode: int) -> str:
    return next((name for name, eps in OBJECTS.items() if episode in eps), "object")


def side_frame(dataset: Path, episodes: "pa.Table", episode: int, out: Path, camera: str = "side") -> None:
    row = episodes.filter(pa.compute.equal(episodes["episode_index"], episode)).to_pylist()[0]
    key = f"videos/observation.images.{camera}"
    video = dataset / key / f"chunk-{row[key + '/chunk_index']:03d}" / f"file-{row[key + '/file_index']:03d}.mp4"
    start = row[key + "/from_timestamp"] + SEEK_S
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-frames:v", "1", str(out)],
        check=True,
    )


def main() -> int:
    from ml.perception import locate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    dataset = Path(args.dataset).expanduser()

    tasks = _tasks(dataset)
    prompts = registry_prompts()
    table = pa.concat_tables(
        [pq.read_table(f) for f in sorted(glob.glob(str(dataset / "data" / "**" / "*.parquet"), recursive=True))]
    )
    episode_col = np.array(table["episode_index"])
    task_col = np.array(table["task_index"])
    actions = np.array(table["action"].to_pylist(), dtype=float)
    meta = pa.concat_tables(
        [pq.read_table(f) for f in sorted(glob.glob(str(dataset / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))]
    )

    tmp = Path(tempfile.mkdtemp(prefix="index_replays_"))

    def place(episode: int, camera: str) -> list[float] | None:
        frame = tmp / f"ep{episode}_{camera}.jpg"
        side_frame(dataset, meta, episode, frame, camera)
        found = locate(str(frame), object_of(episode))
        time.sleep(PAUSE_S)
        return [round(v, 1) for v in found["center"]] if found["found"] else None

    # One request at a time with a pause: parallel calls got this IP rate-limited (Cloudflare 1015).
    # side: object center in C920 pixels. wrist: object center as a fraction of the wrist
    # frame at the take's first pose; the wrist camera rides on the arm, so this survives
    # moving the C920 or the whole setup.
    placed, wrist = {}, {}
    for episode in sorted({int(e) for e in episode_col}):
        placed[episode] = place(episode, "side")
        center = place(episode, "wrist")
        wrist[episode] = None if center is None else [round(center[0] / 640, 4), round(center[1] / 480, 4)]

    REPLAY_DIR.mkdir(exist_ok=True)
    for index, text in sorted(tasks.items()):
        name = prompts.get(text)
        if name is None:
            print(f"skip {text!r}: not in SHOTS", file=sys.stderr)
            continue
        takes = []
        for ep in sorted({int(e) for e in episode_col[task_col == index]}):
            if placed[ep] is None and wrist[ep] is None:
                print(f"{name} episode {ep}: {object_of(ep)} not found in either view, left out")
                continue
            rows = actions[episode_col == ep]
            takes.append(
                {
                    "episode": ep,
                    "object": object_of(ep),
                    "side_xy": placed[ep],
                    "wrist_xy": wrist[ep],
                    "actions": [[round(float(v), 2) for v in row] for row in rows],
                }
            )
        (REPLAY_DIR / f"{name}_all.json").write_text(
            json.dumps({"task": name, "prompt": text, "fps": FPS, "takes": takes}, separators=(",", ":")) + "\n"
        )
        spots = ", ".join(f"{t['episode']}:{t['side_xy']}" for t in takes)
        print(f"{name}: {len(takes)} takes indexed  [{spots}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
