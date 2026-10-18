"""Turn recorded demos into one replay per task.

  python -m robot.make_replays ~/Downloads/duck-shots-pushin-v1

For each task in the dataset it picks the episode closest to the task's average
trajectory (the medoid) and writes its action rows to robot/replays/<task>.json.
The arm replays that file when no policy is deployed. It also prints the typical
demo length, which is the natural shot length for that task.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ml.check_dataset import _tasks, registry_prompts
from robot.poses import REPLAY_DIR

FPS = 30


def _resample(rows: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(rows))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, rows[:, j]) for j in range(rows.shape[1])], axis=1)


def medoid(episodes: list[np.ndarray]) -> int:
    """Index of the episode closest to the mean trajectory."""
    n = int(np.median([len(e) for e in episodes]))
    stack = np.stack([_resample(e, n) for e in episodes])
    mean = stack.mean(axis=0)
    return int(np.argmin(((stack - mean) ** 2).sum(axis=(1, 2))))


def build(dataset: Path) -> dict[str, dict]:
    tasks = _tasks(dataset)
    prompts = registry_prompts()
    table = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(str(dataset / "data" / "**" / "*.parquet"), recursive=True))])
    episode = np.array(table["episode_index"])
    task_idx = np.array(table["task_index"])
    action = np.array(table["action"].to_pylist(), dtype=float)
    out: dict[str, dict] = {}
    for index, text in sorted(tasks.items()):
        name = prompts.get(text)
        if name is None:
            print(f"skip {text!r}: not in SHOTS (add it as a vla_prompt first)", file=sys.stderr)
            continue
        eps = sorted({int(e) for e in episode[task_idx == index]})
        runs = [action[episode == e] for e in eps]
        pick = medoid(runs)
        length = int(np.median([len(r) for r in runs]))
        out[name] = {
            "task": name,
            "prompt": text,
            "fps": FPS,
            "episode": eps[pick],
            "episodes_in_task": len(eps),
            "typical_length_s": round(length / FPS, 1),
            "actions": [[round(float(v), 2) for v in row] for row in runs[pick]],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    replays = build(Path(args.dataset).expanduser())
    if not replays:
        print("no replays written", file=sys.stderr)
        return 1
    REPLAY_DIR.mkdir(exist_ok=True)
    for name, replay in replays.items():
        (REPLAY_DIR / f"{name}.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n")
        start = replay["actions"][0]
        print(
            f"{name}: episode {replay['episode']} of {replay['episodes_in_task']}, "
            f"{replay['typical_length_s']} s typical, start pose {start}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
