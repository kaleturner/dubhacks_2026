"""Offline preflight for a LeRobot dataset before it goes to the Robotics API.

  python -m ml.check_dataset ~/Downloads/duck-shots-pushin-v1

Reads files only. Errors mean the upload would be rejected or the policy would
learn the wrong thing. Warnings are worth a look before you spend a job.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from shared.constants import JOINT_NAMES, SHOTS

ALLOWED_EXT = {".json", ".jsonl", ".parquet", ".mp4", ".png", ".jpg", ".jpeg"}
CAMERAS = ("observation.images.side", "observation.images.wrist")
JOINTS = [f"{name}.pos" for name in JOINT_NAMES]
FPS = 30


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.facts: dict = {}

    @property
    def ok(self) -> bool:
        return not self.errors


def dataset_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))


def _tasks(root: Path) -> dict[int, str]:
    """task_index -> task string, from tasks.parquet (v3) or tasks.jsonl (v2)."""
    parquet = root / "meta" / "tasks.parquet"
    jsonl = root / "meta" / "tasks.jsonl"
    found: dict[int, str] = {}
    if parquet.is_file():
        for row in pq.read_table(parquet).to_pylist():
            text = next((v for v in row.values() if isinstance(v, str)), None)
            index = row.get("task_index")
            if text is not None and index is not None:
                found[int(index)] = text
    elif jsonl.is_file():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                found[int(row["task_index"])] = str(row.get("task", ""))
    return found


def registry_prompts() -> dict[str, str]:
    """sentence -> task name, for every task that has a sentence."""
    return {spec["vla_prompt"]: name for name, spec in SHOTS.items() if spec.get("vla_prompt")}


def check_dataset(root: str | Path, expected_tasks: dict[str, str] | None = None) -> Report:
    """expected_tasks maps sentence -> task name. Default: the SHOTS registry."""
    expected = registry_prompts() if expected_tasks is None else expected_tasks
    root = Path(root).expanduser()
    rep = Report()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        rep.errors.append("meta/info.json is missing")
        return rep
    info = json.loads(info_path.read_text())

    for path in dataset_files(root):
        if path.suffix.lower() not in ALLOWED_EXT:
            rep.errors.append(f"{path.relative_to(root)}: extension not allowed")
        elif path.stat().st_size < 1:
            rep.errors.append(f"{path.relative_to(root)}: empty file")

    if info.get("fps") != FPS:
        rep.errors.append(f"fps is {info.get('fps')}, expected {FPS}")
    features = info.get("features", {})
    for key in ("action", "observation.state"):
        names = features.get(key, {}).get("names")
        if names != JOINTS:
            rep.errors.append(f"{key} names are {names}, expected {JOINTS}")
    for cam in CAMERAS:
        shape = features.get(cam, {}).get("shape")
        if shape != [480, 640, 3]:
            rep.errors.append(f"{cam} shape is {shape}, expected [480, 640, 3]")
        if not list((root / "videos" / cam).rglob("*.mp4")):
            rep.errors.append(f"no videos for {cam}")

    tasks = _tasks(root)
    rep.facts["tasks"] = sorted(tasks.values())
    if not tasks:
        rep.errors.append("no task strings found in meta/tasks")
    for text in tasks.values():
        if text not in expected:
            rep.errors.append(
                f"task string {text!r} is not in SHOTS. Add it as that task's vla_prompt, exactly, "
                f"or fix the dataset. Known: {sorted(expected)}"
            )

    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        rep.errors.append("no data/*.parquet files")
        return rep
    # LeRobot writes action/state as fixed_size_list in some files and list in others.
    list_type = pa.list_(pa.float32())
    tables = []
    for f in data_files:
        t = pq.read_table(f)
        for col in ("action", "observation.state"):
            if col in t.column_names:
                t = t.set_column(t.schema.get_field_index(col), col, t[col].cast(list_type))
        tables.append(t)
    table = pa.concat_tables(tables)
    episode = np.array(table["episode_index"])
    stamp = np.array(table["timestamp"])
    action = np.array(table["action"].to_pylist())
    episodes = sorted(set(episode.tolist()))
    lengths = Counter(episode.tolist())
    rep.facts.update(episodes=len(episodes), frames=len(episode))

    if info.get("total_episodes") not in (None, len(episodes)):
        rep.errors.append(f"info says {info['total_episodes']} episodes, data has {len(episodes)}")
    if info.get("total_frames") not in (None, len(episode)):
        rep.errors.append(f"info says {info['total_frames']} frames, data has {len(episode)}")
    if len(episodes) < 2:
        rep.errors.append("need at least 2 episodes for a train/val split")
    typical = int(np.median(list(lengths.values())))
    for ep, n in sorted(lengths.items()):
        if abs(n - typical) > 0.25 * typical:
            rep.warnings.append(f"episode {ep} has {n} frames, typical is {typical}")

    same = np.diff(episode) == 0
    dt = np.diff(stamp)[same]
    if dt.size and abs(float(np.median(dt)) - 1 / FPS) > 0.002:
        rep.errors.append(f"median frame gap is {np.median(dt) * 1000:.1f} ms, expected {1000 / FPS:.1f} ms")
    if dt.size and float(dt.max()) > 2.5 / FPS:
        rep.warnings.append(f"largest frame gap is {dt.max() * 1000:.0f} ms (dropped frames?)")

    # Episodes per task, from each episode's first frame.
    task_idx = np.array(table["task_index"]) if "task_index" in table.column_names else np.zeros(len(episode), int)
    ep_task = {ep: tasks.get(int(task_idx[episode == ep][0]), "?") for ep in episodes}
    per_task = Counter(ep_task.values())
    rep.facts["episodes_per_task"] = dict(per_task)
    for text, n in per_task.items():
        if n < 10:
            rep.warnings.append(f"only {n} episodes for {text!r}; the policy may not learn it")

    # The gripper holds nothing and stays out of the wrist shot. Demos should all
    # use one value, since inference overrides the gripper with GRIPPER_HOLD.
    grip = action[:, 5]
    means = {ep: float(grip[episode == ep].mean()) for ep in episodes}
    rep.facts["gripper_by_episode"] = {ep: round(v, 1) for ep, v in means.items()}
    if max(means.values()) - min(means.values()) > 5:
        groups = Counter(round(v) for v in means.values())
        rep.warnings.append(
            f"gripper action differs between episodes: {dict(sorted(groups.items()))} "
            "(value: episode count). Pick one GRIPPER_HOLD and re-record or drop the odd group."
        )
    return rep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="folder containing meta/, data/, videos/")
    parser.add_argument("--print-tasks", action="store_true", help="print the task strings and exit")
    args = parser.parse_args()
    if args.print_tasks:
        for index, text in sorted(_tasks(Path(args.dataset).expanduser()).items()):
            print(f"{index}\t{text!r}")
        return 0
    rep = check_dataset(args.dataset)
    f = rep.facts
    print(f"episodes={f.get('episodes')} frames={f.get('frames')} per_task={f.get('episodes_per_task')}")
    for w in rep.warnings:
        print("WARN ", w)
    for e in rep.errors:
        print("ERROR", e)
    print("OK" if rep.ok else "FAILED")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
