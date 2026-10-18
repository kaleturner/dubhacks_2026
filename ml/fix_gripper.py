"""Make the gripper consistent across a LeRobot v3 dataset by writing a patched copy.

  python -m ml.fix_gripper SRC DST [--threshold 60]

Episodes whose mean gripper action is below --threshold used a different jaw
opening from the rest. Inference overrides the gripper with GRIPPER_HOLD, so
their action[5] is set to GRIPPER_HOLD and their observation.state[5] to the
median state of the other episodes. SRC is never modified. Per-episode stats in
meta/episodes and the global meta/stats.json are refreshed for the gripper
channel. Videos are hard-linked (or copied) unchanged.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from shared.constants import GRIPPER_HOLD

GRIP = 5
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _matrix(col: pa.ChunkedArray) -> np.ndarray:
    return np.array(col.to_pylist(), dtype=np.float64)


def _with_matrix(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    idx = table.schema.get_field_index(name)
    typ = table.schema.field(idx).type
    return table.set_column(idx, name, pa.array(values.astype(np.float32).tolist(), type=typ))


def odd_episodes(files: list[Path], threshold: float) -> tuple[set[int], float]:
    """Episodes with a low mean gripper action, and the median gripper state of the others."""
    low: set[int] = set()
    good_state: list[np.ndarray] = []
    for f in files:
        t = pq.read_table(f)
        ep = np.array(t["episode_index"])
        action, state = _matrix(t["action"]), _matrix(t["observation.state"])
        for e in sorted(set(ep.tolist())):
            m = ep == e
            if action[m, GRIP].mean() < threshold:
                low.add(e)
            else:
                good_state.append(state[m, GRIP])
    if not good_state:
        raise SystemExit("every episode is below the threshold; nothing to copy the state value from")
    return low, float(np.median(np.concatenate(good_state)))


def fix_gripper(src: Path, dst: Path, threshold: float = 60.0, hold: float | None = None) -> dict:
    hold = GRIPPER_HOLD if hold is None else hold
    if hold is None:
        raise SystemExit("GRIPPER_HOLD is None; pass a value")
    src, dst = src.resolve(), dst.resolve()
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} is not empty")
    files = sorted((src / "data").rglob("*.parquet"))
    low, state_hold = odd_episodes(files, threshold)

    columns: dict[str, list[np.ndarray]] = {"action": [], "observation.state": []}
    for f in files:
        t = pq.read_table(f)
        ep = np.array(t["episode_index"])
        mask = np.isin(ep, list(low))
        for name, value in (("action", hold), ("observation.state", state_hold)):
            mat = _matrix(t[name])
            mat[mask, GRIP] = value
            t = _with_matrix(t, name, mat)
            columns[name].append(mat)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)

    # Everything else is copied: meta (patched below), videos (linked), any other file.
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_dir() or rel.parts[0] == "data":
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts[0] == "videos":
            try:
                os.link(f, out)
                continue
            except OSError:
                pass
        shutil.copy2(f, out)

    # Per-episode stats: the gripper channel is constant in a patched episode.
    for f in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f)
        eps = np.array(t["episode_index"])
        for name, value in (("action", hold), ("observation.state", state_hold)):
            for stat in ("min", "max", "mean", "std", *QUANTILES):
                col = f"stats/{name}/{stat}"
                if col not in t.column_names:
                    continue
                rows = [list(r) for r in t[col].to_pylist()]
                for i, e in enumerate(eps.tolist()):
                    if e in low:
                        rows[i][GRIP] = 0.0 if stat == "std" else value
                idx = t.schema.get_field_index(col)
                t = t.set_column(idx, col, pa.array(rows, type=t.schema.field(idx).type))
        pq.write_table(t, f)

    # Global stats from the patched data.
    stats_path = dst / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        for name, parts in columns.items():
            grip = np.concatenate(parts)[:, GRIP]
            entry = stats.get(name, {})
            for stat, value in (
                ("min", grip.min()),
                ("max", grip.max()),
                ("mean", grip.mean()),
                ("std", grip.std()),
                *((q, np.quantile(grip, p)) for q, p in QUANTILES.items()),
            ):
                if stat in entry:
                    entry[stat][GRIP] = float(value)
        stats_path.write_text(json.dumps(stats, indent=4))
    return {"episodes": sorted(low), "action_gripper": hold, "state_gripper": state_hold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--threshold", type=float, default=60.0)
    args = parser.parse_args()
    result = fix_gripper(Path(args.src), Path(args.dst), args.threshold)
    print(f"patched episodes {result['episodes']}: action gripper -> {result['action_gripper']}, "
          f"state gripper -> {result['state_gripper']:.1f}")


if __name__ == "__main__":
    main()
