import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ml.fix_gripper import fix_gripper


def build(root):
    (root / "meta").mkdir(parents=True)
    (root / "videos" / "cam" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "cam" / "chunk-000" / "file-000.mp4").write_bytes(b"x")
    (root / "data" / "chunk-000").mkdir(parents=True)
    ep = np.repeat([0, 1, 2], 10)
    grip = {0: 41.0, 1: 97.0, 2: 97.2}
    action = [[1.0, 2, 3, 4, 5, grip[e]] for e in ep]
    state = [[1.0, 2, 3, 4, 5, grip[e] - 0.5] for e in ep]
    pq.write_table(
        pa.table({"episode_index": ep, "action": action, "observation.state": state}),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps({"action": {"min": [0] * 6, "max": [0] * 6, "mean": [0] * 6, "std": [0] * 6, "q01": [0] * 6}})
    )


def test_odd_episode_is_patched_and_source_untouched(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    build(src)
    result = fix_gripper(src, dst, hold=97.1)
    assert result["episodes"] == [0]
    t = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
    action = np.array(t["action"].to_pylist())
    state = np.array(t["observation.state"].to_pylist())
    assert np.allclose(action[:10, 5], 97.1)
    assert np.allclose(action[10:, 5], [97.0] * 10 + [97.2] * 10)
    assert np.allclose(action[:, :5], [1, 2, 3, 4, 5])
    assert np.allclose(state[:10, 5], np.median(state[10:, 5]))
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    assert stats["action"]["min"][5] > 96
    assert (dst / "videos" / "cam" / "chunk-000" / "file-000.mp4").exists()
    assert np.allclose(np.array(pq.read_table(src / "data" / "chunk-000" / "file-000.parquet")["action"].to_pylist())[:10, 5], 41.0)
