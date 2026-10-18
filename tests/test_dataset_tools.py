import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.check_dataset import check_dataset
from ml.upload_and_train import ApiError, batches, content_hash, extract_grants
from shared.constants import SHOTS

TASK = SHOTS["zoom"]["vla_prompt"]
JOINTS = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


def make(tmp_path, fps=30, task=TASK, grip=(85.0, 85.0), n=30, tasks=None):
    info = {
        "fps": fps,
        "total_episodes": 2,
        "total_frames": 2 * n,
        "features": {
            "action": {"names": JOINTS},
            "observation.state": {"names": JOINTS},
            "observation.images.side": {"shape": [480, 640, 3]},
            "observation.images.wrist": {"shape": [480, 640, 3]},
        },
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    names = tasks or [task]
    pq.write_table(
        pa.table({"task_index": list(range(len(names))), "__index_level_0__": names}),
        tmp_path / "meta" / "tasks.parquet",
    )
    for cam in ("side", "wrist"):
        d = tmp_path / "videos" / f"observation.images.{cam}" / "chunk-000"
        d.mkdir(parents=True)
        (d / "file-000.mp4").write_bytes(b"x")
    ep = np.repeat([0, 1], n)
    ts = np.tile(np.arange(n) / 30, 2).astype("float32")
    action = [[0, 0, 0, 0, 0, grip[e]] for e in ep]
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": ep, "timestamp": ts, "action": action, "task_index": ep % len(names)}),
        tmp_path / "data" / "chunk-000" / "file-000.parquet",
    )
    return tmp_path


def test_good_dataset(tmp_path):
    rep = check_dataset(make(tmp_path))
    assert rep.ok, rep.errors
    assert rep.facts["episodes_per_task"] == {TASK: 2}
    assert any("only 2 episodes" in w for w in rep.warnings)


def test_wrong_fps_and_task(tmp_path):
    rep = check_dataset(make(tmp_path, fps=25, task="orbit"))
    assert any("fps" in e for e in rep.errors)
    assert any("not in SHOTS" in e for e in rep.errors)


def test_mixed_gripper_warns(tmp_path):
    rep = check_dataset(make(tmp_path, grip=(41.0, 97.0)))
    assert rep.ok
    assert any("gripper" in w for w in rep.warnings)


def test_batches_and_hash():
    m = [{"path": str(i), "size_bytes": 60, "sha256": "a" * 64} for i in range(5)]
    assert [len(b) for b in batches(m, 100)] == [1, 1, 1, 1, 1]
    assert [len(b) for b in batches(m, 130)] == [2, 2, 1]
    assert content_hash(m) == content_hash(list(m))


def test_extract_grants_shapes():
    reply = {"uploads": [{"path": "a", "url": "/v1/uploads/x", "headers": {"Content-Length": "1"}}]}
    g = extract_grants(reply)
    assert g["a"]["method"] == "PUT" and g["a"]["url"].endswith("/v1/uploads/x")
    with pytest.raises(ApiError):
        extract_grants({"nothing": 1})


def test_two_tasks_counted_when_both_known(tmp_path):
    other = "make a sweep from above"
    reg = {TASK: "zoom", other: "sweep"}
    rep = check_dataset(make(tmp_path, tasks=[TASK, other]), reg)
    assert rep.ok, rep.errors
    assert rep.facts["episodes_per_task"] == {TASK: 1, other: 1}


def test_unknown_second_task_is_an_error(tmp_path):
    rep = check_dataset(make(tmp_path, tasks=[TASK, "some new sentence"]))
    assert any("some new sentence" in e for e in rep.errors)
