"""Register demos with the Robotics API and start an Isaac 0.5 fine-tune.

  python -m ml.upload_and_train check [DIR]            preflight + live capabilities, writes nothing
  python -m ml.upload_and_train upload DIR --name N    register, upload, finalize, wait for ready
  python -m ml.upload_and_train train --steps 1000 --yes
  python -m ml.upload_and_train deployments            wait for inference_ready, print the ids

The team key is read from team.key (mode 600) and only ever sent to the control
API. It is never an argument, never printed, and never sent to a signed upload
URL. Every command can be rerun: progress is kept in ml/.train_state.json.

Quota is two successful jobs. `train` reads the allowed step counts from
GET /v1/capabilities (1000 at the time of writing) and refuses anything else.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

import requests

from ml.check_dataset import check_dataset, dataset_files
from shared.robotics_api import BASE_URL, USER_AGENT, Api, ApiError, read_key  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]

STATE_FILE = ROOT / "ml" / ".train_state.json"
BATCH_BYTES = 150 * 1024 * 1024  # keep each reservation well inside the 15 minute URL life
POLL_S = 10
FAILED = {"failed", "error", "rejected", "invalid", "cancelled", "canceled"}


def load_state() -> dict:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {"datasets": {}, "jobs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path) -> list[dict]:
    return [
        {"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in dataset_files(root)
    ]


def content_hash(manifest: list[dict]) -> str:
    lines = "\n".join(f"{m['path']}:{m['sha256']}" for m in manifest)
    return hashlib.sha256(lines.encode()).hexdigest()


def batches(manifest: list[dict], limit: int = BATCH_BYTES) -> list[list[dict]]:
    out: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for item in manifest:
        if current and size + item["size_bytes"] > limit:
            out.append(current)
            current, size = [], 0
        current.append(item)
        size += item["size_bytes"]
    if current:
        out.append(current)
    return out


def extract_grants(reply) -> dict[str, dict]:
    """Map path -> {url, method, headers} from a reserve reply, tolerating key names."""
    items = None
    if isinstance(reply, list):
        items = reply
    elif isinstance(reply, dict):
        for value in reply.values():
            if isinstance(value, list) and value and isinstance(value[0], dict) and "path" in value[0]:
                items = value
                break
    if not items:
        shape = sorted(reply) if isinstance(reply, dict) else type(reply).__name__
        raise ApiError(f"could not find the upload list in the reserve reply (keys: {shape})")
    grants = {}
    for item in items:
        url = item.get("url") or item.get("upload_url") or item.get("signed_url")
        if not url:
            raise ApiError(f"upload entry for {item.get('path')} has no url (keys: {sorted(item)})")
        grants[item["path"]] = {
            "url": urljoin(BASE_URL + "/", url),
            "method": (item.get("method") or "PUT").upper(),
            "headers": item.get("headers") or {},
        }
    return grants


def put_file(path: Path, grant: dict) -> None:
    """Signed URL only: no team key, only the headers the reply named."""
    with path.open("rb") as handle:
        resp = requests.request(
            grant["method"], grant["url"], data=handle, headers=grant["headers"], timeout=(30, 900)
        )
    if resp.status_code >= 300:
        raise ApiError(f"upload of {path.name} -> {resp.status_code}: {resp.text[:200]}")


def state_of(doc: dict) -> str:
    return str(doc.get("state") or doc.get("status") or "unknown").lower()


def wait_for(fetch, good: set[str], what: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last = None
    while True:
        doc = fetch()
        state = state_of(doc)
        if state != last:
            print(f"{time.strftime('%H:%M:%S')} {what}: {state}")
            last = state
        if state in good:
            return doc
        if state in FAILED:
            raise SystemExit(f"{what} ended as {state}: {json.dumps(doc)[:600]}")
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out waiting for {what} (last state {state})")
        time.sleep(POLL_S)


def cmd_check(args) -> None:
    if args.dir:
        rep = check_dataset(args.dir)
        for w in rep.warnings:
            print("WARN ", w)
        for e in rep.errors:
            print("ERROR", e)
        print("preflight:", "OK" if rep.ok else "FAILED")
    caps = Api(read_key()).call("GET", "/v1/capabilities")
    print(json.dumps(caps, indent=2)[:4000])


def cmd_upload(args) -> None:
    root = Path(args.dir).expanduser().resolve()
    rep = check_dataset(root)
    for w in rep.warnings:
        print("WARN ", w)
    if not rep.ok:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(rep.errors))
    api = Api(read_key())
    state = load_state()
    print("hashing files...")
    manifest = build_manifest(root)
    digest = content_hash(manifest)
    entry = state["datasets"].get(args.name)
    if entry and entry.get("content_hash") != digest:
        raise SystemExit(f"dataset name {args.name!r} already holds different files. Use a new --name.")
    key = f"dataset-{args.name}-{digest[:16]}"
    if not entry:
        reply = api.call("POST", "/v1/datasets", idem=key, body={"name": args.name, "source_kind": "upload"})
        entry = {"id": reply["id"], "content_hash": digest, "finalized": False}
        state["datasets"][args.name] = entry
        save_state(state)
    ds = entry["id"]
    print(f"dataset {ds}")

    if not entry["finalized"]:
        done: set[str] = set(entry.get("uploaded", []))
        for index, batch in enumerate(batches([m for m in manifest if m["path"] not in done])):
            for attempt in range(3):
                reply = api.call("POST", f"/v1/datasets/{ds}/uploads", idem=f"{key}-b{index}-a{attempt}", body={"files": batch})
                grants = extract_grants(reply)
                try:
                    for item in batch:
                        put_file(root / item["path"], grants[item["path"]])
                        done.add(item["path"])
                        entry["uploaded"] = sorted(done)
                        save_state(state)
                        print(f"uploaded {item['path']} ({item['size_bytes'] / 1e6:.1f} MB)")
                    break
                except (ApiError, requests.RequestException) as exc:
                    print(f"upload failed ({exc}); reserving fresh URLs")
                    batch = [m for m in batch if m["path"] not in done]
            else:
                raise SystemExit("uploads kept failing; rerun the command to resume")
        api.call("POST", f"/v1/datasets/{ds}/finalize")
        entry["finalized"] = True
        save_state(state)
        print("finalized")

    wait_for(lambda: api.call("GET", f"/v1/datasets/{ds}"), {"ready"}, "dataset", args.timeout_min * 60)
    print(json.dumps(api.call("GET", f"/v1/datasets/{ds}/validation"), indent=2)[:3000])
    entry["ready"] = True
    save_state(state)


def cmd_train(args) -> None:
    if not args.yes:
        raise SystemExit("Quota is two successful jobs. Rerun with --yes to start this one.")
    state = load_state()
    name = args.name or next(reversed(state["datasets"]), None)
    entry = state["datasets"].get(name or "")
    if not entry:
        raise SystemExit("no uploaded dataset in ml/.train_state.json; run upload first")
    api = Api(read_key())
    ds = entry["id"]
    doc = api.call("GET", f"/v1/datasets/{ds}")
    if state_of(doc) != "ready":
        raise SystemExit(f"dataset is {state_of(doc)}, not ready. Train only on ready.")
    caps = api.call("GET", "/v1/capabilities")
    print("capabilities:", json.dumps(caps)[:1500])
    offered = caps.get("steps") or []
    if args.steps not in offered:
        raise SystemExit(f"steps={args.steps} is not offered. The API allows {offered}.")

    body = {
        "dataset_id": ds,
        "backend": "standard",
        "mode": "expert_only",
        "action_horizon": 30,
        "rtc_enabled": True,
        "steps": args.steps,
    }
    job_key = f"{ds}-{args.steps}"
    job = state["jobs"].get(job_key)
    if not job:
        # Saved before the POST: a retry reuses the key and returns the same job.
        job = {"idem": f"job-{uuid.uuid4()}", "dataset": name, "steps": args.steps}
        state["jobs"][job_key] = job
        save_state(state)
    if "id" not in job:
        job["id"] = api.call("POST", "/v1/jobs", idem=job["idem"], body=body)["id"]
        save_state(state)
    print(f"job {job['id']}")

    def fetch():
        doc = api.call("GET", f"/v1/jobs/{job['id']}")
        extra = {k: doc[k] for k in ("step", "steps_done", "progress", "loss", "train_loss") if k in doc}
        if extra:
            print("  ", extra)
        return doc

    wait_for(fetch, {"succeeded", "completed", "complete", "done", "success"}, "job", args.timeout_min * 60)
    print("training finished. Export and inference readiness come later: run `deployments`.")


def cmd_deployments(args) -> None:
    api = Api(read_key())

    def fetch() -> dict:
        reply = api.call("GET", "/v1/deployments")
        items = reply if isinstance(reply, list) else next((v for v in reply.values() if isinstance(v, list)), [])
        ready = [d for d in items if state_of(d) == "inference_ready"]
        return ready[-1] if ready else {"state": "waiting"}

    dep = wait_for(fetch, {"inference_ready"}, "deployment", args.timeout_min * 60)
    dep_id = dep.get("id") or dep.get("deployment_id")
    contract = api.call("GET", f"/v1/deployments/{dep_id}/contract")
    print(f"DEPLOYMENT_ID={dep_id}")
    print(f"contract_id={contract.get('contract_id')}")
    print("Put DEPLOYMENT_ID in .env. Use the contract_id on every session and predict.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("dir", nargs="?")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("upload")
    p.add_argument("dir")
    p.add_argument("--name", required=True)
    p.add_argument("--timeout-min", type=float, default=45)
    p.set_defaults(fn=cmd_upload)
    p = sub.add_parser("train")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--name")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--timeout-min", type=float, default=120)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("deployments")
    p.add_argument("--timeout-min", type=float, default=60)
    p.set_defaults(fn=cmd_deployments)
    args = parser.parse_args()
    try:
        args.fn(args)
    except ApiError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
