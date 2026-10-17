"""Register orbit demos and start an Isaac 0.5 fine-tune.

Demos are recorded with nothing in the gripper, at GRIPPER_HOLD, with the
wrist camera kept on the apple the whole orbit. That shot is the product.

Prefer a public Hugging Face repo. Direct upload is reserve → PUT signed
URLs (no team key) → finalize, then poll until the dataset state is ready.
Train only on ready.

Job body for the first run: standard, expert_only, horizon 30, RTC on,
1000 updates, with an Idempotency-Key. The default team quota is two
successful jobs, so the 10k final run has to be the other one. Read
capabilities before assuming 10000 is accepted.

Usage:
  python -m ml.upload_and_train --steps 1000
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload orbit demos and start an Isaac fine-tune.")
    parser.add_argument("--steps", type=int, required=True, help="1000 for the check, 10000 only if capabilities and quota allow")
    parser.add_argument("--dataset", default="orbit-apple")
    args = parser.parse_args()
    raise SystemExit(
        "Not implemented. "
        f"dataset={args.dataset} steps={args.steps} "
        "backend=standard mode=expert_only horizon=30 rtc=on. "
        "Poll until the dataset is ready, then POST the job and log status and loss."
    )


if __name__ == "__main__":
    main()
