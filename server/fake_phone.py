"""Tiny phone stand-in. Sends one prompt and prints every server message.

  python server/fake_phone.py
  python server/fake_phone.py --url ws://localhost:8000/ws --text "record a 10 second video of this apple"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--text", default="record a 15 second video of this apple")
    args = parser.parse_args()

    async with websockets.connect(args.url) as ws:
        await ws.send(json.dumps({"type": "prompt", "text": args.text}))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                print(raw)
                message = json.loads(raw)
                if message.get("type") == "result":
                    return
                if message.get("stage") == "ERROR":
                    return
        except asyncio.TimeoutError:
            print("timed out waiting for the next message")


if __name__ == "__main__":
    asyncio.run(main())
