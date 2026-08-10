"""Run the web console.

    python scripts/serve.py            # http://127.0.0.1:8000
    python scripts/serve.py --port 9000

Binds to localhost only. The API asserts identity from the request rather than
authenticating it, so exposing it on a network interface would let anyone
choose their own role — which is the point of the scope note in the README, not
an oversight to be fixed by binding wider.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

UNDERSTANDING = Path("data/understanding")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if not (UNDERSTANDING / "facts.db").exists():
        print("Run `python scripts/build_understanding.py` first.",
              file=sys.stderr)
        return 1

    print(f"Console: http://127.0.0.1:{args.port}")
    uvicorn.run("src.api:app", host="127.0.0.1", port=args.port,
                reload=args.reload, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
