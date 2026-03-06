#!/usr/bin/env python3
"""
Grab one frame of SolarXR world bones and save to a JSON snapshot.
Run this right after SlimeVR reset, while trackers are still on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


def _resolve_solarxr_path(user_path: Optional[str]) -> Path:
    if user_path:
        base = Path(user_path).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parents[1]
        base = (repo_root.parent / "XRoboToolkit-PC-Service-Pybind").resolve()

    candidates = [base / "examples" / "solarxr", base]
    for path in candidates:
        if (path / "solarxr_client.py").exists():
            return path
    raise FileNotFoundError(
        "solarxr_client.py not found. Pass --solarxr-root pointing to "
        "XRoboToolkit-PC-Service-Pybind or its examples/solarxr directory."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a SolarXR bones snapshot to JSON.")
    parser.add_argument("--solarxr-root", type=str, default=None)
    parser.add_argument("--solar-url", type=str, default="ws://127.0.0.1:21110")
    parser.add_argument("--minimum-ms", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="solarxr_snapshot.json",
        help="Output JSON file path (default: solarxr_snapshot.json)",
    )
    args = parser.parse_args()

    solarxr_path = _resolve_solarxr_path(args.solarxr_root)
    sys.path.insert(0, str(solarxr_path))
    from solarxr_client import SolarXRClient  # type: ignore

    client = SolarXRClient(url=args.solar_url, minimum_ms=args.minimum_ms)
    client.start()

    print("[snapshot] Waiting for bones...", flush=True)
    start = time.time()
    bones = None
    try:
        while True:
            bones = client.get_raw_bones()
            if bones:
                break
            if time.time() - start > args.timeout_s:
                print("[snapshot] Timeout — no data received.")
                return
            time.sleep(0.01)
    finally:
        client.stop()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        name: {k: list(v) if hasattr(v, "__iter__") else v for k, v in entry.items()}
        for name, entry in bones.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")

    print(f"[snapshot] Saved {len(snapshot)} bones to {out_path}")
    print(f"[snapshot] Bones: {sorted(snapshot.keys())}")


if __name__ == "__main__":
    main()
