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
        if (path / "solarxr_world.py").exists():
            return path
    raise FileNotFoundError(
        "solarxr_world.py not found. Pass --solarxr-root pointing to "
        "XRoboToolkit-PC-Service-Pybind or its examples/solarxr directory."
    )


def _import_solarxr_world(solarxr_path: Path):
    sys.path.insert(0, str(solarxr_path))
    xr_path = solarxr_path.parent / "xr"
    if xr_path.exists():
        sys.path.insert(0, str(xr_path))
    try:
        from solarxr_world import SolarXRWorld  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "xrobotoolkit_sdk":
            raise RuntimeError(
                "xrobotoolkit_sdk is not installed. Install it from "
                "XRoboToolkit-PC-Service-Pybind (see README.md)."
            ) from exc
        raise
    return SolarXRWorld


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a SolarXR world-bones snapshot to JSON.")
    parser.add_argument("--solarxr-root", type=str, default=None)
    parser.add_argument("--solar-url", type=str, default="ws://127.0.0.1:21110")
    parser.add_argument("--minimum-ms", type=int, default=20)
    parser.add_argument("--reset-hold-s", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="solarxr_snapshot.json",
        help="Output JSON file path (default: solarxr_snapshot.json)",
    )
    args = parser.parse_args()

    solarxr_path = _resolve_solarxr_path(args.solarxr_root)
    SolarXRWorld = _import_solarxr_world(solarxr_path)

    world = SolarXRWorld(
        solar_url=args.solar_url,
        minimum_ms=args.minimum_ms,
        reset_hold_s=args.reset_hold_s,
    )
    world.start()

    print("[snapshot] Waiting for bones...", flush=True)
    start = time.time()
    bones = None
    try:
        while True:
            bones = world.get_world_bones()
            if bones:
                break
            if time.time() - start > args.timeout_s:
                print("[snapshot] Timeout — no data received.")
                return
            time.sleep(0.01)
    finally:
        world.stop()

    out_path = Path(args.output)
    # world bones values may contain non-serialisable types; coerce to plain lists
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
