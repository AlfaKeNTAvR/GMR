#!/usr/bin/env python3
"""
Realtime SolarXR -> GMR parser for G1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

Position = Tuple[float, float, float]
Rotation = Tuple[float, float, float, float]  # wxyz


XR_TO_MJ = np.array(
    [
        [0.0, 0.0, -1.0],  # x_fwd = -z
        [-1.0, 0.0, 0.0],  # y_left = -x
        [0.0, 1.0, 0.0],  # z_up = y
    ],
    dtype=float,
)


def _resolve_solarxr_path(user_path: Optional[str]) -> Path:
    if user_path:
        base = Path(user_path).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parents[1]
        base = (repo_root.parent / "XRoboToolkit-PC-Service-Pybind").resolve()

    candidates = [
        base / "examples" / "solarxr",
        base,
    ]
    for path in candidates:
        if (path / "solarxr_client.py").exists():
            return path
    raise FileNotFoundError(
        "solarxr_client.py not found. Pass --solarxr-root pointing to XRoboToolkit-PC-Service-Pybind "
        "or to its examples/solarxr directory."
    )


def _import_solarxr_client(solarxr_path: Path):
    sys.path.insert(0, str(solarxr_path))
    from solarxr_client import SolarXRClient  # type: ignore

    return SolarXRClient


def _import_xr_bridge(solarxr_path: Path):
    sys.path.insert(0, str(solarxr_path))
    xr_path = solarxr_path.parent / "xr"
    if xr_path.exists():
        sys.path.insert(0, str(xr_path))
    from xr_bridge_sender import XRBridgeSender  # type: ignore

    return XRBridgeSender


def _pos_xr_to_mj(pos: Position) -> np.ndarray:
    return XR_TO_MJ @ np.asarray(pos, dtype=float)


def _quat_xr_to_mj_wxyz(quat_xyzw: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    if not quat_xyzw:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    rot_xr = R.from_quat(quat_xyzw)  # xyzw
    rot_mj = R.from_matrix(XR_TO_MJ @ rot_xr.as_matrix() @ XR_TO_MJ.T)
    q_xyzw = rot_mj.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


ALIAS_MAP: Dict[str, List[str]] = {
    "hip": ["hip", "waist"],
    "chest": ["chest", "upper_chest"],
    "upper_chest": ["upper_chest", "chest"],
    "left_hip": ["left_hip", "left_upper_leg"],
    "right_hip": ["right_hip", "right_upper_leg"],
    "left_lower_leg": ["left_lower_leg"],
    "right_lower_leg": ["right_lower_leg"],
    "left_foot_tail": ["left_foot_tail", "left_foot"],
    "right_foot_tail": ["right_foot_tail", "right_foot"],
    "left_shoulder": ["left_shoulder"],
    "right_shoulder": ["right_shoulder"],
    "left_lower_arm": ["left_lower_arm"],
    "right_lower_arm": ["right_lower_arm"],
    "left_hand": ["left_hand", "left_wrist"],
    "right_hand": ["right_hand", "right_wrist"],
}


def _get_bone(
    bones: Dict[str, Dict[str, object]],
    names: Iterable[str],
) -> Optional[Tuple[Position, Optional[Tuple[float, float, float, float]]]]:
    for name in names:
        entry = bones.get(name)
        if not entry:
            continue
        head = entry.get("head")
        if head is None:
            continue
        rot = entry.get("rot")
        return (tuple(head), tuple(rot) if rot is not None else None)
    return None


def _build_human_data(
    bones: Dict[str, Dict[str, object]],
    required: Iterable[str],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str]]:
    human_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    missing: List[str] = []
    for name in required:
        candidates = ALIAS_MAP.get(name, [name])
        bone = _get_bone(bones, candidates)
        if bone is None:
            missing.append(name)
            continue
        pos_xr, rot_xyzw = bone
        pos_mj = _pos_xr_to_mj(pos_xr)
        quat_wxyz = _quat_xr_to_mj_wxyz(rot_xyzw)
        human_data[name] = (pos_mj, quat_wxyz)
    return human_data, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="SolarXR -> GMR realtime parser.")
    parser.add_argument("--solarxr-root", type=str, default=None)
    parser.add_argument("--solar-url", type=str, default="ws://127.0.0.1:21110")
    parser.add_argument("--minimum-ms", type=int, default=20)
    parser.add_argument("--reset-hold-s", type=float, default=0.5)
    parser.add_argument("--robot", type=str, default="unitree_g1_27dof")
    parser.add_argument("--output", choices=["viewer", "stdout", "none"], default="viewer")
    parser.add_argument("--rate-limit", action="store_true")
    parser.add_argument("--show-human", action="store_true")
    parser.add_argument("--show-frames", action="store_true")
    parser.add_argument("--frame-size", type=float, default=0.15)
    parser.add_argument("--print-fps", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    solarxr_path = _resolve_solarxr_path(args.solarxr_root)
    SolarXRClient = _import_solarxr_client(solarxr_path)
    XRBridgeSender = _import_xr_bridge(solarxr_path)

    bridge = XRBridgeSender(
        solar_url=args.solar_url,
        reset_hold_s=args.reset_hold_s,
    )
    bridge.start()

    client = SolarXRClient(
        url=args.solar_url,
        minimum_ms=args.minimum_ms,
    )
    client.start()

    retarget = GMR(
        src_human="solarxr",
        tgt_robot=args.robot,
        actual_human_height=None,
        verbose=args.verbose,
    )

    required = set(retarget.human_body_to_task1.keys()) | set(retarget.human_body_to_task2.keys())

    viewer = None
    robot_body_frames = None
    if args.output == "viewer":
        viewer = RobotMotionViewer(robot_type=args.robot)
        if args.show_frames:
            robot_body_frames = sorted(set(retarget.ik_match_table1.keys()) | set(retarget.ik_match_table2.keys()))

    fps_counter = 0
    fps_start = time.time()
    fps_interval = 2.0
    last_missing_report = 0.0

    try:
        while True:
            bridge.tick()
            bones = client.get_raw_bones()
            if not bones:
                time.sleep(0.001)
                continue

            human_data, missing = _build_human_data(bones, required)
            if missing:
                if args.verbose and (time.time() - last_missing_report) > 1.0:
                    print(f"[solarxr_parser] missing bones: {sorted(missing)}")
                    last_missing_report = time.time()
                continue

            qpos = retarget.retarget(human_data)

            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data if args.show_human else None,
                    robot_body_frames=robot_body_frames,
                    robot_frame_size=args.frame_size,
                    rate_limit=args.rate_limit,
                    follow_camera=False,
                )

            if args.output == "stdout":
                print(" ".join(f"{v:.6f}" for v in qpos))

            if args.print_fps:
                fps_counter += 1
                now = time.time()
                if now - fps_start >= fps_interval:
                    fps = fps_counter / (now - fps_start)
                    print(f"[solarxr_parser] fps={fps:.2f}")
                    fps_counter = 0
                    fps_start = now
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        client.stop()
        bridge.stop()


if __name__ == "__main__":
    main()
