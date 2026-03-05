#!/usr/bin/env python3
"""
Compute rot_offset entries for solarxr_to_g1.json using rest-pose SolarXR data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import ROBOT_XML_DICT

Position = Tuple[float, float, float]


XR_TO_MJ = np.array(
    [
        [0.0, 0.0, -1.0],  # x_fwd = -z
        [-1.0, 0.0, 0.0],  # y_left = -x
        [0.0, 1.0, 0.0],  # z_up = y
    ],
    dtype=float,
)

ALIAS_MAP: Dict[str, List[str]] = {
    "hip": ["hip", "waist"],
    "chest": ["chest", "upper_chest"],
    "waist": ["waist", "upper_chest", "chest"],
    "upper_chest": ["upper_chest", "chest"],
    "left_hip": ["left_hip", "left_upper_leg"],
    "right_hip": ["right_hip", "right_upper_leg"],
    "left_upper_arm": ["left_upper_arm", "left_shoulder"],
    "right_upper_arm": ["right_upper_arm", "right_shoulder"],
    "left_hand": ["left_hand", "left_wrist"],
    "right_hand": ["right_hand", "right_wrist"],
    "left_foot_tail": ["left_foot_tail", "left_foot"],
    "right_foot_tail": ["right_foot_tail", "right_foot"],
}


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
        if (path / "solarxr_world.py").exists():
            return path
    raise FileNotFoundError(
        "solarxr_world.py not found. Pass --solarxr-root pointing to XRoboToolkit-PC-Service-Pybind "
        "or to its examples/solarxr directory."
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
                "xrobotoolkit_sdk is not installed. Install it inside your conda env from "
                "XRoboToolkit-PC-Service-Pybind (see README.md)."
            ) from exc
        raise
    return SolarXRWorld


def _quat_xr_to_mj_wxyz(quat_xyzw: Tuple[float, float, float, float]) -> np.ndarray:
    rot_xr = R.from_quat(quat_xyzw)  # xyzw
    rot_mj = R.from_matrix(XR_TO_MJ @ rot_xr.as_matrix() @ XR_TO_MJ.T)
    q_xyzw = rot_mj.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _collect_human_quats(
    world,
    names: Iterable[str],
    samples: int,
    sleep_s: float,
    timeout_s: float,
) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    missing = set(names)
    start = time.time()
    while len(missing) > 0:
        bones = world.get_world_bones()
        if not bones:
            time.sleep(0.01)
            continue
        for name in list(missing):
            candidates = ALIAS_MAP.get(name, [name])
            for cand in candidates:
                entry = bones.get(cand)
                if entry is None:
                    continue
                rot = entry.get("rot")
                if rot is None:
                    continue
                out[name].append(_quat_xr_to_mj_wxyz(tuple(rot)))
                if len(out[name]) >= 1:
                    missing.discard(name)
                break
        if missing and (time.time() - start) > timeout_s:
            print(f"[solarxr_calibrate] timeout waiting for bones: {sorted(missing)}")
            break
        time.sleep(0.01)

    # Collect remaining samples
    for _ in range(max(0, samples - 1)):
        bones = world.get_world_bones()
        if not bones:
            time.sleep(0.01)
            continue
        for name in names:
            entry = bones.get(name)
            if entry is None:
                continue
            rot = entry.get("rot")
            if rot is None:
                continue
            out[name].append(_quat_xr_to_mj_wxyz(tuple(rot)))
        time.sleep(sleep_s)
    return out


def _mean_quat_wxyz(quats_wxyz: List[np.ndarray]) -> np.ndarray:
    if not quats_wxyz:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quats_xyzw = np.array([[q[1], q[2], q[3], q[0]] for q in quats_wxyz])
    rot = R.from_quat(quats_xyzw).mean()
    q_xyzw = rot.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _update_rot_offsets(config: dict, offsets: Dict[str, np.ndarray]) -> None:
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = config.get(table_name, {})
        for frame_name, entry in table.items():
            if frame_name in offsets:
                q = offsets[frame_name]
                entry[4] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
                table[frame_name] = entry
        config[table_name] = table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rot_offset for SolarXR -> G1.")
    parser.add_argument("--solarxr-root", type=str, default=None)
    parser.add_argument("--solar-url", type=str, default="ws://127.0.0.1:21110")
    parser.add_argument("--minimum-ms", type=int, default=20)
    parser.add_argument("--reset-hold-s", type=float, default=0.5)
    parser.add_argument("--robot", type=str, default="unitree_g1_27dof")
    parser.add_argument("--ik-config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--robot-qpos",
        action="append",
        default=[],
        help="Override robot joint qpos (e.g. --robot-qpos left_elbow=0.0). Can be repeated.",
    )
    parser.add_argument("--list-joints", action="store_true", help="Print robot joint names and exit.")
    args = parser.parse_args()

    config_path = Path(args.ik_config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Collect frame->human mappings (prefer table1, fallback to table2)
    entries = {}
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = config.get(table_name, {})
        for frame_name, entry in table.items():
            if frame_name not in entries:
                entries[frame_name] = entry[0]

    human_names = sorted(set(entries.values()))

    # Load robot model and get rest-pose body orientations
    xml_path = ROBOT_XML_DICT[args.robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    if args.list_joints:
        for i in range(model.njnt):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            if name:
                print(name)
        return

    for item in args.robot_qpos:
        if "=" not in item:
            raise ValueError(f"Invalid --robot-qpos '{item}', expected joint=value")
        joint_name, value_str = item.split("=", 1)
        joint_name = joint_name.strip()
        value = float(value_str.strip())
        jid = model.joint(joint_name).id
        qpos_adr = model.jnt_qposadr[jid]
        data.qpos[qpos_adr] = value
    mj.mj_forward(model, data)

    robot_rots: Dict[str, R] = {}
    for frame_name in entries.keys():
        try:
            bid = model.body(frame_name).id
        except KeyError:
            continue
        mat = data.xmat[bid].reshape(3, 3)
        robot_rots[frame_name] = R.from_matrix(mat)

    solarxr_path = _resolve_solarxr_path(args.solarxr_root)
    SolarXRWorld = _import_solarxr_world(solarxr_path)
    world = SolarXRWorld(
        solar_url=args.solar_url,
        minimum_ms=args.minimum_ms,
        reset_hold_s=args.reset_hold_s,
    )
    world.start()

    print("[solarxr_calibrate] Hold rest pose (arms down), then press Enter to capture...")
    input()

    try:
        human_quats = _collect_human_quats(world, human_names, args.samples, args.sleep, args.timeout_s)
    finally:
        world.stop()

    offsets: Dict[str, np.ndarray] = {}
    for frame_name, human_name in entries.items():
        if frame_name not in robot_rots or not human_quats.get(human_name):
            continue
        human_mean = _mean_quat_wxyz(human_quats[human_name])
        human_rot = R.from_quat([human_mean[1], human_mean[2], human_mean[3], human_mean[0]])
        robot_rot = robot_rots[frame_name]
        offset_rot = human_rot.inv() * robot_rot
        q_xyzw = offset_rot.as_quat()
        offsets[frame_name] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

    _update_rot_offsets(config, offsets)

    out_path = Path(args.output) if args.output else config_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    print(f"[solarxr_calibrate] Wrote rot_offset values to {out_path}")


if __name__ == "__main__":
    main()
