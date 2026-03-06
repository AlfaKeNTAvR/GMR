#!/usr/bin/env python3
"""Compute human_scale_table for a GMR IK config from a SolarXR T-pose snapshot and robot XML.

For each human bone in the IK match tables, computes:
    scale = dist(robot_root → robot_body) / dist(human_root → human_bone)
both measured in their respective neutral/zero-angle poses.

Usage:
    python3 scripts/compute_scales.py \\
        --snapshot snapshot.json \\
        --ik-config general_motion_retargeting/ik_configs/solarxr_to_g1_27dof.json \\
        --robot unitree_g1_27dof
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import mujoco as mj
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT

ALIAS_MAP: Dict[str, List[str]] = {
    "hip": ["hip", "waist"],
    "chest": ["chest", "upper_chest"],
    "left_hip": ["left_hip", "left_upper_leg"],
    "right_hip": ["right_hip", "right_upper_leg"],
    "left_lower_leg": ["left_lower_leg"],
    "right_lower_leg": ["right_lower_leg"],
    "left_foot_tail": ["left_foot_tail", "left_foot"],
    "right_foot_tail": ["right_foot_tail", "right_foot"],
    "left_upper_arm": ["left_upper_arm", "left_shoulder"],
    "right_upper_arm": ["right_upper_arm", "right_shoulder"],
    "left_lower_arm": ["left_lower_arm"],
    "right_lower_arm": ["right_lower_arm"],
    "left_hand": ["left_hand", "left_wrist"],
    "right_hand": ["right_hand", "right_wrist"],
}


MIN_HUMAN_DIST = 0.05  # m; displacements below this are likely degenerate


def _get_bone_pos(snapshot: dict, bone_name: str) -> Optional[np.ndarray]:
    for cand in ALIAS_MAP.get(bone_name, [bone_name]):
        entry = snapshot.get(cand)
        if entry and entry.get("head") is not None:
            return np.array(entry["head"], dtype=float)
    return None


def _get_bone_pos_with_fallback(
    snapshot: dict, bone_name: str, root_pos: np.ndarray
) -> tuple[Optional[np.ndarray], str]:
    """Return (pos, source_name) for the bone with the largest displacement from root.

    Tries each alias in order, returns the first one whose displacement from
    root exceeds MIN_HUMAN_DIST. Falls back to the first available alias if
    none exceed the threshold.
    """
    first_pos: Optional[np.ndarray] = None
    first_name: str = bone_name
    for cand in ALIAS_MAP.get(bone_name, [bone_name]):
        entry = snapshot.get(cand)
        if entry is None or entry.get("head") is None:
            continue
        pos = np.array(entry["head"], dtype=float)
        if first_pos is None:
            first_pos = pos
            first_name = cand
        if float(np.linalg.norm(pos - root_pos)) >= MIN_HUMAN_DIST:
            return pos, cand
    return first_pos, first_name


def _estimate_floor_y(snapshot: dict) -> Optional[float]:
    """Return the lowest foot Y in SolarXR frame (+Y up) as the floor estimate."""
    ys = []
    for bone_name in ("left_foot_tail", "right_foot_tail", "left_foot", "right_foot"):
        pos = _get_bone_pos(snapshot, bone_name)
        if pos is not None:
            ys.append(float(pos[1]))
    return min(ys) if ys else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute human_scale_table from snapshot + robot XML.")
    parser.add_argument("--snapshot", "-s", required=True, help="SolarXR T-pose snapshot JSON")
    parser.add_argument("--ik-config", required=True, help="IK config JSON to update")
    parser.add_argument("--robot", default="unitree_g1_27dof")
    parser.add_argument("--output", "-o", default=None, help="Output JSON (default: overwrite --ik-config)")
    args = parser.parse_args()

    with open(args.snapshot) as f:
        snapshot = json.load(f)
    with open(args.ik_config) as f:
        ik_cfg = json.load(f)

    xml_path = ROBOT_XML_DICT[args.robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    human_root = ik_cfg["human_root_name"]
    robot_root = ik_cfg["robot_root_name"]

    human_root_pos = _get_bone_pos(snapshot, human_root)
    if human_root_pos is None:
        raise ValueError(f"Human root '{human_root}' not found in snapshot")

    robot_root_pos = data.xpos[model.body(robot_root).id].copy()

    # Collect unique human_bone -> robot_body pairs (first occurrence wins)
    pairs: Dict[str, str] = {}
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for robot_body, entry in ik_cfg.get(table_name, {}).items():
            human_bone = entry[0]
            if human_bone not in pairs:
                pairs[human_bone] = robot_body

    floor_y = _estimate_floor_y(snapshot)
    if floor_y is None:
        print("WARNING: no foot bones found in snapshot — root scale will be inaccurate")

    scales: Dict[str, float] = {}
    print(f"\n{'bone':<22} {'source':<20} {'robot body':<30} {'robot dist':>10} {'human dist':>10} {'scale':>7}")
    print("-" * 103)

    for human_bone, robot_body in pairs.items():
        # Robot distance from root
        try:
            robot_body_pos = data.xpos[model.body(robot_body).id].copy()
        except Exception:
            print(f"  WARNING: robot body '{robot_body}' not found in model, skipping")
            continue

        # Human distance from root
        if human_bone == human_root:
            # Root: use floor-relative hip height (SolarXR Y-up) vs robot Z-up
            if floor_y is not None:
                human_dist = float(human_root_pos[1] - floor_y)
            else:
                human_dist = float(np.linalg.norm(human_root_pos))
            robot_dist = float(robot_root_pos[2])
            source = human_bone
        else:
            human_pos, source = _get_bone_pos_with_fallback(snapshot, human_bone, human_root_pos)
            if human_pos is None:
                print(f"  WARNING: '{human_bone}' not found in snapshot, skipping")
                continue
            human_dist = float(np.linalg.norm(human_pos - human_root_pos))
            robot_dist = float(np.linalg.norm(robot_body_pos - robot_root_pos))
            if human_dist < MIN_HUMAN_DIST:
                print(f"  WARNING: degenerate displacement for '{human_bone}' ({human_dist:.4f}m) — skipping")
                continue

        if human_dist < 1e-6:
            print(f"  WARNING: zero human distance for '{human_bone}', skipping")
            continue

        scale = robot_dist / human_dist
        scales[human_bone] = round(scale, 6)
        flag = " *" if source != human_bone else ""
        print(f"  {human_bone:<22} {source + flag:<20} {robot_body:<30} {robot_dist:>10.4f} {human_dist:>10.4f} {scale:>7.4f}")

    ik_cfg["human_scale_table"] = scales

    out = Path(args.output) if args.output else Path(args.ik_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ik_cfg, f, indent=4)
        f.write("\n")
    print(f"\n[scales] Wrote updated human_scale_table → {out}")


if __name__ == "__main__":
    main()
