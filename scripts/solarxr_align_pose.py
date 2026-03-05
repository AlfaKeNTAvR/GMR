#!/usr/bin/env python3
"""
Interactive pose alignment tool for SolarXR → G1 calibration.

Loads a SolarXR snapshot and overlays bone orientations (in MuJoCo frame) onto
the corresponding robot body frames. Adjust robot joints until the two sets of
frames overlap, then save the qpos — that becomes the robot reference pose for
offline rot_offset calibration.

Controls:
  Tab          — next joint
  N / P        — next / previous joint
  Up / Down    — rotate selected joint ±2°
  Left / Right — rotate selected joint ±10°
  R            — reset all joints to zero
  S            — save qpos to output JSON
  Q            — quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import ROBOT_XML_DICT

# SolarXR world → MuJoCo frame
XR_TO_MJ = np.array(
    [
        [ 0.0,  0.0, -1.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
    ],
    dtype=float,
)

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

# GLFW key codes (avoiding MuJoCo built-ins like arrows, Tab, Space, Backspace)
KEY_COMMA  = 44   # , → previous joint
KEY_PERIOD = 46   # . → next joint
KEY_UP     = 265  # ↑ → +1°
KEY_DOWN   = 264  # ↓ → -1°
KEY_LEFT   = 263  # ← → -10°
KEY_RIGHT  = 262  # → → +10°
KEY_R      = 82   # R → reset
KEY_S      = 83   # S → save
KEY_Q      = 81   # Q → quit


def _quat_xr_to_mj(quat_xyzw: list) -> np.ndarray:
    rot_xr = R.from_quat(quat_xyzw)
    rot_mj = R.from_matrix(XR_TO_MJ @ rot_xr.as_matrix() @ XR_TO_MJ.T)
    return rot_mj.as_matrix()


def _load_snapshot_mats(
    snapshot: dict,
    ik_table: dict,
) -> Dict[str, np.ndarray]:
    """Return {robot_body: rotation_matrix_in_mj} from snapshot for each IK entry."""
    mats: Dict[str, np.ndarray] = {}
    for robot_body, entry in ik_table.items():
        human_name = entry[0]
        candidates = ALIAS_MAP.get(human_name, [human_name])
        for cand in candidates:
            bone = snapshot.get(cand)
            if bone is None:
                continue
            rot = bone.get("rot")
            if rot is None:
                continue
            mats[robot_body] = _quat_xr_to_mj(rot)
            break
    return mats


def _draw_axes(
    scn: mj.MjvScene,
    pos: np.ndarray,
    mat: np.ndarray,
    size: float,
    label: str = "",
    colors=None,  # list of 3 rgba
):
    if colors is None:
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.005, 0.005, 0.005],
            pos=pos,
            mat=mat.flatten(),
            rgba=colors[i],
        )
        geom.label = label if i == 0 else ""
        mj.mjv_connector(
            scn.geoms[scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.004,
            from_=pos,
            to=pos + size * mat[:, i],
        )
        scn.ngeom += 1


class _State:
    def __init__(self, joint_names: List[str]):
        self.joints = joint_names
        self.selected = 0
        self.do_save = False
        self.do_quit = False

    @property
    def selected_name(self) -> str:
        return self.joints[self.selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive SolarXR → robot pose alignment.")
    parser.add_argument("--snapshot", "-s", required=True, help="Path to SolarXR snapshot JSON")
    parser.add_argument("--ik-config", required=True, help="Path to IK config JSON")
    parser.add_argument("--robot", default="unitree_g1_27dof")
    parser.add_argument("--output", "-o", default=None, help="Output IK config JSON (default: overwrite --ik-config)")
    parser.add_argument("--frame-size", type=float, default=0.25)
    parser.add_argument("--ignore-limits", action="store_true", help="Ignore joint limits during alignment")
    args = parser.parse_args()

    with open(args.snapshot, "r") as f:
        snapshot = json.load(f)
    with open(args.ik_config, "r") as f:
        ik_cfg = json.load(f)

    ik_table: dict = {}
    ik_table.update(ik_cfg.get("ik_match_table1", {}))
    ik_table.update(ik_cfg.get("ik_match_table2", {}))

    xml_path = ROBOT_XML_DICT[args.robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    # Collect revolute joint names (skip free joint)
    joint_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE:
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_names.append(name)

    state = _State(joint_names)

    STEP_FINE   = np.deg2rad(1.0)
    STEP_COARSE = np.deg2rad(10.0)

    def key_cb(key: int) -> None:
        n = len(state.joints)
        if key == KEY_PERIOD:
            state.selected = (state.selected + 1) % n
            print(f"  joint [{state.selected+1}/{n}]: {state.selected_name}  "
                  f"= {np.rad2deg(data.qpos[_qpos_adr(model, state.selected_name)]):.1f}°")
        elif key == KEY_COMMA:
            state.selected = (state.selected - 1) % n
            print(f"  joint [{state.selected+1}/{n}]: {state.selected_name}  "
                  f"= {np.rad2deg(data.qpos[_qpos_adr(model, state.selected_name)]):.1f}°")
        elif key in (KEY_UP, KEY_RIGHT):
            step = STEP_COARSE if key == KEY_RIGHT else STEP_FINE
            _adjust_joint(model, data, state.selected_name, +step, args.ignore_limits)
            print(f"  {state.selected_name} = {np.rad2deg(data.qpos[_qpos_adr(model, state.selected_name)]):.1f}°")
        elif key in (KEY_DOWN, KEY_LEFT):
            step = STEP_COARSE if key == KEY_LEFT else STEP_FINE
            _adjust_joint(model, data, state.selected_name, -step, args.ignore_limits)
            print(f"  {state.selected_name} = {np.rad2deg(data.qpos[_qpos_adr(model, state.selected_name)]):.1f}°")
        elif key == KEY_R:
            for jname in state.joints:
                adr = _qpos_adr(model, jname)
                if adr is not None:
                    data.qpos[adr] = 0.0
            print("  Reset all joints to zero.")
        elif key == KEY_S:
            state.do_save = True
        elif key == KEY_Q:
            state.do_quit = True

    viewer = mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=True,
        key_callback=key_cb,
    )

    xr_mats = _load_snapshot_mats(snapshot, ik_table)
    missing = [rb for rb in ik_table if rb not in xr_mats]
    if missing:
        print(f"[align] Warning: no snapshot data for: {missing}")

    print("\n[align] Controls:")
    print("  , / .   — previous / next joint")
    print("  ↑ / ↓   — ±1°   |   → / ← — ±10°")
    print("  R       — reset all to zero")
    print("  S       — compute rot_offsets and save to IK config")
    print("  Q       — quit")
    print(f"\n[align] {len(joint_names)} joints, {len(xr_mats)} SolarXR frames loaded.")
    print(f"[align] Selected: {state.selected_name}\n")

    # Robot frame colors: RGB
    ROBOT_COLORS = [[1, 0, 0, 0.9], [0, 1, 0, 0.9], [0, 0, 1, 0.9]]
    # SolarXR frame colors: CMY
    XR_COLORS    = [[0, 1, 1, 0.9], [1, 0, 1, 0.9], [1, 1, 0, 0.9]]

    while viewer.is_running():
        mj.mj_forward(model, data)
        viewer.user_scn.ngeom = 0

        for robot_body, entry in ik_table.items():
            try:
                bid = model.body(robot_body).id
            except Exception:
                continue

            pos = data.xpos[bid].copy()
            robot_mat = data.xmat[bid].reshape(3, 3).copy()

            is_selected = (entry[0] == state.selected_name or robot_body == state.selected_name)
            alpha = 1.0 if is_selected else 0.6
            rc = [[c[0], c[1], c[2], alpha] for c in ROBOT_COLORS]
            xc = [[c[0], c[1], c[2], alpha] for c in XR_COLORS]

            _draw_axes(viewer.user_scn, pos, robot_mat, args.frame_size, label=robot_body, colors=rc)

            if robot_body in xr_mats:
                _draw_axes(viewer.user_scn, pos, xr_mats[robot_body], args.frame_size, colors=xc)

        viewer.sync()

        if state.do_save:
            state.do_save = False
            _calibrate_and_save(model, data, xr_mats, ik_cfg, args.ik_config, args.output)

        if state.do_quit:
            break

    viewer.close()


def _qpos_adr(model: mj.MjModel, joint_name: str) -> Optional[int]:
    try:
        jid = model.joint(joint_name).id
        return int(model.jnt_qposadr[jid])
    except Exception:
        return None


def _adjust_joint(model: mj.MjModel, data: mj.MjData, joint_name: str, delta: float, ignore_limits: bool = False) -> None:
    adr = _qpos_adr(model, joint_name)
    if adr is None:
        return
    new_val = data.qpos[adr] + delta
    if not ignore_limits:
        jid = model.joint(joint_name).id
        lo, hi = model.jnt_range[jid]
        new_val = float(np.clip(new_val, lo, hi))
    data.qpos[adr] = float(new_val)


def _calibrate_and_save(
    model: mj.MjModel,
    data: mj.MjData,
    xr_mats: Dict[str, np.ndarray],
    ik_cfg: dict,
    ik_config_path: str,
    output_path: Optional[str],
) -> None:
    out = Path(output_path) if output_path else Path(ik_config_path)

    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = ik_cfg.get(table_name, {})
        for robot_body, entry in table.items():
            if robot_body not in xr_mats:
                continue
            try:
                bid = model.body(robot_body).id
            except Exception:
                continue
            robot_mat = data.xmat[bid].reshape(3, 3).copy()
            robot_rot = R.from_matrix(robot_mat)
            human_rot = R.from_matrix(xr_mats[robot_body])
            offset = human_rot.inv() * robot_rot
            q = offset.as_quat()  # xyzw
            entry[4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]  # wxyz

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ik_cfg, f, indent=4)
        f.write("\n")
    print(f"[align] Wrote rot_offsets → {out}")


if __name__ == "__main__":
    main()
