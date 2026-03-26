#!/usr/bin/env python3
"""
SolarXR -> GMR retargeting -> WBC (control_rs) bridge for IT1.

Reads mocopi tracker data from SolarXR, runs GMR inverse kinematics,
and publishes arm joints, torso angle, and height to the whole-body
controller via Zenoh.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

# ---------------------------------------------------------------------------
# SolarXR helpers (copied from solarxr_parser.py)
# ---------------------------------------------------------------------------

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

# qpos index -> WBC joint name (8 arm joints only)
ARM_JOINT_INDICES: Dict[str, int] = {
    "left_shoulder_pitch": 24,
    "left_shoulder_roll": 25,
    "left_shoulder_yaw": 26,
    "left_elbow_pitch": 27,
    "right_shoulder_pitch": 29,
    "right_shoulder_roll": 30,
    "right_shoulder_yaw": 31,
    "right_elbow_pitch": 32,
}

# MuJoCo body names for torso angle computation
PELVIS_BODY = "pelvis_link"
CHEST_BODY = "spine_pitch_link"

HEIGHT_INDEX = 2  # qpos[2] = pelvis Z


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


def _import_solarxr_client(solarxr_path: Path):
    sys.path.insert(0, str(solarxr_path))
    from solarxr_client import SolarXRClient  # type: ignore
    return SolarXRClient


def _import_slimevr_bridge(solarxr_path: Path):
    sys.path.insert(0, str(solarxr_path))
    from slimevr_bridge_sender import SlimeVRBridgeSender  # type: ignore
    return SlimeVRBridgeSender


def _pos_xr_to_mj(pos: Position) -> np.ndarray:
    return XR_TO_MJ @ np.asarray(pos, dtype=float)


def _quat_xr_to_mj_wxyz(
    quat_xyzw: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    if not quat_xyzw:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = np.asarray(quat_xyzw, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    rot_xr = R.from_quat(q / norm)
    rot_mj = R.from_matrix(XR_TO_MJ @ rot_xr.as_matrix() @ XR_TO_MJ.T)
    q_xyzw = rot_mj.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


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


# ---------------------------------------------------------------------------
# Torso angle computation
# ---------------------------------------------------------------------------


def _body_id(model: mj.MjModel, name: str) -> int:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"MuJoCo body '{name}' not found in model")
    return bid


def compute_torso_angle(
    model: mj.MjModel,
    data: mj.MjData,
    pelvis_id: int,
    chest_id: int,
) -> Tuple[float, float, float]:
    """Compute torso roll/pitch/yaw.

    Roll and pitch are chest orientation relative to the control frame
    (z-up, yaw-aligned to pelvis heading). Yaw is the pelvis heading
    in world frame.
    """
    # Body quaternions from MuJoCo (wxyz format) — copy to avoid aliasing
    pelvis_q = np.array(data.xquat[pelvis_id], dtype=float)
    chest_q = np.array(data.xquat[chest_id], dtype=float)

    # Normalize (guard against zero/degenerate quaternions)
    pn = np.linalg.norm(pelvis_q)
    cn = np.linalg.norm(chest_q)
    if not np.isfinite(pn) or not np.isfinite(cn) or pn < 1e-6 or cn < 1e-6:
        return 0.0, 0.0, 0.0
    pelvis_q /= pn
    chest_q /= cn

    # Convert wxyz → xyzw for scipy
    pelvis_xyzw = np.array([pelvis_q[1], pelvis_q[2], pelvis_q[3], pelvis_q[0]])
    chest_xyzw = np.array([chest_q[1], chest_q[2], chest_q[3], chest_q[0]])
    if np.linalg.norm(pelvis_xyzw) < 1e-6 or np.linalg.norm(chest_xyzw) < 1e-6:
        return 0.0, 0.0, 0.0
    pelvis_rot = R.from_quat(pelvis_xyzw)
    chest_rot = R.from_quat(chest_xyzw)

    # Pelvis yaw in world frame
    world_yaw = pelvis_rot.as_euler("ZYX")[0]

    # Chest relative to control frame (z-up, yaw-aligned to pelvis)
    control_rot = R.from_euler("Z", world_yaw)
    relative_rot = control_rot.inv() * chest_rot

    # Roll/pitch from relative rotation, yaw from pelvis world heading
    roll, pitch, _ = relative_rot.as_euler("xyz")

    return float(roll), float(pitch), float(world_yaw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolarXR -> GMR -> WBC bridge for IT1."
    )
    parser.add_argument("--solarxr-root", type=str, default=None)
    parser.add_argument("--solar-url", type=str, default="ws://127.0.0.1:21110")
    parser.add_argument("--minimum-ms", type=int, default=20)
    parser.add_argument("--robot", type=str, default="persona_it1")
    parser.add_argument("--port", type=int, default=4202,
                        help="Zenoh TCP port for control_rs")
    parser.add_argument("--teleop-port", type=int, default=9876,
                        help="UDP port for teleop_rs Quest 3 data")
    parser.add_argument("--viewer", action="store_true",
                        help="Show MuJoCo viewer alongside WBC output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # --- Import ControllerApi from control_rs ---
    control_rs_scripts = Path(__file__).resolve().parents[1].parent / "control_rs" / "scripts"
    sys.path.insert(0, str(control_rs_scripts))
    from utils.controller import ControllerApi  # type: ignore

    # --- SolarXR setup ---
    solarxr_path = _resolve_solarxr_path(args.solarxr_root)
    SolarXRClient = _import_solarxr_client(solarxr_path)
    SlimeVRBridgeSender = _import_slimevr_bridge(solarxr_path)

    # teleop_rs Quest 3 data → SlimeVR bridge (HMD + controllers)
    from teleop_receiver import TeleopReceiver

    teleop = TeleopReceiver(port=args.teleop_port)
    teleop.start()

    slimevr = SlimeVRBridgeSender()
    slimevr.connect()
    print(f"[solarxr_wbc] SlimeVR bridge connected, teleop on :{args.teleop_port}")

    client = SolarXRClient(
        url=args.solar_url,
        minimum_ms=args.minimum_ms,
    )
    client.start()

    # --- GMR retargeter ---
    retarget = GMR(
        src_human="solarxr",
        tgt_robot=args.robot,
        actual_human_height=None,
        verbose=args.verbose,
    )
    required = set(retarget.human_body_to_task1.keys()) | set(
        retarget.human_body_to_task2.keys()
    )

    # Resolve MuJoCo body IDs for torso angle computation
    model = retarget.model
    pelvis_id = _body_id(model, PELVIS_BODY)
    chest_id = _body_id(model, CHEST_BODY)

    # --- WBC controller ---
    controller = ControllerApi(args.port)
    print(f"[solarxr_wbc] Connected to control_rs on port {args.port}")
    print("[solarxr_wbc] Starting walk policy...")
    controller.policy("walk")

    # --- Optional viewer ---
    viewer = None
    if args.viewer:
        viewer = RobotMotionViewer(robot_type=args.robot)

    FORWARD_MAX = 1.0  # m/s
    LATERAL_MAX = 0.5  # m/s
    ANGULAR_MAX = 1.5  # rad/s
    JOYSTICK_DEADZONE = 0.1
    mode = "walk"  # current policy: "wbc" or "walk"
    a_button_prev = False
    last_missing_report = 0.0

    print("[solarxr_wbc] A=toggle WBC/walk, Reset View=reset SlimeVR. Ctrl-C to stop.")

    try:
        while True:
            snap = teleop.latest()

            # --- Forward Quest 3 poses to SlimeVR ---
            if snap.head:
                x, y, z, qw, qx, qy, qz = snap.head
                slimevr.send_hmd((x, y, z), (qx, qy, qz, qw))
            if snap.controller_left:
                c = snap.controller_left
                # 180° Y correction: grip frame → SlimeVR convention
                qw, qx, qy, qz = c.orientation
                slimevr.send_left_controller(
                    c.position, (-qx, qy, -qz, qw)
                )
            if snap.controller_right:
                c = snap.controller_right
                qw, qx, qy, qz = c.orientation
                slimevr.send_right_controller(
                    c.position, (-qx, qy, -qz, qw)
                )

            # --- A button: toggle WBC <-> walk ---
            a_button = (
                snap.controller_right.buttons[0]
                if snap.controller_right
                else False
            )
            if a_button and not a_button_prev:
                if mode == "wbc":
                    mode = "walk"
                    print("[solarxr_wbc] Switching to WALK policy")
                    controller.policy("walk")
                else:
                    mode = "wbc"
                    print("[solarxr_wbc] Switching to WBC policy")
                    controller.policy("wbc")
            a_button_prev = a_button

            # --- Reset SlimeVR skeleton ---
            # Trigger 1: headset "reset view" — reset immediately (OpenXR
            # already waits for the origin change before sending the flag)
            if snap.recenter:
                print("[solarxr_wbc] Resetting SlimeVR skeleton (recenter)")
                client.reset_full()


            # --- Walk mode: joystick control ---
            if mode == "walk":
                left_stick = (
                    snap.controller_left.thumbstick
                    if snap.controller_left
                    else (0.0, 0.0)
                )
                right_stick = (
                    snap.controller_right.thumbstick
                    if snap.controller_right
                    else (0.0, 0.0)
                )

                lx, ly = left_stick
                rx, _ = right_stick

                if abs(lx) < JOYSTICK_DEADZONE:
                    lx = 0.0
                if abs(ly) < JOYSTICK_DEADZONE:
                    ly = 0.0
                if abs(rx) < JOYSTICK_DEADZONE:
                    rx = 0.0

                controller.gait_velocity(
                    forward=ly * FORWARD_MAX,
                    lateral=-lx * LATERAL_MAX,
                    angular=-rx * ANGULAR_MAX,
                )
                time.sleep(0.02)
                continue

            # --- WBC mode: retargeting ---
            bones = client.get_raw_bones()
            if not bones:
                time.sleep(0.001)
                continue

            human_data, missing = _build_human_data(bones, required)
            if missing:
                if args.verbose and (time.time() - last_missing_report) > 1.0:
                    print(f"[solarxr_wbc] missing bones: {sorted(missing)}")
                    last_missing_report = time.time()
                continue

            # --- Retarget ---
            qpos = retarget.retarget(human_data)
            mj_data = retarget.configuration.data

            # Update body poses (xpos/xquat) from qpos — mink doesn't do this.
            mj.mj_forward(model, mj_data)

            # --- Extract arm joints ---
            arm_targets = {
                name: float(qpos[idx]) for name, idx in ARM_JOINT_INDICES.items()
            }
            controller.joint_state(arm_targets)

            # --- Extract height ---
            height = float(qpos[HEIGHT_INDEX])
            controller.height(height)

            # --- Compute and send torso angle ---
            roll, pitch, yaw = compute_torso_angle(
                model, mj_data, pelvis_id, chest_id
            )
            controller.torso_angle(roll=roll, pitch=pitch, yaw=yaw)

            # --- Optional viewer ---
            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    rate_limit=True,
                    follow_camera=False,
                )


    except KeyboardInterrupt:
        print("\n[solarxr_wbc] Stopping...")
    finally:
        controller.stop()
        if viewer is not None:
            viewer.close()
        client.stop()
        teleop.stop()
        slimevr.close()


if __name__ == "__main__":
    main()
