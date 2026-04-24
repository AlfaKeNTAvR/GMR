#!/usr/bin/env python3
"""
SolarXR -> GMR retargeting -> WBC (pcf) bridge for IT1.

Reads mocopi tracker data from SolarXR, runs GMR inverse kinematics,
and publishes arm joints, torso angle, and height to the whole-body
controller via Zenoh.
"""

from __future__ import annotations

import argparse
import math
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

# ---------------------------------------------------------------------------
# IOBT direct mode: Quest 3 body joint index → GMR bone name
# ---------------------------------------------------------------------------

IOBT_TO_BONE: Dict[int, str] = {
    0: "hip",               # Hips
    1: "chest",             # SpineChest
    5: "left_upper_arm",    # ShoulderLeft
    6: "left_lower_arm",    # ElbowLeft
    7: "left_hand",         # WristLeft
    9: "right_upper_arm",   # ShoulderRight
    10: "right_lower_arm",  # ElbowRight
    11: "right_hand",       # WristRight
    17: "left_hip",         # UpperLegLeft
    18: "left_lower_leg",   # LowerLegLeft
    19: "left_foot_tail",   # AnkleLeft
    21: "right_hip",        # UpperLegRight
    22: "right_lower_leg",  # LowerLegRight
    23: "right_foot_tail",  # AnkleRight
}


def _iobt_to_bones(
    body: List[Tuple[float, ...]],
    reference: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, object]]:
    """Convert IOBT upper_body poses to SolarXR-compatible bones dict.

    IOBT pose layout: (x, y, z, qw, qx, qy, qz)
    SolarXR bones expect: {"head": (x,y,z), "rot": (qx,qy,qz,qw)}

    If *reference* is provided (from neutral-pose calibration), rotations are
    expressed relative to the reference: delta = current * inv(reference).
    """
    bones: Dict[str, Dict[str, object]] = {}
    for idx, name in IOBT_TO_BONE.items():
        pose = body[idx]
        pos = (pose[0], pose[1], pose[2])
        rot_xyzw = np.array([pose[4], pose[5], pose[6], pose[3]])  # wxyz → xyzw

        norm = np.linalg.norm(rot_xyzw)
        if norm < 1e-4:
            continue  # No tracking data yet for this joint
        rot_xyzw /= norm  # Normalize before scipy to avoid zero-norm errors

        if reference is not None and name in reference:
            ref_q = reference[name]
            ref_norm = np.linalg.norm(ref_q)
            if ref_norm < 1e-4:
                continue
            ref_q = ref_q / ref_norm
            cur = R.from_quat(rot_xyzw)
            ref = R.from_quat(ref_q)
            rot_xyzw = (cur * ref.inv()).as_quat()

        bones[name] = {"head": pos, "rot": tuple(rot_xyzw)}
    return bones


def _iobt_capture_reference(
    body: List[Tuple[float, ...]],
) -> Dict[str, np.ndarray]:
    """Capture current IOBT rotations as the neutral-pose reference (xyzw)."""
    ref: Dict[str, np.ndarray] = {}
    for idx, name in IOBT_TO_BONE.items():
        pose = body[idx]
        quat = np.array([pose[4], pose[5], pose[6], pose[3]])
        norm = np.linalg.norm(quat)
        if norm < 1e-4:
            continue  # No tracking data for this joint
        ref[name] = quat / norm
    return ref


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

# qpos index -> WBC joint name (shoulder + elbow + wrist_roll per arm).
# Wrist rolls are passthrough: the WBC policy does not command them, so PCF
# takes the GMR-retargeted value and sends it straight to the actuator.
# Prototype caveat: PCF has no "owner" concept per joint. If a future policy
# starts writing wrist_roll, its output and the GMR passthrough will race on
# the same zenoh command and last-write-wins. Revisit then.
ARM_JOINT_INDICES: Dict[str, int] = {
    "left_shoulder_pitch": 24,
    "left_shoulder_roll": 25,
    "left_shoulder_yaw": 26,
    "left_elbow_pitch": 27,
    "left_wrist_roll": 28,
    "right_shoulder_pitch": 29,
    "right_shoulder_roll": 30,
    "right_shoulder_yaw": 31,
    "right_elbow_pitch": 32,
    "right_wrist_roll": 33,
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
    """Compute chest roll/pitch/yaw in world frame.

    All three angles are the chest body orientation in the world frame,
    decomposed as ZYX intrinsic Euler angles (yaw, pitch, roll).
    """
    chest_q = np.array(data.xquat[chest_id], dtype=float)

    cn = np.linalg.norm(chest_q)
    if not np.isfinite(cn) or cn < 1e-6:
        return 0.0, 0.0, 0.0
    chest_q /= cn

    # Convert wxyz → xyzw for scipy
    chest_xyzw = np.array([chest_q[1], chest_q[2], chest_q[3], chest_q[0]])
    chest_rot = R.from_quat(chest_xyzw)

    # ZYX intrinsic: first yaw around Z, then pitch around Y, then roll around X
    yaw, pitch, roll = chest_rot.as_euler("ZYX")

    return float(roll), float(pitch), float(yaw)


# Neck joint limits from iteration_1 MJCF (radians). Clamp commands to
# these so the neck never chases a target past its hardware stops.
NECK_PITCH_LIMIT = (-0.645772, 0.820305)
NECK_YAW_LIMIT = (-2.56563, 2.56563)

# Slew-rate caps for direct-commanded joints (neck + wrist rolls). These
# joints bypass the policy, so a new target (Reset View, mode switch,
# operator snapping their wrist) can be arbitrarily far from where the
# joint currently is, and sending it directly snaps the actuator. We
# clamp the commanded delta per second instead and let it ramp.
NECK_MAX_RATE_RAD_S = 5.0
WRIST_MAX_RATE_RAD_S = 5.0
# If the loop stalls (e.g. PCF RPC timeout), cap the effective dt so we
# don't let a burst-step through that is dt * rate large.
RATE_LIMIT_MAX_DT_S = 0.1
JOINT_MAX_RATE_RAD_S: Dict[str, float] = {
    "neck_pitch": NECK_MAX_RATE_RAD_S,
    "neck_yaw": NECK_MAX_RATE_RAD_S,
    "left_wrist_roll": WRIST_MAX_RATE_RAD_S,
    "right_wrist_roll": WRIST_MAX_RATE_RAD_S,
}


def apply_rate_limit(
    targets: Dict[str, float],
    last_commanded: Dict[str, float],
    dt_s: float,
) -> Dict[str, float]:
    """Return a copy of `targets` with rate-limited values for any joint in
    JOINT_MAX_RATE_RAD_S. `last_commanded` is the per-joint slew state;
    it is updated in place so the next call continues from where this one
    left off. Joints that do not appear in `targets` are skipped. First-
    time targets seed from 0 rad (home pose), so the operator-home flow
    ramps up smoothly rather than snapping from 0 to operator pose.
    """
    limited = dict(targets)
    for name, max_rate in JOINT_MAX_RATE_RAD_S.items():
        if name not in limited:
            continue
        target = limited[name]
        prev = last_commanded.get(name, 0.0)
        max_step = max_rate * dt_s
        delta = target - prev
        if abs(delta) > max_step:
            new_cmd = prev + math.copysign(max_step, delta)
        else:
            new_cmd = target
        limited[name] = new_cmd
        last_commanded[name] = new_cmd
    return limited


def compute_neck_angles_from_headset(
    head_pose_xr: Optional[Tuple[float, ...]],
    torso_pitch_world: float,
    torso_yaw_world: float,
) -> Optional[Tuple[float, float]]:
    """Headset-driven neck targets that stabilize the robot's head to the
    ground. Returns (neck_pitch_cmd, neck_yaw_cmd) in robot joint convention.

    Input is an OpenXR VIEW-in-LOCAL head pose (x, y, z, qw, qx, qy, qz)
    from snap.head. OpenXR frame: +X=right, +Y=up, -Z=forward, right-handed.
    Robot joint convention (iteration_1 MJCF): +neck_pitch rotates head
    down, +neck_yaw rotates head left.

    torso_pitch_world / torso_yaw_world: the pitch and yaw we are about
    to command via `controller.torso_angle(...)`, in the robot's world
    frame (both already in robot sign, +pitch=forward, +yaw=left). Must
    be the final commanded values, including PITCH_OFFSET and any
    retargeting output, so the neck cancels whatever the torso actually
    does.

    Stabilization logic: robot_head_world = robot_torso_world + neck. We
    want robot_head_world to match the operator's head orientation in
    world, so neck = operator_head_world - commanded_torso. Effect: when
    the operator bends forward but keeps the head level, the neck pitches
    back to keep the robot's head parallel to the floor. Neck is clipped
    to MJCF joint limits; big body tilts may hit the clip and leak back
    into the head.

    Chest is deliberately NOT used as a reference here: the IOBT chest
    joint frame is rotated ~90° around up relative to the headset VIEW
    frame, so a direct `chest.inv() * head` injects a 90° bias at rest
    and flips yaw. Working in world and subtracting the commanded torso
    sidesteps that.
    """
    if head_pose_xr is None:
        return None
    head_quat_xyzw = np.array(
        [head_pose_xr[4], head_pose_xr[5], head_pose_xr[6], head_pose_xr[3]],
        dtype=float,
    )
    head_norm = np.linalg.norm(head_quat_xyzw)
    if head_norm < 1e-4:
        return None
    head_rot_xr = R.from_quat(head_quat_xyzw / head_norm)
    head_forward_in_xr_world = head_rot_xr.apply(np.array([0.0, 0.0, -1.0]))
    fwd_x, fwd_y, fwd_z = head_forward_in_xr_world
    # Operator yaw: swing of forward vector around world +Y (up). Forward
    # nominally sits near -Z; a positive operator_yaw_xr (head turning
    # right, so forward tilts toward world +X) is atan2(+X, -(-Z)).
    operator_yaw_xr = math.atan2(fwd_x, -fwd_z)
    # Operator pitch: elevation of forward vector off the world horizontal.
    # Positive = looking up (forward tilts toward world +Y).
    operator_pitch_xr = math.atan2(
        fwd_y, math.sqrt(fwd_x * fwd_x + fwd_z * fwd_z)
    )
    # Flip to robot joint sign (robot: +pitch=down, +yaw=left).
    head_target_pitch_world = -operator_pitch_xr
    head_target_yaw_world = -operator_yaw_xr
    # Cancel whatever the torso is about to do, so the head lands at the
    # operator's world orientation regardless of body pose.
    neck_pitch_cmd = float(np.clip(
        head_target_pitch_world - torso_pitch_world,
        NECK_PITCH_LIMIT[0], NECK_PITCH_LIMIT[1],
    ))
    neck_yaw_cmd = float(np.clip(
        head_target_yaw_world - torso_yaw_world,
        NECK_YAW_LIMIT[0], NECK_YAW_LIMIT[1],
    ))
    return neck_pitch_cmd, neck_yaw_cmd


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
                        help="Zenoh TCP port for pcf")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host/IP where pcf is running")
    parser.add_argument("--teleop-port", type=int, default=9876,
                        help="UDP port for teleop_rs Quest 3 data")
    parser.add_argument("--viewer", action="store_true",
                        help="Show MuJoCo viewer window")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--iobt", action="store_true",
                        help="Use Quest 3 IOBT body tracking directly (skip SolarXR/SlimeVR)")
    args = parser.parse_args()

    # --- Import ControllerApi from pcf ---
    pcf_scripts = Path(__file__).resolve().parents[1].parent / "persona" / "locomotion" / "pcf" / "scripts"
    sys.path.insert(0, str(pcf_scripts))
    from utils.controller import ControllerApi  # type: ignore

    # --- Teleop receiver (always needed for head/controllers) ---
    from teleop_receiver import TeleopReceiver

    teleop = TeleopReceiver(port=args.teleop_port)
    teleop.start()

    # --- SolarXR / SlimeVR setup (skipped in IOBT mode) ---
    client = None
    slimevr = None
    if not args.iobt:
        solarxr_path = _resolve_solarxr_path(args.solarxr_root)
        SolarXRClient = _import_solarxr_client(solarxr_path)
        SlimeVRBridgeSender = _import_slimevr_bridge(solarxr_path)

        slimevr = SlimeVRBridgeSender()
        slimevr.connect()
        print(f"[solarxr_wbc] SlimeVR bridge connected, teleop on :{args.teleop_port}")

        client = SolarXRClient(
            url=args.solar_url,
            minimum_ms=args.minimum_ms,
        )
        client.start()
    else:
        print(f"[solarxr_wbc] IOBT direct mode, teleop on :{args.teleop_port}")

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
    controller = ControllerApi(args.port, args.host)
    print(f"[solarxr_wbc] Connected to pcf at {args.host}:{args.port}")
    print("[solarxr_wbc] Idle — press B on right controller to arm + home.")

    # --- Optional viewer ---
    viewer = None
    if args.viewer:
        viewer = RobotMotionViewer(robot_type=args.robot)

    FORWARD_MAX = 1.0  # m/s
    LATERAL_MAX = 0.5  # m/s
    ANGULAR_MAX = 1.5  # rad/s
    JOYSTICK_DEADZONE = 0.1
    armed = False  # True after first B press
    mode: Optional[str] = None  # None=idle, "walk", or "wbc"
    # None = haven't seen a valid controller frame yet. The first frame seeds
    # prev state without firing edges, so buttons already held when the script
    # starts (e.g. Quest app was running before us) don't look like presses.
    a_button_prev: Optional[bool] = None
    b_button_prev: Optional[bool] = None
    r_trigger_prev: Optional[bool] = None
    recenter_prev = False
    last_missing_report = 0.0
    last_wrist_debug = 0.0  # throttled wrist-roll command print (~2 Hz)
    iobt_debug_frames = 3  # print first 3 frames when --verbose --iobt
    iobt_reference: Optional[Dict[str, np.ndarray]] = None
    # Slew-rate state for direct-commanded joints (see apply_rate_limit).
    last_commanded: Dict[str, float] = {}
    rate_limit_last_t: Optional[float] = None

    if args.iobt:
        print("[solarxr_wbc] Stand in neutral pose (arms down), then press Reset View on Quest 3.")
    print("[solarxr_wbc] B=home (arm), A=start walk / toggle WBC<->walk, R-trigger=stop, Reset View=recenter. Ctrl-C to exit.")

    try:
        while True:
            snap = teleop.latest()

            # --- Forward Quest 3 poses to SlimeVR (SolarXR mode only) ---
            if slimevr is not None:
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

            # --- Right controller: B=arm+home, A=start walk / toggle WBC<->walk, trigger=stop ---
            # Only process when we actually have controller data; otherwise prev state
            # stays frozen so the next valid frame doesn't look like a rising edge.
            if snap.controller_right is not None:
                a_button = snap.controller_right.buttons[0]
                b_button = snap.controller_right.buttons[1]
                r_trigger = snap.controller_right.trigger > 0.5

                if (
                    a_button_prev is None
                    or b_button_prev is None
                    or r_trigger_prev is None
                ):
                    # First valid frame: seed baselines, no edges fire.
                    a_button_prev = a_button
                    b_button_prev = b_button
                    r_trigger_prev = r_trigger
                else:
                    if b_button and not b_button_prev:
                        armed = True
                        print("[solarxr_wbc] B pressed — homing.")
                        controller.pose("home")
                    if a_button and not a_button_prev and armed:
                        if mode is None:
                            mode = "walk"
                            print("[solarxr_wbc] A pressed — starting WALK policy")
                            controller.policy("walk")
                        elif mode == "walk":
                            mode = "wbc"
                            print("[solarxr_wbc] Switching to WBC policy")
                            controller.policy("wbc_1")
                        else:
                            mode = "walk"
                            print("[solarxr_wbc] Switching to WALK policy")
                            controller.policy("walk")
                    if r_trigger and not r_trigger_prev:
                        print("[solarxr_wbc] Right trigger pulled — stopping policy, returning to idle.")
                        controller.stop()
                        mode = None
                        armed = False

                    a_button_prev = a_button
                    b_button_prev = b_button
                    r_trigger_prev = r_trigger

            # --- Recenter / calibrate (edge-detected: upstream may send sticky True) ---
            # Posture calibration (IOBT neutral pose) runs once per script run,
            # on the first Reset View. Subsequent presses only update the local
            # XR frame so the operator can re-center without re-standing the
            # pose.
            if snap.recenter and not recenter_prev:
                if args.iobt and snap.upper_body and iobt_reference is None:
                    iobt_reference = _iobt_capture_reference(snap.upper_body)
                    iobt_debug_frames = 3
                    print("[iobt] Neutral pose captured — calibration done.")
                if client is not None:
                    print("[solarxr_wbc] Resetting SlimeVR skeleton (recenter)")
                    client.reset_full()
            recenter_prev = snap.recenter

            # --- Idle: no policy active yet, just watch buttons ---
            if mode is None:
                time.sleep(0.02)
                continue

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

                # Neck tracking during walking: walk policy controls the torso,
                # so we don't know its exact pose here. Passing zero torso
                # means the walking policy's built-in lean shows up as a
                # small constant pitch bias on the head; acceptable for now.
                neck_walk = compute_neck_angles_from_headset(
                    snap.head,
                    torso_pitch_world=0.0,
                    torso_yaw_world=0.0,
                )
                if neck_walk is not None:
                    now_ts = time.time()
                    dt_s = (
                        0.02 if rate_limit_last_t is None
                        else min(RATE_LIMIT_MAX_DT_S, now_ts - rate_limit_last_t)
                    )
                    rate_limit_last_t = now_ts
                    neck_targets = apply_rate_limit(
                        {"neck_pitch": neck_walk[0], "neck_yaw": neck_walk[1]},
                        last_commanded,
                        dt_s,
                    )
                    controller.joint_states(neck_targets)

                time.sleep(0.02)
                continue

            # --- WBC mode: retargeting ---
            if args.iobt:
                if not snap.upper_body:
                    time.sleep(0.001)
                    continue
                if iobt_reference is None:
                    time.sleep(0.01)
                    continue
                bones = _iobt_to_bones(snap.upper_body, reference=iobt_reference)
                if args.verbose and iobt_debug_frames > 0:
                    iobt_debug_frames -= 1
                    print(f"[iobt] frame {3 - iobt_debug_frames}/3 — calibrated bone data:")
                    for name in sorted(bones.keys()):
                        b = bones[name]
                        p = b["head"]
                        r = b["rot"]
                        print(f"  {name:20s}  pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  rot_xyzw=({r[0]:+.4f},{r[1]:+.4f},{r[2]:+.4f},{r[3]:+.4f})")
            else:
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

            # --- Torso angle (computed first so the neck can cancel its effect) ---
            roll, pitch, yaw = compute_torso_angle(
                model, mj_data, pelvis_id, chest_id
            )

            # --- Neck from headset, stabilized to world (head parallel to ground) ---
            neck = compute_neck_angles_from_headset(
                snap.head,
                torso_pitch_world=pitch,
                torso_yaw_world=yaw,
            )
            if neck is not None:
                arm_targets["neck_pitch"], arm_targets["neck_yaw"] = neck

            # --- Rate-limit direct-commanded joints (neck + wrist rolls) ---
            now_ts = time.time()
            dt_s = (
                0.02 if rate_limit_last_t is None
                else min(RATE_LIMIT_MAX_DT_S, now_ts - rate_limit_last_t)
            )
            rate_limit_last_t = now_ts
            arm_targets = apply_rate_limit(arm_targets, last_commanded, dt_s)

            controller.joint_states(arm_targets)

            # --- Wrist-roll debug print (throttled) ---
            now = time.time()
            if now - last_wrist_debug > 0.5:
                last_wrist_debug = now
                print(
                    f"[wrist_roll] left={arm_targets['left_wrist_roll']:+.3f} "
                    f"right={arm_targets['right_wrist_roll']:+.3f} rad"
                )

            # --- Extract height ---
            height = float(qpos[HEIGHT_INDEX])
            controller.height(height)

            # --- Send torso angle ---
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
        print("\n[solarxr_wbc] Stopping (policy persists on robot)...")
    finally:
        if viewer is not None:
            viewer.close()
        if client is not None:
            client.stop()
        teleop.stop()
        if slimevr is not None:
            slimevr.close()


if __name__ == "__main__":
    main()
