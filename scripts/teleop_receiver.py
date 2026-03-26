"""Receives Quest 3 tracking data from teleop_rs over UDP.

teleop_rs streams TrackingPacket structs serialized with bincode 2.0 (standard
config, little-endian, no alignment padding).  This module parses the compact
wire format and exposes the latest head pose, controller state (buttons,
thumbsticks, triggers), and body skeleton.

Typical usage:

    receiver = TeleopReceiver(port=9876)
    receiver.start()
    snap = receiver.latest()
    if snap.controller_right:
        print(snap.controller_right.buttons)  # (A, B, thumbstick_click)
    receiver.stop()
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CompactController:
    """Quest 3 Touch Plus controller state."""

    position: Tuple[float, float, float]  # x, y, z
    orientation: Tuple[float, float, float, float]  # qw, qx, qy, qz
    thumbstick: Tuple[float, float]  # x, y  (-1..1)
    trigger: float  # 0..1
    squeeze: float  # 0..1
    buttons: Tuple[bool, bool, bool]  # (btn0, btn1, stick_click)
    # Left:  btn0=X, btn1=Y
    # Right: btn0=A, btn1=B


@dataclass
class TrackingSnapshot:
    """Most recent tracking data from a single UDP packet."""

    sequence: int = 0
    timestamp_ns: int = 0
    head: Optional[Tuple[float, ...]] = None  # (x, y, z, qw, qx, qy, qz)
    controller_left: Optional[CompactController] = None
    controller_right: Optional[CompactController] = None
    recenter: bool = False  # True on the frame when the user reset the view


# ---------------------------------------------------------------------------
# Binary sizes (bincode 2.0 standard, no padding)
# ---------------------------------------------------------------------------
_HAND_SIZE = 26 * 7 * 4  # [[f32; 7]; 26] = 728 bytes
_BODY_SIZE = 25 * 7 * 4  # [[f32; 7]; 25] = 700 bytes
_POSE_SIZE = 7 * 4  # [f32; 7] = 28 bytes
_CTRL_SIZE = 7 * 4 + 2 * 4 + 4 + 4 + 3  # 47 bytes
_HEADER_SIZE = 4 + 8  # u32 sequence + u64 timestamp = 12 bytes


class TeleopReceiver:
    """Background thread that receives teleop_rs UDP packets and stores the
    latest parsed snapshot."""

    MAX_PACKET = 4096

    def __init__(self, port: int = 9876) -> None:
        self._port = port
        self._latest = TrackingSnapshot()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, name="teleop_recv", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def latest(self) -> TrackingSnapshot:
        with self._lock:
            snap = self._latest
            if snap.recenter:
                self._latest = TrackingSnapshot(
                    sequence=snap.sequence,
                    timestamp_ns=snap.timestamp_ns,
                    head=snap.head,
                    controller_left=snap.controller_left,
                    controller_right=snap.controller_right,
                    recenter=False,
                )
            return snap

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self._port))
        sock.settimeout(0.1)

        while self._running:
            try:
                data, _ = sock.recvfrom(self.MAX_PACKET)
                snap = self._parse(data)
                if snap is not None:
                    with self._lock:
                        if self._latest.recenter and not snap.recenter:
                            snap = TrackingSnapshot(
                                sequence=snap.sequence,
                                timestamp_ns=snap.timestamp_ns,
                                head=snap.head,
                                controller_left=snap.controller_left,
                                controller_right=snap.controller_right,
                                recenter=True,
                            )
                        self._latest = snap
            except socket.timeout:
                continue
            except Exception:
                continue

        sock.close()

    # ------------------------------------------------------------------
    # Bincode parser
    # ------------------------------------------------------------------

    def _parse(self, data: bytes) -> Optional[TrackingSnapshot]:
        if len(data) < _HEADER_SIZE:
            return None

        off = 0

        # TrackingPacket header
        seq = struct.unpack_from("<I", data, off)[0]
        off += 4
        ts = struct.unpack_from("<Q", data, off)[0]
        off += 8

        # CompactTrackingData — fields in declaration order:
        # 1. hand_left:  Option<[[f32;7];26]>
        off = _skip_option(data, off, _HAND_SIZE)
        # 2. hand_right: Option<[[f32;7];26]>
        off = _skip_option(data, off, _HAND_SIZE)
        # 3. head:       Option<[f32;7]>
        head, off = _parse_option_pose(data, off)
        # 4. controller_left:  Option<CompactController>
        ctrl_l, off = _parse_option_controller(data, off)
        # 5. controller_right: Option<CompactController>
        ctrl_r, off = _parse_option_controller(data, off)
        # 6. upper_body: Option<[[f32;7];25]> — skip
        off = _skip_option(data, off, _BODY_SIZE)
        # 7. recenter: bool
        recenter = bool(data[off]) if off < len(data) else False

        return TrackingSnapshot(
            sequence=seq,
            timestamp_ns=ts,
            head=head,
            controller_left=ctrl_l,
            controller_right=ctrl_r,
            recenter=recenter,
        )


# ------------------------------------------------------------------
# Parsing helpers (module-level for speed)
# ------------------------------------------------------------------


def _skip_option(data: bytes, off: int, size: int) -> int:
    """Skip an Option<T> field, advancing offset past it."""
    tag = data[off]
    off += 1
    if tag:
        off += size
    return off


def _parse_option_pose(
    data: bytes, off: int
) -> Tuple[Optional[Tuple[float, ...]], int]:
    """Parse Option<[f32; 7]> → (x, y, z, qw, qx, qy, qz) or None."""
    tag = data[off]
    off += 1
    if not tag:
        return None, off
    values = struct.unpack_from("<7f", data, off)
    off += _POSE_SIZE
    return values, off


def _parse_option_controller(
    data: bytes, off: int
) -> Tuple[Optional[CompactController], int]:
    """Parse Option<CompactController>."""
    tag = data[off]
    off += 1
    if not tag:
        return None, off

    pose = struct.unpack_from("<7f", data, off)
    off += 28
    thumb = struct.unpack_from("<2f", data, off)
    off += 8
    trigger = struct.unpack_from("<f", data, off)[0]
    off += 4
    squeeze = struct.unpack_from("<f", data, off)[0]
    off += 4
    b0, b1, b2 = data[off], data[off + 1], data[off + 2]
    off += 3

    ctrl = CompactController(
        position=(pose[0], pose[1], pose[2]),
        orientation=(pose[3], pose[4], pose[5], pose[6]),  # qw, qx, qy, qz
        thumbstick=thumb,
        trigger=trigger,
        squeeze=squeeze,
        buttons=(bool(b0), bool(b1), bool(b2)),
    )
    return ctrl, off
