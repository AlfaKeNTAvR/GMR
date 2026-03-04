# SolarXR -> GMR realtime parser plan

## Goal
- Stream SolarXR skeleton in realtime and feed GMR (G1) with MuJoCo-frame SE(3) joint poses for teleoperation.

## Assumptions (current)
- Target robot: G1 only.
- No headset rotation applied in the SolarXR→world transform.
- Create a new `solarxr_to_g1.json` using SolarXRWorld bone names.
- MuJoCo frame: +X forward, +Y left, +Z up; meters; quaternions wxyz.
- Rest pose is arms-down (not T-pose).
- Minimal latency: avoid smoothing unless needed.

## Plan
1) Define the data contract for GMR input:
   - Joint list + root name (human_root_name) must match the IK config.
   - Coordinate frame: MuJoCo (+X forward, +Y left, +Z up), meters, quaternions wxyz.
   - Use SolarXRWorld bone keys as IK-config human names (mostly 1:1); select a subset for IK.
2) Create/extend IK config:
   - Add `general_motion_retargeting/ik_configs/solarxr_to_g1.json`.
   - Tune rot/pos offsets for SolarXR rest pose (arms-down).
   - Start with manual `human_scale_table`.
   - Update `general_motion_retargeting/params.py` with src_human="solarxr".
3) Implement `scripts/solarxr_parser.py`:
   - Use `SolarXRWorld` to get world bones.
   - Convert to MuJoCo frame + target joint naming; output `{joint: (pos, quat)}`.
   - Feed `GeneralMotionRetargeting` and optionally `RobotMotionViewer`.
   - Add CLI flags (url, robot, minimum_ms/rate, reset, output mode).
4) Validate and tune:
   - Rest pose check: verify axes/rot offsets (joints point the expected directions).
   - Floor alignment: feet near z≈0; pelvis height stable.
   - Limb lengths: compare SolarXR vs robot limb distances; adjust scale.
   - Latency: measure end-to-end loop time; check for jitter.
5) Teleop integration:
   - For now, MuJoCo viewer path: `qpos` → `RobotMotionViewer.step`.
   - Optional output hook for `qpos` (stdout or socket) if we want to drive another consumer later.

## Extended goal
- Auto-scale: compute SolarXR limb lengths from a calibration frame and derive `human_scale_table`
  from G1 MuJoCo body distances; compare with manual scaling.

## Open questions
- Confirm target update rate (SolarXRWorld default minimum_ms=20 → ~50 Hz).
