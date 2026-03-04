# SolarXR -> GMR (G1) setup notes

## Step 0: Prepare SolarXR stream (pre-GMR)
- Choose a stable World Frame (WF): Quest 3 global frame (post-reset).
- SolarXR is head-centric: lift the SolarXR skeleton into WF using headset pose + a fixed headset->SolarXR head transform.
- Convert to MuJoCo axes: +X forward, +Y left, +Z up.
- Positions in meters; quaternions are scalar-first (wxyz).
- Pelvis is a root joint pose in WF, not the WF itself.
- Output per-frame dict: {joint_name: (pos_world, quat_world)} in WF.

## Step 1: Create solarxr_to_g1.json (IK config)
- Map robot bodies (MuJoCo body frames) -> human joints (SolarXR names).
- In GMR, "frame_name" is a robot body name, not a joint DOF.
- Set human_root_name (pelvis) and robot_root_name (pelvis).
- rot_offset aligns SolarXR joint frames (neutral pose) to robot body frames.
  - Neutral pose = SolarXR calibration pose; align frames, not posture.
- pos_offset compensates body-frame vs joint-center mismatch (toes often need this).
- Scaling is applied only to human data before IK:
  - Uniform: actual_human_height
  - Per-joint: human_scale_table

## Local scale measuring (human_scale_table)
1) Capture a neutral SolarXR frame (calibration pose).
2) Compute human limb lengths from joint positions:
   - hip->knee, knee->ankle, shoulder->elbow, elbow->wrist, pelvis->neck, etc.
3) Compute robot limb lengths between corresponding MuJoCo body frames.
4) For each joint, scale[joint] = robot_length / human_length.
5) Start coarse (leg/arm/torso groups), then refine.

If you skip a joint, use the distance between the two joints you keep:
- Example: hip and ankle only -> length = ||pos_ankle - pos_hip||.
