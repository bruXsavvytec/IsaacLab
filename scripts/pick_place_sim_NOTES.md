# pick_place_sim.py — design decisions & alternative paths

A running log of the choices behind `scripts/pick_place_sim.py`, why we made them,
and the options we deliberately did **not** take (so we can revisit them later).

## What the demo does now
Unitree G1 walks to a shelf (RSL-RL locomotion policy), then for each of three
cubes (blue → green → red) runs `REACH → GRASP → LIFT → OVER_BIN → LOWER →
RELEASE`, placing all three into an open-top **container that sits on the shelf**.
After the last cube it eases the arm home and **self-closes** the sim.

## Bugs fixed along the way
1. **Crash:** `sim_utils.RectLightCfg` doesn't exist in this Isaac Lab (0.54.3 /
   Isaac Sim 4.5). Replaced with `DiskLightCfg(radius=...)`.
2. **PhysX error on grasp/release:** the direct-GPU pipeline forbids switching a
   body dynamic↔kinematic at runtime. Removed the kinematic toggle; the cube is
   held by driving its pose + zeroing its velocity each frame, and released by
   simply stopping that drive.
3. **Arm "didn't move" / cube floated:** the original scripted joint targets were
   ≈ G1's already-bent default pose and referenced `left_wrist_pitch_joint`, which
   **G1 does not have** (arm = shoulder pitch/roll/yaw → elbow pitch/roll → finger
   joints `left_zero..left_six`; no wrist).

## How the robot is driven (mechanism)
- One loop: state machine picks a velocity `cmd (vx,vy,wz)` → `build_obs` → RSL-RL
  policy → 37 joint actions → `joint_targets = default + 0.25*action` →
  `set_joint_position_target` → PhysX **PD actuators** (`ImplicitActuatorCfg`,
  arms stiffness 40 / damping 10). The PD layer *is* the position controller — we
  never write our own PID.
- For manipulation phases we overwrite the arm entries of `action` so those joints
  track our target pose; legs keep running under the policy for balance.
- Cube attachment: teleport-follow (pose driven to `hand + offset`, velocity
  zeroed) while held; during LOWER the cube is guided to its bin slot so placement
  is reliable regardless of small hand error.

## Key decisions (chosen path in **bold**)

### D1 — Container placement
Options: floor bin beside shelf / bin at the old drop zone (walk back) /
**container on the shelf next to the cubes (no walking)**.
- Chosen: on the shelf. Simplest choreography, no walk-back, short arm moves.

### D2 — Arm control approach
Options: **scripted eased keyframes** / hybrid scripted-arm + authored cube path /
IK-driven (DifferentialIKController).
- Chosen: scripted (option "A"), for smoothness and simplicity (no solver jitter).

### D3 — Reach accuracy (after D2 proved insufficient)
We found open-loop scripted poses **cannot** put the hand on the cube on a
balance-controlled humanoid: the hand landed 0.3–1.1 m off because (a) the balance
policy moves the torso/waist to counterbalance the extended arm (posture differs
from any static calibration) and (b) PD actuator lag means the arm hasn't reached
the commanded pose by grasp time. Evidence: even the first cube was ~0.36 m off;
later cubes (arm returning from the bin) were 0.77–1.16 m off, and a cube was
knocked out of the bin.
Options: **light closed-loop nudge** / DifferentialIKController / accept loose grab.
- Chosen: light closed-loop **P-servo** during REACH — read the live hand
  position and nudge shoulder pitch/roll + elbow to drive the hand onto each cube,
  using probe-derived axis signs. No IK library, no Jacobian. Gate the grasp on
  convergence (hand within `REACH_TOL`) or a frame timeout.

## Calibration tool
`scripts/probe_arm.py` — pins the G1 base at the pick spot, sweeps each left-arm
joint, and prints the resulting hand (left_two_link) world position + joint limits.
This is how we learned the **axis signs/sensitivities**:
- NEGATIVE `shoulder_pitch` reaches forward/up (the original code had it backwards).
- `hand_y ≈ 0.09 + 0.26·shoulder_roll` (roll abducts to the left / +y).
- `elbow_pitch` flexes positive only (limit ≈ [−0.04, 3.24]); straighter → more reach.
- `pitch=-0.4, elbow=0.3, roll≈0 → hand ≈ (4.38, 0.09, 0.82)` on a *static* base.
Re-run it whenever the stance, stop distance, or target heights change.

## Alternative paths we may take later (NOT done)
- **DifferentialIKController for REACH/place** (option C). Proven on this exact G1
  in `scripts/vla/finetune/collect_demos.py` (`command_type="position"`, DLS,
  `lambda_val=0.1`, Jacobian from `robot.root_physx_view.get_jacobians()`, EE in
  base frame via `subtract_frame_transforms`). Most accurate; switch if we need
  precise/clutter-aware reaching. Pink IK (`pink` 3.1.0, whole-body QP) is the
  heavier option used by the `Isaac-PickPlace-…-G1` task / `m0_pinkik_grasp_check.py`.
- **Hybrid authored cube path** (option B): drive the cube along an explicit
  Cartesian spline independent of the hand for a guaranteed-smooth look.
- **Real physical grasp** instead of teleport-follow: would need a working gripper
  contact model; currently the hand pose is cosmetic and the cube is kinematically
  carried.
- **Head camera**: G1 has no head link, so the camera is mounted on `torso_link`
  with a raised z-offset (~0.52) to approximate eye height. A true head POV would
  need a head link or a fixed eye-height frame.
- **Stacking in the bin**: we spread cubes into separate `_BIN_SLOTS` to avoid
  unstable stacking; stacking is possible if desired.

## Verify
```bash
/home/trooperai/isaac-env/bin/python -u scripts/pick_place_sim.py --headless
```
Watch for: no `Traceback`/`RectLightCfg`/`setRigidBodyFlag`; per-cube
`[REACH→GRASP] … gap=` small; `[CHECK]` robot UPRIGHT and all cubes IN BIN;
`[DONE] closing simulation` then clean exit. Add `--enable_cameras` for the YOLO
inspector window (slow first-time boot — shader warmup).
```
```
Note: run with the venv Python above — the `./isaaclab.sh` wrapper is broken on
this box (it looks for a bundled `_isaac_sim/python.sh` that doesn't exist; Isaac
Sim is pip-installed in `/home/trooperai/isaac-env`).
