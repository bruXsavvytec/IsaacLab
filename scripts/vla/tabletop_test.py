"""
GR00T tabletop manipulation test.

A clean capability probe for NVIDIA GR00T N1.7-3B, separate from the greenhouse
inspection demo (main.py). The G1 stands at a table holding a blue cube and a
yellow cube, and GR00T drives BOTH arms toward the task:

    "pick up the blue cube and put it on top of the yellow cube"

The robot base is ANCHORED (fix_root_link=True) — no locomotion policy at all.
A zero-velocity locomotion policy drifts and the robot walks into the table, so
for a pure manipulation test we pin the pelvis and only drive joint targets:
legs hold the default standing pose, GR00T drives the arms. This isolates GR00T's
arm control completely — nothing can walk.

State machine:  STABILISE → READY (ramp arms to a reach-ready pose) → MANIPULATE

Run:
    cd /home/trooperai/IsaacLab
    # GR00T (default) — needs the server running, or it loads the model directly
    ./isaaclab.sh -p scripts/vla/tabletop_test.py --enable_cameras
    # just hold the ready pose (no model) — handy for checking the scene/camera
    ./isaaclab.sh -p scripts/vla/tabletop_test.py --enable_cameras --vla-mode scripted

Annotated camera preview → /tmp/groot_tabletop_latest.png  (feh --auto-reload to watch)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T tabletop manipulation test")
parser.add_argument("--vla-mode", choices=["scripted", "groot"], default="groot",
                    help="groot = GR00T drives the arms; scripted = just hold ready pose")
parser.add_argument("--instruction",
                    default="pick up the blue cube and put it on top of the yellow cube",
                    help="language instruction passed to GR00T")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after AppLauncher ──────────────────────────────────────────────────
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip

_CAMERAS_ENABLED = getattr(args_cli, "enable_cameras", False)
if _CAMERAS_ENABLED:
    from isaaclab.sensors import Camera, CameraCfg

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_VLA_MODE = args_cli.vla_mode

# ── VLA backend ─────────────────────────────────────────────────────────────────
_groot_runner = None
if _VLA_MODE == "groot":
    if not _CAMERAS_ENABLED:
        print("[VLA] groot mode needs --enable_cameras (GR00T is blind without it).")
        print("[VLA] Falling back to scripted mode.")
        _VLA_MODE = "scripted"
    else:
        from vla.groot_runner import GR00TRunner
        _groot_runner = GR00TRunner(instruction=args_cli.instruction)
        _groot_runner.load()

# ── Layout ───────────────────────────────────────────────────────────────────────
# Robot at the origin facing +X (base anchored). Table just ahead, within reach.
_ROBOT_START   = (0.0, 0.0, 0.74)

_TABLE_TOP_Z    = 0.74                 # height of the table's top surface
_TABLE_CENTER   = (0.60, 0.0)          # (x, y) — 0.6 m in front of the robot
_TABLE_TOP_SIZE = (0.70, 0.90, 0.04)   # (x, y, thickness)
_LEG            = 0.06

_CUBE      = 0.05
_CUBE_Z    = _TABLE_TOP_Z + _CUBE / 2 + 0.002      # rest on the table top
# Both cubes biased to the robot's left (+y) so the LEFT arm can reach them.
_BLUE_POS   = (0.50, 0.16, _CUBE_Z)
_YELLOW_POS = (0.55, 0.02, _CUBE_Z)
_BLUE_RGB   = (0.10, 0.20, 0.85)
_YELLOW_RGB = (0.90, 0.80, 0.10)

# Reach-ready pose: lift BOTH arms forward over the table, elbows bent so the
# hands come down toward the table surface. Positive shoulder_pitch = forward
# flexion (default standing pose is +0.35); larger lifts the arm forward/up.
# Roll is mirrored (+left / -right). Tune by eye against the preview.
_READY_JOINTS = {
    "left_shoulder_pitch_joint":  1.20,
    "right_shoulder_pitch_joint": 1.20,
    "left_shoulder_roll_joint":   0.20,
    "right_shoulder_roll_joint": -0.20,
    "left_elbow_pitch_joint":     1.20,
    "right_elbow_pitch_joint":    1.20,
}

STABILISE_FRAMES = 40      # settle the anchored pose
RAMP_FRAMES      = 80      # ramp from default pose to the ready pose
_APPLY_EVERY     = 4       # advance one GR00T chunk step every N sim frames

_PNG_OUT = "/tmp/groot_tabletop_latest.png"


# ── Scene helpers ─────────────────────────────────────────────────────────────────
def _mat(rgb):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.7)


def build_table():
    cx, cy = _TABLE_CENTER
    tw, td, tt = _TABLE_TOP_SIZE
    top_center_z = _TABLE_TOP_Z - tt / 2
    top = sim_utils.CuboidCfg(
        size=(tw, td, tt), visual_material=_mat((0.55, 0.40, 0.25)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    top.func("/World/Table/Top", top, translation=(cx, cy, top_center_z))

    leg_h = _TABLE_TOP_Z - tt
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lx = cx + sx * (tw / 2 - _LEG)
        ly = cy + sy * (td / 2 - _LEG)
        leg = sim_utils.CuboidCfg(
            size=(_LEG, _LEG, leg_h), visual_material=_mat((0.40, 0.28, 0.16)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        leg.func(f"/World/Table/Leg{i}", leg, translation=(lx, ly, leg_h / 2))


def spawn_cube(path, pos, rgb):
    cfg = sim_utils.CuboidCfg(
        size=(_CUBE, _CUBE, _CUBE), visual_material=_mat(rgb),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
    )
    cfg.func(path, cfg, translation=pos)


def design_scene() -> dict:
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sun = sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.98, 0.90))
    sun.func("/World/SunLight", sun, translation=(0, 0, 10),
             orientation=(0.906, 0.0, 0.423, 0.0))
    sim_utils.DomeLightCfg(intensity=600.0, color=(0.85, 0.88, 0.95)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=600.0, color=(0.85, 0.88, 0.95))
    )

    build_table()
    spawn_cube("/World/BlueCube", _BLUE_POS, _BLUE_RGB)
    spawn_cube("/World/YellowCube", _YELLOW_POS, _YELLOW_RGB)

    robot_cfg           = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.init_state.pos = _ROBOT_START
    robot_cfg.spawn.articulation_props.fix_root_link = True   # anchor the pelvis
    robot = Articulation(cfg=robot_cfg)

    camera = None
    if _CAMERAS_ENABLED:
        camera = Camera(cfg=CameraCfg(
            prim_path="/World/G1/torso_link/insp_cam",
            update_period=0.1, height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 30.0)),
            offset=CameraCfg.OffsetCfg(
                # Upright (no roll), pitched 40° down at the table; mounted high on
                # the torso so the raised arms enter the lower frame (GR00T servos
                # to its own hands). Computed, not eyeballed.
                pos=(0.10, 0.0, 0.28), rot=(-0.2988, 0.6409, -0.6409, 0.2988),
                convention="ros",
            ),
        ))

    return {"robot": robot, "camera": camera}


def _show_preview(rgb_np: np.ndarray, label: str) -> None:
    from PIL import Image as PILImage, ImageDraw
    pil  = PILImage.fromarray(rgb_np)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, 639, 26], fill=(20, 20, 20))
    draw.text((6, 8), label, fill=(230, 230, 230))
    pil.save(_PNG_OUT)


# ── Simulation setup ─────────────────────────────────────────────────────────────
sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([1.6, 1.6, 1.4], [0.5, 0.0, 0.8])   # viewer looks at the table

scene  = design_scene()
robot  = scene["robot"]
camera = scene.get("camera")

sim.reset()

if _groot_runner is not None:
    _groot_runner.bind_robot(robot.joint_names)

default_jpos = robot.data.default_joint_pos.clone()
_name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}

# Reach-ready pose (full-body default with the ready arm joints overridden).
ready_pose = default_jpos.clone()
for jname, angle in _READY_JOINTS.items():
    if jname in _name_to_idx:
        ready_pose[0, _name_to_idx[jname]] = angle

# Arm target GR00T writes its absolute joint targets into (starts at ready pose).
arm_target = ready_pose.clone()

# GR00T's bound arm joints (absolute targets get assigned to these in MANIPULATE).
groot_idx = _groot_runner.arm_dof_indices() if _groot_runner is not None else []
# DOF indices we override away from the default standing pose.
arm_drive_idx = sorted(set(groot_idx) |
                       {_name_to_idx[j] for j in _READY_JOINTS if j in _name_to_idx})

# GR00T action chunk being executed (T, DOF) and the step pointer into it.
groot_chunk = None
chunk_step  = 0

# ── State machine ─────────────────────────────────────────────────────────────────
from enum import Enum


class State(Enum):
    STABILISE = "STABILISE"; READY = "READY"; MANIPULATE = "MANIPULATE"


state          = State.STABILISE
frame_in_state = 0
sim_dt         = sim.get_physics_dt()

print(f"\n[VLA] Tabletop test | mode={_VLA_MODE} | cameras={_CAMERAS_ENABLED} | base=ANCHORED")
print(f"[VLA] Instruction: {args_cli.instruction!r}")
print(f"[VLA] State machine: STABILISE → READY → MANIPULATE")
if _CAMERAS_ENABLED:
    print(f"[PREVIEW] {_PNG_OUT}   (feh --auto-reload {_PNG_OUT})\n")


def _drive(cmd: torch.Tensor, source: torch.Tensor) -> None:
    """Copy `source`'s arm-joint values into the command pose."""
    for idx in arm_drive_idx:
        cmd[0, idx] = source[0, idx]


# ── Main loop ──────────────────────────────────────────────────────────────────
while simulation_app.is_running():
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)

    rgb = None
    if camera is not None and _CAMERAS_ENABLED:
        camera.update(sim_dt)
        out = camera.data.output.get("rgb")
        if out is not None:
            rgb = out[0, :, :, :3].cpu().numpy()
            _show_preview(rgb, f"{state.value} | {args_cli.instruction}")
            if _groot_runner is not None:
                _groot_runner.push_frame(rgb)   # build the t-20 video history

    # Command pose: default standing legs/torso, arms set per state below.
    cmd_pose = default_jpos.clone()

    if state == State.STABILISE:
        if frame_in_state >= STABILISE_FRAMES:
            state = State.READY; frame_in_state = 0
            print("[STABILISE→READY]")

    elif state == State.READY:
        alpha  = min(frame_in_state / RAMP_FRAMES, 1.0)
        interp = torch.lerp(default_jpos, ready_pose, alpha)
        _drive(cmd_pose, interp)
        if frame_in_state >= RAMP_FRAMES:
            state = State.MANIPULATE; frame_in_state = 0
            print("[READY→MANIPULATE]  GR00T now driving the arms")

    elif state == State.MANIPULATE:
        if _groot_runner is not None and rgb is not None:
            # Re-query when we've executed the whole chunk (or on first entry).
            if groot_chunk is None or chunk_step >= len(groot_chunk):
                joint_pos   = robot.data.joint_pos[0].cpu().numpy()
                groot_chunk = _groot_runner.act(rgb, joint_pos)
                chunk_step  = 0
                print(f"[GR00T] new chunk T={len(groot_chunk)}  "
                      f"|target|max={np.abs(groot_chunk).max():.3f} rad")
            # GR00T returns ABSOLUTE joint targets (referenced to the arm state we
            # fed in), so ASSIGN them — never accumulate. One step / _APPLY_EVERY.
            if frame_in_state % _APPLY_EVERY == 0:
                target_step = groot_chunk[chunk_step]
                for idx in groot_idx:
                    arm_target[0, idx] = float(target_step[idx])
                chunk_step += 1
        _drive(cmd_pose, arm_target)

    robot.set_joint_position_target(cmd_pose)
    frame_in_state += 1

simulation_app.close()
