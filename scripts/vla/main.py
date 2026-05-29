"""
VLA Greenhouse Inspection — entry point.

Self-contained: all scene/policy/health helpers are inlined here.
The importlib approach was dropped because exec_module ran greenhouse_locomotion.py's
own argparser, which rejected --vla-mode as an unknown argument and exited immediately.

Phase 1 (--vla-mode claude):
    Camera frame + contact/health → Claude Sonnet API → HEALTHY/STRESSED/CONTINUE
Phase 2 (--vla-mode groot):
    Camera frame + joint state + language → GR00T N1.7-3B → arm joint deltas
Fallback (--vla-mode scripted, default):
    Exit INSIDE after _VLA_MAX_INSIDE_FRAMES (same as original greenhouse script)

Run:
    ./isaaclab.sh -p scripts/vla/main.py --enable_cameras
    ./isaaclab.sh -p scripts/vla/main.py --enable_cameras --vla-mode claude
    ./isaaclab.sh -p scripts/vla/main.py --enable_cameras --vla-mode groot

TODO: migrate to a proper IsaacLab extension (source/isaaclab_tasks/)
TODO: separate repo, clean from Isaac Sim, support real hardware
"""

import argparse
import math
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA greenhouse inspection")
parser.add_argument(
    "--vla-mode",
    choices=["scripted", "claude", "groot"],
    default="scripted",
    help="VLA backend (default: scripted)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All imports after AppLauncher ────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip

_CAMERAS_ENABLED = getattr(args_cli, "enable_cameras", False)
if _CAMERAS_ENABLED:
    from isaaclab.sensors import Camera, CameraCfg

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vla.action_space import InspectionAction, VLAMode

_VLA_MODE = VLAMode(args_cli.vla_mode)

# ── VLA backend ───────────────────────────────────────────────────────────────
_claude_planner = None
_groot_runner   = None

if _VLA_MODE == VLAMode.CLAUDE:
    try:
        from vla.planner import ClaudePlanner
        _claude_planner = ClaudePlanner()
    except (ImportError, EnvironmentError) as exc:
        print(f"[VLA] Cannot start Claude planner: {exc}")
        print("[VLA] Falling back to scripted mode.")
        _VLA_MODE = VLAMode.SCRIPTED

elif _VLA_MODE == VLAMode.GROOT:
    from vla.groot_runner import GR00TRunner
    _groot_runner = GR00TRunner()
    _groot_runner.load()


# ── Policy constants ──────────────────────────────────────────────────────────
_CKPT_PATH    = "/home/trooperai/IsaacLab/.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt"
_OBS_DIM      = 310
_ACTION_DIM   = 37
_HIDDEN_DIMS  = [512, 256, 128]
_ACTION_SCALE = 0.25
_HEIGHT_SCAN_N = 187

# ── Layout ────────────────────────────────────────────────────────────────────
_GH_CX, _GH_CY  = 5.0, 0.0
_BUSH_POS        = (5.0, 0.70, 0.0)
_ROBOT_START     = (2.5, 0.0, 0.74)
_INSPECT_X       = 5.0

WALK_VX          = 0.8
ARRIVE_THRESH    = 0.10
STABILISE_FRAMES = 50
RAMP_FRAMES      = 60

_VLA_MAX_INSIDE_FRAMES = 200   # hard cap — VLA decides earlier when confident
_VLA_QUERY_EVERY       = 20    # query VLA every N frames during INSIDE

_REACH_JOINTS = {
    "left_shoulder_roll_joint":  2.00,
    "left_shoulder_pitch_joint": 0.00,
    "left_elbow_pitch_joint":    0.05,
    "left_one_joint":            0.0,
    "left_two_joint":            0.0,
}

# ── Bush constants ────────────────────────────────────────────────────────────
_TRUNK_HEIGHT    = 0.80
_TRUNK_COLOR     = (0.35, 0.20, 0.08)
_CLUSTER_LAYOUT  = [
    ( 0.22,  0.00, 0.75, 0.14), ( 0.07,  0.21, 0.75, 0.14),
    (-0.18,  0.13, 0.75, 0.14), (-0.18, -0.13, 0.75, 0.14),
    ( 0.07, -0.21, 0.75, 0.14),
    ( 0.13,  0.13, 0.95, 0.13), (-0.15,  0.10, 0.95, 0.13),
    (-0.15, -0.10, 0.95, 0.13), ( 0.13, -0.13, 0.95, 0.13),
    ( 0.00,  0.00, 1.10, 0.14),
]
_CLUSTER_COLORS  = [
    (0.18, 0.55, 0.12), (0.20, 0.60, 0.10), (0.15, 0.50, 0.15),
    (0.72, 0.65, 0.10),
    (0.17, 0.52, 0.13), (0.12, 0.48, 0.16),
    (0.20, 0.55, 0.11),
    (0.62, 0.52, 0.08),
    (0.19, 0.57, 0.09), (0.14, 0.50, 0.17),
]
_CLUSTER_MASS    = 0.05
_SPRING_STIFF    = 3.0
_SPRING_DAMP     = 0.8
_SWING_LIMIT_DEG = 75.0

_LEAF_PER_CLUSTER = 3
_LEAF_RADIUS      = 0.045
_LEAF_MASS        = 0.006
_LEAF_STIFFNESS   = 0.5
_LEAF_DAMPING     = 0.10
_LEAF_SWING_DEG   = 90.0
_LEAF_OFFSETS     = [(0.13, 0.00, 0.06), (-0.07, 0.11, 0.08), (-0.07, -0.11, 0.05)]

_PNG_OUT = "/tmp/plant_inspector_latest.png"


# ── Scene helpers ─────────────────────────────────────────────────────────────
def _mat(rgb, opacity=1.0):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.6, opacity=opacity)


def _spawn_box(path, size, pos, rgb, opacity=1.0, orient=None):
    cfg = sim_utils.CuboidCfg(size=size, visual_material=_mat(rgb, opacity))
    kw  = {"translation": pos}
    if orient is not None:
        kw["orientation"] = orient
    cfg.func(path, cfg, **kw)


def build_greenhouse():
    W, D, H = 8.0, 5.0, 3.0
    ox, oy  = _GH_CX, _GH_CY
    FRAME   = (0.50, 0.44, 0.33)
    FLOOR   = (0.80, 0.72, 0.55)
    _spawn_box("/World/Greenhouse/Floor", (W, D, 0.12), (ox, oy, 0.06), FLOOR)
    wall_z = H / 2
    for i, (dx, dy) in enumerate([(-W/2, -D/2), (-W/2, D/2), (W/2, -D/2), (W/2, D/2)]):
        _spawn_box(f"/World/Greenhouse/Pillar{i}", (0.12, 0.12, H),
                   (ox+dx, oy+dy, wall_z), FRAME)


def _attach_leaves(stage, cluster_path, cx, cy, cz, color, base, ci):
    from pxr import UsdPhysics, Gf, Sdf
    for j, (ldx, ldy, ldz) in enumerate(_LEAF_OFFSETS[:_LEAF_PER_CLUSTER]):
        lp  = f"{base}/Leaf_{ci}_{j}"
        lcf = sim_utils.SphereCfg(
            radius=_LEAF_RADIUS,
            visual_material=_mat(color, opacity=0.80),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        )
        lcf.func(lp, lcf, translation=(cx+ldx, cy+ldy, cz+ldz))
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(lp)).CreateMassAttr(_LEAF_MASS)
        d6 = UsdPhysics.Joint.Define(stage, f"{base}/LeafJoint_{ci}_{j}")
        d6.CreateBody0Rel().SetTargets([Sdf.Path(cluster_path)])
        d6.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
        d6.CreateLocalPos0Attr().Set(Gf.Vec3f(ldx, ldy, ldz))
        d6.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        d6.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        d6.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        for ax in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), ax)
            lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)
        for ax in ("rotX", "rotY"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), ax)
            lim.CreateLowAttr(-_LEAF_SWING_DEG); lim.CreateHighAttr(_LEAF_SWING_DEG)
            drv = UsdPhysics.DriveAPI.Apply(d6.GetPrim(), ax)
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(_LEAF_STIFFNESS); drv.CreateDampingAttr(_LEAF_DAMPING)
            drv.CreateTargetPositionAttr(0.0)
        lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), "rotZ")
        lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)


def build_interactive_bush():
    from pxr import UsdPhysics, Gf, Sdf
    bx, by, _ = _BUSH_POS
    base  = "/World/Greenhouse/Plants/Bush"
    stage = sim_utils.get_current_stage()
    stage.DefinePrim(base, "Xform")

    trunk_path = f"{base}/Trunk"
    trunk_z    = _TRUNK_HEIGHT / 2
    tcfg = sim_utils.CuboidCfg(
        size=(0.08, 0.08, _TRUNK_HEIGHT), visual_material=_mat(_TRUNK_COLOR),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    tcfg.func(trunk_path, tcfg, translation=(bx, by, trunk_z))

    for i, (dx, dy, dz, radius) in enumerate(_CLUSTER_LAYOUT):
        cp  = f"{base}/Cluster_{i}"
        col = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        ccf = sim_utils.SphereCfg(
            radius=radius, visual_material=_mat(col, opacity=0.90),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        )
        ccf.func(cp, ccf, translation=(bx+dx, by+dy, dz))
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(cp)).CreateMassAttr(_CLUSTER_MASS)

        d6 = UsdPhysics.Joint.Define(stage, f"{base}/Joint_{i}")
        d6.CreateBody0Rel().SetTargets([Sdf.Path(trunk_path)])
        d6.CreateBody1Rel().SetTargets([Sdf.Path(cp)])
        d6.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, trunk_z))
        d6.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        d6.CreateLocalPos1Attr().Set(Gf.Vec3f(-dx, -dy, _TRUNK_HEIGHT - dz))
        d6.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        for ax in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), ax)
            lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)
        for ax in ("rotX", "rotY"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), ax)
            lim.CreateLowAttr(-_SWING_LIMIT_DEG); lim.CreateHighAttr(_SWING_LIMIT_DEG)
            drv = UsdPhysics.DriveAPI.Apply(d6.GetPrim(), ax)
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(_SPRING_STIFF); drv.CreateDampingAttr(_SPRING_DAMP)
            drv.CreateTargetPositionAttr(0.0)
        lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), "rotZ")
        lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)
        _attach_leaves(stage, cp, bx+dx, by+dy, dz, col, base, i)

    print(f"[INFO] Bush: {len(_CLUSTER_LAYOUT)} clusters + "
          f"{len(_CLUSTER_LAYOUT)*_LEAF_PER_CLUSTER} leaves")
    return base, (bx, by)


def design_scene() -> dict:
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sun = sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.98, 0.90))
    sun.func("/World/SunLight", sun, translation=(0, 0, 10),
             orientation=(0.906, 0.0, 0.423, 0.0))
    sim_utils.DomeLightCfg(intensity=600.0, color=(0.55, 0.72, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=600.0, color=(0.55, 0.72, 1.0))
    )

    build_greenhouse()
    bush_path, bush_xy = build_interactive_bush()

    robot_cfg           = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.init_state.pos = _ROBOT_START
    robot = Articulation(cfg=robot_cfg)

    contact_sensor = ContactSensor(cfg=ContactSensorCfg(
        prim_path="/World/G1/.*", update_period=0.0, history_length=1, debug_vis=False,
    ))

    camera = None
    if _CAMERAS_ENABLED:
        camera = Camera(cfg=CameraCfg(
            prim_path="/World/G1/torso_link/insp_cam",
            update_period=0.1, height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 30.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.1, 0.0, 0.25), rot=(0.7071, 0.0, 0.7071, 0.0), convention="ros",
            ),
        ))

    return {"robot": robot, "bush_xy": bush_xy,
            "contact_sensor": contact_sensor, "camera": camera}


# ── Policy ────────────────────────────────────────────────────────────────────
def load_policy(device: str) -> nn.Module:
    ckpt     = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    model_sd = ckpt["model_state_dict"]
    dims     = [_OBS_DIM] + _HIDDEN_DIMS + [_ACTION_DIM]
    layers: list[nn.Module] = []
    for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(_HIDDEN_DIMS):
            layers.append(nn.ELU())
    actor    = nn.Sequential(*layers)
    actor_sd = {k[len("actor."):]: v for k, v in model_sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd)
    actor.eval()
    print(f"[INFO] Policy loaded from {_CKPT_PATH}")
    return actor.to(device)


def build_obs(robot: Articulation, cmd: torch.Tensor,
              last_action: torch.Tensor) -> torch.Tensor:
    height_scan = torch.zeros(1, _HEIGHT_SCAN_N, device=robot.device)
    cmd_2d      = cmd.unsqueeze(0) if cmd.dim() == 1 else cmd
    return torch.cat([
        robot.data.root_lin_vel_b,
        robot.data.root_ang_vel_b,
        robot.data.projected_gravity_b,
        cmd_2d,
        robot.data.joint_pos - robot.data.default_joint_pos,
        robot.data.joint_vel,
        last_action,
        height_scan,
    ], dim=-1)


# ── Camera / health helpers ───────────────────────────────────────────────────
_last_health = (0.0, 0.0)


def _analyze_health(rgb_np: np.ndarray) -> tuple:
    img = rgb_np.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    veg     = g > 0.20
    healthy = veg & (g > r * 1.35) & (g > b * 1.35)
    stressed = veg & (r > 0.28) & (g > 0.28) & (g < r * 1.20) & (b < 0.25)
    n_veg = float(veg.sum())
    if n_veg < 200:
        return 0.0, 0.0
    return healthy.sum() / n_veg * 100.0, stressed.sum() / n_veg * 100.0


def _show_preview(rgb_np: np.ndarray) -> None:
    from PIL import Image as PILImage, ImageDraw
    h_pct, s_pct = _last_health
    display = rgb_np.copy()
    img_f   = display.astype(np.float32) / 255.0
    r_ch, g_ch, b_ch = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    mask = (r_ch > 0.28) & (g_ch > 0.28) & (g_ch < r_ch * 1.20) & (b_ch < 0.25)
    if mask.any():
        overlay = display.copy()
        overlay[mask] = [220, 60, 60]
        display = (display * 0.55 + overlay * 0.45).astype(np.uint8)
    pil  = PILImage.fromarray(display)
    draw = ImageDraw.Draw(pil)
    verdict = "STRESSED" if s_pct > 15.0 else "HEALTHY"
    col     = (220, 60, 60) if s_pct > 15.0 else (60, 200, 60)
    draw.rectangle([0, 0, 300, 60], fill=(20, 20, 20))
    draw.text((8,  6), verdict, fill=col)
    draw.text((8, 34), f"Green {h_pct:.0f}%   Stressed {s_pct:.0f}%", fill=(200, 200, 200))
    pil.save(_PNG_OUT)


# ── Simulation setup ──────────────────────────────────────────────────────────
sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.0, 6.0, 4.0], [5.0, 0.5, 0.8])

scene        = design_scene()
robot        = scene["robot"]
contact_sens = scene["contact_sensor"]
camera       = scene.get("camera")

if _groot_runner is not None:
    _groot_runner.bind_robot(robot.joint_names)

sim.reset()

device      = sim.device
policy      = load_policy(device)
last_action = torch.zeros(1, _ACTION_DIM, device=device)
cmd         = torch.zeros(1, 3, device=device)

default_jpos  = robot.data.default_joint_pos.clone()
reach_jpos    = default_jpos.clone()
_name_to_idx  = {n: i for i, n in enumerate(robot.joint_names)}

for jname, angle in _REACH_JOINTS.items():
    if jname in _name_to_idx:
        reach_jpos[0, _name_to_idx[jname]] = angle

# ── State machine ─────────────────────────────────────────────────────────────
from enum import Enum

class State(Enum):
    WALK = "WALK"; ARRIVE = "ARRIVE"
    REACH_IN = "REACH_IN"; INSIDE = "INSIDE"; REACH_OUT = "REACH_OUT"; DONE = "DONE"

state          = State.WALK
frame_in_state = 0
_vla_decision: InspectionAction | None = None

sim_dt = sim.get_physics_dt()
print(f"\n[VLA] Mode: {_VLA_MODE.value}  |  Cameras: {_CAMERAS_ENABLED}")
print(f"[VLA] State machine: WALK → ARRIVE → REACH_IN → INSIDE → REACH_OUT → DONE\n")
if _CAMERAS_ENABLED:
    print(f"[PREVIEW] Annotated frames → {_PNG_OUT}")
    print(f"[PREVIEW] View live:  feh --auto-reload {_PNG_OUT}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
while simulation_app.is_running():
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)
    contact_sens.update(sim_dt)

    # Camera
    rgb = None
    if camera is not None and _CAMERAS_ENABLED:
        camera.update(sim_dt)
        out = camera.data.output.get("rgb")
        if out is not None:
            rgb = out[0, :, :, :3].cpu().numpy()
            h_pct, s_pct = _analyze_health(rgb)
            _last_health  = (h_pct, s_pct)
            _show_preview(rgb)

    # Contact
    forces    = contact_sens.data.net_forces_w
    max_force = forces.norm(dim=-1).max().item() if forces is not None else 0.0

    # Policy
    obs = build_obs(robot, cmd[0], last_action)
    with torch.inference_mode():
        action = policy(obs)
    action      = action.clone()
    last_action = action.clone()

    root_x = robot.data.root_pos_w[0, 0].item()

    # ── State transitions ─────────────────────────────────────────────────────
    if state == State.WALK:
        cmd[0, 0] = WALK_VX
        if root_x >= _INSPECT_X - ARRIVE_THRESH:
            state = State.ARRIVE; frame_in_state = 0; cmd[:] = 0.0
            print(f"[WALK→ARRIVE] x={root_x:.2f}")

    elif state == State.ARRIVE:
        if frame_in_state >= STABILISE_FRAMES:
            state = State.REACH_IN; frame_in_state = 0
            print("[ARRIVE→REACH_IN]")

    elif state == State.REACH_IN:
        alpha  = min(frame_in_state / RAMP_FRAMES, 1.0)
        interp = torch.lerp(default_jpos, reach_jpos, alpha)
        for jname in _REACH_JOINTS:
            if jname in _name_to_idx:
                idx = _name_to_idx[jname]
                action[0, idx] = (interp[0, idx] - default_jpos[0, idx]) / _ACTION_SCALE
        if frame_in_state >= RAMP_FRAMES:
            state = State.INSIDE; frame_in_state = 0
            print("[REACH_IN→INSIDE]")

    elif state == State.INSIDE:
        for jname in _REACH_JOINTS:
            if jname in _name_to_idx:
                idx = _name_to_idx[jname]
                action[0, idx] = (reach_jpos[0, idx] - default_jpos[0, idx]) / _ACTION_SCALE

        if frame_in_state % _VLA_QUERY_EVERY == 0 and rgb is not None:
            h_pct, s_pct = _last_health

            if _VLA_MODE == VLAMode.CLAUDE and _claude_planner:
                _vla_decision, _ = _claude_planner.decide(rgb, max_force, h_pct, s_pct)

            elif _VLA_MODE == VLAMode.GROOT and _groot_runner:
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                delta     = _groot_runner.act(rgb, joint_pos)
                for jname, j_idx in _groot_runner._arm_indices.items():
                    g1_idx = _name_to_idx.get(jname)
                    if g1_idx is not None:
                        reach_jpos[0, g1_idx] += float(delta[j_idx])
                if s_pct > 15.0:
                    _vla_decision = InspectionAction.STRESSED
                elif h_pct > 40.0 and s_pct < 8.0:
                    _vla_decision = InspectionAction.HEALTHY

            else:  # scripted
                if frame_in_state >= _VLA_MAX_INSIDE_FRAMES:
                    _vla_decision = InspectionAction.HEALTHY

        done = (
            _vla_decision in (InspectionAction.HEALTHY, InspectionAction.STRESSED)
            or frame_in_state >= _VLA_MAX_INSIDE_FRAMES
        )
        if done:
            h_pct, s_pct = _last_health
            verdict = "STRESSED" if (
                _vla_decision == InspectionAction.STRESSED or s_pct > 15.0
            ) else "HEALTHY"
            print(f"\n[VLA REPORT] mode={_VLA_MODE.value}  decision={_vla_decision}  verdict={verdict}")
            print(f"  Green: {h_pct:.0f}%   Stressed: {s_pct:.0f}%\n")
            state = State.REACH_OUT; frame_in_state = 0
            print("[INSIDE→REACH_OUT]")

    elif state == State.REACH_OUT:
        alpha  = 1.0 - min(frame_in_state / RAMP_FRAMES, 1.0)
        interp = torch.lerp(default_jpos, reach_jpos, alpha)
        for jname in _REACH_JOINTS:
            if jname in _name_to_idx:
                idx = _name_to_idx[jname]
                action[0, idx] = (interp[0, idx] - default_jpos[0, idx]) / _ACTION_SCALE
        if frame_in_state >= RAMP_FRAMES:
            state = State.DONE; frame_in_state = 0
            print("[REACH_OUT→DONE]")

    elif state == State.DONE:
        cmd[:] = 0.0

    robot.set_joint_position_target(action * _ACTION_SCALE + default_jpos)
    frame_in_state += 1

simulation_app.close()
