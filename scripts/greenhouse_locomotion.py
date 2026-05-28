"""
Greenhouse locomotion + inspection demo — IsaacLab 2.x / Isaac Sim 4.x

The Unitree G1 uses its pre-trained RSL-RL rough-terrain policy to walk
naturally through the greenhouse, stops at the inspection position, then
extends its left arm into the interactive spring bush and retracts.

How the two control modes combine
──────────────────────────────────
  WALK / ARRIVE / DONE phase:
      policy(obs) → 37 joint targets → PD actuators → real dynamics

  REACH_IN / INSIDE / REACH_OUT phase:
      policy still runs for LEGS (keeps robot balanced while standing)
      arm joint targets are overridden with interpolated reach values on top

Observation layout (310-dim, matches Isaac-Velocity-Rough-G1-v0 training):
  [0:3]    base_lin_vel_b        body-frame linear velocity
  [3:6]    base_ang_vel_b        body-frame angular velocity
  [6:9]    projected_gravity_b   gravity in body frame (IMU proxy)
  [9:12]   velocity_commands     [vx, vy, wz] — injected by state machine
  [12:49]  joint_pos_rel         joint_pos − default_joint_pos  (37)
  [49:86]  joint_vel             (37)
  [86:123] last_action           previous policy output  (37)
  [123:310] height_scan          zeros — greenhouse floor is flat  (187)

Run from the IsaacLab root:
    ./isaaclab.sh -p scripts/greenhouse_locomotion.py
    ./isaaclab.sh -p scripts/greenhouse_locomotion.py --enable_cameras
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Greenhouse locomotion + inspection demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

_CAMERAS_ENABLED = getattr(args_cli, "enable_cameras", False)
if _CAMERAS_ENABLED:
    from isaaclab.sensors import Camera, CameraCfg

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip


# ---------------------------------------------------------------------------
# Policy constants — must match checkpoint architecture exactly
# ---------------------------------------------------------------------------

_CKPT_PATH      = "/home/trooperai/IsaacLab/.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt"
_OBS_DIM        = 310
_ACTION_DIM     = 37
_HIDDEN_DIMS    = [512, 256, 128]
_ACTION_SCALE   = 0.25      # JointPositionActionCfg scale used during training
_HEIGHT_SCAN_N  = 187       # flat greenhouse floor → zeros is a valid approximation


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_GH_CX, _GH_CY = 5.0, 0.0
_BUSH_POS       = (5.0, 0.70, 0.0)
_ROBOT_START    = (2.5, 0.0, 0.74)
_INSPECT_X      = 5.0           # robot stops when root X reaches this

WALK_VX         = 0.8           # forward velocity command (m/s)
ARRIVE_THRESH   = 0.10          # stop threshold (m)
STABILISE_FRAMES = 50           # frames to let robot slow down before reaching
RAMP_FRAMES     = 60
HOLD_FRAMES     = 80

# Arm reach: shoulder_roll=2.0 rad (115°) angles arm down-and-sideways to z≈0.94 m
_REACH_JOINTS = {
    "left_shoulder_roll_joint":  2.00,
    "left_shoulder_pitch_joint": 0.00,
    "left_elbow_pitch_joint":    0.05,
    "left_one_joint":            0.0,
    "left_two_joint":            0.0,
}


# ---------------------------------------------------------------------------
# Interactive bush geometry (identical to greenhouse_sim.py)
# ---------------------------------------------------------------------------

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
    (0.72, 0.65, 0.10),  # index 3 — stressed yellow (nitrogen deficiency)
    (0.17, 0.52, 0.13), (0.12, 0.48, 0.16),
    (0.20, 0.55, 0.11),
    (0.62, 0.52, 0.08),  # index 7 — stressed yellow-brown (root stress)
    (0.19, 0.57, 0.09),
    (0.14, 0.50, 0.17),
]
_UNHEALTHY_CLUSTER_INDICES = {3, 7}
_CLUSTER_MASS    = 0.05
_SPRING_STIFF    = 3.0    # N·m/rad — soft; clusters deflect easily
_SPRING_DAMP     = 0.8    # N·m·s/rad — low damping → natural oscillation
_SWING_LIMIT_DEG = 75.0

# ── Second-level leaf sub-clusters ──────────────────────────────────────────
_LEAF_PER_CLUSTER = 3
_LEAF_RADIUS      = 0.045
_LEAF_MASS        = 0.006
_LEAF_STIFFNESS   = 0.5
_LEAF_DAMPING     = 0.10
_LEAF_SWING_DEG   = 90.0
_LEAF_OFFSETS     = [
    ( 0.13,  0.00,  0.06),
    (-0.07,  0.11,  0.08),
    (-0.07, -0.11,  0.05),
]


# ---------------------------------------------------------------------------
# Policy — load actor MLP directly from RSL-RL checkpoint
# ---------------------------------------------------------------------------

def load_policy(device: str) -> nn.Module:
    """Build actor MLP and load weights from the pre-trained checkpoint.

    Old rsl_rl (< 4.0.0) format: ckpt["model_state_dict"]["actor.N.weight"].
    We bypass OnPolicyRunner entirely — load just the actor branch.
    """
    ckpt     = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    model_sd = ckpt["model_state_dict"]

    dims = [_OBS_DIM] + _HIDDEN_DIMS + [_ACTION_DIM]
    layers: list[nn.Module] = []
    for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(_HIDDEN_DIMS):
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)

    actor_sd = {k[len("actor."):]: v for k, v in model_sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd)
    actor.eval()
    print(f"[INFO] Policy loaded from {_CKPT_PATH}")
    return actor.to(device)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_obs(robot: Articulation, cmd: torch.Tensor,
              last_action: torch.Tensor) -> torch.Tensor:
    """Assemble 310-dim observation from live robot state."""
    height_scan = torch.zeros(1, _HEIGHT_SCAN_N, device=robot.device)
    cmd_2d      = cmd.unsqueeze(0) if cmd.dim() == 1 else cmd    # → (1, 3)
    return torch.cat([
        robot.data.root_lin_vel_b,                                 # 3
        robot.data.root_ang_vel_b,                                 # 3
        robot.data.projected_gravity_b,                            # 3
        cmd_2d,                                                    # 3  → 12
        robot.data.joint_pos - robot.data.default_joint_pos,      # 37 → 49
        robot.data.joint_vel,                                      # 37 → 86
        last_action,                                               # 37 → 123
        height_scan,                                               # 187 → 310
    ], dim=-1)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _mat(rgb, opacity=1.0):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.6, opacity=opacity)


def _spawn_box(path, size, pos, rgb, opacity=1.0, orient=None):
    cfg = sim_utils.CuboidCfg(size=size, visual_material=_mat(rgb, opacity))
    kw  = {"translation": pos}
    if orient is not None:
        kw["orientation"] = orient
    cfg.func(path, cfg, **kw)


def build_greenhouse(ox=_GH_CX, oy=_GH_CY, oz=0.0):
    W, D, H = 8.0, 5.0, 3.0
    FRAME = (0.50, 0.44, 0.33)
    FLOOR = (0.80, 0.72, 0.55)
    _spawn_box("/World/Greenhouse/Floor", (W, D, 0.12), (ox, oy, oz + 0.06), FLOOR)
    wall_z = oz + H / 2
    # Glass walls commented out — restore when dynamics are validated
    # _spawn_box("/World/Greenhouse/WallFront", (W, 0.08, H), (ox, oy-D/2, wall_z), (0.72,0.94,0.84), opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallBack",  (W, 0.08, H), (ox, oy+D/2, wall_z), (0.72,0.94,0.84), opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallLeft",  (0.08, D, H), (ox-W/2, oy, wall_z), (0.72,0.94,0.84), opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallRight", (0.08, D, H), (ox+W/2, oy, wall_z), (0.72,0.94,0.84), opacity=0.35)
    for i, (dx, dy) in enumerate([(-W/2, -D/2), (-W/2, D/2), (W/2, -D/2), (W/2, D/2)]):
        _spawn_box(f"/World/Greenhouse/Pillar{i}", (0.12, 0.12, H), (ox+dx, oy+dy, wall_z), FRAME)


def _attach_leaves(stage, cluster_path: str, cx: float, cy: float, cz: float,
                   leaf_color: tuple, base: str, ci: int) -> None:
    from pxr import UsdPhysics, Gf, Sdf
    for j, (ldx, ldy, ldz) in enumerate(_LEAF_OFFSETS[:_LEAF_PER_CLUSTER]):
        leaf_path = f"{base}/Leaf_{ci}_{j}"
        leaf_cfg  = sim_utils.SphereCfg(
            radius=_LEAF_RADIUS,
            visual_material=_mat(_CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)], opacity=0.80),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        )
        leaf_cfg.func(leaf_path, leaf_cfg, translation=(cx + ldx, cy + ldy, cz + ldz))
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(leaf_path)).CreateMassAttr(_LEAF_MASS)

        d6 = UsdPhysics.Joint.Define(stage, f"{base}/LeafJoint_{ci}_{j}")
        d6.CreateBody0Rel().SetTargets([Sdf.Path(cluster_path)])
        d6.CreateBody1Rel().SetTargets([Sdf.Path(leaf_path)])
        d6.CreateLocalPos0Attr().Set(Gf.Vec3f(ldx, ldy, ldz))
        d6.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        d6.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        d6.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), axis)
            lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)
        for axis in ("rotX", "rotY"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), axis)
            lim.CreateLowAttr(-_LEAF_SWING_DEG); lim.CreateHighAttr(_LEAF_SWING_DEG)
            drv = UsdPhysics.DriveAPI.Apply(d6.GetPrim(), axis)
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(_LEAF_STIFFNESS); drv.CreateDampingAttr(_LEAF_DAMPING)
            drv.CreateTargetPositionAttr(0.0)
        lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), "rotZ")
        lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)


def build_interactive_bush() -> tuple[str, tuple]:
    from pxr import UsdPhysics, Gf, Sdf
    bx, by, _ = _BUSH_POS
    base       = "/World/Greenhouse/Plants/Bush"
    stage      = sim_utils.get_current_stage()
    stage.DefinePrim(base, "Xform")

    trunk_path = f"{base}/Trunk"
    trunk_z    = _TRUNK_HEIGHT / 2
    trunk_cfg  = sim_utils.CuboidCfg(
        size=(0.08, 0.08, _TRUNK_HEIGHT),
        visual_material=_mat(_TRUNK_COLOR),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    trunk_cfg.func(trunk_path, trunk_cfg, translation=(bx, by, trunk_z))

    for i, (dx, dy, dz, radius) in enumerate(_CLUSTER_LAYOUT):
        cluster_path = f"{base}/Cluster_{i}"
        cluster_cfg  = sim_utils.SphereCfg(
            radius=radius,
            visual_material=_mat(_CLUSTER_COLORS[i % len(_CLUSTER_COLORS)], opacity=0.90),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        )
        cluster_cfg.func(cluster_path, cluster_cfg, translation=(bx+dx, by+dy, dz))

        cluster_prim = stage.GetPrimAtPath(cluster_path)
        UsdPhysics.MassAPI.Apply(cluster_prim).CreateMassAttr(_CLUSTER_MASS)

        joint_path = f"{base}/Joint_{i}"
        d6 = UsdPhysics.Joint.Define(stage, joint_path)
        d6.CreateBody0Rel().SetTargets([Sdf.Path(trunk_path)])
        d6.CreateBody1Rel().SetTargets([Sdf.Path(cluster_path)])
        d6.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, trunk_z))
        d6.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        d6.CreateLocalPos1Attr().Set(Gf.Vec3f(-dx, -dy, _TRUNK_HEIGHT - dz))
        d6.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), axis)
            lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)
        for axis in ("rotX", "rotY"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), axis)
            lim.CreateLowAttr(-_SWING_LIMIT_DEG); lim.CreateHighAttr(_SWING_LIMIT_DEG)
            drv = UsdPhysics.DriveAPI.Apply(d6.GetPrim(), axis)
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(_SPRING_STIFF); drv.CreateDampingAttr(_SPRING_DAMP)
            drv.CreateTargetPositionAttr(0.0)
        lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), "rotZ")
        lim.CreateLowAttr(0.0); lim.CreateHighAttr(0.0)

        _attach_leaves(stage, cluster_path, bx+dx, by+dy, dz,
                       _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)], base, i)

    n_leaves = len(_CLUSTER_LAYOUT) * _LEAF_PER_CLUSTER
    print(f"[INFO] Soft bush: {len(_CLUSTER_LAYOUT)} clusters + {n_leaves} leaves  "
          f"(cluster k={_SPRING_STIFF}, leaf k={_LEAF_STIFFNESS})")
    return base, (bx, by)


# ---------------------------------------------------------------------------
# Full scene
# ---------------------------------------------------------------------------

def design_scene() -> dict:
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sun = sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.98, 0.90))
    sun.func("/World/SunLight", sun, translation=(0, 0, 10),
             orientation=(0.906, 0.0, 0.423, 0.0))
    dome = sim_utils.DomeLightCfg(intensity=600.0, color=(0.55, 0.72, 1.0))
    dome.func("/World/DomeLight", dome)

    build_greenhouse()
    bush_path, bush_xy = build_interactive_bush()

    # G1_MINIMAL_CFG — same joints as G1_CFG, lighter collision meshes, matches checkpoint
    robot_cfg             = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path   = "/World/G1"
    robot_cfg.init_state.pos = _ROBOT_START
    # Root is NOT fixed — physics + policy keep the robot upright during WALK

    robot = Articulation(cfg=robot_cfg)
    contact_sensor = ContactSensor(cfg=ContactSensorCfg(
        prim_path="/World/G1/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    ))

    camera = None
    if _CAMERAS_ENABLED:
        camera = Camera(cfg=CameraCfg(
            prim_path="/World/G1/torso_link/insp_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 30.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.1, 0.0, 0.25),
                rot=(0.7071, 0.0, 0.7071, 0.0),
                convention="ros",
            ),
        ))

    return {"robot": robot, "bush_xy": bush_xy,
            "contact_sensor": contact_sensor, "camera": camera}


# ---------------------------------------------------------------------------
# Contact helper
# ---------------------------------------------------------------------------

_contact_cooldown = 0

# Only report contacts on non-leg links (ankle/foot contact during walking is normal noise).
# Leg link names contain these substrings — suppress them during walk.
_LEG_LINK_SUBSTRINGS = ("hip", "knee", "ankle", "calf", "thigh")


def _check_contact(sensor: ContactSensor, phase: str) -> None:
    """Print contact alerts only during arm-reach phases and only for non-leg links."""
    global _contact_cooldown
    if _contact_cooldown > 0:
        _contact_cooldown -= 1
        return
    if phase not in ("reach_in", "inside", "reach_out"):
        return   # ankle/foot contacts during walking are expected — skip
    if not sensor.is_initialized:
        return
    forces = sensor.data.net_forces_w
    if forces is None:
        return
    magnitudes = forces.norm(dim=-1)[0]   # (N_bodies,)
    for body_idx in range(len(sensor.body_names)):
        name = sensor.body_names[body_idx]
        if any(s in name for s in _LEG_LINK_SUBSTRINGS):
            continue   # ignore foot/leg contacts
        mag = magnitudes[body_idx].item()
        if mag > 5.0:
            print(f"[CONTACT] '{name}' touching cluster — {mag:.1f} N")
            _contact_cooldown = 100
            return


# ---------------------------------------------------------------------------
# YOLO plant health inspector + live preview (omni.ui floating window)
# ---------------------------------------------------------------------------

_YOLO_MODEL        = None
_YOLO_READY        = False
_yolo_call_n       = 0
_YOLO_EVERY        = 10
_health_log: list  = []
_last_yolo_results = None
_last_health       = (0.0, 0.0)
_PNG_OUT           = "/tmp/plant_inspector_latest.png"


def _init_yolo() -> None:
    global _YOLO_MODEL, _YOLO_READY
    try:
        from ultralytics import YOLO  # type: ignore
        _YOLO_MODEL = YOLO("yolov8n.pt")
        _YOLO_READY = True
        print("[YOLO] YOLOv8n loaded — visual inspection active")
    except Exception as exc:
        print(f"[YOLO] Disabled ({exc})")
        print("[YOLO]   pip install ultralytics")
    print(f"[PREVIEW] Annotated frames saved to {_PNG_OUT}")
    print(f"[PREVIEW] View live in a second terminal:  feh --auto-reload {_PNG_OUT}")


def _analyze_health(rgb_np: np.ndarray) -> tuple:
    img = rgb_np.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    veg      = g > 0.20
    healthy  = veg & (g > r * 1.35) & (g > b * 1.35)
    stressed = veg & (r > 0.28) & (g > 0.28) & (g < r * 1.20) & (b < 0.25)
    n_veg = float(veg.sum())
    if n_veg < 200:
        return 0.0, 0.0
    return healthy.sum() / n_veg * 100.0, stressed.sum() / n_veg * 100.0


def _show_preview(rgb_np: np.ndarray) -> None:
    """Save annotated camera frame to /tmp/plant_inspector_latest.png.

    Open in a second terminal with:  feh --auto-reload /tmp/plant_inspector_latest.png
    """
    from PIL import Image as PILImage, ImageDraw
    h_pct, s_pct = _last_health
    display = rgb_np.copy()

    # Tint stressed pixels red
    img_f = display.astype(np.float32) / 255.0
    r_ch, g_ch, b_ch = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    stressed_mask = (r_ch > 0.28) & (g_ch > 0.28) & (g_ch < r_ch * 1.20) & (b_ch < 0.25)
    if stressed_mask.any():
        overlay = display.copy()
        overlay[stressed_mask] = [220, 60, 60]
        display = (display * 0.55 + overlay * 0.45).astype(np.uint8)

    pil_img = PILImage.fromarray(display)
    draw    = ImageDraw.Draw(pil_img)
    if _last_yolo_results is not None:
        boxes = _last_yolo_results[0].boxes
        names = _last_yolo_results[0].names
        for box in boxes:
            if float(box.conf) < 0.35:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            label = f"{names[int(box.cls)]} {float(box.conf):.0%}"
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 80), width=2)
            draw.text((x1 + 3, max(y1 - 14, 0)), label, fill=(0, 220, 80))

    verdict = "STRESSED" if s_pct > 15.0 else "HEALTHY"
    col     = (220, 60, 60) if s_pct > 15.0 else (60, 200, 60)
    draw.rectangle([0, 0, 300, 60], fill=(20, 20, 20))
    draw.text((8,  6), verdict, fill=col)
    draw.text((8, 34), f"Green {h_pct:.0f}%   Stressed {s_pct:.0f}%", fill=(200, 200, 200))

    pil_img.save(_PNG_OUT)


def _run_inspection(camera, phase: str) -> None:
    global _yolo_call_n, _last_yolo_results, _last_health
    if camera is None or not camera.is_initialized:
        return
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return

    rgb_np       = rgb[0, :, :, :3].cpu().numpy()
    h_pct, s_pct = _analyze_health(rgb_np)
    _last_health  = (h_pct, s_pct)

    if phase == "inside" and h_pct + s_pct > 0:
        status = "STRESSED" if s_pct > 15.0 else "HEALTHY"
        print(f"[INSPECT] Green: {h_pct:.0f}%  Stressed: {s_pct:.0f}%  → {status}")
        _health_log.append({"h": h_pct, "s": s_pct})

    _yolo_call_n += 1
    if _YOLO_READY and _yolo_call_n % _YOLO_EVERY == 0:
        rgb_bgr            = rgb_np[:, :, ::-1].copy()
        results            = _YOLO_MODEL(rgb_bgr, verbose=False)
        _last_yolo_results = results
        boxes, names = results[0].boxes, results[0].names
        hits = [(names[int(b.cls)], float(b.conf))
                for b in boxes if float(b.conf) > 0.35]
        if hits:
            print(f"[YOLO] {', '.join(f'{n} ({c:.0%})' for n, c in hits)}")

    _show_preview(rgb_np)


def _print_inspection_report() -> None:
    if not _health_log:
        print("[INSPECT] No vegetation pixels captured during inspection.")
        return
    n     = len(_health_log)
    avg_h = sum(d["h"] for d in _health_log) / n
    avg_s = sum(d["s"] for d in _health_log) / n
    verdict = "NEEDS ATTENTION" if avg_s > 15.0 else "HEALTHY"
    print(f"\n[INSPECT REPORT] ── {n} frames analysed ──")
    print(f"  Avg healthy green  : {avg_h:.1f}%")
    print(f"  Avg stressed/yellow: {avg_s:.1f}%")
    print(f"  Verdict: {verdict}  (unhealthy clusters: {sorted(_UNHEALTHY_CLUSTER_INDICES)})")
    if _YOLO_READY:
        print(f"  YOLO ran {_yolo_call_n // _YOLO_EVERY} time(s)")
    print()


# ---------------------------------------------------------------------------
# Main loop — policy locomotion + arm inspection
# ---------------------------------------------------------------------------

def run_simulator(sim: SimulationContext, robot: Articulation, policy: nn.Module,
                  bush_xy: tuple, sim_dt: float, contact_sensor: ContactSensor,
                  camera=None):
    device = sim.device

    # Joint index map for arm overrides
    name_to_idx  = {name: i for i, name in enumerate(robot.joint_names)}
    default_jpos = robot.data.default_joint_pos.clone()   # (1, 37)

    # Build reach pose tensor (absolute joint angles)
    reach_jpos = default_jpos.clone()
    for jname, val in _REACH_JOINTS.items():
        idx = name_to_idx.get(jname)
        if idx is not None:
            reach_jpos[0, idx] = val

    # Velocity commands
    walk_cmd = torch.tensor([WALK_VX, 0.0, 0.0], device=device)
    stop_cmd = torch.zeros(3, device=device)
    cmd      = walk_cmd.clone()

    last_action = torch.zeros(1, _ACTION_DIM, device=device)

    phase = "walk"
    frame = 0

    bx, by = bush_xy
    print(f"\n[INFO] G1 locomotion policy active | bush at ({bx:.2f}, {by:.2f})")
    print(f"[INFO] WALK → ARRIVE → REACH_IN → INSIDE → REACH_OUT → DONE\n")

    while simulation_app.is_running():
        robot_x = robot.data.root_pos_w[0, 0].item()

        # ── State machine ────────────────────────────────────────────────
        if phase == "walk":
            cmd = walk_cmd
            if robot_x >= _INSPECT_X - ARRIVE_THRESH:
                print(f"[WALK→ARRIVE] robot x={robot_x:.2f} — sending stop command")
                phase = "arrive"; frame = 0

        elif phase == "arrive":
            cmd    = stop_cmd
            frame += 1
            if frame >= STABILISE_FRAMES:
                print("[ARRIVE→REACH_IN] robot stabilised — extending arm")
                phase = "reach_in"; frame = 0

        elif phase == "reach_in":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                print("[REACH_IN→INSIDE] arm fully in bush — clusters pushed aside")
                phase = "inside"; frame = 0

        elif phase == "inside":
            cmd    = stop_cmd
            frame += 1
            if frame >= HOLD_FRAMES:
                _print_inspection_report()
                print("[INSIDE→REACH_OUT] retracting arm — clusters spring back")
                phase = "reach_out"; frame = 0

        elif phase == "reach_out":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                print("[REACH_OUT→DONE] interaction complete — robot standing")
                phase = "done"

        # phase "done": robot stands under policy with zero command

        # ── Policy inference ─────────────────────────────────────────────
        with torch.inference_mode():
            obs    = build_obs(robot, cmd, last_action)
            action = policy(obs)        # (1, 37)
        # Clone outside inference_mode — makes a normal tensor that allows in-place writes
        action      = action.clone()
        last_action = action.clone()

        # ── Arm override (REACH phases only) ─────────────────────────────
        # Policy still runs for legs (balance while standing).
        # We override the arm joint targets by back-solving the action:
        #   joint_target = default + scale * a  →  a = (desired − default) / scale
        if phase == "reach_in":
            alpha = min(frame / RAMP_FRAMES, 1.0)
        elif phase == "inside":
            alpha = 1.0
        elif phase == "reach_out":
            alpha = max(1.0 - frame / RAMP_FRAMES, 0.0)
        else:
            alpha = 0.0

        if alpha > 0.0:
            interp = torch.lerp(default_jpos, reach_jpos, alpha)
            for jname in _REACH_JOINTS:
                idx = name_to_idx.get(jname)
                if idx is not None:
                    action[0, idx] = (interp[0, idx] - default_jpos[0, idx]) / _ACTION_SCALE

        # ── Apply to robot via PD actuators ──────────────────────────────
        joint_targets = default_jpos + _ACTION_SCALE * action
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        contact_sensor.update(sim_dt)
        _check_contact(contact_sensor, phase)

        if camera is not None:
            camera.update(sim_dt)
            _run_inspection(camera, phase)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # dt=0.005 s (200 Hz) — standard for humanoid locomotion in Isaac Lab
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    sim_cfg    = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, render=render_cfg)
    sim        = SimulationContext(sim_cfg)

    bx, by, _ = _BUSH_POS
    sim.set_camera_view(
        eye   =[_ROBOT_START[0] - 2.0, _ROBOT_START[1] - 4.0, 2.5],
        target=[bx, by, 1.0],
    )

    entities       = design_scene()
    robot          = entities["robot"]
    bush_xy        = entities["bush_xy"]
    contact_sensor = entities["contact_sensor"]
    camera         = entities["camera"]

    sim.reset()
    _init_yolo()
    print("[INFO] Scene ready — loading locomotion policy...")
    print(f"[INFO] Unhealthy clusters (yellow): {sorted(_UNHEALTHY_CLUSTER_INDICES)}")
    if camera is not None:
        print("[INFO] Camera active — YOLO + colour-health inspection enabled")
    else:
        print("[INFO] Camera disabled — rerun with --enable_cameras to enable YOLO")
    policy = load_policy(sim.device)
    print("[INFO] Policy ready. G1 will walk under real dynamics.\n")

    run_simulator(sim, robot, policy, bush_xy, sim_cfg.dt, contact_sensor, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()
