"""
Greenhouse interactive bush demo — IsaacLab 2.x / Isaac Sim 4.x

Layout:
  - Greenhouse floor + frame (glass walls commented out for clear view)
  - One procedural interactive bush:
      Trunk  → kinematic rigid body (brown cuboid, 80 cm tall)
      Clusters → 10 dynamic sphere rigid bodies connected to trunk via D6 joints
      Each cluster has an angular spring drive — it deflects when the robot
      pushes it and springs back when released.
  - Unitree G1 humanoid walks to the bush, extends left arm into the clusters,
    holds while they are pushed aside, then retracts.

Sensors:
  - ContactSensor on all G1 body links — prints when robot touches a cluster
  - Camera on torso_link — 640×480 RGB at ~10 Hz, GPU tensor ready for YOLO

Robot motion (state machine):
  WALK → REACH_IN → INSIDE → REACH_OUT → DONE

Run from the IsaacLab root:
    ./isaaclab.sh -p scripts/greenhouse_sim.py
    ./isaaclab.sh -p scripts/greenhouse_sim.py --enable_cameras

Note: --enable_cameras is required for the RGB camera sensor to render frames.
Without it the camera sensor is skipped gracefully.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Greenhouse inspection demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# All non-stdlib imports must come after AppLauncher
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.unitree import G1_CFG  # isort:skip

_CAMERAS_ENABLED = getattr(args_cli, "enable_cameras", False)
if _CAMERAS_ENABLED:
    from isaaclab.sensors import Camera, CameraCfg


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_GH_CX, _GH_CY = 5.0, 0.0

# Bush is 0.70 m to the robot's left (+Y). At shoulder_roll=2.0 rad (115°)
# the G1 hand reaches approximately y=0.78 m, z=0.94 m — into the upper clusters.
_BUSH_POS    = (5.0, 0.70, 0.0)
_ROBOT_START = (2.5, 0.0, 0.74)   # pelvis height 0.74 m
_INSPECT_POS = (5.0, 0.0, 0.74)   # stop here, same X as bush, on the aisle

MOVE_SPEED    = 0.6    # m/s
ARRIVE_THRESH = 0.05   # m

RAMP_FRAMES = 60   # frames to ramp arm in / out (0.6 s at dt=0.01)
HOLD_FRAMES = 80   # frames arm stays inside bush (0.8 s)

# shoulder_roll = 2.0 rad (115°): arm points down-and-out, lowering hand to z≈0.94 m
# which puts it into the upper cluster ring (z=0.95 m). Elbow nearly straight for reach.
_REACH_JOINTS = {
    "left_shoulder_roll_joint":  2.00,   # 115° — arm angled down-and-sideways
    "left_shoulder_pitch_joint": 0.00,   # neutral
    "left_elbow_pitch_joint":    0.05,   # nearly fully extended
    "left_one_joint":  0.0,             # open left hand
    "left_two_joint":  0.0,
}


# ---------------------------------------------------------------------------
# Interactive bush geometry & physics constants
# ---------------------------------------------------------------------------

_TRUNK_HEIGHT = 0.80                    # trunk: z = 0 → 0.80 m
_TRUNK_COLOR  = (0.35, 0.20, 0.08)     # brown

# Each entry: (dx, dy, world_z, sphere_radius)
# All joints pivot around trunk top at (bx, by, _TRUNK_HEIGHT).
# Clusters at z = 0.75–1.10 m sit in the robot arm's reachable zone.
_CLUSTER_LAYOUT = [
    # lower ring — 5 clusters, z ≈ 0.75 m
    ( 0.22,  0.00, 0.75, 0.14),
    ( 0.07,  0.21, 0.75, 0.14),
    (-0.18,  0.13, 0.75, 0.14),
    (-0.18, -0.13, 0.75, 0.14),
    ( 0.07, -0.21, 0.75, 0.14),
    # upper ring — 4 clusters, z ≈ 0.95 m  (primary interaction zone)
    ( 0.13,  0.13, 0.95, 0.13),
    (-0.15,  0.10, 0.95, 0.13),
    (-0.15, -0.10, 0.95, 0.13),
    ( 0.13, -0.13, 0.95, 0.13),
    # top cap — z ≈ 1.10 m
    ( 0.00,  0.00, 1.10, 0.14),
]

_CLUSTER_COLORS = [
    (0.18, 0.55, 0.12), (0.20, 0.60, 0.10), (0.15, 0.50, 0.15),
    (0.72, 0.65, 0.10),  # index 3 — stressed yellow (nitrogen deficiency)
    (0.17, 0.52, 0.13), (0.12, 0.48, 0.16),
    (0.20, 0.55, 0.11),
    (0.62, 0.52, 0.08),  # index 7 — stressed yellow-brown (root stress)
    (0.19, 0.57, 0.09),
    (0.14, 0.50, 0.17),
]
_UNHEALTHY_CLUSTER_INDICES = {3, 7}   # clusters that are visually diseased

_CLUSTER_MASS     = 0.05   # kg
_SPRING_STIFFNESS = 3.0    # N·m/rad — soft spring; clusters deflect easily
_SPRING_DAMPING   = 0.8    # N·m·s/rad — low damping → natural oscillation
_SWING_LIMIT_DEG  = 75.0   # max deflection in degrees

# ── Second-level leaf sub-clusters (attached to clusters, not trunk) ────────
_LEAF_PER_CLUSTER = 3       # leaves per cluster
_LEAF_RADIUS      = 0.045   # m — smaller than clusters
_LEAF_MASS        = 0.006   # kg — very light → responds quickly to touch
_LEAF_STIFFNESS   = 0.5     # N·m/rad — ultra-soft spring
_LEAF_DAMPING     = 0.10    # N·m·s/rad — minimal damping → lingering wobble
_LEAF_SWING_DEG   = 90.0    # degrees — large range for dramatic sway
_LEAF_OFFSETS     = [       # (dx, dy, dz) relative to each cluster centre
    ( 0.13,  0.00,  0.06),
    (-0.07,  0.11,  0.08),
    (-0.07, -0.11,  0.05),
]


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _mat(rgb: tuple, opacity: float = 1.0) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.6, opacity=opacity)


def _spawn_box(path, size, pos, rgb, opacity=1.0, orient=None):
    cfg = sim_utils.CuboidCfg(size=size, visual_material=_mat(rgb, opacity))
    kw = {"translation": pos}
    if orient is not None:
        kw["orientation"] = orient
    cfg.func(path, cfg, **kw)


def _x_rot_quat(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), math.sin(h), 0.0, 0.0)


# ---------------------------------------------------------------------------
# Greenhouse structure
# ---------------------------------------------------------------------------

def build_greenhouse(ox=_GH_CX, oy=_GH_CY, oz=0.0):
    W, D, H = 8.0, 5.0, 3.0
    t       = 0.08
    PITCH   = 25.0
    rise    = (D / 2) * math.tan(math.radians(PITCH))

    GLASS = (0.72, 0.94, 0.84)
    FRAME = (0.50, 0.44, 0.33)
    FLOOR = (0.80, 0.72, 0.55)

    _spawn_box("/World/Greenhouse/Floor", (W, D, 0.12), (ox, oy, oz + 0.06), FLOOR)
    wall_z = oz + H / 2

    # --- Glass walls commented out for clear view of robot/bush dynamics ---
    # Restore these lines to bring back the full greenhouse enclosure.
    # _spawn_box("/World/Greenhouse/WallFront", (W, t, H), (ox,     oy-D/2, wall_z), GLASS, opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallBack",  (W, t, H), (ox,     oy+D/2, wall_z), GLASS, opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallLeft",  (t, D, H), (ox-W/2, oy,     wall_z), GLASS, opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallRight", (t, D, H), (ox+W/2, oy,     wall_z), GLASS, opacity=0.35)

    for i, (dx, dy) in enumerate([(-W/2, -D/2), (-W/2, D/2), (W/2, -D/2), (W/2, D/2)]):
        _spawn_box(f"/World/Greenhouse/Pillar{i}", (0.12, 0.12, H), (ox+dx, oy+dy, wall_z), FRAME)

    # _spawn_box("/World/Greenhouse/RoofFront", ...)
    # _spawn_box("/World/Greenhouse/RoofBack",  ...)
    # _spawn_box("/World/Greenhouse/Ridge",     ...)


# ---------------------------------------------------------------------------
# Interactive bush — trunk + spring-jointed leaf clusters
# ---------------------------------------------------------------------------

def _attach_leaves(stage, cluster_path: str, cx: float, cy: float, cz: float,
                   leaf_color: tuple, base: str, ci: int) -> None:
    """Attach _LEAF_PER_CLUSTER tiny spheres to a cluster via ultra-soft D6 springs.

    Pivot is the cluster centre; leaves sway on top of cluster motion giving a
    two-level cascade: robot pushes cluster → cluster moves → leaves wobble.
    """
    from pxr import UsdPhysics, Gf, Sdf

    for j, (ldx, ldy, ldz) in enumerate(_LEAF_OFFSETS[:_LEAF_PER_CLUSTER]):
        leaf_path = f"{base}/Leaf_{ci}_{j}"
        leaf_cfg  = sim_utils.SphereCfg(
            radius=_LEAF_RADIUS,
            visual_material=_mat(leaf_color, opacity=0.80),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        )
        leaf_cfg.func(leaf_path, leaf_cfg, translation=(cx + ldx, cy + ldy, cz + ldz))
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(leaf_path)).CreateMassAttr(_LEAF_MASS)

        # D6 joint: cluster (dynamic parent) → leaf (dynamic child); pivot at cluster centre
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
    """Build a procedural bush with articulated leaf clusters.

    Trunk:    kinematic rigid body (brown cuboid) — static, immovable.
    Clusters: dynamic rigid body spheres connected to trunk top via D6 joints.
              Each joint locks all translation and allows angular swing (±60°)
              with a spring drive that restores the cluster to rest when the
              robot withdraws its hand.

    Physics note:
      `UsdPhysics.DriveAPI` on a D6 joint's rotX/rotY axes gives a PD spring:
        torque = stiffness * (target_angle − current_angle) − damping * angular_vel
      With target_angle=0, this always pulls the cluster back to rest.
    """
    from pxr import UsdPhysics, Gf, Sdf

    bx, by, _ = _BUSH_POS
    base       = "/World/Greenhouse/Plants/Bush"
    stage      = sim_utils.get_current_stage()

    stage.DefinePrim(base, "Xform")

    # ── Trunk: kinematic, brown cuboid ────────────────────────────────────
    trunk_path = f"{base}/Trunk"
    trunk_z    = _TRUNK_HEIGHT / 2   # prim origin at geometric centre

    trunk_cfg = sim_utils.CuboidCfg(
        size=(0.08, 0.08, _TRUNK_HEIGHT),
        visual_material=_mat(_TRUNK_COLOR),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    trunk_cfg.func(trunk_path, trunk_cfg, translation=(bx, by, trunk_z))

    # ── Leaf clusters + D6 spring joints ──────────────────────────────────
    for i, (dx, dy, dz, radius) in enumerate(_CLUSTER_LAYOUT):
        cluster_path = f"{base}/Cluster_{i}"
        color        = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]

        # Spawn dynamic sphere (kinematic_enabled=False → gravity + forces act on it)
        cluster_cfg = sim_utils.SphereCfg(
            radius=radius,
            visual_material=_mat(color, opacity=0.90),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        )
        cluster_cfg.func(cluster_path, cluster_cfg, translation=(bx+dx, by+dy, dz))

        # Set mass via USD API (keeps SphereCfg usage minimal)
        cluster_prim = stage.GetPrimAtPath(cluster_path)
        mass_api = UsdPhysics.MassAPI.Apply(cluster_prim)
        mass_api.CreateMassAttr(_CLUSTER_MASS)

        # ── D6 Joint ────────────────────────────────────────────────────
        # Pivot placed at trunk top: world pos = (bx, by, _TRUNK_HEIGHT).
        # body0 = trunk (parent, kinematic)
        # body1 = cluster (child, dynamic)
        #
        # LocalPos0: trunk top in trunk-local frame = (0, 0, +trunk_z)
        # LocalPos1: trunk top in cluster-local frame = (−dx, −dy, _TRUNK_HEIGHT − dz)
        # At rest both frames coincide at (bx, by, _TRUNK_HEIGHT) in world space.
        joint_path = f"{base}/Joint_{i}"
        d6 = UsdPhysics.Joint.Define(stage, joint_path)   # D6Joint renamed to Joint in pxr < 4.x
        d6.CreateBody0Rel().SetTargets([Sdf.Path(trunk_path)])
        d6.CreateBody1Rel().SetTargets([Sdf.Path(cluster_path)])

        d6.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, trunk_z))
        d6.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        d6.CreateLocalPos1Attr().Set(Gf.Vec3f(-dx, -dy, _TRUNK_HEIGHT - dz))
        d6.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        # Lock all translational DOFs — cluster can only rotate, not slide
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), axis)
            lim.CreateLowAttr(0.0)
            lim.CreateHighAttr(0.0)

        # Angular swing ±SWING_LIMIT_DEG with spring drive on rotX and rotY
        for axis in ("rotX", "rotY"):
            lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), axis)
            lim.CreateLowAttr(-_SWING_LIMIT_DEG)
            lim.CreateHighAttr(_SWING_LIMIT_DEG)

            drv = UsdPhysics.DriveAPI.Apply(d6.GetPrim(), axis)
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(_SPRING_STIFFNESS)
            drv.CreateDampingAttr(_SPRING_DAMPING)
            drv.CreateTargetPositionAttr(0.0)   # always pull back to rest

        # Lock twist (rotZ) — clusters don't spin
        lim = UsdPhysics.LimitAPI.Apply(d6.GetPrim(), "rotZ")
        lim.CreateLowAttr(0.0)
        lim.CreateHighAttr(0.0)

        # Attach second-level leaf sub-spheres to this cluster
        _attach_leaves(stage, cluster_path, bx+dx, by+dy, dz,
                       _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)], base, i)

    n_leaves = len(_CLUSTER_LAYOUT) * _LEAF_PER_CLUSTER
    print(f"[INFO] Soft bush: {len(_CLUSTER_LAYOUT)} clusters + {n_leaves} leaves  "
          f"(cluster k={_SPRING_STIFFNESS}, leaf k={_LEAF_STIFFNESS})")
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

    # Robot: free-floating root driven kinematically every frame
    robot_cfg = G1_CFG.copy()
    robot_cfg.prim_path      = "/World/G1"
    robot_cfg.init_state.pos = _ROBOT_START

    robot = Articulation(cfg=robot_cfg)

    # Contact sensor — all G1 body links.
    # No filter_prim_paths_expr: the clusters are valid rigid bodies but not
    # registered with PhysxContactReportAPI yet. Any G1 link force > threshold fires.
    contact_sensor = ContactSensor(cfg=ContactSensorCfg(
        prim_path="/World/G1/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    ))

    # Head camera — 640×480 RGB, ~10 Hz, GPU tensor for future YOLO
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

    return {"robot": robot, "bush_path": bush_path, "bush_xy": bush_xy,
            "contact_sensor": contact_sensor, "camera": camera}


# ---------------------------------------------------------------------------
# Sensor helpers
# ---------------------------------------------------------------------------

_contact_print_cooldown = 0
_CONTACT_FORCE_THRESH   = 5.0   # N


def _check_contact(sensor: ContactSensor) -> None:
    global _contact_print_cooldown
    if _contact_print_cooldown > 0:
        _contact_print_cooldown -= 1
        return
    if not sensor.is_initialized:
        return
    forces = sensor.data.net_forces_w
    if forces is None:
        return
    magnitudes = forces.norm(dim=-1)
    max_force  = magnitudes.max().item()
    if max_force > _CONTACT_FORCE_THRESH:
        body_idx  = magnitudes[0].argmax().item()
        body_name = sensor.body_names[body_idx]
        print(f"[CONTACT] '{body_name}' touching cluster — force {max_force:.1f} N")
        _contact_print_cooldown = 100


# ---------------------------------------------------------------------------
# YOLO plant health inspector + live preview (omni.ui floating window)
# ---------------------------------------------------------------------------

_YOLO_MODEL        = None
_YOLO_READY        = False
_yolo_call_n       = 0
_YOLO_EVERY        = 10      # run YOLO every N inspection calls (~10 Hz cam → ~1 s)
_health_log: list  = []
_last_yolo_results = None    # cached between YOLO runs so boxes persist on preview
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
    """Pixel-level colour health analysis.  Returns (healthy_pct, stressed_pct)."""
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
    feh refreshes automatically each time the file is updated (~10 Hz).
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

    # Draw YOLO boxes + labels
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

    # Health HUD — dark bar in top-left corner
    verdict = "STRESSED" if s_pct > 15.0 else "HEALTHY"
    col     = (220, 60, 60) if s_pct > 15.0 else (60, 200, 60)
    draw.rectangle([0, 0, 300, 60], fill=(20, 20, 20))
    draw.text((8,  6), verdict, fill=col)
    draw.text((8, 34), f"Green {h_pct:.0f}%   Stressed {s_pct:.0f}%", fill=(200, 200, 200))

    pil_img.save(_PNG_OUT)


def _run_inspection(camera, phase: str) -> None:
    """Colour health analysis every frame + YOLO every _YOLO_EVERY frames + live preview."""
    global _yolo_call_n, _last_yolo_results, _last_health
    if camera is None or not camera.is_initialized:
        return
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return

    # RGBA (1,H,W,4) GPU tensor → RGB numpy (H,W,3) CPU
    rgb_np = rgb[0, :, :, :3].cpu().numpy()

    # ── Colour health analysis (cheap, always) ──────────────────────────
    h_pct, s_pct = _analyze_health(rgb_np)
    _last_health  = (h_pct, s_pct)
    if phase == "inside" and h_pct + s_pct > 0:
        status = "STRESSED" if s_pct > 15.0 else "HEALTHY"
        print(f"[INSPECT] Green: {h_pct:.0f}%  Stressed: {s_pct:.0f}%  → {status}")
        _health_log.append({"h": h_pct, "s": s_pct})

    # ── YOLO inference (every N calls) ──────────────────────────────────
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

    # ── Live preview window ──────────────────────────────────────────────
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
# Inspection state machine
# ---------------------------------------------------------------------------

def _build_reach_pose(robot: Articulation) -> torch.Tensor:
    """Build joint-position tensor with left arm extended into the bush."""
    pose = robot.data.default_joint_pos.clone()
    name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
    for joint_name, target_val in _REACH_JOINTS.items():
        idx = name_to_idx.get(joint_name)
        if idx is not None:
            pose[0, idx] = target_val
    return pose


def run_simulator(sim: SimulationContext, robot: Articulation,
                  bush_xy: tuple, sim_dt: float,
                  contact_sensor: ContactSensor, camera):
    """
    WALK      — robot glides from _ROBOT_START to _INSPECT_POS
    REACH_IN  — left arm ramps to reach pose over RAMP_FRAMES
                clusters deflect as hand pushes into them
    INSIDE    — arm holds at full reach for HOLD_FRAMES
    REACH_OUT — arm ramps back to default pose over RAMP_FRAMES
                clusters spring back once hand withdraws
    DONE      — robot stands still; viewport stays live (Ctrl+C to quit)
    """
    cur   = list(_ROBOT_START)
    phase = "walk"
    frame = 0

    default_jpos = robot.data.default_joint_pos.clone()
    default_jvel = robot.data.default_joint_vel.clone()
    reach_jpos   = _build_reach_pose(robot)
    cur_jpos     = default_jpos.clone()

    bx, by = bush_xy
    print(f"\n[INFO] Bush at ({bx:.2f}, {by:.2f}) | robot starts at {_ROBOT_START}")
    print(f"[INFO] State machine: WALK → REACH_IN → INSIDE → REACH_OUT → DONE\n")

    while simulation_app.is_running():

        # ── state machine ─────────────────────────────────────────────────
        if phase == "walk":
            tx, ty, tz = _INSPECT_POS
            dx   = tx - cur[0]
            dy   = ty - cur[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < ARRIVE_THRESH:
                print(f"[WALK→REACH_IN] arrived at ({tx:.1f}, {ty:.1f})")
                phase = "reach_in"; frame = 0
            else:
                step    = min(MOVE_SPEED * sim_dt, dist)
                cur[0] += dx / dist * step
                cur[1] += dy / dist * step
            cur_jpos = default_jpos

        elif phase == "reach_in":
            frame   += 1
            alpha    = min(frame / RAMP_FRAMES, 1.0)
            cur_jpos = torch.lerp(default_jpos, reach_jpos, alpha)
            if frame >= RAMP_FRAMES:
                print("[REACH_IN→INSIDE] arm fully extended — clusters pushed aside")
                phase = "inside"; frame = 0

        elif phase == "inside":
            frame   += 1
            cur_jpos = reach_jpos
            if frame >= HOLD_FRAMES:
                _print_inspection_report()
                print("[INSIDE→REACH_OUT] retracting arm")
                phase = "reach_out"; frame = 0

        elif phase == "reach_out":
            frame   += 1
            alpha    = max(1.0 - frame / RAMP_FRAMES, 0.0)
            cur_jpos = torch.lerp(default_jpos, reach_jpos, alpha)
            if frame >= RAMP_FRAMES:
                print("[REACH_OUT→DONE] arm retracted — clusters springing back to rest")
                phase = "done"; frame = 0

        # phase == "done": arm at default, robot stands still

        # ── kinematic root + joint control ────────────────────────────────
        pose = torch.tensor(
            [[cur[0], cur[1], cur[2], 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32, device=sim.device,
        )
        vel = torch.zeros(1, 6, dtype=torch.float32, device=sim.device)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)
        robot.write_joint_state_to_sim(cur_jpos, default_jvel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        contact_sensor.update(sim_dt)
        _check_contact(contact_sensor)

        if camera is not None:
            camera.update(sim_dt)
            _run_inspection(camera, phase)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    sim_cfg    = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, render=render_cfg)
    sim        = SimulationContext(sim_cfg)

    bx, by, _ = _BUSH_POS
    sim.set_camera_view(
        eye   =[_ROBOT_START[0] - 2.0, _ROBOT_START[1] - 3.5, 2.5],
        target=[bx, by, 1.0],
    )

    entities       = design_scene()
    robot          = entities["robot"]
    bush_xy        = entities["bush_xy"]
    contact_sensor = entities["contact_sensor"]
    camera         = entities["camera"]

    sim.reset()
    _init_yolo()
    print(f"[INFO] Scene ready — interactive bush at {_BUSH_POS}, robot at {_ROBOT_START}")
    print(f"[INFO] Unhealthy clusters (yellow): {sorted(_UNHEALTHY_CLUSTER_INDICES)}")
    if camera is not None:
        print("[INFO] Camera active — YOLO + colour-health inspection enabled")
    else:
        print("[INFO] Camera disabled — rerun with --enable_cameras to enable YOLO")

    run_simulator(sim, robot, bush_xy, sim_cfg.dt, contact_sensor, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()
