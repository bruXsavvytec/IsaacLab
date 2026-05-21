"""
Greenhouse single-bush interaction demo — IsaacLab 2.x / Isaac Sim 4.x

Layout:
  - Greenhouse floor + frame (glass walls commented out for clear view)
  - One Bush.usd asset, collidable, kinematic rigid body
  - Unitree G1 humanoid walks to the bush, extends left arm inside, retracts

Sensors:
  - ContactSensor on all G1 body links — prints alert when robot touches the bush
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
import torch  # noqa: E402

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.unitree import G1_CFG  # isort:skip

# ---------------------------------------------------------------------------
# Bush asset paths (from forest generator repo, converted OBJ → USD)
# ---------------------------------------------------------------------------
_BUSH_USD   = "/home/trooperai/dev-bru/Nvidia-Isaac-Sim-Procedual-Forest-Generator/models/Bush_obj/Bush.usd"
_BUSH_SCALE = 1.0   # confirmed good in asset_showcase.py

# Camera requires --enable_cameras; import lazily to avoid hard failure without the flag
_CAMERAS_ENABLED = getattr(args_cli, "enable_cameras", False)
if _CAMERAS_ENABLED:
    from isaaclab.sensors import Camera, CameraCfg


# ---------------------------------------------------------------------------
# Single-bush interaction layout
# ---------------------------------------------------------------------------

_GH_CX, _GH_CY = 5.0, 0.0   # greenhouse centre — used for floor / frame

# Single bush: 0.7 m to the robot's left (robot faces +X, so left = +Y).
# 0.7 m is within the G1's arm reach: shoulder offset ~0.22 m + arm ~0.62 m at 90° roll.
# A real greenhouse aisle is 60-80 cm wide; this matches a plant row at aisle edge.
_BUSH_POS    = (5.0, 0.70, 0.0)   # bush spawned here
_ROBOT_START = (2.5, 0.0, 0.74)   # robot spawn position
_INSPECT_POS = (5.0, 0.0, 0.74)   # robot stops here (same X as bush, on the aisle)

MOVE_SPEED    = 0.6    # m/s
ARRIVE_THRESH = 0.05   # m — "close enough" to waypoint

# Arm-reach timing (physics frames at dt = 0.01 s)
RAMP_FRAMES = 60   # frames to ramp arm in / out
HOLD_FRAMES = 80   # frames to hold arm fully inside the bush

# Left arm extends toward +Y to reach the bush.
# shoulder_roll: abduction (arm moves sideways away from body toward +Y).
#   1.57 rad = 90° → arm points directly sideways, maximising Y reach.
# shoulder_pitch: small positive value keeps hand at approx. bush height.
# elbow_pitch:    near 0 = fully extended; do NOT go negative (hyperextend).
_REACH_JOINTS = {
    "left_shoulder_roll_joint":  1.57,   # 90° abduction — arm straight out to the side
    "left_shoulder_pitch_joint": 0.10,   # slight forward tilt for height alignment
    "left_elbow_pitch_joint":    0.05,   # nearly fully extended
    "left_one_joint":  0.0,             # open left hand
    "left_two_joint":  0.0,
}

# Sphere-bush fallback geometry (used only when Bush.usd is missing)
_BUSH_RADIUS  = 0.22
_BUSH_OFFSETS = [
    ( 0.00,  0.00, 0.00, 1.00),
    ( 0.22,  0.12, 0.20, 0.78),
    (-0.18, -0.10, 0.24, 0.72),
    ( 0.05, -0.22, 0.28, 0.65),
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


def _spawn_sphere(path, radius, pos, rgb, collidable=False):
    """Sphere prim. collidable=True → kinematic rigid body (solid to collision)."""
    cfg = sim_utils.SphereCfg(
        radius=radius,
        visual_material=_mat(rgb),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True) if collidable else None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if collidable else None,
    )
    cfg.func(path, cfg, translation=pos)


def _x_rot_quat(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), math.sin(h), 0.0, 0.0)


# ---------------------------------------------------------------------------
# Greenhouse structure
# ---------------------------------------------------------------------------

def build_greenhouse(ox=_GH_CX, oy=_GH_CY, oz=0.0):
    W, D, H = 8.0, 5.0, 3.0
    t     = 0.08
    PITCH = 25.0
    rise  = (D / 2) * math.tan(math.radians(PITCH))
    pdep  = (D / 2) / math.cos(math.radians(PITCH))

    GLASS = (0.72, 0.94, 0.84)
    FRAME = (0.50, 0.44, 0.33)
    FLOOR = (0.80, 0.72, 0.55)

    _spawn_box("/World/Greenhouse/Floor",     (W, D, 0.12), (ox, oy, oz+0.06), FLOOR)
    wall_z = oz + H / 2
    # --- Glass walls commented out for clear view of robot/bush dynamics ---
    # Restore these lines to bring back the full greenhouse enclosure.
    # _spawn_box("/World/Greenhouse/WallFront", (W, t, H), (ox,      oy-D/2, wall_z), GLASS, opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallBack",  (W, t, H), (ox,      oy+D/2, wall_z), GLASS, opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallLeft",  (t, D, H), (ox-W/2,  oy,     wall_z), GLASS, opacity=0.35)
    # _spawn_box("/World/Greenhouse/WallRight", (t, D, H), (ox+W/2,  oy,     wall_z), GLASS, opacity=0.35)
    for i, (dx, dy) in enumerate([(-W/2,-D/2),(-W/2,D/2),(W/2,-D/2),(W/2,D/2)]):
        _spawn_box(f"/World/Greenhouse/Pillar{i}", (0.12, 0.12, H), (ox+dx, oy+dy, wall_z), FRAME)
    roof_z = oz + H + rise / 2
    # _spawn_box("/World/Greenhouse/RoofFront", (W, pdep, t), (ox, oy-D/4, roof_z), GLASS, opacity=0.35, orient=_x_rot_quat(+PITCH))
    # _spawn_box("/World/Greenhouse/RoofBack",  (W, pdep, t), (ox, oy+D/4, roof_z), GLASS, opacity=0.35, orient=_x_rot_quat(-PITCH))
    # _spawn_box("/World/Greenhouse/Ridge",     (W, 0.15, 0.15), (ox, oy, oz+H+rise), FRAME)


# ---------------------------------------------------------------------------
# Bush — single collidable plant
# ---------------------------------------------------------------------------


def _apply_bush_physics(prim_path: str) -> None:
    """Post-spawn: give the Bush USD a kinematic rigid-body + mesh collision.

    Bush.usd (OBJ→USD) has no physics APIs pre-baked.
    `UsdFileCfg.rigid_props` only *modifies* existing rigid bodies — it cannot
    create them. We use `define_rigid_body_properties` / `define_collision_properties`
    (the "define" variants that *apply* the schemas) to make the bush solid.

    Result: the bush is immovable (kinematic) but the robot cannot pass through it.
    Individual leaf deformation would require a deformable-body solver.
    """
    from pxr import Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return

    sim_utils.define_rigid_body_properties(
        prim_path,
        sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    coll_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    for child in Usd.PrimRange(root):   # iterates root + all descendants
        if child.IsA(UsdGeom.Mesh):
            sim_utils.define_collision_properties(child.GetPath().pathString, coll_cfg)


def build_greenhouse_bushes() -> tuple[str, tuple]:
    """Spawn one Bush USD at _BUSH_POS and return (prim_path, (x, y)).

    Physics APIs are applied post-spawn via _apply_bush_physics() because
    Bush.usd has no RigidBodyAPI/CollisionAPI pre-baked (OBJ→USD conversion
    does not add them, and UsdFileCfg only *modifies* existing physics).

    Falls back to a sphere-cluster proxy when Bush.usd is not found.
    """
    import os
    path = "/World/Greenhouse/Plants/Bush"
    bx, by, bz = _BUSH_POS

    if not os.path.isfile(_BUSH_USD):
        print(f"[WARN] Bush USD not found at {_BUSH_USD} — using sphere proxy")
        _build_sphere_bush_proxy(path, bx, by)
        return path, (bx, by)

    s = _BUSH_SCALE
    bush_cfg = sim_utils.UsdFileCfg(
        usd_path=_BUSH_USD,
        scale=(s, s, s),
        # No rigid_props / collision_props / activate_contact_sensors here:
        # Bush.usd has no physics APIs pre-baked; _apply_bush_physics() adds them.
    )
    bush_cfg.func(path, bush_cfg, translation=(bx, by, bz))
    _apply_bush_physics(path)
    print(f"[INFO] Bush spawned at ({bx:.1f}, {by:.1f}, {bz:.1f})  prim: {path}")
    return path, (bx, by)


def _build_sphere_bush_proxy(base: str, bx: float, by: float) -> None:
    """Fallback: collidable sphere cluster at (bx, by) when Bush.usd is missing."""
    r  = _BUSH_RADIUS
    cz = r * 0.7
    for j, (dx, dy_off, dz_fac, rs) in enumerate(_BUSH_OFFSETS):
        g   = 0.45
        rgb = (g * 0.25, g, g * 0.18)
        _spawn_sphere(f"{base}/s{j}", r * rs,
                      (bx + dx, by + dy_off, cz + dz_fac * r),
                      rgb, collidable=True)


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
    bush_path, bush_xy = build_greenhouse_bushes()

    # Robot: no fixed root — we drive root pose kinematically every frame
    robot_cfg = G1_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.init_state.pos = _ROBOT_START

    robot = Articulation(cfg=robot_cfg)

    # --- Contact sensor — monitors all G1 body links ---
    # G1_CFG already sets activate_contact_sensors=True on the USD.
    # We don't filter to bush prims because filter_prim_paths_expr requires
    # the filtered objects to have PhysxContactReportAPI, which the OBJ-converted
    # Bush.usd doesn't carry. Net force on any G1 link > threshold → contact alert.
    contact_sensor = ContactSensor(cfg=ContactSensorCfg(
        prim_path="/World/G1/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    ))

    # --- Head camera — 640×480 RGB, ~10 Hz, GPU tensor for future YOLO ---
    # Spawned at /World/G1/torso_link/insp_cam.
    # Orientation (convention="ros"): +90° around Y maps camera +Z onto robot +X
    # so the lens faces the direction the robot walks.
    # Exact tilt may need tuning once you can view the rendered frames.
    camera = None
    if _CAMERAS_ENABLED:
        camera = Camera(cfg=CameraCfg(
            prim_path="/World/G1/torso_link/insp_cam",
            update_period=0.1,   # ~10 Hz
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 30.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.1, 0.0, 0.25),           # 10 cm forward, 25 cm up from torso_link origin
                rot=(0.7071, 0.0, 0.7071, 0.0), # look along robot +X (forward)
                convention="ros",
            ),
        ))

    return {"robot": robot, "bush_path": bush_path, "bush_xy": bush_xy,
            "contact_sensor": contact_sensor, "camera": camera}


# ---------------------------------------------------------------------------
# Sensor helpers
# ---------------------------------------------------------------------------

# Throttle contact prints: only emit once per second of sim time (every 100 frames at dt=0.01)
_contact_print_cooldown = 0
_CONTACT_FORCE_THRESH   = 5.0   # N — ignore tiny numerical noise


def _check_contact(sensor: ContactSensor) -> None:
    global _contact_print_cooldown
    if _contact_print_cooldown > 0:
        _contact_print_cooldown -= 1
        return
    if not sensor.is_initialized:
        return
    # net_forces_w shape: (num_envs, num_bodies, 3)
    forces = sensor.data.net_forces_w   # shape (1, N, 3)
    if forces is None:
        return
    magnitudes = forces.norm(dim=-1)    # (1, N)
    max_force   = magnitudes.max().item()
    if max_force > _CONTACT_FORCE_THRESH:
        body_idx  = magnitudes[0].argmax().item()
        body_name = sensor.body_names[body_idx]
        print(f"[CONTACT] Robot body '{body_name}' touching something — "
              f"force {max_force:.1f} N")
        _contact_print_cooldown = 100   # silence for ~1 s


_cam_log_counter = 0
_CAM_LOG_EVERY   = 300  # print camera info every 300 frames (~30 s)


def _log_camera(camera) -> None:
    global _cam_log_counter
    _cam_log_counter += 1
    if _cam_log_counter % _CAM_LOG_EVERY != 0:
        return
    if not camera.is_initialized:
        return
    rgb = camera.data.output.get("rgb")
    if rgb is not None:
        print(f"[CAMERA] RGB frame shape: {tuple(rgb.shape)}  "
              f"dtype: {rgb.dtype}  device: {rgb.device}")


# ---------------------------------------------------------------------------
# Inspection state machine
# ---------------------------------------------------------------------------

def _build_reach_pose(robot: Articulation) -> torch.Tensor:
    """Build a joint-position tensor with the left arm extended toward the bush (+Y).

    Returns a (1, num_joints) tensor on the robot's device.
    """
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
    """State machine: WALK → REACH_IN → INSIDE → REACH_OUT → DONE.

    WALK      — robot glides from _ROBOT_START to _INSPECT_POS
    REACH_IN  — left arm ramps from default pose to reach pose (RAMP_FRAMES)
    INSIDE    — arm holds fully extended inside the bush (HOLD_FRAMES)
    REACH_OUT — arm ramps back to default pose (RAMP_FRAMES)
    DONE      — robot stands still; loop keeps running so viewport stays live
    """
    cur   = list(_ROBOT_START)   # mutable [x, y, z] root position
    phase = "walk"
    frame = 0   # frame counter within the current phase

    default_jpos = robot.data.default_joint_pos.clone()
    default_jvel = robot.data.default_joint_vel.clone()
    reach_jpos   = _build_reach_pose(robot)
    cur_jpos     = default_jpos.clone()

    bx, by = bush_xy
    print(f"\n[INFO] Bush at ({bx:.1f}, {by:.1f})  |  robot starts at {_ROBOT_START}")
    print(f"[INFO] State machine: WALK → REACH_IN → INSIDE → REACH_OUT → DONE\n")

    while simulation_app.is_running():

        # ── state machine ─────────────────────────────────────────────────
        if phase == "walk":
            tx, ty, tz = _INSPECT_POS
            dx   = tx - cur[0]
            dy   = ty - cur[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < ARRIVE_THRESH:
                print(f"[WALK→REACH_IN] arrived at inspection position ({tx:.1f}, {ty:.1f})")
                phase = "reach_in"
                frame = 0
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
                print("[REACH_IN→INSIDE] arm fully extended — hand inside bush")
                phase = "inside"
                frame = 0

        elif phase == "inside":
            frame   += 1
            cur_jpos = reach_jpos
            if frame >= HOLD_FRAMES:
                print("[INSIDE→REACH_OUT] retracting arm")
                phase = "reach_out"
                frame = 0

        elif phase == "reach_out":
            frame   += 1
            alpha    = max(1.0 - frame / RAMP_FRAMES, 0.0)
            cur_jpos = torch.lerp(default_jpos, reach_jpos, alpha)
            if frame >= RAMP_FRAMES:
                print("[REACH_OUT→DONE] arm retracted — interaction complete")
                phase = "done"
                frame = 0

        # phase == "done": arm at default, robot stands still, viewport stays live

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

        # ── sensor updates ────────────────────────────────────────────────
        contact_sensor.update(sim_dt)
        _check_contact(contact_sensor)

        if camera is not None:
            camera.update(sim_dt)
            _log_camera(camera)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    sim_cfg    = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, render=render_cfg)
    sim        = SimulationContext(sim_cfg)

    # Side-angle view: see robot walk toward bush and arm extend sideways.
    # Eye is behind-left of the robot start; target is the bush at ground level.
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
    print(f"[INFO] Scene ready — 1 bush at {_BUSH_POS}, robot starts at {_ROBOT_START}")
    if camera is not None:
        print("[INFO] Camera sensor active — RGB frames available")
    else:
        print("[INFO] Camera disabled — rerun with --enable_cameras to enable RGB output")

    run_simulator(sim, robot, bush_xy, sim_cfg.dt, contact_sensor, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()
