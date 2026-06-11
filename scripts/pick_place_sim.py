"""
Pick and Place demo — IsaacLab 2.x / Isaac Sim 4.x

The Unitree G1 uses its pre-trained RSL-RL locomotion policy to walk to a
storage shelf, identifies the target block via YOLO + colour detection, reaches
down to pick it, carries it to a drop zone, and releases it.

Scene
─────
  Storage room (primitive walls + industrial lighting)
  Metal shelf at x=4.5 m with 3 coloured blocks (red / green / blue)
  Drop zone marker at x=2.0, y=1.5 m (on the floor)

Control split
─────────────
  WALK / ARRIVE / WALK_BACK / DROP_ARRIVE:
      policy(obs) → 37 joint targets → PD actuators (real walking)

  REACH / GRASP / LIFT / LOWER / RELEASE:
      locomotion policy still runs for legs (keeps balance while standing)
      arm joints overridden with scripted targets on top

Block attachment
────────────────
  On GRASP: block is switched to kinematic via USD API; its pose is driven each
  frame to follow the robot's left hand link.
  On RELEASE: block is switched back to dynamic; it falls and lands on the floor.

Run from the IsaacLab root:
    ./isaaclab.sh -p scripts/pick_place_sim.py
    ./isaaclab.sh -p scripts/pick_place_sim.py --enable_cameras
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pick and Place demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip

_CAMERAS_ENABLED = getattr(args_cli, "enable_cameras", False)
if _CAMERAS_ENABLED:
    from isaaclab.sensors import Camera, CameraCfg

# ---------------------------------------------------------------------------
# Policy constants — must match checkpoint architecture exactly
# ---------------------------------------------------------------------------

_CKPT_PATH    = "/home/trooperai/IsaacLab/.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt"
_OBS_DIM      = 310
_ACTION_DIM   = 37
_HIDDEN_DIMS  = [512, 256, 128]
_ACTION_SCALE = 0.25
_HEIGHT_SCAN_N = 187

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

_ROOM_W, _ROOM_D, _ROOM_H = 8.0, 6.0, 3.0

_ROBOT_START = (1.0, 0.0, 0.74)

# Shelf — face of the shelf (the side the robot reaches into) at x=4.5
_SHELF_X      = 4.5
_SHELF_SURF_Z = 0.75     # shelf surface height (m)
_BLOCK_H      = 0.08     # block height (m)
_BLOCK_Z      = _SHELF_SURF_Z + _BLOCK_H / 2.0   # block centre z

# Blocks on shelf — spread in Y
_BLOCKS = {
    "red":   {"color": (0.85, 0.12, 0.12), "pos": (_SHELF_X - 0.15, -0.14, _BLOCK_Z)},
    "green": {"color": (0.12, 0.75, 0.15), "pos": (_SHELF_X - 0.15,  0.00, _BLOCK_Z)},
    "blue":  {"color": (0.12, 0.20, 0.85), "pos": (_SHELF_X - 0.15,  0.14, _BLOCK_Z)},
}
_TARGET_BLOCK = "red"     # current cube label (drives the camera HUD); set per cube at runtime

# Motion parameters
WALK_VX          = 0.8    # forward (m/s)
_PICK_X          = _SHELF_X - 0.35   # robot stops here before reaching (0.35 m from shelf face)
ARRIVE_THRESH    = 0.10
STABILISE_FRAMES = 60
RAMP_FRAMES      = 80     # frames to ramp arm in/out
GRASP_FRAMES     = 40     # frames to close hand and stabilise
RELEASE_FRAMES   = 30
DONE_FRAMES      = 150     # stand & settle after the last cube, then close the sim

# REACH closed-loop servo: ease the arm to a calibrated guess, then nudge
# shoulder pitch/roll + elbow each frame to drive the live hand onto the cube.
# Gains use the probe-derived axis signs (probe_arm.py): +y←+roll, +z←−pitch,
# +x←−elbow/−pitch. Small per-frame steps keep the lagging PD plant stable.
REACH_SETTLE = 90      # frames to ease into the initial guess before servoing
REACH_MAX    = 260     # servo timeout (frames)
REACH_TOL    = 0.06    # hand→cube distance (m) that counts as "hugging"
SERVO_KZ     = 0.20    # pitch step per metre of z error
SERVO_KXP    = 0.10    # pitch step per metre of x error
SERVO_KY     = 0.20    # roll step per metre of y error
SERVO_KXE    = 0.40    # elbow step per metre of x error

# Hand body link — terminal link of left arm; the cube is driven to follow it.
_HAND_BODY_PATTERN = "left_two"

# ── Container (open-top bin) on the shelf, to the robot's left of the blocks ──
_BIN_X       = _SHELF_X - 0.15           # same depth as the blocks
_BIN_Y       = 0.45                       # left end of shelf, within left-arm reach
_BIN_INNER   = 0.24                       # inner footprint (m)
_BIN_WALL_H  = 0.10                       # wall height (m)
_BIN_FLOOR_Z = _SHELF_SURF_Z              # bin floor sits on the shelf surface

# Order the cubes are picked in (blue/green sit left — easiest for the left arm)
_CUBE_ORDER = ["blue", "green", "red"]
# Where each cube is set down inside the bin (small xy spread, no stacking)
_BIN_SLOTS  = [(-0.05, -0.04), (0.05, -0.04), (0.0, 0.05)]
# Cube offset relative to the hand link while carried
_HOLD_OFFSET = (0.04, 0.0, -0.04)

# ── Scripted left-arm key poses (rad), calibrated to G1's REAL default pose ──
# Only joints that exist on the model are applied; missing ones are ignored.
# shoulder_pitch ↑ swings the arm forward/down; shoulder_roll ↑ abducts to the
# left; elbow_pitch ↑ flexes the elbow.  (G1 has NO wrist joint.)
_OPEN, _CLOSE = 0.0, 0.7                   # gripper finger targets (one/two)
# Calibrated from an arm-kinematics sweep (scripts/probe_arm.py):
#   NEGATIVE shoulder_pitch reaches forward/up; roll abducts to the left (+y);
#   hand_y ≈ 0.09 + 0.26·roll, so to aim at cube_y: roll ≈ BASE + GAIN·cube_y.
_REACH_ROLL_BASE = -0.35
_REACH_ROLL_GAIN = 3.85
_POSE_REACH = {
    "left_shoulder_pitch_joint": -0.40,   # reach forward to the shelf (hand x≈4.38)
    "left_shoulder_roll_joint":  _REACH_ROLL_BASE,
    "left_shoulder_yaw_joint":   0.0,
    "left_elbow_pitch_joint":    0.30,    # near-straight: hand out at cube height (z≈0.82)
    "left_one_joint": _OPEN, "left_two_joint": _OPEN,
}
_POSE_LIFT = {
    "left_shoulder_pitch_joint": -0.80,   # raise the cube clear of the shelf (z≈0.95)
    "left_shoulder_roll_joint":  0.20,
    "left_elbow_pitch_joint":    0.30,
    "left_one_joint": _CLOSE, "left_two_joint": _CLOSE,
}
_POSE_OVER_BIN = {
    "left_shoulder_pitch_joint": -0.60,   # swing left, over the container (y≈0.45)
    "left_shoulder_roll_joint":  1.30,
    "left_elbow_pitch_joint":    0.40,
    "left_one_joint": _CLOSE, "left_two_joint": _CLOSE,
}
_POSE_LOWER = {
    "left_shoulder_pitch_joint": -0.40,   # lower the hand down into the container
    "left_shoulder_roll_joint":  1.30,
    "left_elbow_pitch_joint":    0.60,
    "left_one_joint": _CLOSE, "left_two_joint": _CLOSE,
}
# Joints the arm controller drives (union of all poses above)
_CTRL_JOINTS = sorted({*_POSE_REACH, *_POSE_LIFT, *_POSE_OVER_BIN, *_POSE_LOWER})


# ---------------------------------------------------------------------------
# Policy — load actor MLP from RSL-RL checkpoint
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _mat(rgb, opacity=1.0):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.5, opacity=opacity)


def _box(path, size, pos, rgb, opacity=1.0, orient=None):
    cfg = sim_utils.CuboidCfg(size=size, visual_material=_mat(rgb, opacity))
    kw  = {"translation": pos}
    if orient:
        kw["orientation"] = orient
    cfg.func(path, cfg, **kw)


def build_storage_room():
    FLOOR = (0.70, 0.65, 0.58)
    WALL  = (0.82, 0.80, 0.75)
    METAL = (0.45, 0.48, 0.52)

    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    cx, cy = _ROOM_W / 2.0, 0.0
    wz     = _ROOM_H / 2.0
    # Walls (visual, no collision — transparent front wall for clear view)
    _box("/World/Room/WallBack",   (_ROOM_W, 0.12, _ROOM_H), (cx, cy + _ROOM_D/2, wz), WALL)
    _box("/World/Room/WallLeft",   (0.12, _ROOM_D, _ROOM_H), (0.0,      cy, wz), WALL)
    _box("/World/Room/WallRight",  (0.12, _ROOM_D, _ROOM_H), (_ROOM_W,  cy, wz), WALL)
    _box("/World/Room/WallFront",  (_ROOM_W, 0.12, _ROOM_H), (cx, cy - _ROOM_D/2, wz), WALL, opacity=0.25)

    # Metal shelf unit against the back wall
    sx, sy = _SHELF_X, 0.0
    shelf_depth, shelf_width = 0.55, 1.4
    for i, dy in enumerate([-shelf_width/2 + 0.05, shelf_width/2 - 0.05]):
        _box(f"/World/Room/Shelf/Post{i}", (0.06, 0.06, 1.60),
             (sx, sy + dy, 0.80), METAL)
    # Lower shelf surface (our pick shelf) — needs collision so blocks rest on it
    shelf_cfg = sim_utils.CuboidCfg(
        size=(0.55, shelf_width, 0.04),
        visual_material=_mat(METAL),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    shelf_cfg.func("/World/Room/Shelf/LowerBoard", shelf_cfg,
                   translation=(sx - 0.01, sy, _SHELF_SURF_Z - 0.02))
    # Upper shelf (visual, not used for pick)
    _box("/World/Room/Shelf/UpperBoard", (0.55, shelf_width, 0.04),
         (sx - 0.01, sy, 1.35), METAL)

    # Ceiling light strips (RectLightCfg is unavailable in this Isaac Lab version —
    # use a disk light, which has the same intensity/color API plus a radius)
    light = sim_utils.DiskLightCfg(intensity=6000.0, color=(1.0, 0.97, 0.90),
                                   radius=1.0)
    light.func("/World/Room/Light0", light, translation=(2.0, 0.0, _ROOM_H - 0.05))
    light.func("/World/Room/Light1", light, translation=(5.0, 0.0, _ROOM_H - 0.05))

    dome = sim_utils.DomeLightCfg(intensity=400.0, color=(0.85, 0.90, 1.0))
    dome.func("/World/Room/Ambient", dome)

    print(f"[INFO] Storage room built  shelf face x={_SHELF_X:.2f}, "
          f"surface z={_SHELF_SURF_Z:.2f}")


def build_container():
    """Open-top bin on the shelf that the cubes are placed into (with collision)."""
    BIN  = (0.30, 0.55, 0.65)
    t    = 0.02                       # wall thickness
    half = _BIN_INNER / 2.0
    wz   = _BIN_FLOOR_Z + _BIN_WALL_H / 2.0

    def wall(name, size, pos):
        cfg = sim_utils.CuboidCfg(
            size=size, visual_material=_mat(BIN),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        cfg.func(name, cfg, translation=pos)

    wall("/World/Room/Bin/Floor", (_BIN_INNER + 2*t, _BIN_INNER + 2*t, t),
         (_BIN_X, _BIN_Y, _BIN_FLOOR_Z + t / 2))
    wall("/World/Room/Bin/WallN", (_BIN_INNER + 2*t, t, _BIN_WALL_H),
         (_BIN_X, _BIN_Y + half + t/2, wz))
    wall("/World/Room/Bin/WallS", (_BIN_INNER + 2*t, t, _BIN_WALL_H),
         (_BIN_X, _BIN_Y - half - t/2, wz))
    wall("/World/Room/Bin/WallE", (t, _BIN_INNER, _BIN_WALL_H),
         (_BIN_X + half + t/2, _BIN_Y, wz))
    wall("/World/Room/Bin/WallW", (t, _BIN_INNER, _BIN_WALL_H),
         (_BIN_X - half - t/2, _BIN_Y, wz))
    print(f"[INFO] Container built at x={_BIN_X:.2f}, y={_BIN_Y:.2f}")


def build_blocks() -> dict:
    """Spawn red/green/blue rigid-body blocks on the shelf. Returns {label: RigidObject}."""
    blocks = {}
    for label, info in _BLOCKS.items():
        cfg = RigidObjectCfg(
            prim_path=f"/World/Room/Blocks/{label.capitalize()}Block",
            spawn=sim_utils.CuboidCfg(
                size=(0.07, 0.07, _BLOCK_H),
                visual_material=_mat(info["color"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.40),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=info["pos"]),
        )
        blocks[label] = RigidObject(cfg=cfg)
    return blocks


def design_scene() -> dict:
    build_storage_room()
    build_container()
    blocks = build_blocks()

    robot_cfg           = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.init_state.pos = _ROBOT_START
    robot = Articulation(cfg=robot_cfg)

    contact_sensor = ContactSensor(cfg=ContactSensorCfg(
        prim_path="/World/G1/.*",
        update_period=0.0, history_length=1, debug_vis=False,
    ))

    camera = None
    if _CAMERAS_ENABLED:
        camera = Camera(cfg=CameraCfg(
            prim_path="/World/G1/torso_link/head_cam",
            update_period=0.1,
            height=480, width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 20.0)),
            # G1 has no head link, so mount on torso_link but raise the z-offset to
            # ~head/eye height for a first-person "head camera" looking forward+down.
            offset=CameraCfg.OffsetCfg(
                pos=(0.10, 0.0, 0.52),
                rot=(0.7071, 0.0, 0.7071, 0.0),
                convention="ros",
            ),
        ))

    return {"robot": robot, "blocks": blocks,
            "contact_sensor": contact_sensor, "camera": camera}


# ---------------------------------------------------------------------------
# Cube attachment
# ---------------------------------------------------------------------------
# The cube is never switched to kinematic (the direct-GPU pipeline forbids the
# dynamic↔kinematic flip at runtime). Instead, while held, the cube's pose is
# driven each frame and its velocity zeroed (done inline in run_simulator), so a
# still-dynamic body rides with the hand; on release we stop driving it and it
# settles in the bin under gravity.


# ---------------------------------------------------------------------------
# YOLO + colour inspector
# ---------------------------------------------------------------------------

_YOLO_MODEL        = None
_YOLO_READY        = False
_PREVIEW_READY     = False
_yolo_call_n       = 0
_YOLO_EVERY        = 10
_last_yolo_results = None
_last_target_xy    = None   # pixel centre of target block (for HUD)
_last_health       = (0.0, 0.0)
_image_provider    = None
_preview_label     = None


def _init_preview_window(w: int = 640, h: int = 480) -> None:
    global _PREVIEW_READY, _image_provider, _preview_label
    try:
        import omni.ui as ui
        _image_provider = ui.ByteImageProvider()
        _image_provider.set_bytes_data(bytes(w * h * 4), [w, h])
        win = ui.Window("Pick & Place Inspector", width=w + 20, height=h + 52)
        with win.frame:
            with ui.VStack(spacing=2):
                ui.ImageWithProvider(_image_provider, width=w, height=h)
                _preview_label = ui.Label("Waiting for camera...", height=24)
        _PREVIEW_READY = True
        print("[PREVIEW] 'Pick & Place Inspector' window created inside Isaac Sim")
    except Exception as exc:
        print(f"[PREVIEW] Could not create window: {exc}")


def _init_yolo() -> None:
    global _YOLO_MODEL, _YOLO_READY
    try:
        from ultralytics import YOLO  # type: ignore
        _YOLO_MODEL = YOLO("yolov8n.pt")
        _YOLO_READY = True
        print("[YOLO] YOLOv8n loaded")
    except Exception as exc:
        print(f"[YOLO] Disabled ({exc})")
    _init_preview_window()


def _detect_target_color(rgb_np: np.ndarray, target: str):
    """Return pixel (cx, cy) of the target colour block, or None if not visible."""
    img = rgb_np.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if target == "red":
        mask = (r > 0.55) & (g < 0.30) & (b < 0.30)
    elif target == "green":
        mask = (g > 0.55) & (r < 0.30) & (b < 0.30)
    elif target == "blue":
        mask = (b > 0.55) & (r < 0.30) & (g < 0.30)
    else:
        return None
    if mask.sum() < 30:
        return None
    ys, xs = np.where(mask)
    return (int(xs.mean()), int(ys.mean()))


def _run_inspection(camera, phase: str) -> None:
    global _yolo_call_n, _last_yolo_results, _last_target_xy
    if camera is None or not camera.is_initialized:
        return
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return

    rgb_np = rgb[0, :, :, :3].cpu().numpy()

    # Colour-based target detection (every frame, cheap)
    _last_target_xy = _detect_target_color(rgb_np, _TARGET_BLOCK)

    # YOLO (every N calls)
    _yolo_call_n += 1
    if _YOLO_READY and _yolo_call_n % _YOLO_EVERY == 0:
        results            = _YOLO_MODEL(rgb_np[:, :, ::-1].copy(), verbose=False)
        _last_yolo_results = results
        boxes, names = results[0].boxes, results[0].names
        hits = [(names[int(b.cls)], float(b.conf))
                for b in boxes if float(b.conf) > 0.35]
        if hits:
            print(f"[YOLO] {', '.join(f'{n} ({c:.0%})' for n, c in hits)}")

    _show_preview(rgb_np, phase)


def _show_preview(rgb_np: np.ndarray, phase: str) -> None:
    if not _PREVIEW_READY or _image_provider is None:
        return
    from PIL import Image as PILImage, ImageDraw
    display = rgb_np.copy()

    # Highlight target colour pixels
    img_f = display.astype(np.float32) / 255.0
    r_ch, g_ch, b_ch = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    if _TARGET_BLOCK == "red":
        target_mask = (r_ch > 0.55) & (g_ch < 0.30) & (b_ch < 0.30)
        tint = [255, 80, 80]
    elif _TARGET_BLOCK == "green":
        target_mask = (g_ch > 0.55) & (r_ch < 0.30) & (b_ch < 0.30)
        tint = [80, 255, 80]
    else:
        target_mask = (b_ch > 0.55) & (r_ch < 0.30) & (g_ch < 0.30)
        tint = [80, 80, 255]

    if target_mask.any():
        overlay = display.copy()
        overlay[target_mask] = tint
        display = (display * 0.5 + overlay * 0.5).astype(np.uint8)

    # PIL drawing
    pil_img = PILImage.fromarray(display)
    draw    = ImageDraw.Draw(pil_img)

    # YOLO boxes
    if _last_yolo_results is not None:
        for box in _last_yolo_results[0].boxes:
            if float(box.conf) < 0.35:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            name  = _last_yolo_results[0].names[int(box.cls)]
            label = f"{name} {float(box.conf):.0%}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=2)
            draw.text((x1 + 3, max(y1 - 14, 0)), label, fill=(255, 200, 0))

    # Target crosshair
    if _last_target_xy is not None:
        cx, cy = _last_target_xy
        r = 12
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=2)
        draw.line([cx - r - 4, cy, cx + r + 4, cy], fill=(255, 255, 255), width=1)
        draw.line([cx, cy - r - 4, cx, cy + r + 4], fill=(255, 255, 255), width=1)

    # HUD
    status_map = {
        "walk": "WALKING TO SHELF",
        "arrive": "STABILISING",
        "reach": "REACHING FOR CUBE",
        "grasp": "GRASPING",
        "lift": "LIFTING",
        "over_bin": "MOVING TO BIN",
        "lower": "LOWERING INTO BIN",
        "release": "RELEASING",
        "done": "DONE",
    }
    status_text = status_map.get(phase, phase.upper())
    target_found = "TARGET FOUND" if _last_target_xy else "SEARCHING..."
    display = np.array(pil_img)
    H, W = display.shape[:2]
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = display
    rgba[:, :, 3]  = 255
    _image_provider.set_bytes_data(rgba.tobytes(), [W, H])
    if _preview_label is not None:
        try:
            _preview_label.text = (
                f"{status_text}  |  {target_found}  |  Pick: {_TARGET_BLOCK.upper()}"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_LEG_LINK_SUBSTRINGS = ("hip", "knee", "ankle", "calf", "thigh")
_contact_cooldown    = 0


def _check_contact(sensor: ContactSensor, phase: str) -> None:
    global _contact_cooldown
    if _contact_cooldown > 0:
        _contact_cooldown -= 1
        return
    if phase not in ("reach", "grasp", "lift", "over_bin", "lower", "release"):
        return
    if not sensor.is_initialized:
        return
    forces = sensor.data.net_forces_w
    if forces is None:
        return
    magnitudes = forces.norm(dim=-1)[0]
    for body_idx in range(len(sensor.body_names)):
        name = sensor.body_names[body_idx]
        if any(s in name for s in _LEG_LINK_SUBSTRINGS):
            continue
        mag = magnitudes[body_idx].item()
        if mag > 3.0:
            print(f"[CONTACT] '{name}' — {mag:.1f} N")
            _contact_cooldown = 80
            return


def _build_arm_pose(robot: Articulation, joint_dict: dict) -> torch.Tensor:
    pose = robot.data.default_joint_pos.clone()
    n2i  = {name: i for i, name in enumerate(robot.joint_names)}
    for jname, val in joint_dict.items():
        idx = n2i.get(jname)
        if idx is not None:
            pose[0, idx] = val
    return pose


def _smoothstep(t: float) -> float:
    """Ease-in/out blend factor (zero velocity at both ends) for smooth motion."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def run_simulator(sim: SimulationContext, robot: Articulation, policy: nn.Module,
                  blocks: dict, sim_dt: float,
                  contact_sensor: ContactSensor, camera=None) -> None:
    device       = sim.device
    n2i          = {name: i for i, name in enumerate(robot.joint_names)}
    default_jpos = robot.data.default_joint_pos.clone()

    # Find hand body link index
    hand_idx = None
    for i, bname in enumerate(robot.body_names):
        if _HAND_BODY_PATTERN in bname:
            hand_idx = i
            print(f"[INFO] Hand link: '{bname}' (idx {i})")
            break
    if hand_idx is None:
        print(f"[WARN] Could not find hand link '{_HAND_BODY_PATTERN}'. "
              f"Body names: {robot.body_names[:10]}...")

    global _TARGET_BLOCK

    # Shared arm key poses (full 37-dim joint-target tensors)
    lift_pose  = _build_arm_pose(robot, _POSE_LIFT)
    over_pose  = _build_arm_pose(robot, _POSE_OVER_BIN)
    lower_pose = _build_arm_pose(robot, _POSE_LOWER)

    hold_offset = torch.tensor(_HOLD_OFFSET, device=device)
    cubes       = [blocks[c] for c in _CUBE_ORDER]

    # Reach-servo joint limits (clamp the nudged targets to what the joints allow)
    _jl = robot.data.soft_joint_pos_limits[0]
    def _lim(j):
        return (_jl[n2i[j], 0].item(), _jl[n2i[j], 1].item()) if j in n2i else (-3.0, 3.0)
    pitch_lim = _lim("left_shoulder_pitch_joint")
    roll_lim  = _lim("left_shoulder_roll_joint")
    elbow_lim = _lim("left_elbow_pitch_joint")

    def clamp(v, lim):
        return min(max(v, lim[0]), lim[1])

    def servo_reach_pose() -> torch.Tensor:
        return _build_arm_pose(robot, {
            "left_shoulder_pitch_joint": reach_pitch,
            "left_shoulder_roll_joint":  reach_roll,
            "left_shoulder_yaw_joint":   0.0,
            "left_elbow_pitch_joint":    reach_elbow,
            "left_one_joint": _OPEN, "left_two_joint": _OPEN,
        })

    # Velocity commands
    walk_fwd = torch.tensor([WALK_VX, 0.0, 0.0], device=device)
    stop_cmd = torch.zeros(3, device=device)
    cmd      = walk_fwd.clone()
    last_action = torch.zeros(1, _ACTION_DIM, device=device)

    # State
    phase     = "walk"
    frame     = 0
    cube_idx  = 0
    holding   = False

    # Arm-blend state: smoothstep from `pose_from` → `pose_to` over `blend_n` frames
    cur_pose  = default_jpos.clone()
    pose_from = default_jpos.clone()
    pose_to   = default_jpos.clone()
    blend_n   = RAMP_FRAMES

    # Cube-driving anchors (world positions captured at phase entry)
    grasp_start = torch.zeros(3, device=device)
    lower_start = torch.zeros(3, device=device)
    bin_target  = torch.zeros(3, device=device)

    # Reach-servo state (commanded arm angles, refined each frame from hand error)
    reach_pitch = _POSE_REACH["left_shoulder_pitch_joint"]
    reach_roll  = _REACH_ROLL_BASE
    reach_elbow = _POSE_REACH["left_elbow_pitch_joint"]
    reach_hold  = 0
    reach_err   = 9.9

    def cube_y_of(idx: int) -> float:
        return cubes[idx].data.root_pos_w[0, 1].item()

    print(f"\n[INFO] Placing {len(cubes)} cubes ({', '.join(_CUBE_ORDER)}) into the bin "
          f"at x={_BIN_X:.2f}, y={_BIN_Y:.2f} | Stop at x={_PICK_X:.1f}")
    print("[INFO] WALK → ARRIVE → [REACH → GRASP → LIFT → OVER_BIN → LOWER → RELEASE] ×"
          f"{len(cubes)} → DONE\n")

    while simulation_app.is_running():
        robot_x = robot.data.root_pos_w[0, 0].item()
        active  = cubes[cube_idx] if cube_idx < len(cubes) else None

        # ── State machine ────────────────────────────────────────────────────
        if phase == "walk":
            cmd = walk_fwd
            if robot_x >= _PICK_X - ARRIVE_THRESH:
                print(f"[WALK→ARRIVE] x={robot_x:.2f}")
                phase = "arrive"; frame = 0

        elif phase == "arrive":
            cmd    = stop_cmd
            frame += 1
            if frame >= STABILISE_FRAMES:
                _TARGET_BLOCK = _CUBE_ORDER[cube_idx]
                pose_from = cur_pose.clone()
                reach_pitch = _POSE_REACH["left_shoulder_pitch_joint"]
                reach_roll  = _REACH_ROLL_BASE + cube_y_of(cube_idx) * _REACH_ROLL_GAIN
                reach_elbow = _POSE_REACH["left_elbow_pitch_joint"]
                reach_hold  = 0; frame = 0
                print(f"[ARRIVE→REACH] cube {cube_idx+1}/{len(cubes)} ({_CUBE_ORDER[cube_idx]})")
                phase = "reach"

        elif phase == "reach":
            cmd    = stop_cmd
            frame += 1
            # After easing into the initial guess, servo the hand onto the cube.
            if frame > REACH_SETTLE and hand_idx is not None:
                hp = robot.data.body_pos_w[0, hand_idx]
                cp = active.data.root_pos_w[0, :3]
                ex = (cp[0] - hp[0]).item()
                ey = (cp[1] - hp[1]).item()
                ez = (cp[2] - hp[2]).item()
                reach_pitch = clamp(reach_pitch - SERVO_KZ * ez - SERVO_KXP * ex, pitch_lim)
                reach_roll  = clamp(reach_roll  + SERVO_KY  * ey,                 roll_lim)
                reach_elbow = clamp(reach_elbow - SERVO_KXE * ex,                 elbow_lim)
                reach_err   = (ex*ex + ey*ey + ez*ez) ** 0.5
                reach_hold  = reach_hold + 1 if reach_err < REACH_TOL else 0
            if reach_hold >= 8 or frame >= REACH_MAX:
                grasp_pose = {
                    "left_shoulder_pitch_joint": reach_pitch,
                    "left_shoulder_roll_joint":  reach_roll,
                    "left_shoulder_yaw_joint":   0.0,
                    "left_elbow_pitch_joint":    reach_elbow,
                    "left_one_joint": _CLOSE, "left_two_joint": _CLOSE,
                }
                pose_from = cur_pose.clone(); pose_to = _build_arm_pose(robot, grasp_pose)
                blend_n = GRASP_FRAMES; frame = 0
                grasp_start = active.data.root_pos_w[0, :3].clone()
                holding = True
                print(f"[REACH→GRASP] hand on cube | gap={reach_err:.2f}m "
                      f"({'converged' if reach_hold >= 8 else 'timeout'})")
                phase = "grasp"

        elif phase == "grasp":
            cmd    = stop_cmd
            frame += 1
            if frame >= GRASP_FRAMES:
                pose_from = cur_pose.clone(); pose_to = lift_pose
                blend_n = RAMP_FRAMES; frame = 0
                print("[GRASP→LIFT] lifting cube")
                phase = "lift"

        elif phase == "lift":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                pose_from = cur_pose.clone(); pose_to = over_pose
                blend_n = RAMP_FRAMES; frame = 0
                print("[LIFT→OVER_BIN] moving over the container")
                phase = "over_bin"

        elif phase == "over_bin":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                pose_from = cur_pose.clone(); pose_to = lower_pose
                blend_n = RAMP_FRAMES; frame = 0
                lower_start = active.data.root_pos_w[0, :3].clone()
                sx, sy = _BIN_SLOTS[cube_idx % len(_BIN_SLOTS)]
                bin_target = torch.tensor(
                    [_BIN_X + sx, _BIN_Y + sy, _BIN_FLOOR_Z + _BLOCK_H / 2 + 0.01],
                    device=device)
                print("[OVER_BIN→LOWER] lowering cube into the container")
                phase = "lower"

        elif phase == "lower":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                holding = False   # cube now resting at bin_target; let it settle
                print("[LOWER→RELEASE] opening hand")
                phase = "release"; frame = 0

        elif phase == "release":
            cmd    = stop_cmd
            frame += 1
            if frame >= RELEASE_FRAMES:
                print(f"[RELEASE] {_CUBE_ORDER[cube_idx].upper()} cube placed in bin")
                cube_idx += 1
                if cube_idx < len(cubes):
                    _TARGET_BLOCK = _CUBE_ORDER[cube_idx]
                    pose_from = cur_pose.clone()
                    reach_pitch = _POSE_REACH["left_shoulder_pitch_joint"]
                    reach_roll  = _REACH_ROLL_BASE + cube_y_of(cube_idx) * _REACH_ROLL_GAIN
                    reach_elbow = _POSE_REACH["left_elbow_pitch_joint"]
                    reach_hold  = 0; frame = 0
                    print(f"[RELEASE→REACH] next cube {cube_idx+1}/{len(cubes)} "
                          f"({_CUBE_ORDER[cube_idx]})")
                    phase = "reach"
                else:
                    pose_from = cur_pose.clone(); pose_to = default_jpos
                    blend_n = RAMP_FRAMES; frame = 0
                    print("[RELEASE→DONE] all cubes placed — robot standing")
                    # Verification summary: robot upright? cubes inside the bin?
                    rz = robot.data.root_pos_w[0, 2].item()
                    print(f"[CHECK] robot base height z={rz:.2f} m "
                          f"({'UPRIGHT' if rz > 0.5 else 'FALLEN'})")
                    half = _BIN_INNER / 2.0
                    for lbl in _CUBE_ORDER:
                        p = blocks[lbl].data.root_pos_w[0, :3].tolist()
                        inside = (abs(p[0] - _BIN_X) < half + 0.04 and
                                  abs(p[1] - _BIN_Y) < half + 0.04)
                        print(f"[CHECK] {lbl:>5} cube at "
                              f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) "
                              f"{'IN BIN' if inside else 'OUT'}")
                    phase = "done"

        elif phase == "done":
            cmd    = stop_cmd
            frame += 1   # let the arm ease back to its home pose, then exit
            if frame >= DONE_FRAMES:
                print("[DONE] closing simulation")
                break

        # ── Policy inference ─────────────────────────────────────────────────
        with torch.inference_mode():
            obs    = build_obs(robot, cmd, last_action)
            action = policy(obs)
        action      = action.clone()
        last_action = action.clone()

        # ── Arm override (manipulation phases) ────────────────────────────────
        if phase == "reach":
            # Ease into the servo target (which is itself nudged toward the cube).
            a        = _smoothstep(frame / max(REACH_SETTLE, 1))
            cur_pose = torch.lerp(pose_from, servo_reach_pose(), a)
        elif phase in ("grasp", "lift", "over_bin", "lower", "release", "done"):
            a        = _smoothstep(frame / max(blend_n, 1))
            cur_pose = torch.lerp(pose_from, pose_to, a)
        if phase in ("reach", "grasp", "lift", "over_bin", "lower", "release", "done"):
            for jname in _CTRL_JOINTS:
                idx = n2i.get(jname)
                if idx is not None:
                    action[0, idx] = (cur_pose[0, idx] - default_jpos[0, idx]) / _ACTION_SCALE

        # ── Apply to robot ────────────────────────────────────────────────────
        joint_targets = default_jpos + _ACTION_SCALE * action
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()

        # ── Drive the held cube ───────────────────────────────────────────────
        if holding and active is not None and hand_idx is not None:
            hand = robot.data.body_pos_w[0, hand_idx]
            if phase == "grasp":            # smoothly draw the cube into the hand
                tgt = torch.lerp(grasp_start, hand + hold_offset,
                                 _smoothstep(frame / max(GRASP_FRAMES, 1)))
            elif phase in ("lift", "over_bin"):   # cube rides with the hand
                tgt = hand + hold_offset
            elif phase == "lower":          # smoothly set it down into the bin
                tgt = torch.lerp(lower_start, bin_target,
                                 _smoothstep(frame / max(RAMP_FRAMES, 1)))
            else:
                tgt = active.data.root_pos_w[0, :3]
            pose7 = torch.zeros(1, 7, device=device)
            pose7[0, :3] = tgt
            pose7[0, 3]  = 1.0
            active.write_root_pose_to_sim(pose7)
            active.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            active.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)
        for blk in blocks.values():
            blk.update(sim_dt)

        contact_sensor.update(sim_dt)
        _check_contact(contact_sensor, phase)

        if camera is not None:
            camera.update(sim_dt)
            _run_inspection(camera, phase)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005, device=args_cli.device,
        render=sim_utils.RenderCfg(rendering_mode="quality"),
    )
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye   =[_ROBOT_START[0] - 1.5, _ROBOT_START[1] - 3.5, 2.2],
        target=[_SHELF_X, 0.0, 0.9],
    )

    entities       = design_scene()
    robot          = entities["robot"]
    blocks         = entities["blocks"]
    contact_sensor = entities["contact_sensor"]
    camera         = entities["camera"]

    sim.reset()
    _init_yolo()

    print(f"[INFO] Scene ready — {len(blocks)} blocks on shelf, "
          f"target: {_TARGET_BLOCK.upper()}")
    if camera is not None:
        print("[INFO] Camera active — Pick & Place Inspector will open")
    else:
        print("[INFO] Camera disabled — rerun with --enable_cameras to enable YOLO")

    policy = load_policy(sim.device)
    print("[INFO] Policy ready — G1 will walk under real dynamics\n")

    run_simulator(sim, robot, policy, blocks, sim_cfg.dt, contact_sensor, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()
