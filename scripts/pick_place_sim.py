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
_TARGET_BLOCK = "red"     # which block to pick — change to "green" or "blue"

# Drop zone — robot walks backwards to this X, then releases block on floor
_DROP_X  = 2.0
_DROP_Y  = 1.5

# Motion parameters
WALK_VX          = 0.8    # forward (m/s)
WALK_VX_BACK     = -0.6   # backward (m/s) to drop zone
WALK_WZ_DROP     = 0.0    # no turning needed
_PICK_X          = _SHELF_X - 0.35   # robot stops here before reaching (0.35 m from shelf face)
ARRIVE_THRESH    = 0.10
STABILISE_FRAMES = 60
RAMP_FRAMES      = 80     # frames to ramp arm in/out
GRASP_FRAMES     = 40     # frames to close hand and stabilise
RELEASE_FRAMES   = 30

# Hand body link — terminal link of left arm; used to track block during carry.
# If G1_MINIMAL_CFG has different names, adjust this pattern.
_HAND_BODY_PATTERN = "left_two"

# Arm joint targets for each motion phase.
# REACH: arm swings forward+slightly down to shelf level.
# LIFT:  arm pulls object up and holds close to torso.
# LOWER: arm extends downward to release near floor.
_REACH_JOINTS = {
    "left_shoulder_pitch_joint":  0.55,   # forward swing ~31°
    "left_shoulder_roll_joint":  -0.05,   # slight inward
    "left_elbow_pitch_joint":     0.85,   # elbow bend ~49°
    "left_wrist_pitch_joint":    -0.35,   # wrist tilts hand down toward block
    "left_one_joint":             0.0,    # hand open
    "left_two_joint":             0.0,
}
_GRASP_JOINTS = {
    **_REACH_JOINTS,
    "left_one_joint": 0.7,   # close hand
    "left_two_joint": 0.7,
}
_LIFT_JOINTS = {
    "left_shoulder_pitch_joint":  0.15,
    "left_shoulder_roll_joint":  -0.10,
    "left_elbow_pitch_joint":     0.45,
    "left_wrist_pitch_joint":    -0.10,
    "left_one_joint":             0.7,   # keep hand closed
    "left_two_joint":             0.7,
}
_LOWER_JOINTS = {
    "left_shoulder_pitch_joint":  0.40,
    "left_shoulder_roll_joint":  -0.05,
    "left_elbow_pitch_joint":     0.90,
    "left_wrist_pitch_joint":    -0.60,
    "left_one_joint":             0.7,   # still closed
    "left_two_joint":             0.7,
}


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

    # Ceiling light strips
    light = sim_utils.RectLightCfg(intensity=6000.0, color=(1.0, 0.97, 0.90),
                                    width=2.0, height=0.3)
    light.func("/World/Room/Light0", light, translation=(2.0, 0.0, _ROOM_H - 0.05))
    light.func("/World/Room/Light1", light, translation=(5.0, 0.0, _ROOM_H - 0.05))

    dome = sim_utils.DomeLightCfg(intensity=400.0, color=(0.85, 0.90, 1.0))
    dome.func("/World/Room/Ambient", dome)

    # Drop zone marker (flat disc on floor)
    _box("/World/Room/DropZone", (0.50, 0.50, 0.005),
         (_DROP_X, _DROP_Y, 0.003), (0.95, 0.80, 0.15))

    print(f"[INFO] Storage room built  shelf face x={_SHELF_X:.2f}, "
          f"surface z={_SHELF_SURF_Z:.2f}")


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
            prim_path="/World/G1/torso_link/pick_cam",
            update_period=0.1,
            height=480, width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 20.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.12, 0.0, 0.20),
                rot=(0.7071, 0.0, 0.7071, 0.0),
                convention="ros",
            ),
        ))

    return {"robot": robot, "blocks": blocks,
            "contact_sensor": contact_sensor, "camera": camera}


# ---------------------------------------------------------------------------
# Block grasp helpers
# ---------------------------------------------------------------------------

def _set_block_kinematic(stage, block: RigidObject, kinematic: bool) -> None:
    from pxr import UsdPhysics
    prim   = stage.GetPrimAtPath(block.cfg.prim_path)
    rb_api = UsdPhysics.RigidBodyAPI(prim)
    if rb_api:
        rb_api.GetKinematicEnabledAttr().Set(kinematic)


def _drive_block_to_hand(robot: Articulation, block: RigidObject,
                          hand_idx: int, grasp_offset: torch.Tensor,
                          device: str) -> None:
    """Each frame while holding: teleport block to follow the hand link."""
    hand_pos = robot.data.body_pos_w[0, hand_idx]            # (3,)
    block_pos = hand_pos + grasp_offset
    pose = torch.zeros(1, 7, device=device)
    pose[0, :3] = block_pos
    pose[0, 3]  = 1.0   # quaternion w=1 (identity — keep block upright)
    block.write_root_pose_to_sim(pose)
    block.write_data_to_sim()


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
        "reach": "REACHING FOR BLOCK",
        "grasp": "GRASPING",
        "lift": "LIFTING",
        "walk_back": "CARRYING TO DROP",
        "drop_arrive": "AT DROP ZONE",
        "lower": "LOWERING BLOCK",
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
    if phase not in ("reach", "grasp", "lift", "lower", "release"):
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


def run_simulator(sim: SimulationContext, robot: Articulation, policy: nn.Module,
                  blocks: dict, sim_dt: float,
                  contact_sensor: ContactSensor, camera=None) -> None:
    import isaaclab.sim as sim_utils_inner  # already imported — for get_current_stage
    stage = sim_utils_inner.get_current_stage()

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

    # Build arm pose tensors for each motion phase
    reach_jpos   = _build_arm_pose(robot, _REACH_JOINTS)
    grasp_jpos   = _build_arm_pose(robot, _GRASP_JOINTS)
    lift_jpos    = _build_arm_pose(robot, _LIFT_JOINTS)
    lower_jpos   = _build_arm_pose(robot, _LOWER_JOINTS)

    target_block = blocks[_TARGET_BLOCK]

    # Velocity commands
    walk_fwd  = torch.tensor([WALK_VX,      0.0, 0.0], device=device)
    walk_back = torch.tensor([WALK_VX_BACK, 0.0, 0.0], device=device)
    stop_cmd  = torch.zeros(3, device=device)
    cmd       = walk_fwd.clone()

    last_action = torch.zeros(1, _ACTION_DIM, device=device)

    phase          = "walk"
    frame          = 0
    holding_block  = False
    grasp_offset   = torch.zeros(3, device=device)  # hand→block offset at grasp moment

    # --- Per-arm-phase target pose (used to build action override) ---
    arm_target_jpos = default_jpos  # updated at each phase transition

    print(f"\n[INFO] Pick target: {_TARGET_BLOCK.upper()} block | "
          f"Shelf x={_SHELF_X:.1f} | Stop at x={_PICK_X:.1f}")
    print(f"[INFO] WALK → ARRIVE → REACH → GRASP → LIFT → "
          f"WALK_BACK → DROP_ARRIVE → LOWER → RELEASE → DONE\n")

    while simulation_app.is_running():
        robot_x = robot.data.root_pos_w[0, 0].item()
        robot_y = robot.data.root_pos_w[0, 1].item()

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
                arm_target_jpos = reach_jpos
                print("[ARRIVE→REACH] extending arm toward block")
                phase = "reach"; frame = 0

        elif phase == "reach":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                arm_target_jpos = grasp_jpos
                print("[REACH→GRASP] closing hand")
                phase = "grasp"; frame = 0

        elif phase == "grasp":
            cmd    = stop_cmd
            frame += 1
            if frame == GRASP_FRAMES // 2:
                # Attach block: switch to kinematic + compute grasp offset
                _set_block_kinematic(stage, target_block, True)
                holding_block = True
                if hand_idx is not None:
                    hand_pos = robot.data.body_pos_w[0, hand_idx]
                    block_pos = target_block.data.root_pos_w[0, :3]
                    grasp_offset = block_pos - hand_pos
                    grasp_offset[2] = max(grasp_offset[2], -0.12)  # clamp vertical offset
                print(f"[GRASP] Block attached — grasp offset {grasp_offset.tolist()}")
            if frame >= GRASP_FRAMES:
                arm_target_jpos = lift_jpos
                print("[GRASP→LIFT] lifting block")
                phase = "lift"; frame = 0

        elif phase == "lift":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                cmd = walk_back
                print(f"[LIFT→WALK_BACK] carrying block to drop zone x={_DROP_X:.1f}")
                phase = "walk_back"; frame = 0

        elif phase == "walk_back":
            cmd = walk_back
            if robot_x <= _DROP_X + ARRIVE_THRESH:
                cmd   = stop_cmd
                print(f"[WALK_BACK→DROP_ARRIVE] x={robot_x:.2f}")
                phase = "drop_arrive"; frame = 0

        elif phase == "drop_arrive":
            cmd    = stop_cmd
            frame += 1
            if frame >= STABILISE_FRAMES:
                arm_target_jpos = lower_jpos
                print("[DROP_ARRIVE→LOWER] lowering block to release")
                phase = "lower"; frame = 0

        elif phase == "lower":
            cmd    = stop_cmd
            frame += 1
            if frame >= RAMP_FRAMES:
                print("[LOWER→RELEASE] opening hand")
                phase = "release"; frame = 0

        elif phase == "release":
            cmd    = stop_cmd
            frame += 1
            if frame == RELEASE_FRAMES // 2:
                # Detach block: switch back to dynamic
                _set_block_kinematic(stage, target_block, False)
                holding_block = False
                print(f"[RELEASE] Block released — it will fall to floor")
            if frame >= RELEASE_FRAMES:
                arm_target_jpos = default_jpos
                print("[RELEASE→DONE] task complete — robot standing")
                phase = "done"

        # phase "done" — robot stands under policy with zero command

        # ── Policy inference ─────────────────────────────────────────────────
        with torch.inference_mode():
            obs    = build_obs(robot, cmd, last_action)
            action = policy(obs)
        action      = action.clone()
        last_action = action.clone()

        # ── Arm override (all non-walk phases) ───────────────────────────────
        if phase in ("reach", "grasp", "lift", "lower", "release"):
            alpha = min(frame / max(RAMP_FRAMES, 1), 1.0)
            interp = torch.lerp(default_jpos, arm_target_jpos, alpha)
        elif phase in ("walk_back", "drop_arrive", "done") and holding_block:
            interp = lift_jpos   # keep arm raised while carrying
            alpha  = 1.0
        else:
            alpha  = 0.0
            interp = default_jpos

        if alpha > 0.0:
            for jname in {**_REACH_JOINTS, **_LIFT_JOINTS, **_LOWER_JOINTS}:
                idx = n2i.get(jname)
                if idx is not None:
                    action[0, idx] = (interp[0, idx] - default_jpos[0, idx]) / _ACTION_SCALE

        # ── Apply to robot ────────────────────────────────────────────────────
        joint_targets = default_jpos + _ACTION_SCALE * action
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()

        # ── Drive block ───────────────────────────────────────────────────────
        if holding_block and hand_idx is not None:
            _drive_block_to_hand(robot, target_block, hand_idx, grasp_offset, device)

        sim.step()
        robot.update(sim_dt)
        target_block.update(sim_dt)
        for lbl, blk in blocks.items():
            if lbl != _TARGET_BLOCK:
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
