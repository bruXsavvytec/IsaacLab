"""
B3/B4 — collect reach-to-named-cube demonstrations for GR00T fine-tuning.

Anchored G1 at a table with a blue + yellow cube (positions randomized per
episode). A scripted expert uses a differential IK controller to drive the LEFT
hand to whichever cube the (randomly phrased) instruction names. Every step is
recorded into a GR00T-flavored LeRobot v2 dataset via LeRobotV2Recorder.

This is multi-command + language-conditioned from day one: half the episodes
target blue, half yellow, with varied phrasings — so the fine-tuned policy must
read the instruction to pick the right cube.

Run:
    cd /home/trooperai/IsaacLab
    ./isaaclab.sh -p scripts/vla/finetune/collect_demos.py --enable_cameras \
        --num-episodes 40 --out /home/trooperai/g1_reach_demos

Then generate stats and fine-tune:
    python -m gr00t.data.stats --dataset-path <out> --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path scripts/vla/finetune/g1_tabletop_config.py
    (see getting_started/finetune_new_embodiment.md)
"""

import argparse
import math
import os
import random
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect G1 reach demos for GR00T")
parser.add_argument("--num-episodes", type=int, default=40)
parser.add_argument("--steps", type=int, default=60, help="recorded steps per episode")
parser.add_argument("--out", type=str, default="/home/trooperai/g1_reach_demos")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after AppLauncher ───────────────────────────────────────────────────
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip

sys.path.insert(0, os.path.dirname(__file__))
from g1_tabletop_config import SINGLE_ARM_JOINTS, GRIPPER_JOINT
from lerobot_writer import LeRobotV2Recorder, g1_tabletop_modality

random.seed(args_cli.seed)

# ── layout ──────────────────────────────────────────────────────────────────────
_ROBOT_START   = (0.0, 0.0, 0.74)
# G1 left arm is short (≈0.42 m) and reaches the front-LEFT and CLOSE, not far
# forward. Measured (measure_reach.py): shoulder ≈ (0, 0.10, 1.0); reachable
# footprint at table height x[-0.35,0.10] y[0.04,0.36]. The reachable zone in
# FRONT overlaps the robot's own footprint, so a front table engulfs the robot.
# Solution: put the workspace to the front-LEFT (where the body isn't), as a thin
# floating slab (no legs to clip the robot), with cubes inside the reach disk.
_TABLE_TOP_Z   = 0.66
_TABLE_CENTER  = (0.08, 0.30)             # front-left of the robot, clear of body
_TABLE_TOP_SIZE = (0.30, 0.32, 0.04)      # slab only (legs removed)
_CUBE          = 0.05
_CUBE_Z        = _TABLE_TOP_Z + _CUBE / 2 + 0.002
_BLUE_RGB      = (0.10, 0.20, 0.85)
_YELLOW_RGB    = (0.90, 0.80, 0.10)

# Cube sampling = inside the left arm's reach disk, to the front-LEFT.
_SHOULDER_XY = (0.0, 0.104)
_REACH_MIN, _REACH_MAX = 0.16, 0.25      # horizontal dist from shoulder (m)
_SAMPLE_X = (0.00, 0.14)
_SAMPLE_Y = (0.20, 0.30)
_PNG_OUT  = "/tmp/g1_collect_latest.png"

_PHRASINGS = {
    "blue":   ["reach the blue cube", "touch the blue cube",
               "move your hand to the blue cube", "go to the blue block"],
    "yellow": ["reach the yellow cube", "touch the yellow cube",
               "move your hand to the yellow cube", "go to the yellow block"],
}

_EE_CANDIDATES = ["left_palm_link", "left_hand_link", "left_zero_link",
                  "left_rubber_hand", "left_two_link", "left_one_link",
                  "left_wrist_yaw_link", "left_elbow_roll_link"]


def _mat(rgb):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.7)


def build_table():
    # Thin floating slab (static collider, no legs) so nothing clips the anchored
    # robot. Placed to the front-left where the cubes (and the arm's reach) are.
    cx, cy = _TABLE_CENTER
    tw, td, tt = _TABLE_TOP_SIZE
    top = sim_utils.CuboidCfg(size=(tw, td, tt), visual_material=_mat((0.55, 0.40, 0.25)),
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
    top.func("/World/Table/Top", top, translation=(cx, cy, _TABLE_TOP_Z - tt / 2))


def _cube_cfg(path, rgb):
    return RigidObjectCfg(
        prim_path=path,
        spawn=sim_utils.CuboidCfg(
            size=(_CUBE, _CUBE, _CUBE), visual_material=_mat(rgb),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.1, _CUBE_Z)),
    )


def design_scene():
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sun = sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.98, 0.90))
    sun.func("/World/SunLight", sun, translation=(0, 0, 10), orientation=(0.906, 0.0, 0.423, 0.0))
    sim_utils.DomeLightCfg(intensity=600.0, color=(0.85, 0.88, 0.95)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=600.0, color=(0.85, 0.88, 0.95)))

    build_table()
    blue   = RigidObject(cfg=_cube_cfg("/World/BlueCube", _BLUE_RGB))
    yellow = RigidObject(cfg=_cube_cfg("/World/YellowCube", _YELLOW_RGB))

    robot_cfg = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.init_state.pos = _ROBOT_START
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot = Articulation(cfg=robot_cfg)

    camera = Camera(cfg=CameraCfg(
        prim_path="/World/G1/torso_link/insp_cam",
        update_period=0.0, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.10, 0.0, 0.28),
                                   rot=(-0.2988, 0.6409, -0.6409, 0.2988), convention="ros"),
    ))
    return {"robot": robot, "blue": blue, "yellow": yellow, "camera": camera}


# ── setup ───────────────────────────────────────────────────────────────────────
sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 30.0, device=args_cli.device))
sim.set_camera_view([1.6, 1.6, 1.4], [0.5, 0.0, 0.8])
scene  = design_scene()
robot  = scene["robot"]; camera = scene["camera"]
blue   = scene["blue"];  yellow = scene["yellow"]
sim.reset()
sim_dt = sim.get_physics_dt()

# joint / body indices
jn = robot.joint_names
arm_ids = [jn.index(n) for n in SINGLE_ARM_JOINTS]
grip_id = jn.index(GRIPPER_JOINT)
all6    = arm_ids + [grip_id]

ee_name = next((b for b in _EE_CANDIDATES if b in robot.body_names), None)
if ee_name is None:
    ee_name = [b for b in robot.body_names if b.startswith("left")][-1]
ee_body_id  = robot.body_names.index(ee_name)
ee_jacobi   = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
print(f"[COLLECT] EE body = {ee_name} (id {ee_body_id}, jacobi {ee_jacobi}); "
      f"arm_ids={arm_ids} grip_id={grip_id} fixed_base={robot.is_fixed_base}")

ik = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="position", use_relative_mode=False,
                                ik_method="dls", ik_params={"lambda_val": 0.1}),
    num_envs=1, device=sim.device)

# Joint limits for the 5 arm joints — clamp IK output so recorded actions stay
# physical (raw DLS can spike near singularities / unreachable targets).
try:
    _jlim = robot.data.soft_joint_pos_limits[0, arm_ids].clone()   # (5, 2)
except AttributeError:
    _jlim = robot.data.joint_pos_limits[0, arm_ids].clone()
_jlo = _jlim[:, 0].unsqueeze(0); _jhi = _jlim[:, 1].unsqueeze(0)   # (1, 5)

default_jpos = robot.data.default_joint_pos.clone()
grip_open    = float(default_jpos[0, grip_id].item())

recorder = LeRobotV2Recorder(
    args_cli.out, fps=30,
    state_names=SINGLE_ARM_JOINTS + [GRIPPER_JOINT],
    action_names=SINGLE_ARM_JOINTS + [GRIPPER_JOINT],
    modality=g1_tabletop_modality(), image_hw=(480, 640), robot_type="g1_sim")


def _rand_xy(exclude=None):
    """Sample a cube (x,y) inside the left arm's reachable disk on the table."""
    for _ in range(80):
        x = random.uniform(*_SAMPLE_X); y = random.uniform(*_SAMPLE_Y)
        d = math.hypot(x - _SHOULDER_XY[0], y - _SHOULDER_XY[1])
        if not (_REACH_MIN <= d <= _REACH_MAX):
            continue
        if exclude is not None and math.hypot(x - exclude[0], y - exclude[1]) < 0.10:
            continue
        return x, y
    return 0.10, 0.18


def _show_preview(rgb, label):
    from PIL import Image, ImageDraw
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 639, 26], fill=(20, 20, 20)); d.text((6, 8), label, fill=(230, 230, 230))
    pil.save(_PNG_OUT)


def place_cube(obj, x, y):
    pose = torch.tensor([[x, y, _CUBE_Z, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))


def reset_arm():
    robot.write_joint_state_to_sim(default_jpos, torch.zeros_like(default_jpos))


# ── IK helpers ────────────────────────────────────────────────────────────────────
def _ik_arm_target(target_obj):
    """One IK step → clamped arm joint target that moves the hand toward the cube."""
    tgt_w = target_obj.data.root_pos_w[:, 0:3].clone()
    tgt_w[:, 2] += 0.03                                  # approach just above the cube
    root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
    ee_pos_w  = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    tgt_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, tgt_w, ee_quat_w)
    jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi, :, arm_ids]
    ik.set_command(tgt_b, ee_quat=ee_quat_b)
    arm_des = ik.compute(ee_pos_b, ee_quat_b, jac, robot.data.joint_pos[:, arm_ids])
    return torch.clamp(arm_des, _jlo, _jhi)


def _settle(n=5):
    for _ in range(n):
        robot.write_data_to_sim(); sim.step(); robot.update(sim_dt)
    blue.update(sim_dt); yellow.update(sim_dt)


def solve_reach_pose(target_obj, iters=45):
    """Run IK to convergence (NOT recorded) and return the reached arm pose (1,5)."""
    grip = torch.full((1, 1), grip_open, device=sim.device)
    for _ in range(iters):
        targets6 = torch.cat([_ik_arm_target(target_obj), grip], dim=-1)
        robot.set_joint_position_target(targets6, joint_ids=all6)
        robot.write_data_to_sim(); sim.step(); robot.update(sim_dt); target_obj.update(sim_dt)
    return robot.data.joint_pos[:, arm_ids].clone()


# ── collection loop: solve a reach pose, then record a SMOOTH replay to it ─────────
print(f"[COLLECT] {args_cli.num_episodes} episodes → {args_cli.out}")
q_start = default_jpos[:, arm_ids].clone()               # (1,5)
grip    = torch.full((1, 1), grip_open, device=sim.device)
for ep in range(args_cli.num_episodes):
    color = "blue" if ep % 2 == 0 else "yellow"
    instruction = random.choice(_PHRASINGS[color])
    bx, by = _rand_xy(); yx, yy = _rand_xy(exclude=(bx, by))

    # Phase A: place cubes, solve a stable reach pose with IK (not recorded).
    place_cube(blue, bx, by); place_cube(yellow, yx, yy)
    reset_arm(); _settle()
    target_obj = blue if color == "blue" else yellow
    q_goal = solve_reach_pose(target_obj)

    # Phase B: reset + re-place cubes, record a smooth (jitter-free) reach to q_goal.
    reset_arm(); place_cube(blue, bx, by); place_cube(yellow, yx, yy); _settle()
    recorder.start_episode(instruction)
    for t in range(args_cli.steps):
        camera.update(sim_dt)
        rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        _show_preview(rgb, f"ep{ep} {color} | {instruction}")
        a = t / max(args_cli.steps - 1, 1)
        s = a * a * (3 - 2 * a)                          # smoothstep ease in/out
        q_t = q_start + s * (q_goal - q_start)           # (1,5) — smooth, monotonic
        targets6 = torch.cat([q_t, grip], dim=-1)
        state6 = robot.data.joint_pos[0, all6].cpu().numpy()
        recorder.add_frame(state6, targets6[0].cpu().numpy(), rgb)
        robot.set_joint_position_target(targets6, joint_ids=all6)
        robot.write_data_to_sim(); sim.step(); robot.update(sim_dt); target_obj.update(sim_dt)
    recorder.end_episode()
    if (ep + 1) % 5 == 0:
        print(f"[COLLECT] episode {ep + 1}/{args_cli.num_episodes} ({color})")

recorder.close()
print(f"\n[COLLECT] done → {args_cli.out}")
print("[COLLECT] next: generate stats:")
print(f"  python -m gr00t.data.stats --dataset-path {args_cli.out} "
      f"--embodiment-tag NEW_EMBODIMENT "
      f"--modality-config-path scripts/vla/finetune/g1_tabletop_config.py")
print("[COLLECT] DONE — closing the simulator and exiting.")
simulation_app.close()
sys.exit(0)
