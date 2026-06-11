"""
Measure the G1 LEFT-arm reachable workspace on the table plane.

Anchors the G1 (fixed base) and sweeps the 5 arm joints through a coarse grid,
settling at each config and reading the hand's world position. Reports:
  - left shoulder world position
  - the reachable footprint (bounding box of hand x,y) at table height
  - a suggested cube-placement band (_RX, _RY) to paste into collect_demos.py

No recording, no cubes — just geometry. Run:
    cd /home/trooperai/IsaacLab
    ./isaaclab.sh -p scripts/vla/finetune/measure_reach.py --headless
"""

import argparse
import itertools
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure G1 left-arm reach")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip

sys.path.insert(0, os.path.dirname(__file__))
from g1_tabletop_config import SINGLE_ARM_JOINTS

_TABLE_TOP_Z = 0.74
_EE_CANDIDATES = ["left_palm_link", "left_hand_link", "left_zero_link",
                  "left_rubber_hand", "left_two_link", "left_one_link",
                  "left_wrist_yaw_link", "left_elbow_roll_link"]

sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
sim_utils.DistantLightCfg(intensity=3000.0).func("/World/L", sim_utils.DistantLightCfg(intensity=3000.0))

robot_cfg = G1_MINIMAL_CFG.copy()
robot_cfg.prim_path = "/World/G1"
robot_cfg.init_state.pos = (0.0, 0.0, 0.74)
robot_cfg.spawn.articulation_props.fix_root_link = True
robot = Articulation(cfg=robot_cfg)
sim.reset()
dt = sim.get_physics_dt()

jn = robot.joint_names
arm_ids = [jn.index(n) for n in SINGLE_ARM_JOINTS]
ee_name = next((b for b in _EE_CANDIDATES if b in robot.body_names), None)
if ee_name is None:
    ee_name = [b for b in robot.body_names if b.startswith("left")][-1]
ee_id = robot.body_names.index(ee_name)
sh_name = next((b for b in robot.body_names if "left_shoulder_pitch" in b), None) \
          or [b for b in robot.body_names if b.startswith("left_shoulder")][0]
sh_id = robot.body_names.index(sh_name)
print(f"[REACH] EE body = {ee_name} | shoulder body = {sh_name}")
print(f"[REACH] left-arm body names: {[b for b in robot.body_names if b.startswith('left')]}")

default = robot.data.default_joint_pos.clone()

# Sweep grid (radians): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll
GRID = {
    "pitch": [0.4, 0.9, 1.3, 1.6],
    "roll":  [-0.1, 0.2, 0.5],
    "yaw":   [-0.3, 0.0, 0.3],
    "elbow_pitch": [0.2, 0.7, 1.2],
    "elbow_roll":  [0.0],
}

hand_pts = []
shoulder_xyz = None
for p, r, y, ep, er in itertools.product(GRID["pitch"], GRID["roll"], GRID["yaw"],
                                          GRID["elbow_pitch"], GRID["elbow_roll"]):
    q = default.clone()
    for idx, val in zip(arm_ids, [p, r, y, ep, er]):
        q[0, idx] = val
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    for _ in range(12):
        robot.set_joint_position_target(q)
        robot.write_data_to_sim(); sim.step(); robot.update(dt)
    h = robot.data.body_pos_w[0, ee_id].cpu().numpy()
    hand_pts.append(h)
    if shoulder_xyz is None:
        shoulder_xyz = robot.data.body_pos_w[0, sh_id].cpu().numpy()

hand_pts = np.array(hand_pts)
print(f"\n[REACH] shoulder world (x,y,z) = {np.round(shoulder_xyz, 3)}")
print(f"[REACH] hand reached {len(hand_pts)} poses")
print(f"[REACH] hand x range [{hand_pts[:,0].min():.2f}, {hand_pts[:,0].max():.2f}]"
      f"  y [{hand_pts[:,1].min():.2f}, {hand_pts[:,1].max():.2f}]"
      f"  z [{hand_pts[:,2].min():.2f}, {hand_pts[:,2].max():.2f}]")
L = np.linalg.norm(hand_pts - shoulder_xyz, axis=1).max()
print(f"[REACH] max |hand - shoulder| (arm length) = {L:.3f} m")

# Footprint near table height (z within ±8 cm of the table top).
near = hand_pts[np.abs(hand_pts[:, 2] - _TABLE_TOP_Z) < 0.08]
if len(near) >= 4:
    # Suggest a band slightly inside the reachable footprint, in front of the robot.
    rx = (max(0.30, float(near[:,0].min()) + 0.04), float(near[:,0].max()) - 0.04)
    ry = (max(0.0,  float(near[:,1].min()) + 0.02), float(near[:,1].max()) - 0.02)
    print(f"\n[REACH] {len(near)} poses land near table height ({_TABLE_TOP_Z} m).")
    print(f"[REACH] reachable table footprint  x[{near[:,0].min():.2f},{near[:,0].max():.2f}] "
          f"y[{near[:,1].min():.2f},{near[:,1].max():.2f}]")
    print(f"\n[REACH] >>> SUGGESTED in collect_demos.py:")
    print(f"        _RX = ({rx[0]:.2f}, {rx[1]:.2f})")
    print(f"        _RY = ({ry[0]:.2f}, {ry[1]:.2f})")
else:
    print(f"\n[REACH] WARNING: only {len(near)} poses near table height — the hand may not "
          f"reach the table at z={_TABLE_TOP_Z}. Consider raising the table or lowering the robot.")

print("[REACH] DONE — closing the simulator and exiting.")
simulation_app.close()
sys.exit(0)
