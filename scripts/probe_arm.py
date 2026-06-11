"""Probe G1 left-arm kinematics: fix the base at the pick spot, sweep arm joints,
print the resulting hand (left_two_link) world position. One launch, many poses.

Run: /home/trooperai/isaac-env/bin/python -u scripts/probe_arm.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip

# Base pinned where the real robot stands at the shelf; cubes are at x=4.35, z~0.79
BASE = (4.05, 0.0, 0.74)
CUBE_X, CUBE_Z = 4.35, 0.79


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=400.0)
    light.func("/World/Light", light)

    cfg = G1_MINIMAL_CFG.copy()
    cfg.prim_path = "/World/G1"
    cfg.init_state.pos = BASE
    cfg.spawn.articulation_props.fix_root_link = True
    robot = Articulation(cfg=cfg)

    sim.reset()
    n2i = {n: i for i, n in enumerate(robot.joint_names)}
    hand_idx = next(i for i, b in enumerate(robot.body_names) if "left_two" in b)
    default = robot.data.default_joint_pos.clone()

    # Report joint limits for the left-arm joints we care about
    lims = robot.data.soft_joint_pos_limits[0]
    print("\n[LIMITS] left-arm joints:")
    for j in ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
              "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_elbow_roll_joint"]:
        if j in n2i:
            lo, hi = lims[n2i[j]].tolist()
            print(f"   {j:32s} [{lo:+.2f}, {hi:+.2f}]")

    def settle_and_read(pose_dict, frames=120):
        tgt = default.clone()
        for j, v in pose_dict.items():
            if j in n2i:
                tgt[0, n2i[j]] = v
        for _ in range(frames):
            robot.set_joint_position_target(tgt)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
        h = robot.data.body_pos_w[0, hand_idx]
        return (h[0].item(), h[1].item(), h[2].item())

    def show(label, pose):
        x, y, z = settle_and_read(pose)
        # relative to base, plus gap to a cube straight ahead (y=0)
        print(f"{label:42s} hand=({x:.2f},{y:.2f},{z:.2f})  "
              f"rel=({x-BASE[0]:+.2f},{y-BASE[1]:+.2f},{z-BASE[2]:+.2f})  "
              f"dx_to_shelf={CUBE_X-x:+.2f} dz={CUBE_Z-z:+.2f}")

    print(f"\n[BASE] {BASE}   target cube x={CUBE_X}, z={CUBE_Z}\n")
    print("=== shoulder_pitch sweep (roll=0, elbow=0.3) ===")
    for p in [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6]:
        show(f"pitch={p:+.1f}", {"left_shoulder_pitch_joint": p,
                                 "left_shoulder_roll_joint": 0.0,
                                 "left_elbow_pitch_joint": 0.3})

    print("\n=== elbow_pitch sweep (pitch=best-guess, roll=0) ===")
    for e in [-0.5, 0.0, 0.4, 0.8, 1.2, 1.6]:
        show(f"elbow={e:+.1f}", {"left_shoulder_pitch_joint": 0.0,
                                 "left_shoulder_roll_joint": 0.0,
                                 "left_elbow_pitch_joint": e})

    print("\n=== shoulder_roll sweep (pitch=0, elbow=0.5) ===")
    for r in [-0.6, -0.3, 0.0, 0.3, 0.6, 1.0]:
        show(f"roll={r:+.1f}", {"left_shoulder_pitch_joint": 0.0,
                                "left_shoulder_roll_joint": r,
                                "left_elbow_pitch_joint": 0.5})

    print("\n[PROBE DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
