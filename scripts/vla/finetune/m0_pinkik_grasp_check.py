"""
M0 — Trustworthy-solver check for the fixed-base G1 Pink IK env.

Loads the registered `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` environment
and drives ONE wrist through a sequence of Cartesian waypoints using the Pink IK
action term, measuring how accurately the achieved end-effector pose tracks the
commanded target. This is the evidence that the G1 can be moved autonomously and
*accurately* by a real solver (Pink IK / Pinocchio), not by hardcoded joint angles.

Optionally (--grasp) it then descends onto the scene object, closes the active
hand, and lifts — reporting whether the object rose with the hand (i.e. whether a
*physical* PhysX grasp held, no kinematic attach).

Action layout for this env (2 wrist FrameTasks + 14 hand joints) = 28 dims:
    [ 0: 3] left_wrist  target position   (x, y, z)        env-origin frame
    [ 3: 7] left_wrist  target orientation (w, x, y, z)
    [ 7:10] right_wrist target position
    [10:14] right_wrist target orientation
    [14:28] hand joint position targets, in cfg order:
            L_index0, L_mid0, L_thumb0, R_index0, R_mid0, R_thumb0,
            L_index1, L_mid1, L_thumb1, R_index1, R_mid1, R_thumb1,
            L_thumb2, R_thumb2

Run (headless, no cameras needed for the tracking check):
    cd /home/trooperai/IsaacLab
    ./isaaclab.sh -p scripts/vla/finetune/m0_pinkik_grasp_check.py --headless
    ./isaaclab.sh -p scripts/vla/finetune/m0_pinkik_grasp_check.py --headless --grasp --arm left
"""

import argparse

# pinocchio MUST be imported before AppLauncher so the IsaacLab-installed version
# wins over the one Isaac Sim ships (required by the Pink IK controller).
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="M0 Pink IK tracking / physical-grasp check")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
parser.add_argument("--arm", type=str, default="left", choices=["left", "right"],
                    help="Which wrist to drive; the other is held at its reset pose.")
parser.add_argument("--grasp", action="store_true", help="After tracking, descend, close hand, and lift.")
parser.add_argument("--hand-close", type=float, default=0.8, help="Closed hand-joint target (rad).")
parser.add_argument("--steps-per-leg", type=int, default=60, help="Control steps to interpolate each waypoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── after AppLauncher ────────────────────────────────────────────────────────
import gymnasium as gym
import torch

# Registers Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Index of each LEFT / RIGHT hand joint within the 14-dim hand block (cfg order).
_LEFT_HAND_SLOTS = [0, 1, 2, 6, 7, 8, 12]
_RIGHT_HAND_SLOTS = [3, 4, 5, 9, 10, 11, 13]


def _obs_term(obs, name):
    """Fetch one observation term; the policy group is a dict (concatenate_terms=False)."""
    policy = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
    return policy[name]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = env.device

    action_dim = env.action_manager.total_action_dim
    print(f"[M0] task={args_cli.task}  action_dim={action_dim}  device={device}")
    assert action_dim == 28, f"expected 28-dim action, got {action_dim}"

    obs, _ = env.reset()

    # Reset poses for both wrists (env-origin frame). Hold the inactive wrist here.
    l_pos0 = _obs_term(obs, "left_eef_pos")[0].clone()
    l_quat0 = _obs_term(obs, "left_eef_quat")[0].clone()
    r_pos0 = _obs_term(obs, "right_eef_pos")[0].clone()
    r_quat0 = _obs_term(obs, "right_eef_quat")[0].clone()
    object_pos0 = _obs_term(obs, "object")[0, :3].clone()

    active_pos0 = l_pos0 if args_cli.arm == "left" else r_pos0
    print(f"[M0] {args_cli.arm} wrist reset pos = {active_pos0.tolist()}")
    print(f"[M0] object pos             = {object_pos0.tolist()}")

    # Cartesian waypoints for the active wrist: a small box around its start pose
    # (pure translation; orientation held at reset quat) to measure tracking.
    deltas = [
        torch.tensor([0.00, 0.00, 0.10], device=device),   # up
        torch.tensor([0.10, 0.00, 0.10], device=device),   # up + forward
        torch.tensor([0.10, 0.10, 0.00], device=device),   # forward + side
        torch.tensor([0.00, 0.00, 0.00], device=device),   # back to start
    ]
    waypoints = [active_pos0 + d for d in deltas]

    def build_action(active_pos, hand_vec):
        a = torch.zeros(1, 28, device=device)
        if args_cli.arm == "left":
            a[0, 0:3] = active_pos
            a[0, 3:7] = l_quat0
            a[0, 7:10] = r_pos0
            a[0, 10:14] = r_quat0
        else:
            a[0, 0:3] = l_pos0
            a[0, 3:7] = l_quat0
            a[0, 7:10] = active_pos
            a[0, 10:14] = r_quat0
        a[0, 14:28] = hand_vec
        return a

    hand_open = torch.zeros(14, device=device)
    cur = active_pos0.clone()
    errors = []

    # ── Cartesian tracking sweep ─────────────────────────────────────────────
    for wp_i, wp in enumerate(waypoints):
        start = cur.clone()
        for s in range(args_cli.steps_per_leg):
            alpha = (s + 1) / args_cli.steps_per_leg
            target = torch.lerp(start, wp, alpha)
            obs, _, _, _, _ = env.step(build_action(target, hand_open))
        cur = wp.clone()
        achieved = _obs_term(obs, f"{args_cli.arm}_eef_pos")[0]
        err = torch.norm(achieved - wp).item()
        errors.append(err)
        print(f"[M0] waypoint {wp_i}: target={wp.tolist()}  achieved={achieved.tolist()}  err={err*1000:.1f} mm")

    print(f"[M0] tracking error  mean={1000*sum(errors)/len(errors):.1f} mm  max={1000*max(errors):.1f} mm")

    # ── optional physical grasp + lift ───────────────────────────────────────
    if args_cli.grasp:
        hand_slots = _LEFT_HAND_SLOTS if args_cli.arm == "left" else _RIGHT_HAND_SLOTS
        hand_closed = torch.zeros(14, device=device)
        for slot in hand_slots:
            hand_closed[slot] = args_cli.hand_close

        above = object_pos0 + torch.tensor([0.0, 0.0, 0.12], device=device)
        at = object_pos0 + torch.tensor([0.0, 0.0, 0.02], device=device)

        def sweep(p_from, p_to, hand_vec, n):
            nonlocal obs
            for s in range(n):
                alpha = (s + 1) / n
                env.step(build_action(torch.lerp(p_from, p_to, alpha), hand_vec))
            obs = _last_obs(env)

        print("[M0] grasp: approach above object")
        sweep(cur, above, hand_open, args_cli.steps_per_leg)
        print("[M0] grasp: descend onto object")
        sweep(above, at, hand_open, args_cli.steps_per_leg)
        print("[M0] grasp: close hand")
        for _ in range(args_cli.steps_per_leg):
            obs, _, _, _, _ = env.step(build_action(at, hand_closed))
        obj_before = _obs_term(obs, "object")[0, 2].item()
        print("[M0] grasp: lift")
        sweep(at, above, hand_closed, args_cli.steps_per_leg)
        obj_after = _obs_term(obs, "object")[0, 2].item()
        rise = obj_after - obj_before
        print(f"[M0] object z before lift={obj_before:.3f}  after={obj_after:.3f}  rise={rise*1000:.0f} mm")
        print(f"[M0] PHYSICAL GRASP {'HELD' if rise > 0.04 else 'FAILED'} "
              f"(threshold 40 mm rise)")

    env.close()


def _last_obs(env):
    """Re-read current observations without stepping (compute the obs manager)."""
    return env.observation_manager.compute()


if __name__ == "__main__":
    main()
    simulation_app.close()
