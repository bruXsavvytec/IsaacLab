"""
Interactive locomotion demo for the Unitree G1 humanoid.

Modeled on scripts/demos/h1_locomotion.py — same architecture:
  - ManagerBasedRLEnv wraps the physics + observation/command pipeline
  - Pre-trained RSL-RL policy runs at inference (checkpoint auto-downloaded from Nucleus)
  - Arrow keys inject velocity commands into the observation; policy handles balance

Controls (click a robot in the viewport to select it first):
  UP    — walk forward
  LEFT  — turn left
  RIGHT — turn right
  DOWN  — stop
  C     — toggle third-person / free camera
  ESC   — deselect robot

Run from the IsaacLab root:
    ./isaaclab.sh -p scripts/demos/g1_locomotion.py

If the G1 checkpoint is not yet on Nucleus you will get a download error.
Train one first with:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0

--- Architecture note ---
G1 policy input is 310-dim, output is 37 joints:
  obs[0:3]   base_lin_vel       (3)
  obs[3:6]   base_ang_vel       (3)
  obs[6:9]   projected_gravity  (3)
  obs[9:12]  velocity_commands  (3) ← vx, vy, wz — this is what keyboard injects
  obs[12:49] joint_pos_rel      (37)
  obs[49:86] joint_vel_rel      (37)
  obs[86:123] last_action       (37)
  obs[123:]  height_scan        (187)

The H1 script uses obs[:,9:13] (4 values) — this accidentally clobbers joint_pos[0].
H1's 19-joint policy survives it. G1's 37-joint policy does not — hence the collapse.
Fixed here: inject exactly obs[:,9:12] with a 3-wide command vector.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive G1 locomotion demo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import torch.nn as nn

import carb
import omni
from pxr import Gf, Sdf

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg_PLAY

# Viewport utilities are GUI-only — not available in headless mode
_HEADLESS = getattr(args_cli, "headless", False)
if not _HEADLESS:
    from omni.kit.viewport.utility import get_viewport_from_window_name
    from omni.kit.viewport.utility.camera_state import ViewportCameraState

TASK       = "Isaac-Velocity-Rough-G1-v0"
RL_LIBRARY = "rsl_rl"

# Velocity command slice in the G1 observation vector (indices 9-11, 3 values)
_CMD_SLICE = slice(9, 12)

# G1 actor architecture (matches checkpoint: actor.0.weight [512,310] … actor.6.weight [37,128])
_OBS_DIM      = 310
_ACTION_DIM   = 37
_HIDDEN_DIMS  = [512, 256, 128]


def _load_actor_from_old_checkpoint(path: str, device: str) -> nn.Module:
    """Build a plain MLP actor and load weights from an old-format rsl_rl checkpoint.

    Old rsl_rl (< 4.0.0) saved a single ActorCritic with keys like:
        model_state_dict / actor.0.weight, actor.0.bias, actor.2.weight, …

    New rsl_rl (>= 4.0.0) expects actor_state_dict / critic_state_dict (different format).
    We bypass OnPolicyRunner entirely and load just the actor weights directly.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_sd = ckpt["model_state_dict"]

    # Build Sequential MLP matching the old ActorCritic actor branch
    layers: list[nn.Module] = []
    dims = [_OBS_DIM] + _HIDDEN_DIMS + [_ACTION_DIM]
    for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(_HIDDEN_DIMS):   # ELU after every hidden layer, not the output
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)

    # Strip the "actor." prefix and load
    actor_sd = {k[len("actor."):]: v for k, v in model_sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd)
    actor.eval()
    return actor.to(device)


class G1RoughDemo:
    """Interactive demo for the G1 rough-terrain locomotion task.

    Keyboard → 3-D velocity command → obs injection → policy → 37 joint targets → physics.
    Click a robot in the viewport to select it, then use arrow keys.
    """

    def __init__(self):
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
        if checkpoint is None:
            raise RuntimeError(
                f"No pre-trained checkpoint found for {TASK}.\n"
                "Train one with:\n"
                f"  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task {TASK}"
            )

        env_cfg = G1RoughEnvCfg_PLAY()
        env_cfg.scene.num_envs      = 1
        env_cfg.episode_length_s    = 1_000_000
        env_cfg.curriculum          = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.ranges.heading   = (-1.0, 1.0)
        # Spawn on the flattest terrain patch so the policy isn't immediately overwhelmed
        env_cfg.scene.terrain.max_init_terrain_level = 0
        # Minimal terrain grid for a single-env run (saves VRAM and load time)
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 2
            env_cfg.scene.terrain.terrain_generator.num_cols = 2

        self.env    = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        # Load actor weights directly — bypasses OnPolicyRunner which is incompatible
        # with the old-format (pre-4.0.0 rsl_rl) published checkpoints.
        self.policy = _load_actor_from_old_checkpoint(checkpoint, self.device)

        # 3-wide command: [vx, vy, wz] — matches velocity_commands obs slice exactly
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)

        self._camera_local_transform = torch.tensor([-2.0, 0.0, 0.7], device=self.device)
        self._selected_id            = None
        self._previous_selected_id   = None

        if not _HEADLESS:
            self.create_camera()
            self.set_up_keyboard()
            self._prim_selection = omni.usd.get_context().get_selection()
        else:
            print("[INFO] Headless mode — keyboard/camera controls disabled.")

    def create_camera(self):
        stage = get_current_stage()
        self.viewport         = get_viewport_from_window_name("Viewport")
        self.camera_path      = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        self._input    = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1.0   # forward speed command (m/s)
        R = 0.5   # yaw rate command (rad/s)
        # 3 values: [vx, vy, wz] — no heading dimension, so joint_pos[0] is never clobbered
        self._key_to_control = {
            "UP":    torch.tensor([ T,  0.0,  0.0], device=self.device),
            "DOWN":  torch.tensor([0.0, 0.0,  0.0], device=self.device),
            "LEFT":  torch.tensor([ T,  0.0, -R  ], device=self.device),
            "RIGHT": torch.tensor([ T,  0.0,  R  ], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0,  0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                if self._selected_id is not None:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    active = self.viewport.get_active_camera()
                    self.viewport.set_active_camera(
                        self.perspective_path if active == self.camera_path else self.camera_path
                    )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id is not None:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        if _HEADLESS:
            return
        self._previous_selected_id = self._selected_id
        selected_paths = self._prim_selection.get_selected_prim_paths()

        if len(selected_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_paths) > 1:
            print("Multiple prims selected — please select just one.")
        else:
            parts = selected_paths[0].split("/")
            if len(parts) >= 4 and parts[3][:4] == "env_":
                self._selected_id = int(parts[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("Selected prim is not a G1 robot.")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands = self.env.unwrapped.command_manager.get_command("base_velocity").clone()

    def _update_camera(self):
        base_pos  = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id]
        camera_pos   = quat_apply(base_quat, self._camera_local_transform) + base_pos
        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye    = Gf.Vec3d(*[v.item() for v in camera_pos])
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.5)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    demo = G1RoughDemo()
    obs_td, _ = demo.env.reset()   # TensorDict — policy obs is under key "policy"

    print("\n[INFO] G1 locomotion demo running.")
    if not _HEADLESS:
        print("[INFO] Click the robot in the viewport, then use arrow keys to move.")
    print()

    while simulation_app.is_running():
        demo.update_selected_object()
        with torch.inference_mode():
            # Extract flat [1, 310] tensor from TensorDict for our plain MLP actor
            obs_flat = obs_td["policy"]
            # Inject keyboard velocity command into the 3-value velocity_commands slot
            obs_flat[:, _CMD_SLICE] = demo.commands
            action = demo.policy(obs_flat)
            obs_td, _, _, _ = demo.env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
