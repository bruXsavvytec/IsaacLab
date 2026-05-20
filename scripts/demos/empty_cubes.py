import argparse

import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Empty Isaac Lab environment with optional objects.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


def design_scene(sim: SimulationContext) -> torch.Tensor:
    """Designs the scene. Returns spawn origins."""

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Dome light
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # --- Optional: spawn a static rigid cube as a scene object ---
    # Uncomment below to add a box to the scene
    #
    # cfg = sim_utils.CuboidCfg(
    #     size=(0.5, 0.5, 0.5),
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #     collision_props=sim_utils.CollisionPropertiesCfg(),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
    # )
    # cfg.func("/World/Box", cfg, translation=(1.0, 0.0, 0.25))

    # --- Optional: spawn a USD asset from file ---
    #
    # cfg = sim_utils.UsdFileCfg(usd_path="path/to/your/asset.usd")
    # cfg.func("/World/MyAsset", cfg, translation=(0.0, 2.0, 0.0))

    # Define a single origin at world centre
    origins = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    return origins


def run_simulator(sim: SimulationContext, origins: torch.Tensor):
    """Minimal simulation loop — just steps physics, nothing else."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print(f"[INFO]: Running simulation. Physics dt = {sim_dt:.4f} s")

    while simulation_app.is_running():
        # --- Optional periodic reset hook ---
        if count % 500 == 0:
            sim_time = 0.0
            count = 0
            print(f"[INFO]: Reset at step {count}. Origins:\n{origins}")

        # Step the physics
        sim.step()

        sim_time += sim_dt
        count += 1


def main():
    """Main function."""
    # Build simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,           # 200 Hz physics
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)

    # Optional: set a default camera view
    sim.set_camera_view(eye=[3.0, 3.0, 2.5], target=[0.0, 0.0, 0.5])

    # Build the scene
    origins = design_scene(sim)

    # Start the simulation (loads all prims into PhysX)
    sim.reset()
    print("[INFO]: Setup complete — empty environment is live.")

    # Hand off to the run loop
    run_simulator(sim, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()