"""
Visual showcase of all locally available USD plant/rock assets.

Spawns every asset in a grid so you can orbit around, pick what looks good,
then copy its path into greenhouse_sim.py or g1_locomotion.py.

Run (GUI required — you need to see the viewport):
    ./isaaclab.sh -p scripts/tools/asset_showcase.py

Controls in the viewport:
    Middle-mouse drag  — orbit
    Scroll             — zoom
    Right-mouse drag   — pan

Asset positions are printed to the terminal with their prim paths.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Showcase local USD assets in a grid.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

_REPO = "/home/trooperai/dev-bru/Nvidia-Isaac-Sim-Procedual-Forest-Generator/models"
_POT  = "/home/trooperai/isaac-env/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physx.internal/data/usd/assets/pot_plant.usda"

# ---------------------------------------------------------------------------
# Asset catalogue  (label, usd_path, scale, z_offset)
# scale:    uniform scale — adjust per asset after first look
# z_offset: lift if asset origin is underground
# ---------------------------------------------------------------------------
_ASSETS = [
    # ── Plants (small, greenhouse-scale) ──────────────────────────────────
    ("Bush",         f"{_REPO}/Bush_obj/Bush.usd",                                   1.0,  0.0),
    ("Blueberry",    f"{_REPO}/Blueberry_obj/Blueberry.usd",                         5.0,  0.0),
    #("Spruce",       f"{_REPO}/Spruce_obj/Spruce.usd",                               0.15, 0.0),
    #("Maple",        f"{_REPO}/maple_obj/maple.usd",                                 0.15, 0.0),
    # ── Trees (larger) ────────────────────────────────────────────────────
    #("Birch",        f"{_REPO}/Birch_obj/Birch.usd",                                 0.15, 0.0),
    #("Pine",         f"{_REPO}/Pine_obj/Pine.usd",                                   0.15, 0.0),
    # ── Ground details ────────────────────────────────────────────────────
    #("ForestLeaves", f"{_REPO}/forest_leaves_02_4k.blend/forest_leaves_02_4k.usd",   1.0, 0.0),
    # ── Rocks ─────────────────────────────────────────────────────────────
    #("Rock",         f"{_REPO}/Rock_obj/Rock.usd",                                   0.15, 0.0),
    #("BigRock",      f"{_REPO}/Rock_obj/big_rock.usd",                               0.15, 0.0),
    # ── Bundled with Isaac Sim ─────────────────────────────────────────────
    #("PotPlant",     _POT,                                                            5.0,  0.0),
]

_COLS    = 5      # assets per row
_SPACING = 1.5    # metres between asset centres


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spawn_asset(label: str, usd_path: str, pos: tuple, scale: float) -> bool:
    if not os.path.isfile(usd_path):
        print(f"  [SKIP] Not found: {usd_path}")
        return False
    prim_path = f"/World/Showcase/{label}"
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=(scale, scale, scale))
    try:
        cfg.func(prim_path, cfg, translation=pos)
        print(f"  [OK]   {label:<20s}  pos=({pos[0]:+.1f}, {pos[1]:+.1f}, {pos[2]:+.1f})  scale={scale}")
        return True
    except Exception as e:
        print(f"  [ERR]  {label}: {e}")
        return False


def _add_label_marker(label: str, pos: tuple):
    """Create a named Xform at ground level — shows up in the stage tree panel."""
    sim_utils.create_prim(f"/World/Labels/{label}", "Xform",
                          translation=(pos[0], pos[1], 0.02))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    sim_cfg    = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, render=render_cfg)
    sim        = SimulationContext(sim_cfg)

    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    dome = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    dome.func("/World/DomeLight", dome)
    sun = sim_utils.DistantLightCfg(intensity=5000.0, color=(1.0, 0.98, 0.90))
    sun.func("/World/SunLight", sun, translation=(0, 0, 10),
             orientation=(0.906, 0.0, 0.423, 0.0))

    print("\n" + "=" * 60)
    print("  Asset Showcase — spawning assets")
    print("=" * 60)

    for idx, (label, usd_path, scale, z_off) in enumerate(_ASSETS):
        col = idx % _COLS
        row = idx // _COLS
        pos = (col * _SPACING, row * _SPACING, z_off)
        _spawn_asset(label, usd_path, pos, scale)
        _add_label_marker(label, pos)

    n_rows = (len(_ASSETS) - 1) // _COLS + 1
    cx = (_COLS  - 1) * _SPACING / 2.0
    cy = (n_rows - 1) * _SPACING / 2.0
    sim.set_camera_view(
        eye   =[cx - _SPACING, cy - _SPACING * 4.0, _SPACING * 3.0],
        target=[cx, cy, 1.0],
    )

    sim.reset()

    print("\n" + "=" * 60)
    print("  Viewport live — middle-mouse to orbit, scroll to zoom.")
    print("  Scale is 0.01 for repo assets — edit _ASSETS if too big/small.")
    print("  Ctrl+C in terminal to quit.")
    print("=" * 60)
    print(f"\n{'Label':<22} {'Prim path':<36} {'File'}")
    print("-" * 90)
    for idx, (label, usd_path, scale, _) in enumerate(_ASSETS):
        col = idx % _COLS
        row = idx // _COLS
        print(f"{label:<22} /World/Showcase/{label:<20} {os.path.basename(usd_path)}  (scale={scale})")
    print()

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
