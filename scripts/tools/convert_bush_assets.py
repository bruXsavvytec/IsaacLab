"""
Convert .obj plant models from the forest generator repo to USD with correct local paths.

Isaac Sim's asset converter re-resolves all texture paths relative to the output file,
so the resulting .usd will work on Linux regardless of the original Windows paths in the .mtl.

Run from the IsaacLab root (GUI required for the converter to initialise renderers):
    ./isaaclab.sh -p scripts/tools/convert_bush_assets.py

Output: writes <model_dir>/<Name>_converted.usd next to each original .usd.
        Copies textures into a textures/ sub-folder beside the output.

Usage notes:
  - Run once, then reference the output USDs from greenhouse_sim.py.
  - The converter is async-based inside Isaac Sim; the script waits for each job.
"""

import argparse
import asyncio
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert forest generator OBJ models to USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.asset_converter as converter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = "/home/trooperai/dev-bru/Nvidia-Isaac-Sim-Procedual-Forest-Generator/models"

_ASSETS = [
    ("Bush_obj",      "Bush.obj",      "Bush_local.usd"),
    ("Blueberry_obj", "Blueberry.obj", "Blueberry_local.usd"),
]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

async def _convert(src: str, dst: str) -> bool:
    """Convert src OBJ → dst USD using Isaac Sim's asset converter."""
    ctx = converter.AssetConverterContext()
    ctx.ignore_materials       = False
    ctx.ignore_animations      = True
    ctx.ignore_camera          = True
    ctx.ignore_light           = True
    ctx.single_mesh            = False
    ctx.smooth_normals         = True
    ctx.export_preview_surface = True   # OmniPBR materials, not raw UsdPreviewSurface
    ctx.use_meter_as_world_unit = True
    ctx.create_world_as_default_root_prim = False

    task = converter.get_instance().create_converter_task(src, dst, None, ctx)
    success = await task.wait_until_finished()
    if not success:
        print(f"  [ERROR] {task.get_error_message()}")
    return success


def main():
    for folder, src_name, dst_name in _ASSETS:
        src = os.path.join(_REPO, folder, src_name)
        dst = os.path.join(_REPO, folder, dst_name)

        if not os.path.isfile(src):
            print(f"[SKIP] Source not found: {src}")
            continue

        print(f"\n[CONVERT] {src_name} → {dst_name}")
        print(f"  src: {src}")
        print(f"  dst: {dst}")

        ok = asyncio.get_event_loop().run_until_complete(_convert(src, dst))
        if ok:
            print(f"  [OK] Written: {dst}")
        else:
            print(f"  [FAIL] See error above.")

    print("\n[DONE] Conversion complete.")
    print("\nUpdate greenhouse_sim.py to reference these paths:")
    for folder, _, dst_name in _ASSETS:
        dst = os.path.join(_REPO, folder, dst_name)
        print(f"  {dst}")


if __name__ == "__main__":
    main()
    simulation_app.close()
