"""
Browse NVIDIA Nucleus asset directories from the command line.

Usage:
    # List vegetation assets
    ./isaaclab.sh -p scripts/tools/browse_nucleus.py

    # List a custom path
    ./isaaclab.sh -p scripts/tools/browse_nucleus.py --path "NVIDIA/Assets/Vegetation"

    # Recursively list all .usd files under a path (use carefully — can be slow)
    ./isaaclab.sh -p scripts/tools/browse_nucleus.py --path "NVIDIA/Assets/Vegetation/Plants" --recursive
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Browse NVIDIA Nucleus directories.")
parser.add_argument(
    "--path",
    default="NVIDIA/Assets/Vegetation",
    help="Sub-path under the Nucleus root to list (default: NVIDIA/Assets/Vegetation)",
)
parser.add_argument(
    "--recursive",
    action="store_true",
    help="Recursively list all .usd files under the path",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Headless is fine — we don't need rendering
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.client
import carb

NUCLEUS_ROOT = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")


def list_dir(url: str, indent: int = 0) -> None:
    """List entries in a Nucleus directory."""
    result, entries = omni.client.list(url)
    if result != omni.client.Result.OK:
        print(f"{'  ' * indent}[ERROR] Cannot list '{url}' — result: {result}")
        print(f"{'  ' * indent}  Make sure Isaac Sim can reach Nucleus (check your connection).")
        return

    dirs  = []
    files = []
    for e in entries:
        full = f"{url}/{e.relative_path}"
        if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            dirs.append((e.relative_path, full))
        else:
            files.append((e.relative_path, full))

    for name, full in sorted(dirs):
        print(f"{'  ' * indent}📁 {name}/")
        if args_cli.recursive:
            list_dir(full, indent + 1)

    for name, full in sorted(files):
        if name.endswith(".usd") or name.endswith(".usda") or name.endswith(".usdc"):
            print(f"{'  ' * indent}  {name}")
            if not args_cli.recursive:
                print(f"{'  ' * indent}    → {full}")


def main():
    target = f"{NUCLEUS_ROOT}/{args_cli.path.strip('/')}"
    print(f"\nNucleus root : {NUCLEUS_ROOT}")
    print(f"Browsing     : {target}")
    print(f"Recursive    : {args_cli.recursive}")
    print("-" * 60)

    if not NUCLEUS_ROOT or NUCLEUS_ROOT == "":
        print("[ERROR] Nucleus root path is empty.")
        print("Isaac Sim is not connected to a Nucleus server.")
        print("Try: omniverse://localhost  or check your Isaac Sim connection settings.")
        return

    list_dir(target)
    print("-" * 60)
    print("\nTip: copy any path above and use it as:")
    print('  sim_utils.UsdFileCfg(usd_path="<full path here>")')


if __name__ == "__main__":
    main()
    simulation_app.close()
