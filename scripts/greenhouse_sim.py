"""
Greenhouse inspection demo — IsaacLab 2.x / Isaac Sim 4.x

Layout:
  - Greenhouse at (5, 0, 0) — 8×5×3 m, glass walls, peaked roof
  - 2 rows of collidable bush clusters flanking a central walkway
  - Unitree G1 humanoid walks the aisle kinematically, inspecting each column

Robot motion:
  - Kinematic root control: root pose + zero velocity written every frame
  - Joints held in default standing pose throughout
  - State machine: MOVE → INSPECT → MOVE → ... → DONE

Run from the IsaacLab root:
    ./isaaclab.sh -p scripts/greenhouse_sim.py
    ./isaaclab.sh -p scripts/greenhouse_sim.py --headless
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Greenhouse inspection demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# All non-stdlib imports must come after AppLauncher
import torch  # noqa: E402

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.unitree import G1_CFG  # isort:skip


# ---------------------------------------------------------------------------
# Greenhouse + inspection layout constants
# ---------------------------------------------------------------------------

_GH_CX, _GH_CY = 5.0, 0.0          # greenhouse center (x, y)
_ROW_Y_OFFSETS  = [-1.5, 1.5]       # two plant rows flanking the central walkway
_N_PLANTS       = 6                  # plants per row
_PLANT_SPACING  = 1.0               # meters between plants along X
_BUSH_RADIUS    = 0.22              # base radius of each bush sphere (m)

# Derived plant X positions (centred on greenhouse)
_HALF_SPAN  = (_N_PLANTS - 1) * _PLANT_SPACING / 2.0
_PLANT_XS   = [_GH_CX - _HALF_SPAN + i * _PLANT_SPACING for i in range(_N_PLANTS)]

# Robot path: entrance → each plant column → back to entrance
_ENTRANCE   = (_GH_CX - _HALF_SPAN - 1.2, _GH_CY, 0.74)
_WAYPOINTS  = [(x, _GH_CY, 0.74) for x in _PLANT_XS] + [_ENTRANCE]

# Inspection parameters
MOVE_SPEED      = 0.6   # m/s
INSPECT_FRAMES  = 120   # physics frames to pause at each column (~1.2 s at dt=0.01)
ARRIVE_THRESH   = 0.05  # metres — "close enough" to waypoint


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _mat(rgb: tuple, opacity: float = 1.0) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.6, opacity=opacity)


def _spawn_box(path, size, pos, rgb, opacity=1.0, orient=None):
    cfg = sim_utils.CuboidCfg(size=size, visual_material=_mat(rgb, opacity))
    kw = {"translation": pos}
    if orient is not None:
        kw["orientation"] = orient
    cfg.func(path, cfg, **kw)


def _spawn_sphere(path, radius, pos, rgb, collidable=False):
    """Sphere prim. collidable=True → kinematic rigid body (solid to collision)."""
    cfg = sim_utils.SphereCfg(
        radius=radius,
        visual_material=_mat(rgb),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True) if collidable else None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if collidable else None,
    )
    cfg.func(path, cfg, translation=pos)


def _x_rot_quat(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), math.sin(h), 0.0, 0.0)


# ---------------------------------------------------------------------------
# Greenhouse structure
# ---------------------------------------------------------------------------

def build_greenhouse(ox=_GH_CX, oy=_GH_CY, oz=0.0):
    W, D, H = 8.0, 5.0, 3.0
    t     = 0.08
    PITCH = 25.0
    rise  = (D / 2) * math.tan(math.radians(PITCH))
    pdep  = (D / 2) / math.cos(math.radians(PITCH))

    GLASS = (0.72, 0.94, 0.84)
    FRAME = (0.50, 0.44, 0.33)
    FLOOR = (0.80, 0.72, 0.55)

    _spawn_box("/World/Greenhouse/Floor",     (W, D, 0.12), (ox, oy, oz+0.06), FLOOR)
    wall_z = oz + H / 2
    _spawn_box("/World/Greenhouse/WallFront", (W, t, H), (ox,      oy-D/2, wall_z), GLASS, opacity=0.35)
    _spawn_box("/World/Greenhouse/WallBack",  (W, t, H), (ox,      oy+D/2, wall_z), GLASS, opacity=0.35)
    _spawn_box("/World/Greenhouse/WallLeft",  (t, D, H), (ox-W/2,  oy,     wall_z), GLASS, opacity=0.35)
    _spawn_box("/World/Greenhouse/WallRight", (t, D, H), (ox+W/2,  oy,     wall_z), GLASS, opacity=0.35)
    for i, (dx, dy) in enumerate([(-W/2,-D/2),(-W/2,D/2),(W/2,-D/2),(W/2,D/2)]):
        _spawn_box(f"/World/Greenhouse/Pillar{i}", (0.12, 0.12, H), (ox+dx, oy+dy, wall_z), FRAME)
    roof_z = oz + H + rise / 2
    _spawn_box("/World/Greenhouse/RoofFront", (W, pdep, t), (ox, oy-D/4, roof_z), GLASS, opacity=0.35, orient=_x_rot_quat(+PITCH))
    _spawn_box("/World/Greenhouse/RoofBack",  (W, pdep, t), (ox, oy+D/4, roof_z), GLASS, opacity=0.35, orient=_x_rot_quat(-PITCH))
    _spawn_box("/World/Greenhouse/Ridge",     (W, 0.15, 0.15), (ox, oy, oz+H+rise), FRAME)


# ---------------------------------------------------------------------------
# Bush clusters — collidable, grid layout
# ---------------------------------------------------------------------------

_BUSH_OFFSETS = [
    ( 0.00,  0.00, 0.00, 1.00),
    ( 0.22,  0.12, 0.20, 0.78),
    (-0.18, -0.10, 0.24, 0.72),
    ( 0.05, -0.22, 0.28, 0.65),
]


def build_greenhouse_bushes() -> dict:
    """Spawn two collidable rows of bushes and return a state dict per plant.

    State dict format:
        "R1P3": {"row": 1, "col": 3, "pos": (x, y), "healthy": bool, "inspected": bool}
    """
    states = {}
    for row_idx, dy in enumerate(_ROW_Y_OFFSETS):
        row_y = _GH_CY + dy
        for col_idx, x in enumerate(_PLANT_XS):
            bid  = f"R{row_idx}P{col_idx}"
            base = f"/World/Greenhouse/Plants/{bid}"
            r    = _BUSH_RADIUS
            cz   = r * 0.7  # partially embed in ground
            for j, (dx, dy_off, dz_fac, rs) in enumerate(_BUSH_OFFSETS):
                g   = 0.45
                rgb = (g * 0.25, g, g * 0.18)
                _spawn_sphere(f"{base}/s{j}", r * rs,
                              (x + dx, row_y + dy_off, cz + dz_fac * r),
                              rgb, collidable=True)
            states[bid] = {"row": row_idx, "col": col_idx,
                           "pos": (x, row_y), "healthy": True, "inspected": False}
    return states


# ---------------------------------------------------------------------------
# Full scene
# ---------------------------------------------------------------------------

def design_scene() -> dict:
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    sun = sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.98, 0.90))
    sun.func("/World/SunLight", sun, translation=(0, 0, 10),
             orientation=(0.906, 0.0, 0.423, 0.0))
    dome = sim_utils.DomeLightCfg(intensity=600.0, color=(0.55, 0.72, 1.0))
    dome.func("/World/DomeLight", dome)

    build_greenhouse()
    bush_states = build_greenhouse_bushes()

    # Robot: no fixed root — we drive root pose kinematically every frame
    robot_cfg = G1_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.init_state.pos = _ENTRANCE  # start at walkway entrance

    robot = Articulation(cfg=robot_cfg)
    return {"robot": robot, "bush_states": bush_states}


# ---------------------------------------------------------------------------
# Inspection state machine
# ---------------------------------------------------------------------------

def run_simulator(sim: SimulationContext, robot: Articulation,
                  bush_states: dict, sim_dt: float):
    # Mutable robot position tracked in Python
    cur = list(_ENTRANCE)   # [x, y, z]
    wp_idx         = 0
    phase          = "move"
    inspect_count  = 0

    default_jpos = robot.data.default_joint_pos.clone()
    default_jvel = robot.data.default_joint_vel.clone()

    print(f"\n[INFO] Starting inspection of {_N_PLANTS * len(_ROW_Y_OFFSETS)} plants "
          f"({len(_ROW_Y_OFFSETS)} rows × {_N_PLANTS} columns).")
    print(f"[INFO] Waypoints: {len(_WAYPOINTS)} (last = return to entrance)\n")

    while simulation_app.is_running():
        tx, ty, tz = _WAYPOINTS[wp_idx]

        # --- state machine ---
        if phase == "move":
            dx   = tx - cur[0]
            dy   = ty - cur[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < ARRIVE_THRESH:
                is_last = (wp_idx == len(_WAYPOINTS) - 1)
                if is_last:
                    phase = "done"
                    print("[DONE] Inspection complete — robot back at entrance.")
                    _print_summary(bush_states)
                else:
                    phase         = "inspect"
                    inspect_count = 0
                    print(f"[ARRIVE] Column {wp_idx} — inspecting plants "
                          f"R0P{wp_idx} and R1P{wp_idx} ...")
            else:
                step    = min(MOVE_SPEED * sim_dt, dist)
                cur[0] += dx / dist * step
                cur[1] += dy / dist * step

        elif phase == "inspect":
            inspect_count += 1
            if inspect_count >= INSPECT_FRAMES:
                col = wp_idx
                for row_idx in range(len(_ROW_Y_OFFSETS)):
                    bid = f"R{row_idx}P{col}"
                    if bid in bush_states:
                        bush_states[bid]["inspected"] = True
                        tag = "OK" if bush_states[bid]["healthy"] else "SICK"
                        px, py = bush_states[bid]["pos"]
                        print(f"  [{tag}] {bid}  pos=({px:.1f}, {py:.1f})")
                wp_idx += 1
                phase   = "move"

        # phase == "done": robot just stands at entrance, loop continues

        # --- kinematic root control ---
        # Write root pose + zero velocity every frame so physics never takes over
        pose = torch.tensor(
            [[cur[0], cur[1], cur[2], 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32, device=sim.device,
        )
        vel = torch.zeros(1, 6, dtype=torch.float32, device=sim.device)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)
        robot.write_joint_state_to_sim(default_jpos, default_jvel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


def _print_summary(bush_states: dict):
    print("\n--- Inspection summary ---")
    for bid, s in sorted(bush_states.items()):
        tag = "OK  " if s["healthy"] else "SICK"
        chk = "✓" if s["inspected"] else " "
        px, py = s["pos"]
        print(f"  [{chk}] {bid}  {tag}  pos=({px:.1f}, {py:.1f})")
    print("--------------------------\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim     = SimulationContext(sim_cfg)

    # Camera outside front wall, looking down the central aisle
    sim.set_camera_view(
        eye=[_GH_CX - 6.5, _GH_CY, 2.5],
        target=[_GH_CX, _GH_CY, 1.0],
    )

    entities    = design_scene()
    robot       = entities["robot"]
    bush_states = entities["bush_states"]

    sim.reset()
    print(f"[INFO] Scene ready — {len(bush_states)} plants, "
          f"{_N_PLANTS} per row, row spacing {abs(_ROW_Y_OFFSETS[1]-_ROW_Y_OFFSETS[0]):.1f} m")

    run_simulator(sim, robot, bush_states, sim_cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
