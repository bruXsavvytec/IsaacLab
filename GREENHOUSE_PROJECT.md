# Greenhouse Inspection Robot — Project Documentation

> **Status:** Active development · Isaac Lab 2.3.2 · Isaac Sim 4.x · Python 3.10  
> **Robot:** Unitree G1 Humanoid  
> **Goal:** Autonomous greenhouse plant inspection with touch detection, vision (YOLO), and eventual locomotion policy

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Environment & Requirements](#2-environment--requirements)
3. [Repository Structure](#3-repository-structure)
4. [How to Run](#4-how-to-run)
5. [Task List](#5-task-list)
6. [What Has Been Built](#6-what-has-been-built)
7. [Architecture & State Machine](#7-architecture--state-machine)
8. [Key Technical Learnings & Gotchas](#8-key-technical-learnings--gotchas)
9. [Known Limitations](#9-known-limitations)
10. [Roadmap](#10-roadmap)
11. [Asset Catalogue](#11-asset-catalogue)

---

## 1. Project Goal

Build a simulation of a **humanoid robot (Unitree G1) inspecting plants inside a greenhouse** using NVIDIA Isaac Lab.

The full vision (in order of implementation):

```
Kinematic scripted walk
       ↓
Contact detection (is the robot touching a bush?)
       ↓
RGB camera + YOLO (can the robot see unhealthy plants?)
       ↓
Pre-trained locomotion policy (real walking, not kinematic glide)
       ↓
VLA (Vision-Language-Action model) for high-level task planning
```

The current milestone: **scripted single-bush interaction** — robot walks to one bush, reaches its arm inside, holds, retracts.

---

## 2. Environment & Requirements

### 2.1 Software Versions

| Component | Version |
|---|---|
| Isaac Lab | 2.3.2 |
| Isaac Sim | 4.x (bundled with Isaac Lab) |
| Python | 3.10.12 |
| PyTorch | bundled with Isaac Sim |
| USD / pxr | bundled with Isaac Sim |
| OS | Ubuntu 22.04 |

### 2.2 Installation Paths

| What | Path |
|---|---|
| IsaacLab root | `/home/trooperai/IsaacLab/` |
| Isaac Sim Python env | `/home/trooperai/isaac-env/lib/python3.10/` |
| 3D asset models repo | `/home/trooperai/dev-bru/Nvidia-Isaac-Sim-Procedual-Forest-Generator/models/` |
| IsaacLab launcher | `/home/trooperai/IsaacLab/isaaclab.sh` |

### 2.3 How Isaac Lab Scripts Are Run

**All scripts must be launched through the isaaclab.sh wrapper**, not plain Python:

```bash
# From the IsaacLab root directory
cd /home/trooperai/IsaacLab

./isaaclab.sh -p scripts/greenhouse_sim.py
./isaaclab.sh -p scripts/greenhouse_sim.py --enable_cameras
./isaaclab.sh -p scripts/greenhouse_sim.py --headless
```

> **Why:** `isaaclab.sh` sets up the Isaac Sim Python environment, Omniverse Carbonite settings, GPU rendering, and Nucleus connection. Running with plain `python3` will silently produce dummy modules and fail at simulation time.

### 2.4 Critical Import Rule

All `import isaaclab.*` and `import omni.*` statements **must come after** `AppLauncher` is instantiated:

```python
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)      # ← Isaac Sim boots here
simulation_app = app_launcher.app

# ONLY after this line:
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
# etc.
```

---

## 3. Repository Structure

```
IsaacLab/
├── scripts/
│   ├── greenhouse_sim.py          ← MAIN: single-bush interaction demo
│   ├── demos/
│   │   ├── g1.py                  ← Minimal G1 stand demo (falls — no balance)
│   │   ├── g1_locomotion.py       ← G1 with pre-trained locomotion policy
│   │   ├── h1_locomotion.py       ← H1 locomotion reference (working)
│   │   └── sensors/
│   │       ├── contact_sensor.py  ← Contact sensor reference demo
│   │       └── cameras.py         ← Camera sensor reference demo
│   └── tools/
│       ├── asset_showcase.py      ← [NEW] Spawn all USD assets in a grid for visual review
│       ├── browse_nucleus.py      ← [NEW] Browse NVIDIA Nucleus asset directories via CLI
│       └── convert_bush_assets.py ← [NEW] OBJ→USD converter with texture path fix
│
├── source/
│   ├── isaaclab/isaaclab/
│   │   ├── assets/articulation/   ← Articulation, ArticulationCfg
│   │   ├── sensors/               ← ContactSensor, Camera, ContactSensorCfg, CameraCfg
│   │   ├── sim/                   ← SimulationContext, SimulationCfg, RenderCfg
│   │   └── sim/schemas/           ← define_rigid_body_properties, CollisionPropertiesCfg …
│   └── isaaclab_assets/isaaclab_assets/robots/
│       └── unitree.py             ← G1_CFG, G1_MINIMAL_CFG, H1_CFG (joint defaults, PD gains)
│
├── GREENHOUSE_PROJECT.md          ← This file
└── isaaclab.sh                    ← Run all scripts through this

External asset repo (not in IsaacLab):
~/dev-bru/Nvidia-Isaac-Sim-Procedual-Forest-Generator/models/
├── Bush_obj/        Bush.usd, Bush.obj, Bush.mtl, textures/
├── Blueberry_obj/   Blueberry.usd, Blueberry.obj …
├── Birch_obj/
├── Pine_obj/
├── Rock_obj/
├── Spruce_obj/
└── maple_obj/
```

---

## 4. How to Run

### 4.1 Main Demo — Single-Bush Arm Interaction

```bash
cd /home/trooperai/IsaacLab

# GUI mode (required to see the viewport)
./isaaclab.sh -p scripts/greenhouse_sim.py

# With RGB camera sensor active
./isaaclab.sh -p scripts/greenhouse_sim.py --enable_cameras

# Headless (no window, useful for SSH)
./isaaclab.sh -p scripts/greenhouse_sim.py --headless
```

**What you will see:**
1. Greenhouse floor + corner pillars (glass walls hidden for clear view)
2. One Bush.usd at world position `(5.0, 0.70, 0.0)` — 70 cm to the robot's left
3. G1 robot spawns at `(2.5, 0.0, 0.74)` facing +X
4. Robot walks to `(5.0, 0.0, 0.74)` — level with the bush
5. Left arm slowly extends sideways (90° roll) into the bush (~0.6 s)
6. Arm holds inside for ~0.8 s
7. Arm retracts (~0.6 s)
8. Robot stands still; viewport stays live (Ctrl+C to quit)

**Terminal output during run:**
```
[INFO] Bush spawned at (5.0, 0.7, 0.0)  prim: /World/Greenhouse/Plants/Bush
[INFO] Scene ready — 1 bush at (5.0, 0.7, 0.0), robot starts at (2.5, 0.0, 0.74)
[WALK→REACH_IN] arrived at inspection position (5.0, 0.0)
[REACH_IN→INSIDE] arm fully extended — hand inside bush
[INSIDE→REACH_OUT] retracting arm
[REACH_OUT→DONE] arm retracted — interaction complete
[CONTACT] Robot body 'left_elbow_pitch_link' touching something — force 12.3 N
```

### 4.2 Asset Showcase Tool

Preview all locally available USD plant assets in a grid before adding them to the main scene:

```bash
./isaaclab.sh -p scripts/tools/asset_showcase.py
```

Currently active assets in the grid: **Bush** (scale 1.0) and **Blueberry** (scale 5.0).  
Edit `_ASSETS` in the script to add/remove assets.  
Middle-mouse drag to orbit, scroll to zoom.

### 4.3 Browse Nucleus

```bash
# List the vegetation directory
./isaaclab.sh -p scripts/tools/browse_nucleus.py

# Custom path
./isaaclab.sh -p scripts/tools/browse_nucleus.py --path "NVIDIA/Assets/Vegetation/Plants"

# Recursive listing of all USD files
./isaaclab.sh -p scripts/tools/browse_nucleus.py --path "NVIDIA/Assets" --recursive
```

> **Note:** This system has no `NVIDIA/Assets/Vegetation` folder in Nucleus. Local assets from the forest generator repo are used instead.

### 4.4 Convert OBJ Assets to USD

Fixes Windows absolute texture paths that break on Linux:

```bash
./isaaclab.sh -p scripts/tools/convert_bush_assets.py
```

Outputs `Bush_local.usd` and `Blueberry_local.usd` next to the originals.

---

## 5. Task List

### Done ✅

| # | Task | Notes |
|---|---|---|
| 1 | Greenhouse structure | 8×5×3 m box, glass walls (35% opacity), peaked roof (25°), corner pillars, sandy floor |
| 2 | G1 robot in scene | Spawned at hip z=0.74 m in default standing pose |
| 3 | Understand G1 fall | PD gains tuned for locomotion policy, not passive balance — documented with fix options |
| 4 | Kinematic root control | `write_root_pose_to_sim()` + `write_root_velocity_to_sim(zeros)` every frame |
| 5 | Kinematic joint control | `write_joint_state_to_sim()` freezes pose; `torch.lerp` for smooth arm motion |
| 6 | Bush asset discovery | Local OBJ→USD files in forest generator repo at `~/dev-bru/…/models/` |
| 7 | Asset showcase tool | `tools/asset_showcase.py` — visual grid of all local USD plants |
| 8 | Nucleus browser tool | `tools/browse_nucleus.py` — headless CLI to list Nucleus directories |
| 9 | OBJ→USD converter | `tools/convert_bush_assets.py` — fixes Windows texture paths in .mtl files |
| 10 | ContactSensor on G1 | All body links monitored; prints link name + force (N) when contact > 5 N |
| 11 | RGB Camera on torso | 640×480, ~10 Hz, GPU tensor, ROS convention, faces forward — enabled with `--enable_cameras` |
| 12 | Arm reach motion | `torch.lerp` ramps joints between default and reach pose smoothly over 60 frames |
| 13 | State machine | WALK → REACH_IN → INSIDE → REACH_OUT → DONE, clean phase transitions with terminal logs |
| 14 | Glass walls hidden | Commented out (not deleted) — 3 lines to restore full enclosure |
| 15 | Single-bush focus | Simplified to 1 bush for physics debugging |
| 16 | **Interactive bush** ✨ | Procedural: kinematic trunk + 10 dynamic sphere clusters connected via `UsdPhysics.Joint` spring D6 joints — clusters deflect on contact and spring back (stiffness=15 N·m/rad, damping=3) |
| 17 | GitHub repo | Pushed to `github.com/bruXsavvytec/IsaacLab`, SSH key configured on server |

### In Progress 🔄

- [ ] **Verify contact sensor** — check terminal for `[CONTACT] '...' touching cluster` during INSIDE phase; if silent, add `PhysxContactReportAPI` to clusters
- [ ] **Tune arm angles** — confirm hand visually enters upper clusters (z≈0.95 m); adjust `left_shoulder_roll_joint` (currently 2.0 rad) if needed

### Next Up 📋

**Scene completeness:**
- [ ] Re-enable glass walls and roof once single-bush interaction is confirmed solid
- [ ] Restore multi-bush grid — 2 rows × 6 plants, each fully interactive (same spring-joint setup)
- [ ] Per-plant health state — each bush gets `{"healthy": bool, "inspected": bool}`; healthy = green clusters, sick = yellow/brown tint

**Vision:**
- [ ] YOLO integration — pass `camera.data.output["rgb"]` (GPU tensor, shape `[1, 480, 640, 4]`) to `ultralytics` YOLO model; detect colour anomalies as proxy for plant disease

**Locomotion (new priority):**
- [ ] **Find G1 locomotion checkpoint** — search `scripts/reinforcement_learning/` and Nucleus for a pre-trained G1 policy; reference implementation is `scripts/demos/h1_locomotion.py` (H1 version)
- [ ] **Wire up G1 walking policy** — replace kinematic root glide with policy network: load checkpoint → observe IMU + joint state → output joint targets at 50 Hz → robot walks naturally
- [ ] **Integrate policy with state machine** — policy runs during WALK phase; switch to arm-reach control when robot arrives at inspection position
- [ ] **VLA model** (future) — Vision-Language-Action model for high-level task planning on top of the locomotion policy

### Parallel / Background 📌

- [ ] Deformable leaf physics — individual leaf movement requires rigged USD or PhysX FEM deformable body solver; explore as a research spike
- [ ] Blueberry.usd in scene — scale always 5× bush scale; add alongside interactive bush once grid is restored
- [ ] Texture fix — run `convert_bush_assets.py` if Bush.usd textures appear grey (Windows .mtl paths)

---

## 6. What Has Been Built

### 6.1 `greenhouse_sim.py` — Main Scene

**Scene layout (current):**
- Greenhouse floor (8×5 m, sandy colour) + 4 corner pillars
- Glass walls + roof **commented out** (easy to restore — see `build_greenhouse()`)
- 1× Bush.usd at `(5.0, 0.70, 0.0)` — kinematic rigid body, collidable (70 cm from aisle centre — within G1 arm reach)
- G1 humanoid spawned at `(2.5, 0.0, 0.74)`

**Key constants (tune these):**

```python
_BUSH_POS    = (5.0, 0.70, 0.0)  # 70 cm to robot's left — within G1 arm reach
_ROBOT_START = (2.5, 0.0, 0.74)  # robot spawn
_INSPECT_POS = (5.0, 0.0, 0.74)  # robot stops to interact

RAMP_FRAMES  = 60    # frames to extend / retract arm  (0.6 s at dt=0.01)
HOLD_FRAMES  = 80    # frames with arm inside the bush  (0.8 s)
MOVE_SPEED   = 0.6   # m/s

# Tune these angles if hand doesn't reach the bush or clips the torso.
# G1 arm reach: shoulder offset ~0.22 m + arm length ~0.62 m at 90° roll ≈ 0.84 m total Y.
# Bush at Y=0.70 gives ~14 cm of clearance — hand should enter comfortably.
_REACH_JOINTS = {
    "left_shoulder_roll_joint":  1.57,   # 90° abduction — arm straight out to the side
    "left_shoulder_pitch_joint": 0.10,   # slight forward tilt for height alignment
    "left_elbow_pitch_joint":    0.05,   # nearly fully extended (do NOT go negative)
    "left_one_joint":  0.0,             # open hand
    "left_two_joint":  0.0,
}
```

**State machine:**

```
WALK      robot glides from _ROBOT_START → _INSPECT_POS at MOVE_SPEED
REACH_IN  left arm interpolates default_pose → reach_pose over RAMP_FRAMES
INSIDE    arm holds at full reach for HOLD_FRAMES
REACH_OUT arm interpolates reach_pose → default_pose over RAMP_FRAMES
DONE      robot stands still, viewport live (Ctrl+C to quit)
```

**To restore glass walls**, un-comment these lines in `build_greenhouse()`:
```python
# _spawn_box("/World/Greenhouse/WallFront", ...)
# _spawn_box("/World/Greenhouse/WallBack",  ...)
# _spawn_box("/World/Greenhouse/WallLeft",  ...)
# _spawn_box("/World/Greenhouse/WallRight", ...)
# _spawn_box("/World/Greenhouse/RoofFront", ...)
# _spawn_box("/World/Greenhouse/RoofBack",  ...)
# _spawn_box("/World/Greenhouse/Ridge",     ...)
```

### 6.2 `tools/asset_showcase.py`

Spawns a grid of USD assets so you can visually inspect and choose which to use. Runs in GUI mode only. Prints prim paths and scale to terminal.

### 6.3 `tools/browse_nucleus.py`

CLI tool to list NVIDIA Nucleus directories without opening the full Isaac Sim UI.  
Runs headless. Useful for discovering what vegetation assets Nucleus has.

### 6.4 `tools/convert_bush_assets.py`

Uses `omni.kit.asset_converter` to re-convert `.obj` files to USD, fixing Windows-style texture paths embedded in `.mtl` files. Output is `Bush_local.usd` and `Blueberry_local.usd`.

---

## 7. Architecture & State Machine

### 7.1 Typical Script Skeleton

Every IsaacLab script follows this strict ordering:

```python
# 1. Parse args + boot Isaac Sim (MUST be first)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Import everything AFTER boot
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# 3. Configure physics world
sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0",
                                   render=sim_utils.RenderCfg(rendering_mode="quality"))
sim = SimulationContext(sim_cfg)

# 4. Populate scene (before sim.reset())
def design_scene():
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    robot = Articulation(cfg=G1_CFG.replace(prim_path="/World/G1"))
    return robot

robot = design_scene()

# 5. Start physics engine (triggers sensor + articulation initialisation)
sim.reset()

# 6. Main loop
while simulation_app.is_running():
    robot.write_root_pose_to_sim(pose)
    robot.write_joint_state_to_sim(jpos, jvel)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)
```

### 7.2 Kinematic Control Explained

The G1 is **not** using physics-based balance. Instead:

| Call | Effect |
|---|---|
| `write_root_pose_to_sim(pose)` | Teleports pelvis to exact (x, y, z, quat) every frame |
| `write_root_velocity_to_sim(zeros)` | Sets root velocity to zero — prevents physics drift |
| `write_joint_state_to_sim(jpos, jvel)` | Sets all joint angles directly, bypassing PD actuators |

This makes the robot behave like a moving mannequin: perfectly stable, no dynamics, but no real physics feedback either.

### 7.3 Bush Physics Setup

`Bush.usd` is an OBJ→USD conversion with **no physics APIs pre-baked**.  
Isaac Lab's `UsdFileCfg.rigid_props` calls `modify_rigid_body_properties()`, which only *edits* existing rigid bodies — it **cannot create** them.

The fix (in `_apply_bush_physics()`):
```python
# Step 1: apply RigidBodyAPI to root prim (makes it a physics actor)
sim_utils.define_rigid_body_properties(prim_path,
    sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))

# Step 2: apply CollisionAPI to every Mesh descendant
for child in Usd.PrimRange(root):       # correct USD subtree traversal
    if child.IsA(UsdGeom.Mesh):
        sim_utils.define_collision_properties(child.GetPath().pathString, coll_cfg)
```

Key distinction: **`modify_*`** vs **`define_*`**

| Function | Behaviour |
|---|---|
| `modify_rigid_body_properties` | Edits existing rigid body — returns `False` silently if no `RigidBodyAPI` present |
| `define_rigid_body_properties` | Creates `RigidBodyAPI` if absent, then sets properties |
| `modify_collision_properties` | Edits existing collider — no-op if no `CollisionAPI` |
| `define_collision_properties` | Creates `CollisionAPI` if absent, then sets properties |

### 7.4 Contact Sensor

```python
contact_sensor = ContactSensor(cfg=ContactSensorCfg(
    prim_path="/World/G1/.*",   # regex → all G1 body links
    update_period=0.0,           # every physics step
    history_length=1,
    debug_vis=False,
))
```

- `G1_CFG` has `activate_contact_sensors=True` → all G1 links get `PhysxContactReportAPI`
- No `filter_prim_paths_expr` — filtering requires `PhysxContactReportAPI` on the filtered objects too; Bush.usd doesn't have this unless we add it after applying `define_rigid_body_properties`
- `sensor.data.net_forces_w` — shape `(1, N_bodies, 3)` — total force on each G1 link

### 7.5 RGB Camera

```python
camera = Camera(cfg=CameraCfg(
    prim_path="/World/G1/torso_link/insp_cam",
    spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 30.0)),
    offset=CameraCfg.OffsetCfg(
        pos=(0.1, 0.0, 0.25),
        rot=(0.7071, 0.0, 0.7071, 0.0),  # +90° around Y → looks along robot +X
        convention="ros",
    ),
    height=480, width=640,
    data_types=["rgb"],
    update_period=0.1,   # ~10 Hz
))
```

Access the GPU tensor each step:
```python
rgb = camera.data.output["rgb"]   # torch.Tensor (480, 640, 4) RGBA, uint8, on GPU
# Pass directly to YOLO:
# results = yolo_model(rgb[..., :3])
```

Requires `--enable_cameras` flag to render.

---

## 8. Key Technical Learnings & Gotchas

### USD Python API

| Wrong | Correct |
|---|---|
| `prim.GetAllDescendants()` ❌ — does not exist | `Usd.PrimRange(prim)` ✅ — yields root + all descendants |
| `prim.GetChildren()` for subtree ❌ — direct children only | `Usd.PrimRange(prim)` ✅ |

### Physics Schema Creation

> **`activate_contact_sensors=True` on a `UsdFileCfg` for an OBJ-converted USD will raise:**
> ```
> ValueError: No contact sensors added to the prim '...'.
> This means that no rigid bodies are present under this prim.
> ```
> **Root cause:** The OBJ→USD pipeline produces visual-only meshes with no `UsdPhysics.RigidBodyAPI`. `activate_contact_sensors()` walks the prim tree looking for rigid bodies and finds none.  
> **Fix:** Call `define_rigid_body_properties()` first to apply the API, then optionally `activate_contact_sensors()`.

### filter_prim_paths_expr in ContactSensorCfg

Internally converts `.*` → `*` (glob), so regex patterns work.  
But the filtered objects **must have `PhysxContactReportAPI`** — i.e. they must already be rigid bodies with contact reporting enabled. If the bush hasn't been through `activate_contact_sensors()`, the filter produces an empty contact view (no error, just silent misses).

### Why G1 Falls Without a Policy

```
torque = Kp × (target_pos − current_pos) − Kd × current_vel
```

At t=0, positions match targets → zero torque → gravity immediately wins.  
`G1_CFG` gains (Kp=150–200, Kd=5) are tuned for a **running locomotion policy**, not passive balance.  
Solutions (simplest to hardest):
1. `fix_root_link = True` — pins pelvis to world
2. `write_root_pose_to_sim()` every frame — kinematic teleport
3. Much higher PD gains (Kp ≥ 500) — better passive stability, still falls
4. Pre-trained locomotion policy — the real solution

### UsdFileCfg.rigid_props Silently Fails for Visual-Only USDs

`UsdFileCfg.rigid_props` uses `modify_rigid_body_properties()` which returns `False` (no-op + warning) if the prim has no `UsdPhysics.RigidBodyAPI`. No exception is raised. Use `define_rigid_body_properties()` instead.

### configclass `.copy()` is a Shallow Copy

```python
robot_cfg = G1_CFG.copy()
robot_cfg.init_state.pos = _ROBOT_START  # mutates G1_CFG.init_state too!
```

`configclass.copy()` calls `dataclasses.replace()` which does a shallow copy — nested objects are shared. Mutating `init_state.pos` also mutates the original `G1_CFG`. Use `.replace()` with nested replace for safety:
```python
robot_cfg = G1_CFG.replace(
    prim_path="/World/G1",
    init_state=G1_CFG.init_state.replace(pos=_ROBOT_START)
)
```

### OBJ→USD Texture Paths on Linux

`Bush.mtl` contains Windows absolute paths:
```
map_Kd D:\temp_downloads\...\textures\Branches_08.jpg
```
The binary `Bush.usdc` may embed relative paths (working) or Windows paths (broken). If the bush renders as grey/white, run `convert_bush_assets.py` and switch to `Bush_local.usd`.

### Leaf Movement Physics

**Individual leaf movement is not possible** with the current setup.  
The OBJ-converted Bush.usd is a single rigid body — trunk + branches + leaves all move together.  
To get leaf deflection you need either:
- A **deformable body** (PhysX FEM solver) — requires deformable mesh USD format
- A **rigged asset** with an animation graph driven by contact forces
- PhysX cloth for flat leaf planes

---

## 9. Known Limitations

| Limitation | Cause | Planned Fix |
|---|---|---|
| Robot walks through walls if root is kinematic | Kinematic root ignores collision | Re-enable walls + add collision to greenhouse geometry |
| Leaves don't move when arm enters bush | Single rigid-body bush, no deformable solver | Use rigged asset or PhysX deformable |
| No bush-specific contact filter | Bush needs `PhysxContactReportAPI` first | Call `activate_contact_sensors()` after `_apply_bush_physics()` |
| Camera renders only with `--enable_cameras` | Isaac Sim renderer disabled in default headless mode | Always pass `--enable_cameras` when you need RGB frames |
| Arm may not reach bush center exactly | Joint angles approximate, depend on G1 segment lengths | Tune `_REACH_JOINTS` values after visual inspection |
| Glass walls are not collidable | `CuboidCfg` spawns visual-only prims by default | Add `collision_props` + `rigid_props` to `_spawn_box()` |

---

## 10. Roadmap

### Phase 1 — Scripted Interaction (Current)
- [x] Kinematic walk + arm reach state machine
- [x] Single-bush scene for debugging
- [ ] Tune arm joints to reliably reach bush center
- [ ] Re-enable glass walls + add collision to walls

### Phase 2 — Sensing
- [ ] Per-bush contact filter (apply `PhysxContactReportAPI` post-`define_rigid_body_properties`)
- [ ] YOLO integration — `ultralytics` model on `camera.data.output["rgb"]` tensor
- [ ] Healthy/sick plant state — randomise per-plant, detect via YOLO label
- [ ] Restore multi-bush grid (2 rows × 6 columns)

### Phase 3 — Real Locomotion
- [ ] Find/download pre-trained G1 locomotion checkpoint (reference: `h1_locomotion.py`)
- [ ] Replace kinematic root with locomotion policy output
- [ ] Handle transitions: loco policy → arm reach → loco policy (hierarchy)

### Phase 4 — VLA
- [ ] Connect VLA model (language prompt → action tokens)
- [ ] Task: "Inspect all bushes and report sick ones"
- [ ] Data collection pipeline (record demos → imitation learning)

---

## 11. Asset Catalogue

### Locally Available (Forest Generator Repo)

| Asset | USD Path | Notes |
|---|---|---|
| Bush | `…/Bush_obj/Bush.usd` | Scale 1.0 — confirmed good visually |
| Blueberry | `…/Blueberry_obj/Blueberry.usd` | Scale 5.0 (always 5× bush scale) |
| Birch | `…/Birch_obj/Birch.usd` | Tree, scale ~0.15 |
| Pine | `…/Pine_obj/Pine.usd` | Tree, scale ~0.15 |
| Spruce | `…/Spruce_obj/Spruce.usd` | Tree, scale ~0.15 |
| Maple | `…/maple_obj/maple.usd` | Tree, scale ~0.15 |
| Rock | `…/Rock_obj/Rock.usd` | Ground detail |
| BigRock | `…/Rock_obj/big_rock.usd` | Ground detail |

All assets have Windows-path texture references in their `.mtl` files. Run `convert_bush_assets.py` if textures appear grey/white.

### Bundled with Isaac Sim

| Asset | USD Path |
|---|---|
| Pot Plant | `/home/trooperai/isaac-env/lib/python3.10/…/data/usd/assets/pot_plant.usda` |

### Nucleus Cloud

No `NVIDIA/Assets/Vegetation` folder found on this system's Nucleus connection.  
Use `browse_nucleus.py` to check: `./isaaclab.sh -p scripts/tools/browse_nucleus.py`

---

## Quick Reference — Common Patterns

```python
# Spawn a kinematic collidable USD (e.g. a plant)
bush_cfg = sim_utils.UsdFileCfg(usd_path="...", scale=(1.0, 1.0, 1.0))
bush_cfg.func("/World/Bush", bush_cfg, translation=(x, y, 0.0))
# Then apply physics manually:
sim_utils.define_rigid_body_properties("/World/Bush",
    sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
for child in Usd.PrimRange(stage.GetPrimAtPath("/World/Bush")):
    if child.IsA(UsdGeom.Mesh):
        sim_utils.define_collision_properties(child.GetPath().pathString,
            sim_utils.CollisionPropertiesCfg(collision_enabled=True))

# Walk a USD subtree (correct API)
from pxr import Usd, UsdGeom
for child in Usd.PrimRange(root_prim):       # NOT root_prim.GetAllDescendants()
    if child.IsA(UsdGeom.Mesh):
        ...

# Interpolate between two joint poses (smooth arm motion)
alpha    = frame / RAMP_FRAMES               # 0.0 → 1.0
cur_jpos = torch.lerp(default_jpos, reach_jpos, alpha)

# Read contact force on robot body links
forces = contact_sensor.data.net_forces_w    # (1, N_bodies, 3)
max_f  = forces.norm(dim=-1).max().item()    # scalar N
```

---

*Last updated: 2026-05-21 | Maintained by: trooperai*
