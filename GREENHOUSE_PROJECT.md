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
7. [Architecture & Key Patterns](#7-architecture--key-patterns)
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
Interactive spring-jointed bush (clusters deflect, spring back)    ✅ DONE
       ↓
Contact detection (is the robot touching a cluster?)               ✅ DONE
       ↓
RGB camera + YOLO + colour health analysis                         ✅ DONE
       ↓
Pre-trained locomotion policy (real walking dynamics)              ✅ DONE
       ↓
Soft / deformable bush asset (PhysX cloth leaves)                  🔜 next
       ↓
VLA (Vision-Language-Action model) for high-level planning         📋 future
```

---

## 2. Environment & Requirements

### 2.1 Software Versions

| Component | Version |
|---|---|
| Isaac Lab | 2.3.2 |
| Isaac Sim | 4.x (bundled) |
| Python | 3.10.12 |
| PyTorch | bundled with Isaac Sim |
| ultralytics (YOLO) | installed in isaac-env |
| opencv-python | 4.12.0 — headless build (no GUI) |
| Pillow | bundled with Isaac Sim |
| OS | Ubuntu 22.04 |

### 2.2 Installation Paths

| What | Path |
|---|---|
| IsaacLab root | `/home/trooperai/IsaacLab/` |
| Isaac Sim Python env | `/home/trooperai/isaac-env/` |
| Pre-trained G1 checkpoint | `/home/trooperai/IsaacLab/.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt` |
| 3D asset models | `/home/trooperai/dev-bru/Nvidia-Isaac-Sim-Procedual-Forest-Generator/models/` |
| IsaacLab launcher | `/home/trooperai/IsaacLab/isaaclab.sh` |

### 2.3 How to Run Scripts

**Always use `isaaclab.sh`**, never plain `python3`:

```bash
cd /home/trooperai/IsaacLab

# Kinematic demo (scripted walk, no policy)
./isaaclab.sh -p scripts/greenhouse_sim.py

# With camera + YOLO + Plant Inspector window
./isaaclab.sh -p scripts/greenhouse_sim.py --enable_cameras

# Locomotion policy demo (real walking dynamics)
./isaaclab.sh -p scripts/greenhouse_locomotion.py

# Locomotion + camera + YOLO
./isaaclab.sh -p scripts/greenhouse_locomotion.py --enable_cameras
```

### 2.4 Critical Import Rule

All `import isaaclab.*` and `import omni.*` must come **after** `AppLauncher`:

```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)   # Isaac Sim boots here
simulation_app = app_launcher.app

# Only AFTER this:
import isaaclab.sim as sim_utils
import omni.ui as ui
```

---

## 3. Repository Structure

```
IsaacLab/
├── scripts/
│   ├── greenhouse_sim.py          ← Kinematic demo: interactive bush + YOLO
│   ├── greenhouse_locomotion.py   ← Locomotion policy demo: RSL-RL G1 + bush + YOLO
│   ├── demos/
│   │   ├── g1.py                  ← Minimal G1 stand (falls — no balance)
│   │   ├── g1_locomotion.py       ← G1 locomotion reference
│   │   ├── h1_locomotion.py       ← H1 locomotion reference (used as template)
│   │   └── sensors/
│   │       ├── contact_sensor.py  ← Contact sensor reference
│   │       └── cameras.py         ← Camera sensor reference
│   └── tools/
│       ├── asset_showcase.py      ← Spawn all USD assets in a grid for review
│       ├── browse_nucleus.py      ← Browse NVIDIA Nucleus directories via CLI
│       └── convert_bush_assets.py ← OBJ→USD converter with texture path fix
│
├── .pretrained_checkpoints/
│   └── rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt   ← RSL-RL G1 policy
│
├── source/
│   ├── isaaclab/isaaclab/
│   │   ├── assets/articulation/   ← Articulation, ArticulationCfg
│   │   ├── sensors/               ← ContactSensor, Camera, CameraCfg
│   │   └── sim/                   ← SimulationContext, SimulationCfg, RenderCfg
│   └── isaaclab_assets/isaaclab_assets/robots/
│       └── unitree.py             ← G1_CFG, G1_MINIMAL_CFG (joint defaults, PD gains)
│
├── GREENHOUSE_PROJECT.md          ← This file
└── isaaclab.sh                    ← Always run scripts through this
```

---

## 4. How to Run

### 4.1 `greenhouse_sim.py` — Kinematic Demo

```bash
./isaaclab.sh -p scripts/greenhouse_sim.py               # basic
./isaaclab.sh -p scripts/greenhouse_sim.py --enable_cameras  # + YOLO + Inspector window
```

**What you see:**
1. Greenhouse floor + pillars
2. One procedural interactive bush at `(5.0, 0.70, 0.0)` — 10 spring-jointed sphere clusters, 2 of them yellow (simulated disease)
3. G1 robot glides from `(2.5, 0.0)` to `(5.0, 0.0)` — kinematic root, no policy
4. Left arm ramps in → clusters deflect → arm holds → clusters spring back → arm retracts
5. `--enable_cameras`: **"Plant Inspector"** panel appears inside Isaac Sim showing the camera feed with stressed pixels tinted red + YOLO boxes + health HUD

**Terminal output:**
```
[YOLO] YOLOv8n loaded — visual inspection active
[PREVIEW] 'Plant Inspector' window created inside Isaac Sim
[WALK→REACH_IN] arrived at (5.0, 0.0)
[REACH_IN→INSIDE] arm fully extended — clusters pushed aside
[INSPECT] Green: 62%  Stressed: 24%  → STRESSED
[YOLO] sports ball (78%), person (61%)
[INSPECT REPORT] ── 8 frames analysed ──
  Avg healthy green  : 64.2%
  Avg stressed/yellow: 22.1%
  Verdict: NEEDS ATTENTION  (unhealthy clusters: [3, 7])
[INSIDE→REACH_OUT] retracting arm — clusters spring back
```

### 4.2 `greenhouse_locomotion.py` — Policy Demo

```bash
./isaaclab.sh -p scripts/greenhouse_locomotion.py                # basic
./isaaclab.sh -p scripts/greenhouse_locomotion.py --enable_cameras  # + YOLO
```

**What you see:**
1. Same scene, but robot walks naturally under the **pre-trained RSL-RL rough-terrain policy**
2. G1 walks to the bush (`WALK → ARRIVE → REACH_IN → INSIDE → REACH_OUT → DONE`)
3. During `REACH_IN/INSIDE/REACH_OUT`: locomotion policy still runs for legs (balance); arm joints are overridden separately
4. Contact sensor only reports during arm-reach phases (leg contacts during walking are suppressed)

**State machine:**
```
WALK      → policy(vx=0.8, vy=0, wz=0) until robot_x ≥ 5.0 m
ARRIVE    → stop command for 50 frames — robot decelerates
REACH_IN  → arm ramps in over 60 frames (policy still holds balance)
INSIDE    → arm holds for 80 frames, YOLO + health analysis runs
REACH_OUT → arm retracts over 60 frames
DONE      → zero command, robot stands
```

---

## 5. Task List

### Done ✅

| # | Task | Notes |
|---|---|---|
| 1 | Greenhouse structure | 8×5×3 m, sandy floor, peaked roof, corner pillars; glass walls commented out for clear view |
| 2 | G1 robot in scene | Spawned at hip z=0.74 m in default standing pose |
| 3 | Understand G1 fall | PD gains tuned for locomotion policy — documented 4 fix options |
| 4 | Kinematic root control | `write_root_pose_to_sim()` + zero velocity every frame |
| 5 | Kinematic joint control | `write_joint_state_to_sim()` + `torch.lerp` for smooth arm motion |
| 6 | Bush asset discovery | Local OBJ→USD in forest generator repo |
| 7 | Asset showcase tool | `tools/asset_showcase.py` |
| 8 | Nucleus browser tool | `tools/browse_nucleus.py` |
| 9 | OBJ→USD converter | `tools/convert_bush_assets.py` — fixes Windows texture paths |
| 10 | ContactSensor on G1 | All body links; prints link name + force (N) when contact > 5 N |
| 11 | RGB Camera on torso | 640×480, ~10 Hz, GPU tensor; `--enable_cameras` flag required |
| 12 | Arm reach motion | `torch.lerp` ramps joints over 60 frames |
| 13 | Kinematic state machine | WALK → REACH_IN → INSIDE → REACH_OUT → DONE |
| 14 | **Interactive spring bush** | Procedural trunk (kinematic) + 10 dynamic sphere clusters connected via `UsdPhysics.Joint` D6 spring joints. Clusters deflect on contact, spring back when released. Stiffness=15 N·m/rad, damping=3, limit=±60° |
| 15 | GitHub repo | `github.com/bruXsavvytec/IsaacLab`, SSH key configured |
| 16 | **Locomotion policy** | RSL-RL actor MLP loaded from checkpoint; 310-dim obs (IMU + joints + last action + zeros for height scan); 37-dim action scaled by 0.25; arms overridden during reach phases |
| 17 | Phase-filtered contacts | Ankle/leg contacts suppressed during walk; only arm-phase contacts printed |
| 18 | **YOLO integration** | `ultralytics` YOLOv8n on camera frames (~1 Hz); colour health analysis every frame (green vs stressed-yellow pixel ratio); inspection report printed at end of INSIDE phase |
| 19 | Unhealthy plant sim | Clusters 3 and 7 coloured yellow/brown to simulate nitrogen deficiency and root stress |
| 20 | **Plant Inspector window** | `omni.ui.ByteImageProvider` + `ImageWithProvider` → floating panel inside Isaac Sim; stressed pixels tinted red; YOLO boxes drawn with PIL; health HUD label |

### In Progress 🔄

- [ ] Tune `left_shoulder_roll_joint` (currently 2.0 rad) — confirm hand visually enters upper clusters (z≈0.95 m)

### Known Broken / TODO 🔴

- [ ] **Live camera preview window** — three approaches tried, all failed:
  - `cv2.imshow()` → not implemented (Isaac Sim ships headless OpenCV, no GTK+)
  - `omni.ui.ByteImageProvider` → window created but never visible in UI
  - `feh --auto-reload /tmp/plant_inspector_latest.png` → saves PNG to disk (implemented), `feh` installed, but not yet confirmed working end-to-end
  - **Next attempt:** use Isaac Sim's native Viewport 2 (`Window → Viewport → Viewport 2`, switch camera to `/World/G1/torso_link/insp_cam`) for raw feed; for annotated view consider a ROS topic or a proper omni.kit extension

### Next Up 📋

**Soft bush:**
- [ ] Deformable leaves via PhysX Particle Cloth — apply `PhysxParticleClothAPI` to leaf mesh prims in `Bush_local.usd` (run `convert_bush_assets.py` with `single_mesh=False` first to get per-material prims)

**Scene completeness:**
- [ ] Re-enable glass walls + roof once single-bush interaction is fully confirmed
- [ ] Restore 2×6 bush grid — each with full spring-joint setup + per-bush health state dict
- [ ] Per-plant health state: `{"healthy": bool, "inspected": bool}` — randomise at spawn

**Locomotion refinement:**
- [ ] Tune `WALK_VX` if robot drifts
- [ ] Fine-tune arm override angle if hand misses clusters

**Future:**
- [ ] VLA model — language prompt → action tokens on top of locomotion policy

---

## 6. What Has Been Built

### 6.1 Interactive Spring Bush (`build_interactive_bush()`)

Replaces the old static `Bush.usd` with a fully procedural physics object:

```
/World/Greenhouse/Plants/Bush/
├── Trunk          kinematic cuboid (0.08×0.08×0.80 m, brown)
├── Cluster_0…9    dynamic spheres, r=0.13–0.14 m
│   (indices 3,7 are yellow → simulated disease)
└── Joint_0…9      UsdPhysics.Joint D6 spring joints
```

**Joint physics per cluster:**
```python
# Translation: all 3 axes locked → cluster can only rotate, not slide
# Rotation rotX, rotY: ±60° swing limit + spring drive
#   torque = stiffness * (0 - current_angle) - damping * angular_vel
# Rotation rotZ: locked (no twist)

stiffness = 15.0   # N·m/rad — spring-back torque per radian
damping   =  3.0   # N·m·s/rad — kills oscillation
limit     = 60.0   # degrees max swing
mass      =  0.05  # kg per cluster
```

**Joint anchor math:**
```python
# Pivot placed at trunk top: world pos = (bx, by, TRUNK_HEIGHT=0.80)
# LocalPos0 (on trunk) = (0, 0, trunk_z)          # trunk origin → trunk top
# LocalPos1 (on cluster) = (-dx, -dy, 0.80 - dz)  # cluster origin → trunk top
# At rest both frames coincide at (bx, by, 0.80) in world space
```

### 6.2 Locomotion Policy (`greenhouse_locomotion.py`)

Bypasses `OnPolicyRunner` entirely — loads only the actor MLP:

```python
# Old RSL-RL (< 4.0) checkpoint format:
# ckpt["model_state_dict"]["actor.0.weight"], ["actor.0.bias"], ...
# Network: 310 → 512 → 256 → 128 → 37 (ELU activations)

dims = [310, 512, 256, 128, 37]
layers = []
for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
    layers.append(nn.Linear(in_d, out_d))
    if i < 3:   # 3 hidden layers
        layers.append(nn.ELU())
actor = nn.Sequential(*layers)

actor_sd = {k[len("actor."):]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("actor.")}
actor.load_state_dict(actor_sd)
```

**310-dim observation layout:**

| Slice | Content | Dim |
|---|---|---|
| `[0:3]` | `root_lin_vel_b` — body-frame linear velocity | 3 |
| `[3:6]` | `root_ang_vel_b` — body-frame angular velocity | 3 |
| `[6:9]` | `projected_gravity_b` — gravity in body frame | 3 |
| `[9:12]` | velocity command `[vx, vy, wz]` | 3 |
| `[12:49]` | `joint_pos - default_joint_pos` | 37 |
| `[49:86]` | `joint_vel` | 37 |
| `[86:123]` | `last_action` | 37 |
| `[123:310]` | height scan → **zeros** (flat floor) | 187 |

**Arm override during reach phases:**
```python
# Back-solve: joint_target = default + scale * action
# → action[idx] = (desired_angle - default_angle) / ACTION_SCALE
ACTION_SCALE = 0.25
for jname in _REACH_JOINTS:
    idx = name_to_idx[jname]
    action[0, idx] = (interp[0, idx] - default_jpos[0, idx]) / ACTION_SCALE
# Policy still runs for leg joints → robot stays balanced while arm moves
```

### 6.3 YOLO + Plant Inspector

```
Camera → (1,480,640,4) RGBA GPU tensor
         ↓
rgb_np = tensor[0,:,:,:3].cpu().numpy()   # (480,640,3) RGB
         ↓
_analyze_health()        ← every frame, cheap numpy
  green mask: G > 0.20, G > R*1.35, G > B*1.35
  stressed mask: R>0.28, G>0.28, G < R*1.20, B<0.25
         ↓
YOLOv8n inference        ← every 10 calls (~1 s)
  input: rgb_np[:,:,::-1]  (RGB→BGR)
  output: boxes with cls, conf, xyxy
         ↓
_show_preview()          ← every frame with new camera data
  1. Tint stressed pixels red (numpy blend)
  2. Draw YOLO boxes with PIL ImageDraw (cached between runs)
  3. Push RGBA bytes → omni.ui.ByteImageProvider
  4. Update label text
```

**Why `omni.ui` not `cv2.imshow()`:**  
The OpenCV bundled in Isaac Sim's Python env is a **headless build** (no GTK+/Qt GUI support). `cv2.imshow()` raises `error: (-2) The function is not implemented`. Use `omni.ui.ByteImageProvider` instead — it renders inside Isaac Sim's own Qt window.

---

## 7. Architecture & Key Patterns

### 7.1 Script Skeleton

```python
# 1. Boot (must be first)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. All imports after boot
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

# 3. Physics world
sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device,
                                   render=sim_utils.RenderCfg(rendering_mode="quality"))
sim = SimulationContext(sim_cfg)

# 4. Populate scene (before sim.reset())
def design_scene(): ...

# 5. Start physics
sim.reset()

# 6. Main loop
while simulation_app.is_running():
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)
    contact_sensor.update(sim_dt)
```

### 7.2 Kinematic vs Policy Control

| Mode | Root | Joints | File |
|---|---|---|---|
| Kinematic | `write_root_pose_to_sim()` every frame | `write_joint_state_to_sim()` every frame | `greenhouse_sim.py` |
| Policy | Free-floating (physics) | `set_joint_position_target()` → PD actuators | `greenhouse_locomotion.py` |

### 7.3 Contact Sensor

```python
ContactSensorCfg(prim_path="/World/G1/.*", update_period=0.0, history_length=1)
# sensor.data.net_forces_w: (1, N_bodies, 3)
# G1_MINIMAL_CFG has activate_contact_sensors=True → all links get PhysxContactReportAPI
```

### 7.4 Camera

```python
CameraCfg(
    prim_path="/World/G1/torso_link/insp_cam",
    spawn=sim_utils.PinholeCameraCfg(focal_length=8.5, clipping_range=(0.1, 30.0)),
    offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.25), rot=(0.7071, 0, 0.7071, 0), convention="ros"),
    height=480, width=640, data_types=["rgb"], update_period=0.1,
)
# camera.data.output["rgb"] → (1, 480, 640, 4) RGBA uint8 GPU tensor
# Requires --enable_cameras flag
```

---

## 8. Key Technical Learnings & Gotchas

### `UsdPhysics.D6Joint` does not exist

```python
# WRONG — raises AttributeError in Isaac Sim 4.x
d6 = UsdPhysics.D6Joint.Define(stage, path)

# CORRECT — generic Joint IS the D6 joint in this pxr version
d6 = UsdPhysics.Joint.Define(stage, path)
# All LimitAPI / DriveAPI calls are identical either way
```

### Inference tensor in-place write error

```python
# WRONG — action is an inference tensor inside the context manager
with torch.inference_mode():
    action = policy(obs)
action[0, idx] = value   # RuntimeError: Inplace update to inference tensor

# CORRECT — clone immediately after the with block
with torch.inference_mode():
    action = policy(obs)
action = action.clone()          # now a normal tensor
last_action = action.clone()
action[0, idx] = value           # in-place write works
```

### Headless OpenCV — `cv2.imshow()` not implemented

The Isaac Sim Python env ships OpenCV **without GUI support** (`highgui` compiled without GTK+/Qt).  
`cv2.imshow()` raises `error: (-2) The function is not implemented`.  
**Use `omni.ui.ByteImageProvider` instead** — it is always available inside Isaac Sim.

```python
import omni.ui as ui
provider = ui.ByteImageProvider()
provider.set_bytes_data(bytes(W * H * 4), [W, H])   # initial blank
win = ui.Window("My Window", width=W+20, height=H+52)
with win.frame:
    with ui.VStack():
        ui.ImageWithProvider(provider, width=W, height=H)
        label = ui.Label("status text")

# Each frame: push RGBA bytes
rgba = np.zeros((H, W, 4), dtype=np.uint8)
rgba[:,:,:3] = rgb_frame; rgba[:,:,3] = 255
provider.set_bytes_data(rgba.tobytes(), [W, H])
```

### `configclass.copy()` is shallow

```python
robot_cfg = G1_CFG.copy()
robot_cfg.init_state.pos = _ROBOT_START   # ALSO mutates G1_CFG.init_state!

# Safe pattern:
robot_cfg = G1_CFG.replace(
    prim_path="/World/G1",
    init_state=G1_CFG.init_state.replace(pos=_ROBOT_START)
)
```

### USD Python subtree traversal

```python
# WRONG — method does not exist
prim.GetAllDescendants()

# CORRECT
from pxr import Usd
for child in Usd.PrimRange(prim):   # yields root + all descendants
    if child.IsA(UsdGeom.Mesh): ...
```

### Physics API: `modify_*` vs `define_*`

| Function | Behaviour |
|---|---|
| `modify_rigid_body_properties` | Edits existing RigidBodyAPI — **silent no-op** if none present |
| `define_rigid_body_properties` | Creates RigidBodyAPI if absent, then sets properties |

Always use `define_*` when adding physics to assets converted from OBJ.

### Why G1 Falls Without a Policy

```
torque = Kp × (target_pos − current_pos) − Kd × current_vel
```
At t=0 positions match targets → zero torque → gravity wins immediately.  
`G1_CFG` gains (Kp=150–200, Kd=5) are tuned for a running policy, not passive balance.

Fix options (simplest → hardest):
1. `fix_root_link = True` — pins pelvis to world
2. `write_root_pose_to_sim()` every frame — kinematic teleport
3. Higher PD gains (Kp ≥ 500) — still falls eventually
4. Pre-trained locomotion policy — the real solution ✅

---

## 9. Known Limitations

| Limitation | Cause | Planned Fix |
|---|---|---|
| Clusters are spheres, not real leaves | Procedural geometry — no USD leaf mesh | Deformable cloth on real Bush.usd leaf prims |
| Camera may not frame the bush well during INSIDE | Camera on torso facing forward; bush is to the side | Mount second camera on the wrist/hand |
| YOLO detects COCO classes (ball, person) not plants | YOLOv8n trained on COCO, no plant disease classes | Fine-tune on plant disease dataset (PlantVillage) |
| Colour health analysis is a heuristic | Simple RGB channel ratio, not a real model | Fine-tuned YOLO or segmentation model |
| Glass walls not collidable | `CuboidCfg` spawns visual-only prims | Add `collision_props` to `_spawn_box()` |
| Height scan = zeros (flat floor assumption) | Policy was trained on rough terrain | Robot still walks — good enough for flat greenhouse |

---

## 10. Roadmap

### Phase 1 — Scripted Interaction ✅ COMPLETE
- [x] Kinematic walk + arm reach state machine
- [x] Procedural interactive spring bush
- [x] Contact sensor (phase-filtered)

### Phase 2 — Sensing ✅ COMPLETE
- [x] RGB camera sensor on torso
- [x] YOLO integration (YOLOv8n)
- [x] Colour-based health analysis (green vs yellow pixel ratio)
- [x] Plant Inspector window (omni.ui)
- [x] Simulated unhealthy clusters (yellow colour)

### Phase 3 — Real Locomotion ✅ COMPLETE
- [x] Pre-trained RSL-RL G1 policy loaded from checkpoint
- [x] Policy drives legs; arm override during reach phases
- [x] Full WALK → ARRIVE → REACH_IN → INSIDE → REACH_OUT → DONE cycle

### Phase 4 — Soft Bush (Next)
- [ ] Run `convert_bush_assets.py` with `single_mesh=False` to get per-material USD prims
- [ ] Apply `PhysxParticleClothAPI` to leaf prims
- [ ] Apply `PhysxAutoAttachmentAPI` to attach leaves to branch rigid bodies

### Phase 5 — Full Scene
- [ ] 2×6 bush grid, each with spring joints + health state
- [ ] Per-bush inspection log — record which plants are stressed
- [ ] Re-enable glass walls + collision

### Phase 6 — VLA (Future)
- [ ] Connect VLA model: language prompt → action tokens
- [ ] Task: "Walk to each bush, inspect, report sick ones"
- [ ] Data collection pipeline (demo recording → imitation learning)

---

## 11. Asset Catalogue

### Local (Forest Generator Repo)

| Asset | Path | Notes |
|---|---|---|
| Bush | `…/Bush_obj/Bush.usd` | Scale 1.0 |
| Blueberry | `…/Blueberry_obj/Blueberry.usd` | Scale 5.0 |
| Birch / Pine / Spruce / Maple | `…/*/` | Trees, scale ~0.15 |
| Rock / BigRock | `…/Rock_obj/` | Ground detail |

All have Windows-path `.mtl` texture references — run `convert_bush_assets.py` if textures appear grey.

### Bundled with Isaac Sim

| Asset | Path |
|---|---|
| Pot Plant | `…/isaac-env/…/data/usd/assets/pot_plant.usda` |

---

## Quick Reference

```python
# Interactive bush: check spring joint constants
_SPRING_STIFFNESS = 15.0   # N·m/rad — increase for stiffer response
_SPRING_DAMPING   =  3.0   # N·m·s/rad — increase to kill oscillation faster
_SWING_LIMIT_DEG  = 60.0   # max deflection angle
_CLUSTER_MASS     =  0.05  # kg — decrease for easier deflection

# Locomotion policy: tune these
WALK_VX          = 0.8    # forward speed command (m/s)
ARRIVE_THRESH    = 0.10   # stop threshold (m) from target X
STABILISE_FRAMES = 50     # frames to let robot decelerate
ACTION_SCALE     = 0.25   # must match training — do not change

# Arm reach: tune if hand misses clusters
_REACH_JOINTS = {
    "left_shoulder_roll_joint":  2.00,   # 115° — arm out and down
    "left_shoulder_pitch_joint": 0.00,
    "left_elbow_pitch_joint":    0.05,
}

# Push a frame to the Plant Inspector window
rgba = np.zeros((H, W, 4), dtype=np.uint8)
rgba[:, :, :3] = rgb_frame_np
rgba[:, :, 3]  = 255
_image_provider.set_bytes_data(rgba.tobytes(), [W, H])

# Smooth joint interpolation
alpha    = min(frame / RAMP_FRAMES, 1.0)   # 0.0 → 1.0
cur_jpos = torch.lerp(default_jpos, reach_jpos, alpha)

# USD subtree traversal
from pxr import Usd, UsdGeom
for prim in Usd.PrimRange(root_prim):
    if prim.IsA(UsdGeom.Mesh): ...
```

---

*Last updated: 2026-05-22 | Maintained by: trooperai / p.miltrup@savvytec.de*
