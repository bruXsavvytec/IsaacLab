# Greenhouse Inspection Robot — Project Status

## What We Built

### Environment
| File | Status | Notes |
|---|---|---|
| `scripts/greenhouse_sim.py` | ✅ Working | Main scene — G1 + greenhouse + plants |
| `scripts/demos/g1.py` | ✅ Working | Minimal G1 standalone demo |
| `scripts/demos/g1_locomotion.py` | ✅ Working (headless confirmed) | G1 with pre-trained locomotion policy |

---

## Completed Steps

### Step 1 — Baseline scene ✅
- Greenhouse structure: 8×5×3 m glass walls, peaked roof at world pos (5, 0, 0)
- G1 humanoid standing with fixed root link

### Step 2 — Grid bush layout ✅
- Replaced random outdoor bushes with 2 organized rows inside the greenhouse
- 6 plants per row, 1.0 m spacing along X, rows at y = ±1.5 m (flanking a central walkway)
- Each bush = cluster of 4 overlapping spheres

### Step 3 — Physics colliders on bushes ✅
- Each sphere: `CollisionPropertiesCfg(collision_enabled=True)` + `RigidBodyPropertiesCfg(kinematic_enabled=True)`
- Bushes are solid — robot cannot pass through them

### Step 4 — Kinematic waypoint walking ✅
- Robot glides down the central aisle at 0.6 m/s
- Stops at each plant column for ~1.2 s (120 frames), marks both row plants as inspected
- State machine: `move → inspect → move → ... → done`
- Inspection summary printed to terminal
- State dict per plant: `{"row", "col", "pos", "healthy", "inspected"}`

### Step 6 — Contact sensor + camera ✅
- `ContactSensor` on `/World/G1/.*` — every G1 body link monitored; prints body name + force (throttled to 1 msg/s) when force > 5 N
- `Camera` at `/World/G1/torso_link/insp_cam` — 640×480 RGB, 10 Hz, forward-facing (ros convention, +90° Y rotation)
- Camera requires `--enable_cameras` flag; skipped gracefully without it
- Camera logs frame shape every 300 frames to confirm data flow; tensor ready for YOLO passthrough

### Step 5 — G1 locomotion policy (pre-trained) ✅
- `scripts/demos/g1_locomotion.py` — keyboard-controlled G1 using Nucleus checkpoint
- **Key fixes discovered:**
  - `obs[:, 9:13]` (H1 script) was wrong for G1 — clobbers `joint_pos[0]`, causes collapse. Fixed to `obs[:, 9:12]`
  - `omni.kit.viewport` is GUI-only — guarded behind headless check
  - rsl_rl >= 4.0.0 checkpoint format (`actor_state_dict`) incompatible with old published checkpoints (`model_state_dict`) — bypassed `OnPolicyRunner` entirely, load actor weights directly into plain `nn.Sequential`
  - Terrain difficulty `max_init_terrain_level = None` → robot spawns on hardest patch → fixed to `= 0`
  - Viewport wrapper now returns `TensorDict` not plain tensor → extract with `obs_td["policy"]`
- **How it works:** Arrow keys → inject `[vx, vy, wz]` into `obs[9:12]` → 310-dim obs → 37-joint action

---

## To-Do List (Ordered)

| Priority | Step | Description |
|---|---|---|
| ✅ Done | **Contact sensor** | `ContactSensorCfg` on all G1 body links (`/World/G1/.*`) — prints body name + force when robot touches anything above 5 N |
| ✅ Done | **RGB camera on torso** | `CameraCfg` at `/World/G1/torso_link/insp_cam` — 640×480 RGB, 10 Hz, forward-facing. Run with `--enable_cameras` |
| 📋 Next | **Wire locomotion into greenhouse** | Replace kinematic glide in `greenhouse_sim.py` with the G1 policy from `g1_locomotion.py` — robot walks the aisle with real footsteps |
| 📋 Later | **YOLO integration** | Pass `camera.data.output["rgb"]` tensor to `ultralytics` YOLO — detect unhealthy bushes, update `bush_states["healthy"]` |
| 📋 Later | **VLA policy** | Vision-Language-Action model on top of locomotion — receives visual + language goal, outputs navigation commands |
| 💡 Low priority | **Photorealistic bushes** | Replace sphere clusters with USD plant assets from Nucleus (`{NVIDIA_NUCLEUS_DIR}/Assets/Vegetation/`). Keep spheres during dev for fast iteration. |

---

## Key Architecture Notes

### Observation vector (G1, 310-dim)
```
obs[0:3]    base_lin_vel       (3)
obs[3:6]    base_ang_vel       (3)
obs[6:9]    projected_gravity  (3)
obs[9:12]   velocity_commands  (3)  ← keyboard injects here
obs[12:49]  joint_pos_rel      (37)
obs[49:86]  joint_vel_rel      (37)
obs[86:123] last_action        (37)
obs[123:]   height_scan        (187)
```

### Locomotion policy loop
```
Keyboard arrow keys
    ↓ [vx, vy, wz] command
obs[9:12] overwrite
    ↓
Policy MLP (310 → 512 → 256 → 128 → 37)
    ↓ joint position targets
env.step(action)           ← physics + terrain
    ↓ new TensorDict obs
obs_td["policy"]           ← extract flat tensor
```

### Checkpoint compatibility
Published G1 checkpoint (Nucleus) was saved with rsl_rl < 4.0.0.
Current install is rsl_rl >= 4.0.0 which has a different checkpoint format.
**Workaround:** Load `model_state_dict["actor.*"]` directly into `nn.Sequential`, bypass `OnPolicyRunner`.
If a new checkpoint is trained with current rsl_rl, use standard `OnPolicyRunner.load()`.

---

## How to Run

```bash
# Greenhouse inspection — contact sensor + camera (GUI)
./isaaclab.sh -p scripts/greenhouse_sim.py --enable_cameras

# Greenhouse inspection — contact sensor only, headless
./isaaclab.sh -p scripts/greenhouse_sim.py --headless

# Greenhouse inspection — full sensors, headless
./isaaclab.sh -p scripts/greenhouse_sim.py --headless --enable_cameras

# G1 locomotion with keyboard control (GUI required)
./isaaclab.sh -p scripts/demos/g1_locomotion.py

# G1 locomotion headless (policy runs, no keyboard/camera)
./isaaclab.sh -p scripts/demos/g1_locomotion.py --headless
```

> **Environment note:** Always activate the Isaac Sim environment before running:
> `source /home/trooperai/isaac-env/bin/activate` or set `VIRTUAL_ENV=/home/trooperai/isaac-env`
