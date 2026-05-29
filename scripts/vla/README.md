# VLA Greenhouse Inspection

Unitree G1 humanoid inspecting plants using a Vision-Language-Action model.
The robot walks to a bush, extends its arm, and the VLA decides what to do based
on what the camera sees and a language instruction.

---

## Project Structure

```
scripts/vla/
├── main.py            — entry point, full sim loop + state machine
├── groot_runner.py    — Phase 2: GR00T N1.7-3B wrapper (server + direct mode)
├── planner.py         — Phase 1: Claude API high-level planner
├── action_space.py    — shared enums (InspectionAction, VLAMode)
└── README.md          — this file
```

External repos:
```
/home/trooperai/Isaac-GR00T/   — NVIDIA GR00T source (cloned from GitHub)
```

---

## How It Works

```
Camera frame (640×480 RGB)
        │
        ├─► Phase 1: Claude API
        │       Sends image + sensor readings to Claude Sonnet
        │       Gets back: HEALTHY / STRESSED / CONTINUE / REPOSITION
        │       Drives state machine transition
        │
        └─► Phase 2: GR00T N1.7-3B
                Two systems running together:
                  System 2 (slow): Cosmos-Reason2-2B VLM
                      image + language instruction → latent goal
                  System 1 (fast): DiT flow-matching policy
                      goal + joint state → joint position deltas
                Outputs 7-DOF left arm deltas every 20 frames
                Colour heuristic decides HEALTHY / STRESSED

RSL-RL locomotion policy runs every frame for legs (balance + walking).
GR00T / Claude only control the arm during the INSIDE phase.
```

State machine:
```
WALK → ARRIVE → REACH_IN → INSIDE → REACH_OUT → DONE
                                ↑
                         VLA decides when to exit
                         (or hard cap at 200 frames)
```

---

## Quick Start

### Baseline (no VLA, scripted behaviour)

```bash
cd /home/trooperai/IsaacLab
./isaaclab.sh -p scripts/vla/main.py --enable_cameras
```

### Phase 1 — Claude API planner

Requires an Anthropic API key. Get one at https://console.anthropic.com/settings/api-keys

> **Note:** Your claude.ai subscription is separate from API access.
> The API is pay-per-token and requires its own key.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./isaaclab.sh -p scripts/vla/main.py --enable_cameras --vla-mode claude
```

### Phase 2 — GR00T N1.7-3B

**Recommended: server mode** (model loads once, sim restarts instantly)

Terminal 1 — start server, leave it running:
```bash
source /home/trooperai/isaac-env/bin/activate
cd /home/trooperai/Isaac-GR00T
python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag REAL_G1 \
    --server-port 5555
```

Terminal 2 — run sim (connects to server automatically, no model reload):
```bash
cd /home/trooperai/IsaacLab
./isaaclab.sh -p scripts/vla/main.py --enable_cameras --vla-mode groot
```

If no server is running, the sim falls back to loading the model directly (~30-60 s).

---

## Model Details — GR00T N1.7-3B

| Property | Value |
|---|---|
| HuggingFace ID | `nvidia/GR00T-N1.7-3B` |
| Parameters | 3B |
| VRAM | ~16 GB (leaves ~8 GB for Isaac Sim on a 24 GB card) |
| Embodiment tag | `REAL_G1` (zero-shot, no fine-tuning needed) |
| Action space | 7-DOF left arm joint deltas (radians) |
| Action horizon | up to 16 steps |
| Language input | fixed instruction string in `groot_runner.py` |

First run downloads ~7 GB of weights to `~/.cache/huggingface/hub/`.
All subsequent runs load from cache.

**Changing the language instruction:**

Edit `_DEFAULT_INSTRUCTION` in [groot_runner.py](groot_runner.py):
```python
_DEFAULT_INSTRUCTION = (
    "Inspect the plant leaf cluster for signs of disease. "
    "Look for yellowing or brown patches on the leaves."
)
```

---

## Installation

These steps have already been completed on this machine.
Documented here for reproducibility.

### 1. Clone Isaac-GR00T

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T \
    /home/trooperai/Isaac-GR00T
```

### 2. Add to Isaac Sim Python path

```bash
echo "/home/trooperai/Isaac-GR00T" > \
    /home/trooperai/isaac-env/lib/python3.10/site-packages/groot.pth
```

### 3. Install dependencies into Isaac Sim env

```bash
/home/trooperai/isaac-env/bin/pip install \
    "tyro==0.9.17" \
    "typeguard==3.0.2" \
    "dm-tree" \
    "albumentations==1.4.18" \
    "av==16.1.0" \
    "diffusers==0.35.1" \
    "lmdb==1.7.5" \
    "msgpack==1.1.0" \
    "msgpack-numpy==0.4.8" \
    "pandas==2.2.3" \
    "peft==0.17.1" \
    "termcolor==3.2.0" \
    "click==8.1.8" \
    "einops==0.8.1" \
    "jsonlines==4.0.0" \
    "omegaconf==2.3.0" \
    "scipy==1.15.3" \
    "pyzmq==27.0.1" \
    "gymnasium==1.2.2" \
    "gitpython==3.1.46" \
    "boto3"
```

### 4. Verify

```bash
/home/trooperai/isaac-env/bin/python -c \
    "from gr00t.policy import Gr00tPolicy; print('gr00t OK')"
```

---

## Camera Preview

With `--enable_cameras`, annotated frames are saved to `/tmp/plant_inspector_latest.png`.

View live in a third terminal:
```bash
feh --auto-reload /tmp/plant_inspector_latest.png
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Scripted baseline | ✅ Done | Hard-coded arm reach, colour health analysis |
| Phase 1: Claude planner | ✅ Ready | Needs API key |
| Phase 2: GR00T zero-shot | ✅ Running | `--vla-mode groot` |
| Phase 3: Data collection | 📋 Next | Record scripted demos as training data |
| Phase 4: Fine-tuning | 📋 Future | Fine-tune GR00T on greenhouse inspection demos |
| Migrate to IsaacLab extension | 📋 Future | Move from `scripts/vla/` to `source/isaaclab_tasks/` |
| Separate repo | 📋 Future | Clean from Isaac Sim, support real hardware |
