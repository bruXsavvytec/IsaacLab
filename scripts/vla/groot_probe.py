"""
GR00T observability probe — talks directly to the running GR00T server and
prints exactly what the REAL_G1 policy ingests and emits. No Isaac Sim needed.

This is a diagnostic, not part of the sim loop. It exists to answer:
"what does GR00T actually output, and what does it expect as input?"

Run (with the server already up on :5555):
    /home/trooperai/isaac-env/bin/python scripts/vla/groot_probe.py
    /home/trooperai/isaac-env/bin/python scripts/vla/groot_probe.py --image /tmp/groot_tabletop_latest.png

Ground-truth REAL_G1 schema (from the checkpoint's processor_config.json + statistics.json):

  INPUT observation
    video.ego_view              (B, 2, H, W, 3) uint8   # horizon 2: frames t-20 and t=0
    state.left_wrist_eef_9d     (B, 1, 9)  f32          # EEF pose: xyz + 6D rotation
    state.right_wrist_eef_9d    (B, 1, 9)  f32
    state.left_hand             (B, 1, 7)  f32          # dexterous hand joints
    state.right_hand            (B, 1, 7)  f32
    state.left_arm              (B, 1, 7)  f32          # arm joints
    state.right_arm             (B, 1, 7)  f32
    state.waist                 (B, 1, 3)  f32
    language."annotation.human.task_description"  [[str]]   # one string per batch item

  OUTPUT action  (40-step chunk per key — RELATIVE arm/eef deltas, ABSOLUTE hand/waist/base/nav)
    left_wrist_eef_9d (9) | right_wrist_eef_9d (9) | left_hand (7) | right_hand (7)
    left_arm (7) | right_arm (7) | waist (3) | base_height_command (1) | navigate_command (3)

NOTE: this is a *whole-body bimanual* policy. groot_runner.py currently assumes a
single 7-DOF left arm with keys eef_9d/gripper_position/joint_position — that schema
belongs to the DROID/OXE embodiment, NOT REAL_G1. That mismatch is why the sim arm
did nothing meaningful. This probe is the reference for fixing groot_runner.py.
"""

import argparse

import numpy as np

_INSTR = "pick up the blue cube and put it on top of the yellow cube"

# (key, dim) for REAL_G1 state, in config order.
_STATE_SPEC = [
    ("left_wrist_eef_9d", 9), ("right_wrist_eef_9d", 9),
    ("left_hand", 7), ("right_hand", 7),
    ("left_arm", 7), ("right_arm", 7), ("waist", 3),
]
# Identity 9D EEF pose: xyz=0, rot6d = first two columns of the identity matrix.
_EEF_IDENTITY = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def build_observation(image_path: str | None, instruction: str) -> dict:
    # --- video: (B=1, T=2, H, W, 3) uint8 ---
    if image_path:
        from PIL import Image
        frame = np.asarray(Image.open(image_path).convert("RGB").resize((224, 224)),
                           dtype=np.uint8)
        print(f"[probe] ego_view from {image_path}")
    else:
        # synthetic gradient so the model sees *something* structured
        gx = np.linspace(0, 255, 224, dtype=np.uint8)
        frame = np.stack([np.tile(gx, (224, 1)),
                          np.tile(gx[::-1], (224, 1)),
                          np.full((224, 224), 128, np.uint8)], axis=-1)
        print("[probe] ego_view = synthetic gradient (no --image given)")
    ego = np.stack([frame, frame])[None]          # (1, 2, 224, 224, 3)

    # --- state: each (B=1, T=1, D) f32 ---
    state = {}
    for key, dim in _STATE_SPEC:
        if key.endswith("eef_9d"):
            state[key] = _EEF_IDENTITY.reshape(1, 1, 9).copy()
        else:
            state[key] = np.zeros((1, 1, dim), dtype=np.float32)

    return {
        "video": {"ego_view": ego},
        "state": state,
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def describe(name: str, arr) -> None:
    a = np.asarray(arr)
    flat = a.astype(np.float64).ravel()
    print(f"  {name:24s} shape={str(a.shape):16s} dtype={str(a.dtype):8s} "
          f"min={flat.min():+.3f} max={flat.max():+.3f} mean={flat.mean():+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--image", default=None, help="PNG to use as ego_view (else synthetic)")
    ap.add_argument("--instruction", default=_INSTR)
    args = ap.parse_args()

    from gr00t.policy.server_client import PolicyClient

    print(f"[probe] connecting to GR00T server {args.host}:{args.port} ...")
    client = PolicyClient(host=args.host, port=args.port)

    # What the server says it expects:
    try:
        mc = client.get_modality_config()
        print("\n=== server modality_config (what it expects) ===")
        for mod, cfg in mc.items():
            keys = getattr(cfg, "modality_keys", cfg)
            di   = getattr(cfg, "delta_indices", "?")
            print(f"  {mod:10s} keys={keys}  delta_indices_len={len(di) if di!='?' else '?'}")
    except Exception as e:
        print(f"[probe] get_modality_config unavailable: {e}")

    obs = build_observation(args.image, args.instruction)
    print("\n=== observation we send ===")
    for k, v in obs["video"].items():    describe(f"video.{k}", v)
    for k, v in obs["state"].items():    describe(f"state.{k}", v)
    print(f"  language                 {obs['language']['annotation.human.task_description']}")

    print("\n[probe] calling get_action ...")
    result = client.get_action(obs)
    action = result[0] if isinstance(result, tuple) else result

    print("\n=== ACTION returned by GR00T ===")
    if isinstance(action, dict):
        for k in sorted(action):
            describe(k, action[k])
        # Show the first predicted step of the left/right arm chunk if present.
        for armkey in [k for k in action if "left_arm" in k or "right_arm" in k]:
            chunk = np.asarray(action[armkey])
            print(f"\n  {armkey} chunk shape {chunk.shape} — step[0] = "
                  f"{np.array2string(chunk.reshape(-1, chunk.shape[-1])[0], precision=3)}")
    else:
        print(f"  unexpected action type: {type(action)} -> {action}")


if __name__ == "__main__":
    main()
