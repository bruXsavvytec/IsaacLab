"""
Phase 2 VLA — NVIDIA GR00T N1.7-3B  (embodiment REAL_G1, zero-shot)

REAL_G1 is a WHOLE-BODY BIMANUAL humanoid policy. Verified I/O contract
(see scripts/vla/groot_probe.py, checkpoint processor_config.json + statistics.json):

  INPUT  observation
    video.ego_view              (B, 2, H, W, 3) uint8   # horizon 2: frame ~t-20 and now
    state.left_wrist_eef_9d     (B, 1, 9)  f32          # EEF pose xyz+rot6d
    state.right_wrist_eef_9d    (B, 1, 9)  f32
    state.left_hand/right_hand  (B, 1, 7)  f32
    state.left_arm/right_arm    (B, 1, 7)  f32          # arm joint positions
    state.waist                 (B, 1, 3)  f32
    language."annotation.human.task_description"  [[str]]

  OUTPUT action  (40-step chunk per key)
    left_arm / right_arm        (B, 40, 7)  RELATIVE joint deltas   <-- we use these
    left_hand / right_hand      (B, 40, 7)  ABSOLUTE
    left_wrist_eef_9d/right_..  (B, 40, 9)  RELATIVE eef
    waist (3) | base_height_command (1) | navigate_command (3)

SCOPE (this version): BOTH ARMS ONLY. We apply left_arm + right_arm targets to
the sim; legs stay on the RSL-RL locomotion policy; hands/waist/base/nav are
ignored for now. act() returns the full 40-step CHUNK mapped onto the robot's DOF
vector, and the caller replays it step by step.

NOTE on action semantics: although the embodiment name says "relative_joints",
get_action returns ABSOLUTE joint-position targets — the policy has already
converted relative→absolute against the arm `state` we pass in (verified: feeding
left_arm=0.5 makes the output track to ~0.5). So callers must ASSIGN the targets,
never accumulate them.

Server mode (recommended) — model stays in GPU across sim restarts:
    cd /home/trooperai/Isaac-GR00T
    python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.7-3B --embodiment-tag REAL_G1 --port 5555
"""

from collections import deque

import numpy as np

_MODEL_ID = "nvidia/GR00T-N1.7-3B"
_DEVICE   = "cuda:0"

_DEFAULT_INSTRUCTION = "pick up the blue cube and put it on top of the yellow cube"

# GR00T REAL_G1 video horizon is [-20, 0]; we keep ~20 frames of history.
_VIDEO_HISTORY = 20

# Identity 9D EEF pose: xyz=0, rot6d = first two columns of identity matrix.
_EEF_IDENTITY = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)

# Per-arm joint order GR00T's left_arm/right_arm (7 DOF) is assumed to follow
# (Unitree G1 convention). Each slot lists acceptable joint-name suffixes so we
# tolerate the elbow/wrist naming differences between IsaacLab G1 configs.
_ARM_SLOTS = [
    ["{s}_shoulder_pitch_joint"],
    ["{s}_shoulder_roll_joint"],
    ["{s}_shoulder_yaw_joint"],
    ["{s}_elbow_joint", "{s}_elbow_pitch_joint"],
    ["{s}_wrist_roll_joint", "{s}_elbow_roll_joint"],
    ["{s}_wrist_pitch_joint"],
    ["{s}_wrist_yaw_joint"],
]


class GR00TRunner:
    """REAL_G1 GR00T wrapper. Server mode (PolicyClient) preferred, else direct load."""

    def __init__(self, instruction: str = _DEFAULT_INSTRUCTION,
                 server_host: str = "localhost", server_port: int = 5555):
        self._instruction = instruction
        self._server_host = server_host
        self._server_port = server_port
        self._policy      = None
        self._dof         = 0
        self._left_map: list[int | None]  = []   # 7 entries, robot DOF index or None
        self._right_map: list[int | None] = []
        self._frames: deque = deque(maxlen=_VIDEO_HISTORY)

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        try:
            from gr00t.policy import Gr00tPolicy
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy.server_client import PolicyClient
        except ImportError:
            raise ImportError("gr00t package not found. See /home/trooperai/Isaac-GR00T/groot.pth")

        try:
            client = PolicyClient(host=self._server_host, port=self._server_port)
            client.get_modality_config()           # probe — raises if not connected
            self._policy = client
            print(f"[VLA] GR00T connected to server at {self._server_host}:{self._server_port}")
            return
        except Exception:
            print(f"[VLA] No GR00T server at {self._server_host}:{self._server_port} — loading directly")

        print("[VLA] Loading GR00T N1.7-3B directly (REAL_G1, zero-shot)...")
        self._policy = Gr00tPolicy(
            model_path=_MODEL_ID, embodiment_tag=EmbodimentTag.REAL_G1,
            device=_DEVICE, strict=True,
        )
        print("[VLA] GR00T ready.")

    def bind_robot(self, joint_names: list[str]) -> None:
        """Map GR00T's 7-DOF left/right arm order onto this robot's DOF indices."""
        self._dof = len(joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}

        def build(side: str) -> list[int | None]:
            out: list[int | None] = []
            for slot in _ARM_SLOTS:
                idx = None
                for cand in slot:
                    nm = cand.format(s=side)
                    if nm in name_to_idx:
                        idx = name_to_idx[nm]; break
                out.append(idx)
            return out

        self._left_map  = build("left")
        self._right_map = build("right")
        n_l = sum(i is not None for i in self._left_map)
        n_r = sum(i is not None for i in self._right_map)
        print(f"[VLA] Arm mapping — left {n_l}/7, right {n_r}/7 joints bound (DOF={self._dof})")
        if n_l < 7 or n_r < 7:
            print(f"[VLA] WARNING: unmapped arm slots — left={self._left_map} right={self._right_map}")

    def arm_dof_indices(self) -> list[int]:
        """Robot DOF indices of all bound arm joints (both arms)."""
        return [i for i in (self._left_map + self._right_map) if i is not None]

    def reset(self) -> None:
        self._frames.clear()
        if self._policy is not None and hasattr(self._policy, "reset"):
            try:
                self._policy.reset()
            except Exception:
                pass

    def push_frame(self, rgb_np: np.ndarray) -> None:
        """Feed one camera frame into the history buffer (call every sim step so
        the t-20 video horizon is temporally meaningful)."""
        self._frames.append(rgb_np.astype(np.uint8))

    # ------------------------------------------------------------------ inference
    def act(self, rgb_np: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        """Run one GR00T inference.

        Args:
            rgb_np:    (H, W, 3) uint8 — current camera frame.
            joint_pos: (DOF,) float32 — current robot joint positions.

        Returns:
            chunk: (T, DOF) float32 — ABSOLUTE joint-position targets for the
                   predicted horizon, non-arm joints zero. Execute sequentially:
                   target = chunk[step]  (assign, do NOT accumulate).
        """
        if self._policy is None:
            raise RuntimeError("Call load() before act()")

        self._frames.append(rgb_np.astype(np.uint8))   # ensure current frame present
        obs = self._build_obs(joint_pos)

        result = self._policy.get_action(obs)
        action = result[0] if isinstance(result, tuple) else result

        left  = np.asarray(action["left_arm"])      # (1, T, 7) or (T, 7)
        right = np.asarray(action["right_arm"])
        left  = left[0]  if left.ndim  == 3 else left
        right = right[0] if right.ndim == 3 else right
        T = left.shape[0]

        chunk = np.zeros((T, self._dof), dtype=np.float32)
        for slot, gidx in enumerate(self._left_map):
            if gidx is not None:
                chunk[:, gidx] = left[:, slot]
        for slot, gidx in enumerate(self._right_map):
            if gidx is not None:
                chunk[:, gidx] = right[:, slot]
        return chunk

    # ------------------------------------------------------------------ helpers
    def _build_obs(self, joint_pos: np.ndarray) -> dict:
        # video: frame ~t-20 and current, (1, 2, H, W, 3) uint8
        past    = self._frames[0]              # oldest available (≈ t-20)
        current = self._frames[-1]
        ego     = np.stack([past, current])[None].astype(np.uint8)

        def arm_state(arm_map):
            v = np.zeros(7, dtype=np.float32)
            for slot, gidx in enumerate(arm_map):
                if gidx is not None:
                    v[slot] = joint_pos[gidx]
            return v.reshape(1, 1, 7)

        state = {
            "left_wrist_eef_9d":  _EEF_IDENTITY.reshape(1, 1, 9).copy(),
            "right_wrist_eef_9d": _EEF_IDENTITY.reshape(1, 1, 9).copy(),
            "left_hand":  np.zeros((1, 1, 7), dtype=np.float32),
            "right_hand": np.zeros((1, 1, 7), dtype=np.float32),
            "left_arm":  arm_state(self._left_map),
            "right_arm": arm_state(self._right_map),
            "waist": np.zeros((1, 1, 3), dtype=np.float32),
        }
        return {
            "video": {"ego_view": ego},
            "state": state,
            "language": {"annotation.human.task_description": [[self._instruction]]},
        }
