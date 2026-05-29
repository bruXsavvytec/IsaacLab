"""
Phase 2 VLA — NVIDIA GR00T N1.7-3B

HuggingFace:  nvidia/GR00T-N1.7-3B
Embodiment:   EmbodimentTag.REAL_G1  (zero-shot with base model, no fine-tuning)
Output space: relative joint deltas for left arm (7 DOF) + relative EEF delta

Why GR00T over OpenVLA:
  - Explicitly validated on Unitree G1 hardware
  - Output is already in joint space — no IK wrapper needed
  - Isaac Sim native (NVIDIA ecosystem)
  - Only 3B params vs 7B → fits easily alongside Isaac Sim on 24 GB

Installation (do once before running):
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T /home/trooperai/Isaac-GR00T
    cd /home/trooperai/Isaac-GR00T && pip install -e .

The policy is loaded once in load() and is stateless between steps (call reset()
between episodes as good practice).
"""

import numpy as np

_MODEL_ID    = "nvidia/GR00T-N1.7-3B"
_DEVICE      = "cuda:0"
_IMG_SIZE    = 224     # GR00T vision encoder expects 224×224

_DEFAULT_INSTRUCTION = (
    "Inspect the plant leaf cluster for signs of disease. "
    "Look for yellowing or brown patches on the leaves."
)

# Left-arm joint names in the order GR00T's REAL_G1 embodiment expects (7 DOF).
# These match the joint names in G1_MINIMAL_CFG.
_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]


class GR00TRunner:
    """Wrapper around Gr00tPolicy (direct) or PolicyClient (server mode).

    Server mode is strongly recommended for development — the model loads once
    and stays in GPU memory. The sim can restart instantly without reloading weights.

    Start the server in a separate terminal (do this once):
        cd /home/trooperai/Isaac-GR00T
        python gr00t/eval/run_gr00t_server.py \
            --model-path nvidia/GR00T-N1.7-3B \
            --embodiment-tag REAL_G1 \
            --server-port 5555

    Then run the sim normally — it will auto-connect to the server.
    """

    def __init__(self, instruction: str = _DEFAULT_INSTRUCTION,
                 server_host: str = "localhost", server_port: int = 5555):
        self._instruction  = instruction
        self._server_host  = server_host
        self._server_port  = server_port
        self._policy       = None
        self._arm_indices: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Connect to a running GR00T server, or fall back to loading directly.

        Server mode (preferred): model stays in GPU across sim restarts.
        Direct mode: loads 3B params from disk every run (~30-60 s).
        """
        try:
            from gr00t.policy import Gr00tPolicy
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy.server_client import PolicyClient
        except ImportError:
            raise ImportError(
                "gr00t package not found.\n"
                "  See /home/trooperai/Isaac-GR00T/groot.pth\n"
            )

        # Try server first
        try:
            client = PolicyClient(
                host=self._server_host,
                port=self._server_port,
            )
            client.get_action  # probe — raises if not connected
            self._policy = client
            print(f"[VLA] GR00T connected to server at {self._server_host}:{self._server_port}")
            return
        except Exception:
            print(f"[VLA] No GR00T server at {self._server_host}:{self._server_port} — loading directly")
            print("[VLA] TIP: start the server once to avoid reloading weights every run:")
            print(f"[VLA]   cd /home/trooperai/Isaac-GR00T && python gr00t/eval/run_gr00t_server.py \\")
            print(f"[VLA]       --model-path {_MODEL_ID} --embodiment-tag REAL_G1 --server-port {self._server_port}")

        # Fall back to direct load
        print(f"[VLA] Loading GR00T N1.7-3B directly (REAL_G1, zero-shot)...")
        self._policy = Gr00tPolicy(
            model_path=_MODEL_ID,
            embodiment_tag=EmbodimentTag.REAL_G1,
            device=_DEVICE,
            strict=True,
        )
        cfg = self._policy.get_modality_config()
        print(f"[VLA] GR00T ready | modality keys: {list(cfg.keys())}")

    def bind_robot(self, joint_names: list[str]) -> None:
        """Map G1's 37 joint names to the 7-DOF arm slice GR00T controls."""
        self._arm_indices = {
            name: idx
            for idx, name in enumerate(joint_names)
            if name in _ARM_JOINT_NAMES
        }
        missing = [n for n in _ARM_JOINT_NAMES if n not in self._arm_indices]
        if missing:
            print(f"[VLA] WARNING: arm joints not found in robot: {missing}")
        else:
            print(f"[VLA] Arm joints bound: {list(self._arm_indices.keys())}")

    def reset(self) -> None:
        if self._policy is not None:
            self._policy.reset()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, rgb_np: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        """Run one GR00T inference step.

        Args:
            rgb_np:    (H, W, 3) uint8 RGB — robot camera frame (any resolution)
            joint_pos: (37,) float32 — current G1 joint positions in radians

        Returns:
            delta: (37,) float32 — joint position deltas; non-arm joints are zero.
                   Apply as:  new_target = current_joint_pos + delta
        """
        if self._policy is None:
            raise RuntimeError("Call load() before act()")

        img_224  = self._resize(rgb_np)
        arm_pos  = self._extract_arm_state(joint_pos)
        obs      = self._build_obs(img_224, arm_pos)

        action, _ = self._policy.get_action(obs)

        # action["joint_position"] → (1, T, 7) relative deltas; take first step
        delta_arm = action.get("joint_position", np.zeros((1, 1, 7), dtype=np.float32))
        delta_arm = np.asarray(delta_arm[0, 0], dtype=np.float32)   # (7,)

        return self._expand_to_full(delta_arm)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(rgb_np: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(rgb_np, (_IMG_SIZE, _IMG_SIZE))

    def _extract_arm_state(self, joint_pos: np.ndarray) -> np.ndarray:
        """Pull 7 arm joint angles out of the 37-DOF G1 state vector."""
        arm = np.zeros(7, dtype=np.float32)
        for j, name in enumerate(_ARM_JOINT_NAMES):
            if name in self._arm_indices:
                arm[j] = joint_pos[self._arm_indices[name]]
        return arm

    def _build_obs(self, img_224: np.ndarray, arm_pos: np.ndarray) -> dict:
        """Assemble the observation dict in the format Gr00tPolicy expects."""
        return {
            "video": {
                # (B=1, T=1, H=224, W=224, C=3) uint8
                "ego_view": img_224[None, None].astype(np.uint8),
            },
            "state": {
                "eef_9d":           np.zeros((1, 1, 9), dtype=np.float32),
                "gripper_position": np.zeros((1, 1, 1), dtype=np.float32),
                "joint_position":   arm_pos[None, None],   # (1, 1, 7)
            },
            "language": {
                # (B=1, T=1) list of lists of strings
                "task": [[self._instruction]],
            },
        }

    def _expand_to_full(self, delta_arm: np.ndarray) -> np.ndarray:
        """Map 7-DOF arm delta back to a 37-DOF G1 delta vector."""
        delta = np.zeros(37, dtype=np.float32)
        for j, name in enumerate(_ARM_JOINT_NAMES):
            if name in self._arm_indices:
                delta[self._arm_indices[name]] = delta_arm[j]
        return delta
