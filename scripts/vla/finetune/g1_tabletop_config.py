"""
GR00T NEW_EMBODIMENT modality config for our IsaacLab sim G1 (tabletop tasks).

This defines OUR robot's observation/action contract for fine-tuning — we are
NOT using the pretrained REAL_G1 schema (whole-body bimanual, real-world data).
Instead we register a minimal single-arm + gripper embodiment that matches the
demonstrations we record in sim, and fine-tune GR00T onto it.

Passed to fine-tuning via:
    --embodiment-tag NEW_EMBODIMENT
    --modality-config-path scripts/vla/finetune/g1_tabletop_config.py

Embodiment (6-DOF, mirrors the shipped examples/SO100 template):
    state/action vector layout (must match meta/modality.json in the dataset):
        [0:5] single_arm  = left_shoulder_pitch, _roll, _yaw, _elbow_pitch, _elbow_roll
        [5:6] gripper     = left_one_joint  (1 = closed-ish, follows G1 init pose)
    one camera: ego_view  (observation.images.ego_view)
    language:   annotation.human.task_description

Keep these names in sync with the recorder (scripts/vla/finetune/recorder.py).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Joint order for the 5-DOF single_arm group (G1_MINIMAL_CFG left arm).
SINGLE_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
]
GRIPPER_JOINT = "left_one_joint"
ACTION_HORIZON = 16

g1_tabletop_config = {
    # One ego-view camera; current frame only.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    # Current proprioception; keys must match meta/modality.json "state" groups.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    # Predict ACTION_HORIZON future steps.
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # single_arm: RELATIVE joint deltas (better generalization).
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE target (open/close works better absolute).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Custom embodiments MUST register under NEW_EMBODIMENT.
register_modality_config(g1_tabletop_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
