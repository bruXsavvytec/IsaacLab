"""
Shared action and mode definitions for the VLA inspection project.
"""

from enum import Enum


class InspectionAction(str, Enum):
    """High-level actions the VLA planner can output."""
    CONTINUE    = "CONTINUE"    # keep inspecting, more frames needed
    HEALTHY     = "HEALTHY"     # plant looks healthy — retract and move on
    STRESSED    = "STRESSED"    # plant shows disease/stress — flag for attention
    REPOSITION  = "REPOSITION"  # camera angle unclear — adjust arm


class VLAMode(str, Enum):
    SCRIPTED = "scripted"   # original hardcoded behavior (fallback / baseline)
    CLAUDE   = "claude"     # Phase 1: Claude API as high-level planner
    GROOT    = "groot"      # Phase 2: nvidia/GR00T-N1.7-3B (Unitree G1 native)
