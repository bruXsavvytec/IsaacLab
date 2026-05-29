"""
Phase 1 VLA — Claude Sonnet 4.6 as high-level visual planner.

Each call to ClaudePlanner.decide() sends the robot's camera frame + sensor
readings to Claude and gets back a structured inspection decision.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   (from console.anthropic.com)

Prompt caching is applied to the system prompt (ephemeral, 5-min TTL).
Typical latency: ~1.5 s first call, ~0.4 s cached.
"""

import base64
import io
import os
import re

import numpy as np

from .action_space import InspectionAction

_SYSTEM_PROMPT = """You are the high-level inspection planner for a Unitree G1 humanoid robot \
inspecting plants inside a greenhouse.

You receive:
  - A 640×480 RGB image from the robot's head camera (torso-mounted, facing forward).
  - Numeric sensor readings: contact force and colour-based health percentages.

Your task is to decide what the robot should do next during the arm-reach inspection phase.

Available actions and when to use them:
  CONTINUE   — You can see the plant but need more frames to be sure. Use if image is clear \
but inconclusive.
  HEALTHY    — The plant looks healthy (vibrant green, no yellowing, no brown spots). \
Robot should retract arm and move on.
  STRESSED   — The plant shows signs of stress: yellowing leaves, brown patches, \
discolouration, wilting. Robot should retract arm and flag this plant for human attention.
  REPOSITION — The camera view is too dark, obscured, or the plant is barely visible. \
Robot should adjust its arm angle.

Respond with exactly one line in this format:
  ACTION: <action> | REASON: <one sentence explanation, max 20 words>

Example:
  ACTION: STRESSED | REASON: Clear yellowing on leaf clusters indicates nitrogen deficiency.
"""

_MODEL = "claude-sonnet-4-6"


class ClaudePlanner:
    """Wraps the Anthropic API for greenhouse plant inspection decisions."""

    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed.\n"
                "  pip install anthropic\n"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set.\n"
                "  Get a key at: https://console.anthropic.com/settings/api-keys\n"
                "  Then: export ANTHROPIC_API_KEY=sk-ant-...\n"
                "Note: claude.ai subscription is separate — API access requires its own key."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self._call_count = 0
        print(f"[VLA] ClaudePlanner ready (model={_MODEL})")

    def decide(
        self,
        rgb_np: np.ndarray,
        contact_force: float,
        healthy_pct: float,
        stressed_pct: float,
    ) -> tuple[InspectionAction, str]:
        """Send one camera frame + sensor readings to Claude and parse the response.

        Returns (action, reason). Falls back to CONTINUE on any API error.
        """
        image_b64 = self._encode(rgb_np)
        sensor_text = (
            f"Contact force: {contact_force:.1f} N\n"
            f"Colour health: green={healthy_pct:.0f}%, stressed/yellow={stressed_pct:.0f}%"
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=128,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Sensor readings:\n{sensor_text}\n\nDecide:",
                            },
                        ],
                    }
                ],
            )
            self._call_count += 1
            raw = response.content[0].text.strip()
            return self._parse(raw)

        except Exception as exc:
            print(f"[VLA] Claude API error: {exc} — defaulting to CONTINUE")
            return InspectionAction.CONTINUE, f"API error: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(rgb_np: np.ndarray) -> str:
        """RGB numpy array → base64-encoded JPEG string."""
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb_np)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _parse(text: str) -> tuple[InspectionAction, str]:
        """Parse 'ACTION: X | REASON: Y' response. Falls back to CONTINUE."""
        m = re.search(r"ACTION:\s*(\w+)", text, re.IGNORECASE)
        r = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
        action_str = m.group(1).upper() if m else "CONTINUE"
        reason     = r.group(1).strip() if r else text

        try:
            action = InspectionAction(action_str)
        except ValueError:
            action = InspectionAction.CONTINUE

        print(f"[VLA] Claude → {action.value}: {reason}")
        return action, reason
