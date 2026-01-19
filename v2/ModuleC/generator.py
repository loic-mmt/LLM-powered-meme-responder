from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
import json

from v2.ModuleB.policy_engine import ReactionPlan, normalize_token



@dataclass(frozen=True)
class GenerationConstraints:
    max_chars: int = 120
    forbid_mentions: bool = True
    forbid_hashtags: bool = True


def load_templates(path: str) -> list[str]:
    """Load response templates with slots."""
    templates_path = Path(path)
    raw = templates_path.read_text(encoding="utf-8")

    # Accept JSON list or newline-delimited text.
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("templates file must contain a JSON array")
        candidates = [str(item) for item in data]
    except json.JSONDecodeError:
        candidates = [line.strip() for line in raw.splitlines()]

    # Normalize whitespace and drop empty entries.
    templates = [normalize_token(t) for t in candidates if t.strip()]
    if not templates:
        raise ValueError("no templates found")

    # Validate required slots at least once to ensure compatibility.
    required_slots = ("{tone}", "{acts}", "{intensity}", "{format}")
    for slot in required_slots:
        if not any(slot in template for template in templates):
            raise ValueError(f"at least one template must contain {slot}")

    return templates


def render_from_template(
    prompt: str,
    reaction_plan: ReactionPlan,
    template: str,
) -> str:
    """Render a response using a template and reaction plan."""
    # TODO: Suggested path:
    # 1) Build a slot dict with: prompt, tone, intensity, format, acts.
    # 2) Convert acts list to a readable string:
    #    - if empty -> "acknowledge"
    #    - if one -> that act
    #    - if many -> "act1 + act2 + act3"
    # 3) Normalize all slot values (lowercase, trim).
    # 4) Render with template.format_map(slots).
    # 5) If a slot is missing, either:
    #    - fill with "" or a default value, or
    #    - raise a ValueError with the missing key.
    raise NotImplementedError("TODO: implement template rendering")


def apply_constraints(text: str, constraints: GenerationConstraints) -> str:
    """Enforce output constraints (length, banned patterns)."""
    # TODO: Suggested path:
    # 1) Remove @mentions and #hashtags if forbidden.
    # 2) Collapse multiple spaces and strip.
    # 3) Truncate to max_chars without cutting mid-word if possible.
    raise NotImplementedError("TODO: implement constraint enforcement")


def generate_response(
    prompt: str,
    reaction_plan: ReactionPlan,
    templates: Sequence[str],
    constraints: GenerationConstraints | None = None,
) -> str:
    """Generate the final response text."""
    # TODO: Suggested path:
    # 1) Choose a template (random or rule-based).
    # 2) Render text with render_from_template.
    # 3) Optionally pass through LLM for paraphrase.
    # 4) Apply constraints and return final string.
    raise NotImplementedError("TODO: implement response generation")
