from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
import json
import re

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
    # Build the "acts" string based on how many acts we have.
    if not reaction_plan.acts:
        acts_text = "acknowledge"
    elif len(reaction_plan.acts) == 1:
        acts_text = reaction_plan.acts[0]
    else:
        acts_text = " + ".join(reaction_plan.acts)

    # Prepare slots using the reaction plan (template is a string).
    slots = {
        "prompt": normalize_token(prompt),
        "tone": normalize_token(reaction_plan.tone),
        "acts": normalize_token(acts_text),
        "intensity": normalize_token(reaction_plan.intensity),
        "format": normalize_token(reaction_plan.format),
    }

    try:
        return template.format_map(slots)
    except KeyError as exc:
        raise ValueError(f"template is missing slot: {exc}") from exc
    


def remove_emojis(text: str) -> str:
    """Remove most emoji and pictographic unicode characters."""
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, "", text)


def apply_constraints(text: str, constraints: GenerationConstraints) -> str:
    """Enforce output constraints (length, banned patterns)."""
    if constraints.forbid_mentions:
        # Remove tokens starting with @
        text = re.sub(r"@\w+", "", text)
    if constraints.forbid_hashtags:
        # Remove tokens starting with #
        text = re.sub(r"#\w+", "", text)

    # Remove emojis and normalize whitespace.
    text = remove_emojis(text)
    text = " ".join(text.split()).strip()

    # Truncate to max_chars without cutting mid-word when possible.
    if len(text) > constraints.max_chars:
        cut = text[:constraints.max_chars].rstrip()
        last_space = cut.rfind(" ")
        return cut if last_space == -1 else cut[:last_space]
    return text


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
    # More detailed path:
    # - If templates is empty, raise ValueError.
    # - Pick a template deterministically (e.g., hash(prompt) % len(templates))
    #   or randomly for variety.
    # - Render with render_from_template(prompt, reaction_plan, template).
    # - If constraints is None, use GenerationConstraints().
    # - Call apply_constraints and return the final string.
    raise NotImplementedError("TODO: implement response generation")
