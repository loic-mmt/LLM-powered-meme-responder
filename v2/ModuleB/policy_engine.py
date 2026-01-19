from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import json
from pathlib import Path


@dataclass(frozen=True)
class ReactionPlan:
    tone: list[str]
    acts: list[str]
    intensity: str
    format: str


def load_policy_rules(path: str) -> list[dict]:
    """Load rule definitions that map prompt tags to reaction tags."""
    # Minimal rule schema:
    # {
    #   "when": ["sad", "self_deprecating"],
    #   "tone": "supportive",
    #   "acts": ["reassure"],
    #   "intensity": "low",
    #   "format": "short",
    #   "weight": 1.0
    # }

    # Read and parse JSON array of rules from disk.
    rule_path = Path(path)
    data = json.loads(rule_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("policy rules file must contain a JSON array")

    # Validate and normalize each rule.
    normalized_rules: list[dict] = []
    for idx, rule in enumerate(data):
        # Check required structure and types first.
        if not isinstance(rule, dict):
            raise ValueError(f"rule {idx} must be an object")
        for key in ("when", "tone", "acts", "intensity", "format"):
            if key not in rule:
                raise ValueError(f"rule {idx} missing required key: {key}")

        when = rule["when"]
        acts = rule["acts"]
        if not isinstance(when, list) or not all(isinstance(t, str) for t in when):
            raise ValueError(f"rule {idx} 'when' must be a list of strings")
        if not isinstance(acts, list) or not all(isinstance(t, str) for t in acts):
            raise ValueError(f"rule {idx} 'acts' must be a list of strings")
        if not isinstance(rule["tone"], str):
            raise ValueError(f"rule {idx} 'tone' must be a string")
        if not isinstance(rule["intensity"], str):
            raise ValueError(f"rule {idx} 'intensity' must be a string")
        if not isinstance(rule["format"], str):
            raise ValueError(f"rule {idx} 'format' must be a string")

        # Optional weight controls rule strength.
        weight = rule.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"rule {idx} 'weight' must be a non-negative number")

        # Normalize tag strings to match Module A output.
        normalized_rules.append(
            {
                "when": [t.strip().lower() for t in when if t.strip()],
                "tone": rule["tone"].strip().lower(),
                "acts": [t.strip().lower() for t in acts if t.strip()],
                "intensity": rule["intensity"].strip().lower(),
                "format": rule["format"].strip().lower(),
                "weight": float(weight),
            }
        )

    return normalized_rules


def score_rules(prompt_tags: Sequence[str], rules: Iterable[dict]) -> list[tuple[dict, float]]:
    """Score rules for a given set of prompt tags."""
    # Use a set for fast overlap checks.
    prompt_set = {tag.strip().lower() for tag in prompt_tags if tag}
    scored: list[tuple[dict, float]] = []
    for rule in rules:
        when_tags = rule.get("when", [])
        if not when_tags:
            continue
        overlap = prompt_set.intersection(when_tags)
        base_score = len(overlap) / len(when_tags)
        weight = float(rule.get("weight", 1.0))
        scored.append((rule, base_score * weight))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def derive_reaction_plan(
    prompt_tags: Sequence[str],
    rules: Iterable[dict],
    threshold: float = 0.5,
) -> ReactionPlan:
    """Derive reaction tags (tone/acts/intensity/format) from prompt tags."""
    # Score all rules and keep the best match.
    scored = score_rules(prompt_tags=prompt_tags, rules=rules)
    if not scored:
        return ReactionPlan(tone="neutral", acts=["acknowledge"], intensity="low", format="short")

    best_rule, best_score = scored[0]
    # If the best score is too weak, fall back to neutral defaults.
    if best_score < threshold:
        return ReactionPlan(tone="neutral", acts=["acknowledge"], intensity="low", format="short")

    # Build the reaction plan from the selected rule.
    tone = best_rule.get("tone", "neutral")
    acts = list(best_rule.get("acts", ["acknowledge"]))
    intensity = best_rule.get("intensity", "low")
    fmt = best_rule.get("format", "short")

    return ReactionPlan(tone=tone, acts=acts, intensity=intensity, format=fmt)



def reaction_plan_to_tags(plan: ReactionPlan) -> list[str]:
    """Flatten a reaction plan into response tags."""
    # TODO: Suggested path:
    # 1) Standardize output tags, e.g.:
    #    ["tone:sarcastic", "act:roast", "intensity:high", "format:short"].
    # 2) Keep acts as multiple entries.
    # 3) Use this output to drive Module C and Module D.
    raise NotImplementedError("TODO: implement tag flattening")
