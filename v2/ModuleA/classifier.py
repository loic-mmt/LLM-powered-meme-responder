from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class TagPrediction:
    tag: str
    score: float


@dataclass(frozen=True)
class MultiLabelModel:
    vectorizer: TfidfVectorizer
    classifier: OneVsRestClassifier
    tag_dictionary: list[str]


@dataclass(frozen=True)
class MultiClassModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    class_labels: list[str]


@dataclass(frozen=True)
class GroupedClassifier:
    tone: MultiLabelModel
    act: MultiLabelModel
    intensity: MultiClassModel
    format: MultiClassModel


def load_tag_dictionary(path: str) -> list[str]:
    """Load the allowed tag dictionary from disk."""
    # TODO: Parse tags.jsonl or a similar source and return a stable tag list.
    items: list[str] = []
    seen: set[str] = set()
    tag_path = Path(path)
    with tag_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                values = obj.values()
            else:
                values = [obj]
            for value in values:
                if isinstance(value, list):
                    candidates = value
                else:
                    candidates = [value]
                for candidate in candidates:
                    if not isinstance(candidate, str):
                        continue
                    if candidate not in seen:
                        seen.add(candidate)
                        items.append(candidate)
    return items


def load_tag_groups(path: str) -> dict[str, list[str]]:
    """Load tag groups (tone/act/intensity/format) from tags.jsonl."""
    groups: dict[str, list[str]] = {}
    tag_path = Path(path)
    with tag_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, list):
                        groups[key] = [v for v in value if isinstance(v, str)]
    return groups


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt text before vectorization or inference."""
    # TODO: Add consistent normalization (lowercasing, punctuation cleanup).
    prompt = prompt.lower().strip()
    prompt = re.sub(r"[^\w\s']", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt


def _train_multilabel(
    normalized_texts: Sequence[str],
    labels: Sequence[Sequence[str]],
    tag_dictionary: Sequence[str],
) -> MultiLabelModel:
    # Build tag index and multi-hot labels for multi-label training.
    tag_list = list(tag_dictionary)
    tag_to_index = {tag: idx for idx, tag in enumerate(tag_list)}

    y = np.zeros((len(labels), len(tag_list)), dtype=np.float32)
    for row, sample_labels in enumerate(labels):
        for label in sample_labels:
            idx = tag_to_index.get(label)
            if idx is not None:
                y[row, idx] = 1.0

    # TF-IDF baseline with One-vs-Rest logistic regression.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=200))
    x = vectorizer.fit_transform(normalized_texts)
    classifier.fit(x, y)
    return MultiLabelModel(vectorizer=vectorizer, classifier=classifier, tag_dictionary=tag_list)


def _train_multiclass(
    normalized_texts: Sequence[str],
    labels: Sequence[str],
    class_labels: Sequence[str],
) -> MultiClassModel:
    # Map each label to a class index for multi-class training.
    class_list = list(class_labels)
    class_to_index = {label: idx for idx, label in enumerate(class_list)}

    y = np.array([class_to_index[label] for label in labels], dtype=np.int64)
    # TF-IDF baseline with multinomial logistic regression.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    classifier = LogisticRegression(max_iter=200)
    x = vectorizer.fit_transform(normalized_texts)
    classifier.fit(x, y)
    return MultiClassModel(vectorizer=vectorizer, classifier=classifier, class_labels=class_list)


def train_grouped_classifiers(
    texts: Sequence[str],
    labels: Sequence[dict[str, object]],
    tag_groups: dict[str, list[str]],
) -> GroupedClassifier:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("texts cannot be empty")

    # Normalize once so all heads share the same preprocessing.
    normalized = [normalize_prompt(text) for text in texts]

    # Split labels per group so each head sees only its own targets.
    tone_labels = [list(row.get("tags_ton", [])) for row in labels]
    act_labels = [list(row.get("tags_act", [])) for row in labels]
    intensity_labels = [row.get("tags_intensity") for row in labels]
    format_labels = [row.get("tags_format") for row in labels]

    if any(label is None for label in intensity_labels):
        raise ValueError("tags_intensity must be provided for every sample")
    if any(label is None for label in format_labels):
        raise ValueError("tags_format must be provided for every sample")

    # Two multi-label heads for tone and act.
    tone_model = _train_multilabel(normalized, tone_labels, tag_groups.get("tags_ton", []))
    act_model = _train_multilabel(normalized, act_labels, tag_groups.get("tags_act", []))
    # Two multi-class heads for intensity and format (single choice).
    intensity_model = _train_multiclass(
        normalized,
        [str(label) for label in intensity_labels],
        tag_groups.get("tags_intensity", []),
    )
    format_model = _train_multiclass(
        normalized,
        [str(label) for label in format_labels],
        tag_groups.get("tags_format", []),
    )

    return GroupedClassifier(
        tone=tone_model,
        act=act_model,
        intensity=intensity_model,
        format=format_model,
    )



def predict_tags(
    prompt: str,
    model,
    tag_dictionary: Sequence[str],
    threshold: float = 0.5,
) -> list[TagPrediction]:
    """Predict tags from a prompt using a pre-trained classifier."""
    if not prompt:
        raise ValueError("prompt cannot be empty")
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")

    # Normalize prompt to match training preprocessing.
    normalized = normalize_prompt(prompt)
    # Prefer the model's stored dictionary to keep tag order consistent.
    tag_list = list(tag_dictionary)
    if hasattr(model, "tag_dictionary"):
        tag_list = list(model.tag_dictionary)

    # Extract trained components from the model wrapper.
    vectorizer = getattr(model, "vectorizer", None)
    classifier = getattr(model, "classifier", None)
    if vectorizer is None or classifier is None:
        raise ValueError("model must expose vectorizer and classifier")

    # Vectorize once; no fitting during inference.
    x = vectorizer.transform([normalized])
    # Prefer predict_proba when available; otherwise map decision scores to [0,1].
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(x)[0]
    else:
        raw_scores = classifier.decision_function(x)[0]
        scores = 1 / (1 + np.exp(-raw_scores))

    # Keep only tags above threshold and sort by confidence.
    predictions = [
        TagPrediction(tag=tag, score=float(score))
        for tag, score in zip(tag_list, scores)
        if score >= threshold
    ]
    predictions.sort(key=lambda item: item.score, reverse=True)
    return predictions


def _predict_multiclass(prompt: str, model: MultiClassModel) -> TagPrediction:
    # Predict a single class label with its probability-like score.
    normalized = normalize_prompt(prompt)
    x = model.vectorizer.transform([normalized])
    if hasattr(model.classifier, "predict_proba"):
        scores = model.classifier.predict_proba(x)[0]
    else:
        raw_scores = model.classifier.decision_function(x)[0]
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        scores = exp_scores / exp_scores.sum()
    best_idx = int(np.argmax(scores))
    return TagPrediction(tag=model.class_labels[best_idx], score=float(scores[best_idx]))


def predict_grouped_tags(
    prompt: str,
    model: GroupedClassifier,
    threshold: float = 0.5,
) -> dict[str, object]:
    # Return per-group predictions with the correct output shapes.
    return {
        "tags_ton": predict_tags(prompt, model.tone, model.tone.tag_dictionary, threshold),
        "tags_act": predict_tags(prompt, model.act, model.act.tag_dictionary, threshold),
        "tags_intensity": _predict_multiclass(prompt, model.intensity),
        "tags_format": _predict_multiclass(prompt, model.format),
    }

    
def filter_tags(predictions: Iterable[TagPrediction], top_k: int | None = None) -> list[TagPrediction]:
    """Filter predictions to a stable top-k or thresholded list."""
    ordered = sorted(predictions, key=lambda item: item.score, reverse=True)
    if top_k is None:
        return ordered
    return ordered[:top_k]
