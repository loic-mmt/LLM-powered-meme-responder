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
class TrainedClassifier:
    vectorizer: TfidfVectorizer
    classifier: OneVsRestClassifier
    tag_dictionary: list[str]


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


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt text before vectorization or inference."""
    # TODO: Add consistent normalization (lowercasing, punctuation cleanup).
    prompt = prompt.lower().strip()
    prompt = re.sub(r"[^\w\s']", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt


def train_multilabel_classifier(
    texts: Sequence[str],
    labels: Sequence[Sequence[str]],
    tag_dictionary: Sequence[str],
):
    """Train a multi-label classifier (TF-IDF+LR or DistilBERT)."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("texts cannot be empty")

    # Normalize text once so train/infer share the same preprocessing.
    normalized = [normalize_prompt(text) for text in texts]
    # Stable tag list + index lookup for multi-hot encoding.
    tag_list = list(tag_dictionary)
    tag_to_index = {tag: idx for idx, tag in enumerate(tag_list)}

    # Multi-hot label matrix: rows = samples, cols = tags.
    y = np.zeros((len(labels), len(tag_list)), dtype=np.float32)
    for row, sample_labels in enumerate(labels):
        for label in sample_labels:
            idx = tag_to_index.get(label)
            if idx is not None:
                y[row, idx] = 1.0

    # Simple baseline model: TF-IDF features + One-vs-Rest logistic regression.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=200))
    x = vectorizer.fit_transform(normalized)
    classifier.fit(x, y)

    return TrainedClassifier(vectorizer=vectorizer, classifier=classifier, tag_dictionary=tag_list)



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

    



def filter_tags(predictions: Iterable[TagPrediction], top_k: int | None = None) -> list[TagPrediction]:
    """Filter predictions to a stable top-k or thresholded list."""
    # TODO: Sort by score and optionally keep top_k.
    raise NotImplementedError("TODO: implement filtering")
