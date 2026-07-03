"""
Evaluation metrics for energy-guided generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Optional


def coherence_score(text: str, evaluator) -> float:
    """
    Semantic coherence: cosine similarity between the first and second half
    of the text. High = consistent topic. Low = topic drift.

    Args:
        text: Generated text.
        evaluator: SentenceTransformer model for embedding.
    """
    words = text.split()
    if len(words) < 10:
        return 0.0
    mid = len(words) // 2
    h1, h2 = " ".join(words[:mid]), " ".join(words[mid:])
    device = next(evaluator.parameters()).device if hasattr(evaluator, "parameters") else "cpu"
    e1 = evaluator.encode([h1], convert_to_tensor=True, normalize_embeddings=True, device=str(device))
    e2 = evaluator.encode([h2], convert_to_tensor=True, normalize_embeddings=True, device=str(device))
    return round(F.cosine_similarity(e1, e2).item(), 4)


def repetition_ratio(text: str) -> float:
    """
    Lexical diversity: 1.0 = no word repeated, 0.0 = single word repeated.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return round(1.0 - Counter(words).most_common(1)[0][1] / len(words), 4)


def target_similarity(
    text: str,
    target_texts: list[str],
    evaluator,
) -> float:
    """
    Cosine similarity between generated text and target texts (mean over targets).
    """
    device = next(evaluator.parameters()).device if hasattr(evaluator, "parameters") else "cpu"
    resp_emb = evaluator.encode(
        [text], convert_to_tensor=True, normalize_embeddings=True, device=str(device)
    )
    target_embs = evaluator.encode(
        target_texts, convert_to_tensor=True, normalize_embeddings=True, device=str(device)
    )
    return round(
        F.cosine_similarity(resp_emb, target_embs).mean().item(), 4
    )


def evaluate_guidance(
    response: str,
    target_texts: Optional[list[str]] = None,
    suppress_texts: Optional[list[str]] = None,
    evaluator=None,
) -> dict:
    """
    Full evaluation of a single guided generation.

    Returns dict with: target_sim, suppress_sim, coherence, diversity, repetition.
    """
    if evaluator is None:
        raise ValueError("evaluator (SentenceTransformer) is required")

    metrics = {
        "coherence": coherence_score(response, evaluator),
        "diversity": repetition_ratio(response),
    }

    if target_texts:
        metrics["target_sim"] = target_similarity(response, target_texts, evaluator)
    if suppress_texts:
        metrics["suppress_sim"] = target_similarity(response, suppress_texts, evaluator)

    return metrics
