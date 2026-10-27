
import numpy as np


def min_max_normalize(values):
    """
    Normalize scores to [0, 1].
    If all scores are equal, return zeros because
    there is no ranking information in their differences.
    """
    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Scores contain invalid values.")

    min_value = values.min()
    max_value = values.max()

    if np.isclose(max_value, min_value):
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def calculate_structure_aware_scores(
    chunks,
    structural_support,
    alpha=0.7
):
    """
    Combine normalized query relevance and structural support.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    if not chunks:
        return []

    relevance = np.asarray(
        [chunk["relevance_score"] for chunk in chunks],
        dtype=np.float64
    )

    support = np.asarray(
        [structural_support[chunk["id"]] for chunk in chunks],
        dtype=np.float64
    )

    normalized_relevance = min_max_normalize(relevance)
    normalized_support = min_max_normalize(support)

    for i, chunk in enumerate(chunks):
        chunk["normalized_relevance"] = float(
            normalized_relevance[i]
        )
        chunk["normalized_structural_support"] = float(
            normalized_support[i]
        )
        chunk["structure_aware_score"] = float(
            alpha * normalized_relevance[i]
            + (1.0 - alpha) * normalized_support[i]
        )

    return chunks


def rank_by_structure_aware_score(chunks):
    """
    Rank chunks by the combined relevance and structural score.
    """
    return sorted(
        chunks,
        key=lambda chunk: chunk["structure_aware_score"],
        reverse=True
    )
