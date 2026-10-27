
import re
import string
from collections import Counter


def normalize_answer(text):
    """Match the official LongBench English QA normalization."""
    text = text.lower()

    # Remove punctuation
    punctuation = set(string.punctuation)
    text = "".join(
        char for char in text
        if char not in punctuation
    )

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Normalize whitespace
    return " ".join(text.split())


def qa_f1_score(prediction, reference):
    """Token-level F1 used by LongBench for English QA datasets."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    common_count = sum(common.values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction, references):
    """Return the highest F1 across all reference answers."""
    if isinstance(references, str):
        references = [references]

    score = 0.0

    for reference in references:
        score = max(
            score,
            qa_f1_score(prediction, reference),
        )

    return score
