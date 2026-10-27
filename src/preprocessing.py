import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Clean obvious PDF/extraction formatting artifacts while
    preserving the actual document content.
    """

    # Normalize line breaks
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove obvious formatting-size artifacts such as:
    # 10pt, 1.10pt
    text = re.sub(r"\b\d+(?:\.\d+)?pt\b", "", text)

    # Remove isolated square brackets caused by extraction.
    # Do not remove brackets containing meaningful content.
    text = re.sub(r"(?<!\S)\[\s*", "", text)
    text = re.sub(r"\s*\](?!\S)", "", text)

    # Remove a standalone page/section number at the very beginning.
    text = re.sub(r"^\s*\d+\.\s+", "", text)

    # Normalize whitespace again
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split cleaned text into sentence-level units.
    """

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


def create_chunks(
    sentences: List[str],
    tokenizer,
    target_tokens: int = 50,
    max_tokens: int = 70
) -> List[Dict]:
    """
    Group consecutive sentences into chunks.

    Each chunk attempts to stay near target_tokens
    without exceeding max_tokens.
    """

    chunks = []

    current_sentences = []
    current_tokens = 0
    start_sentence = 0

    for i, sentence in enumerate(sentences):

        sentence_tokens = len(
            tokenizer.encode(
                sentence,
                add_special_tokens=False
            )
        )

        # Handle a sentence that is already larger than max_tokens
        if sentence_tokens > max_tokens:

            if current_sentences:
                chunks.append({
                    "id": len(chunks),
                    "start_sentence": start_sentence,
                    "end_sentence": i - 1,
                    "text": " ".join(current_sentences)
                })

                current_sentences = []
                current_tokens = 0

            chunks.append({
                "id": len(chunks),
                "start_sentence": i,
                "end_sentence": i,
                "text": sentence
            })

            start_sentence = i + 1

            continue

        # If adding the sentence would exceed max_tokens,
        # finalize the current chunk.
        if (
            current_sentences
            and current_tokens + sentence_tokens > max_tokens
        ):

            chunks.append({
                "id": len(chunks),
                "start_sentence": start_sentence,
                "end_sentence": i - 1,
                "text": " ".join(current_sentences)
            })

            current_sentences = []
            current_tokens = 0
            start_sentence = i

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

        # If we have reached the target size,
        # finalize the chunk.
        if current_tokens >= target_tokens:

            chunks.append({
                "id": len(chunks),
                "start_sentence": start_sentence,
                "end_sentence": i,
                "text": " ".join(current_sentences)
            })

            current_sentences = []
            current_tokens = 0
            start_sentence = i + 1

    # Add remaining sentences
    if current_sentences:

        chunks.append({
            "id": len(chunks),
            "start_sentence": start_sentence,
            "end_sentence": len(sentences) - 1,
            "text": " ".join(current_sentences)
        })

    return chunks


def create_text_chunks(
    text: str,
    tokenizer,
    target_tokens: int = 50,
    max_tokens: int = 70,
) -> List[Dict]:
    """
    Preprocessing pipeline:
    clean -> sentence splitting -> chunking -> token counting.

    target_tokens: desired approximate chunk size.
    max_tokens: maximum preferred chunk size.
    """

    if target_tokens <= 0 or max_tokens <= 0:
        raise ValueError("Chunk sizes must be positive.")

    if target_tokens > max_tokens:
        raise ValueError(
            "target_tokens cannot exceed max_tokens."
        )

    sentences = split_into_sentences(text)

    chunks = create_chunks(
        sentences,
        tokenizer,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )

    for chunk in chunks:
        chunk["token_count"] = len(
            tokenizer.encode(
                chunk["text"],
                add_special_tokens=False,
            )
        )

    return chunks