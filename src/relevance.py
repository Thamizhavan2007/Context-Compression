
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_relevance_scores(chunks, query_embedding):
    """
    Calculate query-to-chunk cosine similarity.

    Each chunk must contain an 'embedding' field.
    The query embedding and chunk embeddings must
    have the same dimensionality.
    """
    if not chunks:
        return []

    chunk_embeddings = np.asarray(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    if chunk_embeddings.ndim != 2:
        raise ValueError("Chunk embeddings must be a 2D array.")

    if chunk_embeddings.shape[1] != query_embedding.shape[1]:
        raise ValueError(
            "Query and chunk embedding dimensions do not match."
        )

    if not np.isfinite(chunk_embeddings).all():
        raise ValueError("Chunk embeddings contain invalid values.")

    if not np.isfinite(query_embedding).all():
        raise ValueError("Query embedding contains invalid values.")

    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    for chunk, score in zip(chunks, scores):
        chunk["relevance_score"] = float(score)

    return chunks


def rank_by_relevance(chunks):
    """
    Return chunks sorted from highest to lowest relevance.
    The original chunk IDs are preserved.
    """
    return sorted(
        chunks,
        key=lambda chunk: chunk["relevance_score"],
        reverse=True
    )
