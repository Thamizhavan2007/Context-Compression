
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_semantic_graph(chunks, k=5):
    """
    Build an undirected weighted semantic graph.

    Nodes represent chunks.
    Edges represent strong semantic similarity.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    graph = nx.Graph()

    if not chunks:
        return graph

    embeddings = np.asarray(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32
    )

    if embeddings.ndim != 2:
        raise ValueError("Chunk embeddings must be a 2D array.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    similarity_matrix = cosine_similarity(embeddings)

    # Add one node per chunk, preserving the original chunk ID.
    for chunk in chunks:
        graph.add_node(
            chunk["id"],
            text=chunk["text"],
            token_count=chunk["token_count"],
            relevance_score=chunk["relevance_score"]
        )

    # Connect each node to its k most similar other nodes.
    for i, chunk in enumerate(chunks):
        similarities = similarity_matrix[i].copy()
        similarities[i] = -np.inf

        neighbor_count = min(k, len(chunks) - 1)

        if neighbor_count <= 0:
            continue

        neighbor_indices = np.argsort(similarities)[
            -neighbor_count:
        ]

        for j in neighbor_indices:
            neighbor = chunks[j]
            weight = float(similarity_matrix[i, j])

            # Avoid adding self-loops or non-positive edges.
            if weight <= 0:
                continue

            graph.add_edge(
                chunk["id"],
                neighbor["id"],
                weight=weight
            )

    return graph


def calculate_structural_support(graph):
    """
    Calculate query-aware structural support.

    A node receives support from its connected neighbors,
    weighted by edge strength and neighbor relevance.
    """
    support_scores = {}

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))

        if not neighbors:
            support_scores[node] = 0.0
            continue

        weighted_relevance = 0.0
        total_weight = 0.0

        for neighbor in neighbors:
            edge_weight = graph[node][neighbor]["weight"]
            neighbor_relevance = graph.nodes[neighbor][
                "relevance_score"
            ]

            weighted_relevance += (
                edge_weight * neighbor_relevance
            )
            total_weight += edge_weight

        support_scores[node] = (
            weighted_relevance / total_weight
            if total_weight > 0 else 0.0
        )

    return support_scores
