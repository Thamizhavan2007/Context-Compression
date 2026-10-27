
from src.data_loader import load_longbench_dataset, normalize_sample
from src.preprocessing import create_text_chunks
from src.tokenization import load_tokenizer
from src.embeddings import EmbeddingModel
from src.relevance import (
    calculate_relevance_scores,
    rank_by_relevance,
)
from src.graph import (
    build_semantic_graph,
    calculate_structural_support,
)


def main():
    dataset = load_longbench_dataset("qasper")
    sample = normalize_sample(dataset[0], "qasper")

    tokenizer = load_tokenizer()
    chunks = create_text_chunks(sample["context"], tokenizer)

    embedding_model = EmbeddingModel()
    query_embedding = embedding_model.encode_query(sample["query"])
    chunks = embedding_model.add_embeddings(chunks)

    chunks = calculate_relevance_scores(chunks, query_embedding)

    graph = build_semantic_graph(chunks, k=5)
    support_scores = calculate_structural_support(graph)

    print("\n" + "=" * 60)
    print("SEMANTIC GRAPH RESULTS")
    print("=" * 60)

    print(f"Query: {sample['query']}")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Connected components: {nx_component_count(graph)}")

    print("\nTOP 5 CHUNKS BY RELEVANCE")
    for chunk in rank_by_relevance(chunks)[:5]:
        print(
            f"Chunk {chunk['id']}: "
            f"relevance={chunk['relevance_score']:.4f}, "
            f"structural_support={support_scores[chunk['id']]:.4f}"
        )

    print("\nTOP 5 CHUNKS BY STRUCTURAL SUPPORT")
    for chunk_id, score in sorted(
        support_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]:
        print(f"Chunk {chunk_id}: support={score:.4f}")


def nx_component_count(graph):
    import networkx as nx
    return nx.number_connected_components(graph)


if __name__ == "__main__":
    main()
