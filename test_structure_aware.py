
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
from src.compression import compress_by_relevance
from src.structure_aware import (
    calculate_structure_aware_scores,
    rank_by_structure_aware_score,
)


def main():
    dataset = load_longbench_dataset("qasper")
    sample = normalize_sample(dataset[0], "qasper")

    tokenizer = load_tokenizer()
    chunks = create_text_chunks(sample["context"], tokenizer)

    embedding_model = EmbeddingModel()
    query_embedding = embedding_model.encode_query(sample["query"])
    chunks = embedding_model.add_embeddings(chunks)

    # Baseline relevance scores.
    chunks = calculate_relevance_scores(chunks, query_embedding)

    # Semantic graph and structural support.
    graph = build_semantic_graph(chunks, k=5)
    structural_support = calculate_structural_support(graph)

    # Combined scores.
    chunks = calculate_structure_aware_scores(
        chunks,
        structural_support,
        alpha=0.7
    )

    relevance_ranked = rank_by_relevance(chunks)
    structure_ranked = rank_by_structure_aware_score(chunks)

    original_tokens = sum(
        chunk["token_count"] for chunk in chunks
    )

    print("\n" + "=" * 70)
    print("RELEVANCE-ONLY VS STRUCTURE-AWARE COMPRESSION")
    print("=" * 70)
    print(f"Query: {sample['query']}")
    print(f"Chunks: {len(chunks)}")
    print(f"Original processed tokens: {original_tokens}")
    print(f"Alpha: 0.7 (relevance 70%, structure 30%)")

    print("\nTOP 5 STRUCTURE-AWARE CHUNKS")
    for chunk in structure_ranked[:5]:
        print(
            f"Chunk {chunk['id']} | "
            f"Relevance={chunk['relevance_score']:.4f} | "
            f"Structure={chunk['normalized_structural_support']:.4f} | "
            f"Combined={chunk['structure_aware_score']:.4f}"
        )

    for ratio in [4, 8, 16]:
        budget = original_tokens // ratio

        baseline = compress_by_relevance(
            relevance_ranked,
            budget
        )

        proposed = compress_by_relevance(
            structure_ranked,
            budget
        )

        baseline_ids = {
            chunk["id"] for chunk in baseline["selected_chunks"]
        }
        proposed_ids = {
            chunk["id"] for chunk in proposed["selected_chunks"]
        }

        print(f"\n{'-' * 70}")
        print(f"TARGET COMPRESSION: {ratio}x")
        print(f"Shared token budget: {budget}")

        print("\nRelevance-only:")
        print(f"  Tokens selected: {baseline['selected_tokens']}")
        print(f"  Chunks selected: {len(baseline_ids)}")
        print(
            "  IDs: "
            f"{[c['id'] for c in baseline['selected_chunks']]}"
        )

        print("\nStructure-aware:")
        print(f"  Tokens selected: {proposed['selected_tokens']}")
        print(f"  Chunks selected: {len(proposed_ids)}")
        print(
            "  IDs: "
            f"{[c['id'] for c in proposed['selected_chunks']]}"
        )

        print(
            "\nChunks selected by both methods:",
            sorted(baseline_ids & proposed_ids)
        )
        print(
            "Chunks selected only by structure-aware:",
            sorted(proposed_ids - baseline_ids)
        )
        print(
            "Chunks selected only by relevance-only:",
            sorted(baseline_ids - proposed_ids)
        )

        assert baseline["selected_tokens"] <= budget
        assert proposed["selected_tokens"] <= budget


if __name__ == "__main__":
    main()
