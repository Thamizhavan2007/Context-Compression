
from src.data_loader import load_longbench_dataset, normalize_sample
from src.preprocessing import create_text_chunks
from src.tokenization import load_tokenizer
from src.embeddings import EmbeddingModel
from src.relevance import (
    calculate_relevance_scores,
    rank_by_relevance,
)
from src.compression import compress_by_relevance


def main():
    print("1. Loading Qasper sample...")
    dataset = load_longbench_dataset("qasper")
    sample = normalize_sample(dataset[0], "qasper")

    print("2. Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("3. Creating context chunks...")
    chunks = create_text_chunks(
        sample["context"],
        tokenizer
    )

    print("4. Generating embeddings...")
    embedding_model = EmbeddingModel()

    query_embedding = embedding_model.encode_query(
        sample["query"]
    )

    chunks = embedding_model.add_embeddings(chunks)

    print("5. Calculating relevance scores...")
    chunks = calculate_relevance_scores(
        chunks,
        query_embedding
    )

    ranked_chunks = rank_by_relevance(chunks)

    original_tokens = sum(
        chunk["token_count"] for chunk in chunks
    )

    print("\n" + "=" * 65)
    print("FIXED TOKEN-BUDGET COMPRESSION")
    print("=" * 65)
    print(f"Query: {sample['query']}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Processed context tokens: {original_tokens}")

    for ratio in [4, 8, 16]:
        budget = max(1, original_tokens // ratio)

        result = compress_by_relevance(
            ranked_chunks,
            budget
        )

        print(f"\n{'-' * 65}")
        print(f"TARGET COMPRESSION: {ratio}x")
        print(f"Token budget: {result['token_budget']}")
        print(f"Tokens selected: {result['selected_tokens']}")
        print(f"Chunks selected: {len(result['selected_chunks'])}")

        if result["compression_ratio"] is not None:
            print(
                "Actual compression ratio: "
                f"{result['compression_ratio']:.2f}x"
            )
        else:
            print("Actual compression ratio: undefined")

        print(
            "Budget respected:",
            result["selected_tokens"] <= result["token_budget"]
        )

        print(
            "Selected chunk IDs in document order:",
            [c["id"] for c in result["selected_chunks"]]
        )

        print("\nCompressed context preview:")
        print(result["compressed_context"][:700])


if __name__ == "__main__":
    main()
