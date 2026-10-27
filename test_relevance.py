
from src.data_loader import load_longbench_dataset, normalize_sample
from src.preprocessing import create_text_chunks
from src.tokenization import load_tokenizer
from src.embeddings import EmbeddingModel
from src.relevance import (
    calculate_relevance_scores,
    rank_by_relevance,
)


def main():
    print("1. Loading Qasper dataset...")
    dataset = load_longbench_dataset("qasper")
    sample = normalize_sample(dataset[0], "qasper")

    print("2. Loading Qwen tokenizer...")
    tokenizer = load_tokenizer()

    print("3. Creating context chunks...")
    chunks = create_text_chunks(
        sample["context"],
        tokenizer
    )

    print("4. Generating BGE embeddings...")
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

    print("\n" + "=" * 65)
    print("RELEVANCE BASELINE RESULTS")
    print("=" * 65)
    print(f"Dataset: {sample['dataset']}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Query: {sample['query']}")

    print("\nTOP 5 RELEVANT CHUNKS\n")

    for rank, chunk in enumerate(ranked_chunks[:5], start=1):
        print(f"Rank: {rank}")
        print(f"Chunk ID: {chunk['id']}")
        print(f"Relevance score: {chunk['relevance_score']:.4f}")
        print(f"Token count: {chunk['token_count']}")
        print(f"Text: {chunk['text'][:300]}")
        print("-" * 65)

    scores = [chunk["relevance_score"] for chunk in chunks]

    print("\nSCORE SUMMARY")
    print(f"Highest score: {max(scores):.4f}")
    print(f"Lowest score:  {min(scores):.4f}")
    print(f"Mean score:    {sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
