from src.data_loader import load_longbench_dataset, normalize_sample
from src.preprocessing import create_text_chunks
from src.tokenization import load_tokenizer
from src.embeddings import EmbeddingModel


def main():

    dataset_name = "qasper"

    print("Loading dataset...")

    dataset = load_longbench_dataset(
        dataset_name
    )

    sample = normalize_sample(
        dataset[0],
        dataset_name
    )

    print("Loading tokenizer...")

    tokenizer = load_tokenizer()

    chunks = create_text_chunks(
        sample["context"],
        tokenizer
    )

    print("Loading BGE model...")

    embedding_model = EmbeddingModel()

    # Query embedding
    query_embedding = embedding_model.encode_query(
        sample["query"]
    )

    # Chunk embeddings
    chunks = embedding_model.add_embeddings(
        chunks
    )

    print("\n" + "=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Number of chunks: {len(chunks)}"
    )

    print(
        f"Query embedding shape: "
        f"{query_embedding.shape}"
    )

    print(
        f"Chunk embedding shape: "
        f"{chunks[0]['embedding'].shape}"
    )

    print("\nFirst chunk:")

    print(
        f"ID: {chunks[0]['id']}"
    )

    print(
        f"Tokens: {chunks[0]['token_count']}"
    )

    print(
        f"Embedding dimensions: "
        f"{len(chunks[0]['embedding'])}"
    )

    print(
        "\nFirst 10 embedding values:"
    )

    print(
        chunks[0]["embedding"][:10]
    )


if __name__ == "__main__":
    main()