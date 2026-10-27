from src.data_loader import load_longbench_dataset, normalize_sample
from src.preprocessing import create_text_chunks
from src.tokenization import load_tokenizer


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

    original_tokens = len(
        tokenizer.encode(
            sample["context"],
            add_special_tokens=False
        )
    )

    processed_tokens = sum(
        chunk["token_count"]
        for chunk in chunks
    )

    print("\n" + "=" * 60)
    print("FINAL CHUNKING TEST")
    print("=" * 60)

    print(f"Dataset: {dataset_name}")
    print(f"Sample ID: {sample['id']}")

    print(f"\nOriginal tokens: {original_tokens}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Processed tokens: {processed_tokens}")

    print("\nFirst 5 chunks:")
    print("-" * 60)

    for chunk in chunks[:5]:

        print(f"\nChunk ID: {chunk['id']}")
        print(
            f"Sentences: "
            f"{chunk['start_sentence']} → "
            f"{chunk['end_sentence']}"
        )

        print(
            f"Tokens: {chunk['token_count']}"
        )

        print(
            f"Text: {chunk['text'][:500]}"
        )


if __name__ == "__main__":
    main()