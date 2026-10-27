from src.data_loader import load_longbench_dataset, normalize_sample
from src.preprocessing import create_text_units
from src.tokenization import load_tokenizer, add_token_counts


def main():

    dataset_name = "qasper"

    print("Loading dataset...")
    dataset = load_longbench_dataset(dataset_name)

    sample = normalize_sample(
        dataset[0],
        dataset_name
    )

    print("Loading Qwen tokenizer...")
    tokenizer = load_tokenizer()

    units = create_text_units(
        sample["context"]
    )

    units = add_token_counts(
        units,
        tokenizer
    )

    original_tokens = sum(
        unit["token_count"]
        for unit in units
    )

    print("\n" + "=" * 60)
    print("TOKENIZATION TEST")
    print("=" * 60)

    print(f"Dataset: {dataset_name}")
    print(f"Sample ID: {sample['id']}")
    print(f"Text units: {len(units)}")
    print(f"Total tokens: {original_tokens}")

    print("\nFirst 5 units:")
    print("-" * 60)

    for unit in units[:5]:

        print(f"\nID: {unit['id']}")
        print(f"Position: {unit['position']}")
        print(f"Tokens: {unit['token_count']}")
        print(f"Text: {unit['text'][:200]}")


if __name__ == "__main__":
    main()