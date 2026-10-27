from src.data_loader import load_all_datasets, normalize_sample


def main():

    datasets = load_all_datasets()

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for name, dataset in datasets.items():

        print(f"\nDataset: {name}")
        print(f"Number of samples: {len(dataset)}")

        sample = normalize_sample(dataset[0], name)

        print(f"Sample ID: {sample['id']}")
        print(f"Query: {sample['query']}")
        print(f"Number of answers: {len(sample['answers'])}")
        print(f"Context characters: {len(sample['context'])}")
        print(f"Language: {sample['language']}")

        print("\nContext preview:")
        print(sample["context"][:500])
        print("-" * 60)


if __name__ == "__main__":
    main()