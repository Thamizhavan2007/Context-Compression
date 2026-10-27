import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


DATASETS = {
    "qasper": "qasper.jsonl",
    "hotpotqa": "hotpotqa.jsonl",
    "multifieldqa_en": "multifieldqa_en.jsonl"
}


def load_longbench_dataset(dataset_name):
    """
    Load one LongBench dataset from a local JSONL file.
    """

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Choose from {list(DATASETS.keys())}"
        )

    file_path = DATA_DIR / DATASETS[dataset_name]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{file_path}\n\n"
            f"Please download the LongBench data first."
        )

    samples = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def load_all_datasets():

    datasets = {}

    for dataset_name in DATASETS:

        print(f"Loading {dataset_name}...")

        datasets[dataset_name] = load_longbench_dataset(
            dataset_name
        )

        print(
            f"{dataset_name}: "
            f"{len(datasets[dataset_name])} samples"
        )

    return datasets


def normalize_sample(sample, dataset_name):

    return {
        "dataset": dataset_name,
        "id": sample["_id"],
        "context": sample["context"],
        "query": sample["input"],
        "answers": sample["answers"],
        "length": sample["length"],
        "language": sample["language"],
        "all_classes": sample.get("all_classes"),
    }