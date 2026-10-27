
import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.data_loader import load_longbench_dataset, normalize_sample
from src.preprocessing import create_text_chunks
from src.tokenization import load_tokenizer
from src.embeddings import EmbeddingModel
from src.relevance import calculate_relevance_scores, rank_by_relevance
from src.graph import build_semantic_graph, calculate_structural_support
from src.structure_aware import (
    calculate_structure_aware_scores,
    rank_by_structure_aware_score,
)
from src.compression import compress_by_relevance
from src.evaluation import best_qa_f1


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = OUTPUT_DIR / "batch_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

DATASETS = ["qasper", "hotpotqa", "multifieldqa_en"]
METHODS = ["relevance_only", "structure_aware"]


def load_completed_keys():
    """Read saved results so completed generations are not repeated."""
    completed = set()

    if not RESULTS_FILE.exists():
        return completed

    with open(RESULTS_FILE, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            completed.add(
                (
                    row["dataset"],
                    row["sample_index"],
                    row["target_ratio"],
                    row["method"],
                )
            )

    return completed


def append_result(row):
    """Save each answer immediately to support recovery after interruption."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0

    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def prepare_sample(sample, dataset_name, tokenizer, embedding_model, ratios):
    """Calculate both rankings and prepare compressed contexts for each ratio."""
    chunks = create_text_chunks(sample["context"], tokenizer)

    if not chunks:
        return []

    query_embedding = embedding_model.encode_query(sample["query"])
    chunks = embedding_model.add_embeddings(chunks)
    chunks = calculate_relevance_scores(chunks, query_embedding)

    graph = build_semantic_graph(chunks, k=5)
    support = calculate_structural_support(graph)

    chunks = calculate_structure_aware_scores(
        chunks, support, alpha=0.7
    )

    rankings = {
        "relevance_only": rank_by_relevance(chunks),
        "structure_aware": rank_by_structure_aware_score(chunks),
    }

    ordered_chunks = sorted(chunks, key=lambda chunk: chunk["id"])
    original_context = "\n\n".join(
        chunk["text"] for chunk in ordered_chunks
    )
    original_tokens = len(
        tokenizer.encode(original_context, add_special_tokens=False)
    )

    plans = []

    for ratio in ratios:
        budget = original_tokens // ratio

        for method, ranked_chunks in rankings.items():
            compressed = compress_by_relevance(
                ranked_chunks,
                budget,
                tokenizer=tokenizer,
            )

            # Validate the assembled context, not just the sum of chunk sizes.
            actual_tokens = len(
                tokenizer.encode(
                    compressed["compressed_context"],
                    add_special_tokens=False,
                )
            )

            if actual_tokens > budget:
                raise ValueError(
                    f"{dataset_name}: token budget exceeded "
                    f"for {method}, target {ratio}x."
                )

            plans.append({
                "dataset": dataset_name,
                "sample_id": str(sample["id"]),
                "query": sample["query"],
                "answers": sample["answers"],
                "target_ratio": ratio,
                "token_budget": budget,
                "original_tokens": original_tokens,
                "selected_tokens": actual_tokens,
                "selected_chunk_ids": [
                    chunk["id"]
                    for chunk in compressed["selected_chunks"]
                ],
                "method": method,
                "compressed_context": compressed["compressed_context"],
            })

    return plans


def write_summary():
    """Recalculate mean QA F1 from all saved predictions."""
    if not RESULTS_FILE.exists():
        return

    groups = defaultdict(list)

    with open(RESULTS_FILE, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = (
                row["dataset"],
                int(row["target_ratio"]),
                row["method"],
            )
            groups[key].append(float(row["f1"]))

    with open(SUMMARY_FILE, "w", encoding="utf-8", newline="") as file:
        fields = [
            "dataset",
            "target_ratio",
            "method",
            "sample_count",
            "mean_f1",
            "mean_f1_percent",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for (dataset, ratio, method), scores in sorted(groups.items()):
            average = mean(scores)

            writer.writerow({
                "dataset": dataset,
                "target_ratio": ratio,
                "method": method,
                "sample_count": len(scores),
                "mean_f1": round(average, 6),
                "mean_f1_percent": round(average * 100, 2),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate relevance-only and structure-aware compression."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Samples per dataset. Use 0 for all available samples.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=int,
        choices=[4, 8, 16],
        default=[4, 8, 16],
        help="Target compression ratios.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or greater.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    embedding_model = EmbeddingModel()
    completed = load_completed_keys()

    plans = []

    # Prepare compressed contexts before loading the larger generation model.
    for dataset_name in args.datasets:
        print(f"\nPreparing dataset: {dataset_name}")

        raw_samples = load_longbench_dataset(dataset_name)

        if args.limit > 0:
            raw_samples = raw_samples[:args.limit]

        for index, raw_sample in enumerate(raw_samples):
            expected_keys = {
                (
                    dataset_name,
                    str(index),
                    str(ratio),
                    method,
                )
                for ratio in args.ratios
                for method in METHODS
            }

            if expected_keys.issubset(completed):
                print(f"Sample {index}: already complete; skipping.")
                continue

            sample = normalize_sample(raw_sample, dataset_name)

            print(
                f"Sample {index + 1}/{len(raw_samples)} | "
                f"Question: {sample['query'][:100]}"
            )

            sample_plans = prepare_sample(
                sample,
                dataset_name,
                tokenizer,
                embedding_model,
                args.ratios,
            )

            for plan in sample_plans:
                plan["sample_index"] = index

                key = (
                    plan["dataset"],
                    str(index),
                    str(plan["target_ratio"]),
                    plan["method"],
                )

                if key not in completed:
                    plans.append(plan)

    del embedding_model
    gc.collect()

    if not plans:
        print("\nNo new evaluations to run.")
        write_summary()
        print(f"Results: {RESULTS_FILE}")
        print(f"Summary: {SUMMARY_FILE}")
        return

    # Import here so we load the LLM only when new answers are required.
    from src.llm import LocalLLM

    llm = LocalLLM()

    print(f"\nNew answers to generate: {len(plans)}")

    for position, plan in enumerate(plans, start=1):
        print(
            f"\n[{position}/{len(plans)}] "
            f"{plan['dataset']} | "
            f"sample {plan['sample_index']} | "
            f"{plan['method']} | "
            f"{plan['target_ratio']}x"
        )

        prediction = llm.generate_answer(
            question=plan["query"],
            context=plan["compressed_context"],
        )

        score = best_qa_f1(prediction, plan["answers"])

        row = {
            "dataset": plan["dataset"],
            "sample_index": plan["sample_index"],
            "sample_id": plan["sample_id"],
            "target_ratio": plan["target_ratio"],
            "method": plan["method"],
            "original_tokens": plan["original_tokens"],
            "token_budget": plan["token_budget"],
            "selected_tokens": plan["selected_tokens"],
            "actual_compression_ratio": (
                round(
                    plan["original_tokens"] / plan["selected_tokens"], 4
                )
                if plan["selected_tokens"] > 0
                else ""
            ),
            "selected_chunk_ids": json.dumps(
                plan["selected_chunk_ids"]
            ),
            "query": plan["query"],
            "reference_answers": json.dumps(
                plan["answers"], ensure_ascii=False
            ),
            "prediction": prediction,
            "f1": round(score, 6),
        }

        # Save immediately after every generated answer.
        append_result(row)

        print(f"Prediction: {prediction}")
        print(f"QA F1: {score:.4f}")

    write_summary()

    print("\n" + "=" * 70)
    print("BATCH EVALUATION FINISHED")
    print("=" * 70)
    print(f"Detailed results: {RESULTS_FILE}")
    print(f"Mean-score summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

