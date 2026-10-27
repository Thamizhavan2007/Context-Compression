
import gc

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
from src.llm import LocalLLM
from src.evaluation import best_qa_f1


def main():
    # 1. Load one Qasper sample
    dataset = load_longbench_dataset("qasper")
    sample = normalize_sample(dataset[0], "qasper")

    print("=" * 70)
    print("CONTEXT COMPRESSION AND ANSWER EVALUATION")
    print("=" * 70)

    # 2. Tokenize and create chunks
    tokenizer = load_tokenizer()
    chunks = create_text_chunks(sample["context"], tokenizer)

    # 3. Generate embeddings
    embedding_model = EmbeddingModel()
    query_embedding = embedding_model.encode_query(sample["query"])
    chunks = embedding_model.add_embeddings(chunks)

    # 4. Calculate query relevance
    chunks = calculate_relevance_scores(chunks, query_embedding)

    # 5. Build graph and calculate structural support
    graph = build_semantic_graph(chunks, k=5)
    support = calculate_structural_support(graph)

    # 6. Calculate structure-aware scores
    chunks = calculate_structure_aware_scores(
        chunks,
        support,
        alpha=0.7,
    )

    # 7. Rank chunks using both methods
    relevance_ranked = rank_by_relevance(chunks)
    structure_ranked = rank_by_structure_aware_score(chunks)

    # 8. Calculate the original context token count
    ordered_chunks = sorted(chunks, key=lambda c: c["id"])
    original_context = "\n\n".join(
        chunk["text"] for chunk in ordered_chunks
    )
    original_tokens = len(
        tokenizer.encode(
            original_context,
            add_special_tokens=False,
        )
    )

    # Target 8x compression
    budget = original_tokens // 8

    # 9. Compress using the same token budget
    baseline = compress_by_relevance(
        relevance_ranked,
        budget,
        tokenizer=tokenizer,
    )

    proposed = compress_by_relevance(
        structure_ranked,
        budget,
        tokenizer=tokenizer,
    )

    print(f"\nDataset: Qasper")
    print(f"Question: {sample['query']}")
    print(f"Original context tokens: {original_tokens}")
    print(f"Target compression: 8x")
    print(f"Shared token budget: {budget}")

    # Release the embedding model before loading the LLM
    del embedding_model
    del graph
    gc.collect()

    # 10. Load the local language model
    llm = LocalLLM()

    results = [
        ("RELEVANCE-ONLY", baseline),
        ("STRUCTURE-AWARE", proposed),
    ]

    scores = {}

    # 11. Generate and evaluate both answers
    for name, result in results:
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print(f"Actual context tokens: {result['selected_tokens']}")
        print(
            "Selected chunk IDs:",
            [chunk["id"] for chunk in result["selected_chunks"]],
        )

        # Verify the token budget
        assert result["selected_tokens"] <= budget, (
            f"{name} exceeded the token budget."
        )

        answer = llm.generate_answer(
            question=sample["query"],
            context=result["compressed_context"],
        )

        # Compare the answer with LongBench reference answers
        score = best_qa_f1(
            answer,
            sample["answers"],
        )

        scores[name] = score

        print("\nGenerated answer:")
        print(answer)

        print("\nReference answers:")
        print(sample["answers"])

        print(f"\nQA F1 score: {score:.4f}")
        print(f"QA F1 percentage: {score * 100:.2f}%")

    # 12. Compare the methods
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"Relevance-only F1: {scores['RELEVANCE-ONLY']:.4f}")
    print(f"Structure-aware F1: {scores['STRUCTURE-AWARE']:.4f}")

    difference = (
        scores["STRUCTURE-AWARE"] - scores["RELEVANCE-ONLY"]
    )

    print(f"F1 difference: {difference:+.4f}")

    if difference > 0:
        print("Structure-aware scored higher on this sample.")
    elif difference < 0:
        print("Relevance-only scored higher on this sample.")
    else:
        print("Both methods achieved the same F1 score.")

    print("\nNote: This is one sample, not a final research conclusion.")


if __name__ == "__main__":
    main()
