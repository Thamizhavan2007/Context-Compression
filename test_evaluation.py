
from src.evaluation import normalize_answer, qa_f1_score, best_qa_f1


def main():
    reference = "Ground truth is not established in the paper"

    answer_1 = (
        "The ground truth for fake news is established through "
        "the categorization provided by Rubin et al."
    )

    answer_2 = (
        "The paper does not establish the ground truth."
    )

    print("Reference:", reference)

    for name, answer in [
        ("Previous generated answer", answer_1),
        ("Direct answer", answer_2),
    ]:
        score = qa_f1_score(answer, reference)
        print(f"\n{name}:")
        print("Answer:", answer)
        print(f"Official-style QA F1: {score:.4f}")

    assert normalize_answer("The Paper!") == "paper"
    assert qa_f1_score("", reference) == 0.0
    assert best_qa_f1(answer_2, [reference]) > 0.0

    print("\nEvaluation tests passed.")


if __name__ == "__main__":
    main()
