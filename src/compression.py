
def compress_by_relevance(
    ranked_chunks,
    token_budget,
    tokenizer=None,
):
    """
    Compress ranked chunks using greedy, budget-constrained selection.

    Chunks are considered in the order supplied by the ranking function.
    A chunk is selected only if the combined context stays within the
    token budget.

    The same function can be used for relevance-only and structure-aware
    rankings, ensuring that both methods use the same selection procedure.
    """

    if token_budget < 0:
        raise ValueError("Token budget cannot be negative.")

    ranked = list(ranked_chunks)

    for chunk in ranked:
        if chunk["token_count"] < 0:
            raise ValueError("Chunk token count cannot be negative.")

    def assemble(chunks):
        ordered = sorted(chunks, key=lambda chunk: chunk["id"])
        return "\n\n".join(chunk["text"] for chunk in ordered)

    def count_context_tokens(chunks):
        if not chunks:
            return 0

        if tokenizer is None:
            return sum(chunk["token_count"] for chunk in chunks)

        context = assemble(chunks)

        return len(
            tokenizer.encode(
                context,
                add_special_tokens=False,
            )
        )

    # Greedy selection: preserve the incoming ranking order.
    selected_chunks = []

    print("\n===== SELECTOR DEBUG =====")
    print("Input ranking:", [chunk["id"] for chunk in ranked])
    print("Token budget:", token_budget)

    for chunk in ranked:
        candidate = selected_chunks + [chunk]
        candidate_tokens = count_context_tokens(candidate)

        accepted = candidate_tokens <= token_budget

        print(
            f"Chunk {chunk['id']} | "
            f"Tokens if added: {candidate_tokens} | "
            f"Accepted: {accepted}"
        )

        if accepted:
            selected_chunks = candidate

    # Sort by original document position when assembling the context.
    selected_chunks.sort(key=lambda chunk: chunk["id"])

    compressed_context = assemble(selected_chunks)
    selected_tokens = count_context_tokens(selected_chunks)

    # Count the original document using the same counting method.
    original_tokens = count_context_tokens(ranked)

    print("Selected IDs:", [c["id"] for c in selected_chunks])
    print("Selected tokens:", selected_tokens)
    print("==========================\n")

    return {
        "selected_chunks": selected_chunks,
        "compressed_context": compressed_context,
        "original_tokens": original_tokens,
        "selected_tokens": selected_tokens,
        "token_budget": token_budget,
        "compression_ratio": (
            original_tokens / selected_tokens
            if selected_tokens > 0
            else None
        ),
    }
