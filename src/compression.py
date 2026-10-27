
def compress_by_relevance(
    ranked_chunks,
    token_budget,
    tokenizer=None,
):
    if token_budget < 0:
        raise ValueError("Token budget cannot be negative.")

    def assemble(chunks):
        ordered = sorted(chunks, key=lambda chunk: chunk["id"])
        return "\n\n".join(chunk["text"] for chunk in ordered)

    def count_context_tokens(chunks):
        if tokenizer is None:
            return sum(chunk["token_count"] for chunk in chunks)

        context = assemble(chunks)
        return len(
            tokenizer.encode(context, add_special_tokens=False)
        )

    selected_chunks = []

    for chunk in ranked_chunks:
        chunk_tokens = chunk["token_count"]

        if chunk_tokens < 0:
            raise ValueError("Chunk token count cannot be negative.")

        candidate = selected_chunks + [chunk]

        if count_context_tokens(candidate) <= token_budget:
            selected_chunks = candidate

    selected_chunks.sort(key=lambda chunk: chunk["id"])
    compressed_context = assemble(selected_chunks)

    selected_tokens = count_context_tokens(selected_chunks)

    original_chunks = list(ranked_chunks)
    original_tokens = count_context_tokens(original_chunks)

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
