from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer():
    """
    Load the tokenizer used by the final LLM.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    return tokenizer


def count_tokens(text, tokenizer):
    """
    Count the number of tokens in a text.
    """

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    return len(tokens)


def add_token_counts(units, tokenizer):
    """
    Add Qwen token counts to every text unit.
    """

    for unit in units:

        unit["token_count"] = count_tokens(
            unit["text"],
            tokenizer
        )

    return units