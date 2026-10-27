
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalLLM:
    def __init__(self):
        print(f"Loading language model: {MODEL_NAME}")
        print("Using CPU inference. Initial loading may take time.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        self.model.to("cpu")
        self.model.eval()

        print("Language model loaded successfully.")

    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 128,
    ) -> str:
        if not context.strip():
            return "Insufficient context to answer the question."

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied context. "
                    "Give a concise, factual answer. "
                    "Do not invent information. "
                    "If the context does not contain enough information, "
                    "state that the information is insufficient."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Answer:"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Remove the prompt tokens; keep only the generated answer.
        input_length = inputs["input_ids"].shape[1]
        answer_tokens = output[0, input_length:]

        answer = self.tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()
