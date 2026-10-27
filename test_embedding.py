from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

texts = [
    "Context compression reduces the amount of information given to a language model.",
    "Long documents can contain redundant and irrelevant information."
]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)
print("Embedding dimension:", embeddings.shape[1])