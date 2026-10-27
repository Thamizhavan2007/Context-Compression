from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:

    def __init__(self):
        print("Loading embedding model...")
        
        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

    def encode_query(self, query):
        """
        Generate an embedding for the query.
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding

    def encode_chunks(self, chunks):
        """
        Generate embeddings for all text chunks.
        """

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    def add_embeddings(self, chunks):
        """
        Add embeddings to the chunk objects.
        """

        embeddings = self.encode_chunks(chunks)

        for chunk, embedding in zip(chunks, embeddings):

            chunk["embedding"] = embedding

        return chunks