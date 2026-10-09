from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings

    def generate_query_embedding(self, query):
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding