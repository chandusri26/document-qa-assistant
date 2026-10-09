from embeddings.embedding_model import EmbeddingModel
import numpy as np


embedding_model = EmbeddingModel()


sentences = [
    "The Phase 2 budget was increased to 120 million dollars.",
    "How much money was allocated for the second phase?",
    "The headquarters is located in Zurich, Switzerland.",
    "What is the capital of France?"
]


embeddings = embedding_model.generate_embeddings(sentences)


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b)


print("Similarity between budget statements:")
print(
    cosine_similarity(
        embeddings[0],
        embeddings[1]
    )
)


print("\nSimilarity between budget and headquarters:")
print(
    cosine_similarity(
        embeddings[0],
        embeddings[2]
    )
)


print("\nSimilarity between budget and France:")
print(
    cosine_similarity(
        embeddings[0],
        embeddings[3]
    )
)