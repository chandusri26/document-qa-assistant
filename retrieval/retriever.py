from embeddings.embedding_model import EmbeddingModel
from retrieval.faiss_store import FAISSStore
from retrieval.query_corrector import QueryCorrector


class Retriever:

    def __init__(
        self,
        faiss_store,
        embedding_model
    ):

        self.faiss_store = faiss_store
        self.embedding_model = embedding_model

        self.query_corrector = QueryCorrector(
            faiss_store.metadata
        )

    def correct_query(self, query):

        return self.query_corrector.correct_query(
            query
        )

    def retrieve(
        self,
        query,
        top_k=3
    ):

        query_embedding = (
            self.embedding_model
            .generate_query_embedding(query)
        )

        results = self.faiss_store.search(
            query_embedding,
            top_k=top_k
        )

        return results