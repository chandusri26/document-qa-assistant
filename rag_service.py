import os

from retrieval.faiss_store import FAISSStore
from retrieval.retriever import Retriever
from embeddings.embedding_model import EmbeddingModel
from llm.generator import LLMGenerator
from rag import RAGPipeline


class RAGService:

    def __init__(
        self,
        index_path,
        metadata_path
    ):

        # Validate saved index
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        # Load FAISS index
        print("Loading FAISS index...")

        self.faiss_store = FAISSStore.load(
            index_path,
            metadata_path
        )

        # Load embedding model
        print("Loading embedding model...")

        self.embedding_model = EmbeddingModel()

        # Create retriever
        self.retriever = Retriever(
            faiss_store=self.faiss_store,
            embedding_model=self.embedding_model
        )

        # Initialize LLM
        print("Initializing LLM...")

        self.llm_generator = LLMGenerator()

        # Create RAG pipeline
        self.rag = RAGPipeline(
            retriever=self.retriever,
            llm_generator=self.llm_generator
        )

    def ask(
        self,
        question,
        top_k=3
    ):

        return self.rag.answer(
            question,
            top_k=top_k
        )

    @property
    def vector_count(self):

        return self.faiss_store.index.ntotal

    @property
    def document_count(self):

        document_ids = {
            chunk["document_id"]
            for chunk in self.faiss_store.metadata
        }

        return len(document_ids)