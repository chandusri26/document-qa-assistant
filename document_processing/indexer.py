import os

from document_processing.loader import load_pdf
from document_processing.chunker import chunk_pages
from embeddings.embedding_model import EmbeddingModel
from retrieval.faiss_store import FAISSStore


class DocumentIndexer:

    def __init__(
        self,
        chunk_size=800,
        chunk_overlap=100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_model = EmbeddingModel()

    def process_document(self, pdf_path):

        document_id = os.path.basename(pdf_path)

        print(f"\nProcessing: {document_id}")

        # ------------------------------------------
        # 1. Load PDF
        # ------------------------------------------

        pages = load_pdf(pdf_path)

        print(
            f"Loaded {len(pages)} pages."
        )

        # ------------------------------------------
        # 2. Create chunks
        # ------------------------------------------

        chunks = chunk_pages(
            pages,
            document_id=document_id,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        return chunks

    def build_index(self, pdf_paths):

        all_chunks = []

        # ------------------------------------------
        # Process every PDF
        # ------------------------------------------

        for pdf_path in pdf_paths:

            chunks = self.process_document(
                pdf_path
            )

            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError(
                "No document chunks were created."
            )

        # ------------------------------------------
        # Generate embeddings
        # ------------------------------------------

        print(
            "\nGenerating embeddings..."
        )

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        embeddings = (
            self.embedding_model
            .generate_embeddings(texts)
        )

        print(
            f"Embedding shape: "
            f"{embeddings.shape}"
        )

        # ------------------------------------------
        # Create FAISS store
        # ------------------------------------------

        dimension = embeddings.shape[1]

        faiss_store = FAISSStore(
            dimension=dimension
        )

        faiss_store.add_embeddings(
            embeddings,
            all_chunks
        )

        print(
            f"FAISS index contains "
            f"{faiss_store.index.ntotal} vectors."
        )

        return faiss_store