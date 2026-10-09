import faiss
import numpy as np
import pickle
import os


class FAISSStore:

    def __init__(self, dimension):
        """
        Initialize FAISS vector store.

        dimension:
            Number of values in each embedding vector.
            For all-MiniLM-L6-v2, this is 384.
        """

        self.dimension = dimension

        # Inner Product + normalized embeddings = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        # Stores the metadata associated with each vector
        self.metadata = []

    def add_embeddings(self, embeddings, chunks):
        """
        Add embeddings and their corresponding chunk metadata.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.metadata.extend(chunks)

    def search(self, query_embedding, top_k=3):
        """
        Search for the most similar chunks.
        """

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            # FAISS returns -1 if no result exists
            if index == -1:
                continue

            results.append({
                "score": float(score),
                "chunk": self.metadata[index]
            })

        return results

    def save(self, index_path, metadata_path):
        """
        Save FAISS index and metadata to disk.
        """

        faiss.write_index(
            self.index,
            index_path
        )

        with open(metadata_path, "wb") as file:
            pickle.dump(
                self.metadata,
                file
            )

    @classmethod
    def load(cls, index_path, metadata_path):
        """
        Load an existing FAISS index and metadata.
        """

        index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as file:
            metadata = pickle.load(file)

        store = cls(index.d)

        store.index = index
        store.metadata = metadata

        return store