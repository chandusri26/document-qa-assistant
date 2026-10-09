from document_processing.loader import load_pdf
from document_processing.chunker import chunk_pages
from embeddings.embedding_model import EmbeddingModel
from retrieval.faiss_store import FAISSStore


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

pdf_path = "data/uploads/test.pdf"

pages = load_pdf(pdf_path)

print(f"Loaded {len(pages)} pages.")


# --------------------------------------------------
# 2. Create chunks
# --------------------------------------------------

chunks = chunk_pages(
    pages,
    chunk_size=50,
    overlap=10
)

print(f"Created {len(chunks)} chunks.")


# --------------------------------------------------
# 3. Extract chunk text
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

embedding_model = EmbeddingModel()

embeddings = embedding_model.generate_embeddings(
    texts
)

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# 5. Create FAISS store
# --------------------------------------------------

dimension = embeddings.shape[1]

faiss_store = FAISSStore(
    dimension=dimension
)


# --------------------------------------------------
# 6. Add embeddings + metadata
# --------------------------------------------------

faiss_store.add_embeddings(
    embeddings,
    chunks
)

print(
    f"FAISS index contains "
    f"{faiss_store.index.ntotal} vectors."
)


# --------------------------------------------------
# 7. Create query
# --------------------------------------------------

query = "What was the Phase 2 budget?"


# --------------------------------------------------
# 8. Convert query into embedding
# --------------------------------------------------

query_embedding = (
    embedding_model.generate_query_embedding(
        query
    )
)


# --------------------------------------------------
# 9. Search FAISS
# --------------------------------------------------

results = faiss_store.search(
    query_embedding,
    top_k=3
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n")
print("=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)

for i, result in enumerate(results):

    print(f"\nResult {i + 1}")
    print("-" * 70)

    print(
        f"Similarity Score: "
        f"{result['score']:.4f}"
    )

    print(
        f"Page: "
        f"{result['chunk']['page_number']}"
    )

    print(
        f"Text:\n"
        f"{result['chunk']['text']}"
    )