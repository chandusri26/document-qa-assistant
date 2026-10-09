from document_processing.loader import load_pdf
from document_processing.chunker import chunk_pages
from embeddings.embedding_model import EmbeddingModel


# 1. Load PDF
pdf_path = "data/uploads/test.pdf"

pages = load_pdf(pdf_path)


# 2. Create chunks
chunks = chunk_pages(
    pages,
    chunk_size=50,
    overlap=10
)


# 3. Extract only the text from each chunk
texts = [chunk["text"] for chunk in chunks]


# 4. Load embedding model
embedding_model = EmbeddingModel()


# 5. Generate embeddings
embeddings = embedding_model.generate_embeddings(texts)


# 6. Display information
print(f"Number of chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")

print("\nFirst chunk:")
print(texts[0])

print("\nFirst embedding:")
print(embeddings[0])