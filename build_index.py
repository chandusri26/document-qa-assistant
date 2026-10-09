import os

from document_processing.indexer import DocumentIndexer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

UPLOAD_DIR = "data/uploads"
INDEX_DIR = "data/indexes"

INDEX_PATH = os.path.join(
    INDEX_DIR,
    "document.index"
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.pkl"
)


# --------------------------------------------------
# Create directories
# --------------------------------------------------

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    INDEX_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Find PDF files
# --------------------------------------------------

pdf_paths = []

for filename in os.listdir(UPLOAD_DIR):

    if filename.lower().endswith(".pdf"):

        pdf_paths.append(
            os.path.join(
                UPLOAD_DIR,
                filename
            )
        )


if not pdf_paths:

    raise FileNotFoundError(
        "No PDF files found in data/uploads/"
    )


print(
    f"Found {len(pdf_paths)} PDF file(s)."
)


# --------------------------------------------------
# Build index
# --------------------------------------------------

indexer = DocumentIndexer(
    chunk_size=800,
    chunk_overlap=100
)

faiss_store = indexer.build_index(
    pdf_paths
)


# --------------------------------------------------
# Save index
# --------------------------------------------------

print("\nSaving FAISS index...")

faiss_store.save(
    INDEX_PATH,
    METADATA_PATH
)


print("\nIndex successfully saved!")

print(
    f"FAISS index: {INDEX_PATH}"
)

print(
    f"Metadata: {METADATA_PATH}"
)