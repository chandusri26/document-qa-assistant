from rag_service import RAGService


INDEX_PATH = "data/indexes/document.index"
METADATA_PATH = "data/indexes/metadata.pkl"


print("=" * 70)
print("INITIALIZING RAG SERVICE")
print("=" * 70)

rag_service = RAGService(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print("\n")
print("=" * 70)
print("INDEX INFORMATION")
print("=" * 70)

print(
    f"Documents: "
    f"{rag_service.document_count}"
)

print(
    f"Vectors: "
    f"{rag_service.vector_count}"
)


question = "What was the Phase 2 budget?"


print("\n")
print("=" * 70)
print("QUESTION")
print("=" * 70)

print(question)


result = rag_service.ask(
    question,
    top_k=2
)


print("\n")
print("=" * 70)
print("RAG ANSWER")
print("=" * 70)

print(result["answer"])


print("\n")
print("=" * 70)
print("SOURCES")
print("=" * 70)


for i, source in enumerate(
    result["sources"]
):

    print(
        f"\nSource {i + 1}"
    )

    print("-" * 70)

    print(
        f"Document: "
        f"{source['document_id']}"
    )

    print(
        f"Page: "
        f"{source['page_number']}"
    )

    print(
        f"Chunk ID: "
        f"{source['chunk_id']}"
    )

    print(
        f"Similarity Score: "
        f"{source['score']:.4f}"
    )

    print(
        f"Text:\n"
        f"{source['text']}"
    )