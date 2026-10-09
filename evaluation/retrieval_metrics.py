
from retrieval.faiss_store import FAISSStore
from embeddings.embedding_model import EmbeddingModel
from retrieval.retriever import Retriever


INDEX_PATH = "data/indexes/document.index"
METADATA_PATH = "data/indexes/metadata.pkl"

TEST_CASES = [
    {
        "question": "What was the Phase 2 budget?",
        "expected_document": "test.pdf",
    },
    {
        "question": "What was AlphaCorp's total revenue in Q3 2025?",
        "expected_document": "second_document.pdf",
    },
    {
        "question": "What database does the Omega-3 architecture use?",
        "expected_document": "second_document.pdf",
    },
    {
        "question": "Who is the manga artist of Doraemon?",
        "expected_document": "Doraemon.pdf",
    },
]


def evaluate_retrieval(top_k=2):
    store = FAISSStore.load(
        INDEX_PATH,
        METADATA_PATH
    )

    embedding_model = EmbeddingModel()

    retriever = Retriever(
        faiss_store=store,
        embedding_model=embedding_model
    )

    hits = 0

    print(f"\nRetrieval evaluation: Top-K = {top_k}")
    print("=" * 65)

    for case in TEST_CASES:
        results = retriever.retrieve(
            case["question"],
            top_k=top_k
        )

        retrieved_documents = [
            result["chunk"]["document_id"]
            for result in results
        ]

        hit = (
            case["expected_document"]
            in retrieved_documents
        )

        hits += int(hit)

        print(f"\nQuestion: {case['question']}")
        print(f"Expected: {case['expected_document']}")
        print(f"Retrieved: {retrieved_documents}")
        print(f"Result: {'PASS' if hit else 'FAIL'}")

    hit_rate = hits / len(TEST_CASES)

    print("\n" + "=" * 65)
    print(f"Hits: {hits}/{len(TEST_CASES)}")
    print(f"Hit Rate @ {top_k}: {hit_rate:.1%}")


if __name__ == "__main__":
    evaluate_retrieval(top_k=2)
