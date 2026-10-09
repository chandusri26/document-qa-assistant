from document_processing.loader import load_pdf
from document_processing.chunker import chunk_pages


pdf_path = "data/uploads/test.pdf"

pages = load_pdf(pdf_path)

chunks = chunk_pages(
    pages,
    chunk_size=800,
    chunk_overlap=100
)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")


for i, chunk in enumerate(chunks):

    print("\n" + "=" * 70)

    print(f"CHUNK {i + 1}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page_number']}")

    print("=" * 70)

    print(chunk["text"])