from document_processing.loader import load_pdf


pdf_path = "data/uploads/test.pdf"

pages = load_pdf(pdf_path)

print(f"Total pages: {len(pages)}")

for page in pages:
    print("\n" + "=" * 60)
    print(f"PAGE {page['page_number']}")
    print("=" * 60)

    print(page["text"][:1000])