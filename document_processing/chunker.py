from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages,
    document_id,
    chunk_size=800,
    chunk_overlap=100
):
    """
    Split document pages into overlapping chunks
    while preserving document and page metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        page_chunks = splitter.split_text(text)

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            chunks.append({
                "document_id": document_id,
                "chunk_id": (
                    f"{document_id}_"
                    f"page_{page_number}_"
                    f"chunk_{chunk_number}"
                ),
                "page_number": page_number,
                "text": chunk_text
            })

    return chunks