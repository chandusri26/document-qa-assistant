
# Enterprise Document Intelligence & RAG Assistant

A document question-answering application built using Retrieval-Augmented Generation (RAG). It allows users to upload PDF documents, retrieve relevant passages using semantic search, and generate answers grounded in the retrieved context.

## Features

- PDF text extraction with PyMuPDF
- Recursive text chunking with configurable chunk size and overlap
- Semantic embeddings using Sentence Transformers
- FAISS vector indexing and similarity search
- Configurable Top-K retrieval
- Multi-document question answering
- LLM-based answer generation through OpenRouter
- Document and page information for retrieved sources
- Query spelling correction using indexed document vocabulary
- Streamlit user interface
- Basic document-level retrieval evaluation

## Architecture

PDF documents → Text extraction → Chunking → Embeddings → FAISS index

User question → Query correction → Query embedding → Top-K retrieval → Context construction → LLM answer → Sources

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- OpenAI Python SDK
- OpenRouter API
- LangChain Text Splitters

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`.
5. Set your OpenRouter API key and model in `.env`.
6. Start the application:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload one or more PDF documents.
2. Process the documents to build the FAISS index.
3. Enter a question about the uploaded documents.
4. Review the generated answer and its retrieved sources.
5. Adjust Top-K to change the number of chunks supplied to the LLM.

## Evaluation

A small retrieval evaluation script measures Hit Rate@K over four test questions. The current local test set achieved 4/4 hits at K=2 (100%).

This is a small development test set, not a guarantee of performance on unseen documents or questions.

## Security

- Store API credentials in `.env`.
- Never commit `.env`, virtual environments, private PDFs, or generated local indexes.

## Limitations

- PDF text extraction may not work well on scanned PDFs without OCR.
- Retrieval quality depends on document content, chunking, embeddings, and Top-K.
- The LLM can still produce incorrect answers; verify important information against the sources.
- The current evaluation dataset is small.

## Author

Garika chandu sri
