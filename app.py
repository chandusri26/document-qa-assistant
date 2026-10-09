import os
import streamlit as st

from rag_service import RAGService
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


# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 Enterprise Document Intelligence")

st.write(
    "Ask questions about your uploaded documents "
    "using Retrieval-Augmented Generation (RAG)."
)


# --------------------------------------------------
# Document Upload Section
# --------------------------------------------------

st.subheader("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    st.write(
        f"Selected {len(uploaded_files)} document(s):"
    )

    for uploaded_file in uploaded_files:

        st.write(
            f"📄 {uploaded_file.name}"
        )


    process_button = st.button(
        "🔨 Process Documents",
        type="primary"
    )


    if process_button:

        try:

            # ------------------------------------------
            # Save uploaded files
            # ------------------------------------------

            with st.spinner(
                "Saving uploaded documents..."
            ):

                for uploaded_file in uploaded_files:

                    file_path = os.path.join(
                        UPLOAD_DIR,
                        uploaded_file.name
                    )

                    with open(
                        file_path,
                        "wb"
                    ) as file:

                        file.write(
                            uploaded_file.getvalue()
                        )


            # ------------------------------------------
            # Build FAISS index
            # ------------------------------------------

            with st.spinner(
                "Processing documents and building "
                "FAISS index..."
            ):

                indexer = DocumentIndexer(
                    chunk_size=800,
                    chunk_overlap=100
                )

                faiss_store = indexer.build_index(
                    [
                        os.path.join(
                            UPLOAD_DIR,
                            filename
                        )
                        for filename in os.listdir(
                            UPLOAD_DIR
                        )
                        if filename.lower().endswith(
                            ".pdf"
                        )
                    ]
                )


                faiss_store.save(
                    INDEX_PATH,
                    METADATA_PATH
                )


            # ------------------------------------------
            # Clear cached RAG service
            # ------------------------------------------

            st.cache_resource.clear()


            st.success(
                "Documents processed successfully!"
            )


            st.rerun()


        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )


# --------------------------------------------------
# Check whether index exists
# --------------------------------------------------

index_exists = (
    os.path.exists(INDEX_PATH)
    and os.path.exists(METADATA_PATH)
)


# --------------------------------------------------
# Initialize RAG service
# --------------------------------------------------

if index_exists:

    @st.cache_resource
    def load_rag_service():

        return RAGService(
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH
        )


    try:

        rag_service = load_rag_service()

    except Exception as error:

        st.error(
            f"Failed to initialize RAG system: {error}"
        )

        st.stop()

else:

    rag_service = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📊 System Information")

    if rag_service:

        st.metric(
            "Documents",
            rag_service.document_count
        )

        st.metric(
            "Indexed Chunks",
            rag_service.vector_count
        )

    else:

        st.warning(
            "No document index found."
        )


    st.divider()


    st.header("⚙️ Retrieval Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=5,
        value=2
    )


# --------------------------------------------------
# Question Interface
# --------------------------------------------------

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What was the Phase 2 budget?"
    )
)


ask_button = st.button(
    "🔍 Ask Question",
    type="primary"
)


# --------------------------------------------------
# Question Processing
# --------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif rag_service is None:

        st.error(
            "No document index is available. "
            "Upload and process a PDF first."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = rag_service.ask(
                    question,
                    top_k=top_k
                )
                corrected_question = result[
                    "corrected_question"
                    ]
                if corrected_question != question:
                     st.info(
                         f"✏️ Corrected query: "
                         f"{corrected_question}"
                         )


                # ----------------------------------
                # Answer
                # ----------------------------------

                st.subheader("🤖 Answer")

                st.write(
                    result["answer"]
                )


                # ----------------------------------
                # Sources
                # ----------------------------------

                st.subheader("📚 Sources")


                for i, source in enumerate(
                    result["sources"],
                    start=1
                ):

                    with st.expander(
                        f"Source {i} — "
                        f"{source['document_id']} "
                        f"(Page {source['page_number']})"
                    ):

                        st.write(
                            f"**Similarity Score:** "
                            f"{source['score']:.4f}"
                        )

                        st.write(
                            f"**Document:** "
                            f"{source['document_id']}"
                        )

                        st.write(
                            f"**Page:** "
                            f"{source['page_number']}"
                        )

                        st.write(
                            f"**Chunk ID:** "
                            f"{source['chunk_id']}"
                        )

                        st.write(
                            "**Retrieved Text:**"
                        )

                        st.write(
                            source["text"]
                        )


            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )