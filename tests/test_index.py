from retrieval.retriever import Retriever
from llm.generator import LLMGenerator


class RAGPipeline:

    def __init__(
        self,
        retriever,
        llm_generator
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator

    def answer(
        self,
        question,
        top_k=3
    ):

        # Step 1: Retrieve relevant chunks
        results = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        # Step 2: Build context
        context_parts = []

        for result in results:

            document_id = (
                result["chunk"]["document_id"]
            )

            page_number = (
                result["chunk"]["page_number"]
            )

            text = result["chunk"]["text"]

            context_parts.append(
                f"[Document: {document_id} | "
                f"Page: {page_number}]\n"
                f"{text}"
            )

        context = "\n\n".join(context_parts)

        # Step 3: Create LLM prompt
        prompt = f"""
You are an enterprise document assistant.

Answer the question using ONLY the information
provided in the context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. If the answer is not present in the context,
   say that you cannot find the answer in the document.
4. Give a concise and clear answer.
5. When relevant, mention the document and page
   containing the answer.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""

        # Step 4: Generate answer
        answer = self.llm_generator.generate(
            prompt
        )

        # Step 5: Prepare source information
        sources = []

        for result in results:

            sources.append({
                "document_id": (
                    result["chunk"]["document_id"]
                ),
                "page_number": (
                    result["chunk"]["page_number"]
                ),
                "chunk_id": (
                    result["chunk"]["chunk_id"]
                ),
                "score": result["score"],
                "text": (
                    result["chunk"]["text"]
                )
            })

        return {
            "answer": answer,
            "sources": sources
        }