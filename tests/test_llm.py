from llm.generator import LLMGenerator


llm = LLMGenerator()


prompt = """
Explain what Retrieval-Augmented Generation (RAG)
is in three simple sentences.
"""


answer = llm.generate(prompt)


print("\nLLM RESPONSE")
print("=" * 60)
print(answer)