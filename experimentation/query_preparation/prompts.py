from langchain_core.prompts import PromptTemplate

# Direct Answer Prompt
DIRECT_ANSWER_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template="""Given the following question, provide a direct answer without reasoning or context.
The answer should be concise and directly address the user's query.

Here is an example:

Question: Chromadb from_documents function giving error: The following function was working till a few days ago but now gives this error: ValueError: Expected EmbeddingFunction._call_ to have the following signature...
Answer:
{{
    "answer": "I slightly modify your code, using HuggingFaceEmbeddings instead of SentenceTransformerEmbeddings..."
}}

---

Question: {question}

Return your response in the following JSON format:
{{
    "answer": "your direct answer here"
}}
"""
)

# Decomposition Prompt
DECOMPOSITIONAL_QUERIES_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template="""Given the following complex question, decompose it into sub-questions that would help in answering it.
These sub-questions should help a retrieval system find relevant information.

Here is an example:

Question: Chromadb from_documents function giving error: The following function was working till a few days ago but now gives this error: ValueError: Expected EmbeddingFunction._call_ to have the following signature...
Answer:
{{
    "subquestions": [
        "What changes have been made to the SentenceTransformerEmbeddings interface according to the provided documentation?",
        "How is the EmbeddingFunction currently defined and used in the `create_chromadb` function?",
        "What should the type annotation 'D' be in the context of the new EmbeddingFunction?",
        "How should the `Chroma.from_documents` function be called with the updated EmbeddingFunction to avoid the error?"
    ]
}}

---

Question: {question}

Return your response in the following JSON format:
{{
    "subquestions": [
        "sub-question 1",
        "sub-question 2",
        "..."
    ]
}}
"""
)