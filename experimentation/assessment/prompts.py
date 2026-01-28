"""
Prompts for nugget-level relevance assessment.
"""

from langchain_core.prompts import PromptTemplate

FRESHRAG_LISTWISE_NUGGET_V4 = PromptTemplate(
    input_variables=["question", "answer", "nuggets", "context", "count"],
    template="""In this task, you will provide the exhaustive list of documents (either a code snippet or documentation) which sufficiently supports a decompositional fact.

You will first read the question and the answer, next read each decompositional fact carefully one by one which will be provided to you. You must carefully analyze every one of {count} documents provided to you and judge each document and whether it sufficiently supports or does not support the decompositional fact. Read every fact and document pair carefully as you would when proofreading.

It may be helpful to ask yourself, "Which list of documents can provide sufficient evidence required to support the decompositional fact?" Be sure to check all of the information in the document. You will be given two options to choose from:

- "Full Support": The document is sufficient in supporting the answer for the question and entailing *all* necessary parts of the decompositional fact.
- "No Support": The document does not support the answer for the question and *does not* provide information in entailing the decompositional fact.

** Output Format **:
Return a JSON object with the following structure:
{{
  "assessments": [
    {{
      "nugget_index": 1,
      "supported_documents": [
        {{
          "doc_index": 1,
          "reasoning": "Reason why document 1 supports this nugget..."
        }},
        {{
          "doc_index": 4,
          "reasoning": "Reason why document 4 supports this nugget..."
        }}
      ] 
    }},
    ...
  ]
}}

** Important **:
- You must provide an assessment for EVERY nugget. If there are 3 nuggets, you must provide 3 assessments.
- The nuggets are provided in the 'Decompositional Facts' section. 1-based indexing.
- Ensure `nugget_index` matches the provided nuggets order.
- `supported_documents` must contain a list of objects, each with a "doc_index" (1-based integer from "Document (i)") and a "reasoning" string.
- If no documents support a nugget, `supported_documents` should be an empty list [].

** Question **: {question}

** Answer **: {answer}

** Decompositional Facts **: {nuggets}

** Documents **: {context}

** Output **:"""
)
