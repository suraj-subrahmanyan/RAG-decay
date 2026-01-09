"""
Prompts for nugget-level relevance assessment.
"""

from langchain_core.prompts import PromptTemplate

NUGGET_SUPPORT_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["nuggets", "documents"],
    template="""You are an expert at assessing whether documents contain information that supports specific factual statements (nuggets).

Your task is to determine if each retrieved document supports any of the given nuggets from a Stack Overflow question/answer.

**Nuggets (factual statements to verify):**
{nuggets}

**Retrieved Documents:**
{documents}

**Instructions:**
1. For each document, carefully read its content
2. For each nugget, determine if the document contains information that supports or confirms that nugget
3. A document supports a nugget if it contains the same information, explains the concept, or provides evidence for the claim
4. Be strict: only mark as supporting (1) if there is clear evidence in the document
5. Mark as not supporting (0) if the information is absent, contradictory, or only tangentially related

**Output Format:**
Return a JSON object with the following structure:
{{
  "assessments": [
    {{
      "doc_id": "document_id_1",
      "nugget_judgments": [1, 0, 1, ...]  // Binary list: 1 if doc supports nugget i, 0 otherwise
    }},
    ...
  ]
}}

The nugget_judgments list must have exactly {num_nuggets} elements (one per nugget), in the same order as the nuggets listed above.

Do not include any reasoning or explanations. Output ONLY the JSON object.

Output:"""
)
