DIRECT_ANSWER_PROMPT = """
Given the following long-description of a question asked by a user on StackOverflow, please try to directly answer the question in a concise manner.
You do not need to provide any reasoning or context, just provide the answer to the question; the answer should be a direct response to the question asked.
Read the whole question line by line, and first provide a reasoning to attempt to solve the question and then provide the answer.

---
Here is an example and strictly follow the format below: 

** Question **: Chromadb from_documents function giving error: The following function was working till a few days ago but now gives this error:
ValueError: Expected EmbeddingFunction._call_ to have the following signature: odict_keys(['self', 'input']), got odict_keys(['args', 'kwargs']) 
Please see https://docs.trychroma.com/embeddings for details of the EmbeddingFunction interface. 
Please note the recent change to the EmbeddingFunction interface: https://docs.trychroma.com/migration#migration-to-0416---november-7-2023
I am not sure what changes are necessary to work with this.
``` 
def create_chromadb(link): 
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = TextLoader(link)
    documents = loader.load()
    
    # Split the documents into chunks (no changes needed here)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    
    # Update for new EmbeddingFunction definition
    # D is set to the type of documents (Text in this case)
    D = Union[str, List[str]]  # Adjust based on your document format (single string or list of strings)
    embedding_function: EmbeddingFunction[D] = embedding_function
    
    # Initialize Chroma with the embedding function and persist the database
    db = Chroma.from_documents(chunks, embedding_function, ids=None, collection_name="langchain", persist_directory="./chroma_db")
    db.persist()
    return db
```

** Reasoning **: Let's do a step-by-step analysis, the user finds a ValueError of incorrect signature using chromadb in the question with SentenceTransformerEmbeddings. 
The user should reference documentation about SentenceTransformerEmbeddings required to answer the question, and review which embedding function is compatible with ChromaDB.from_documents.

** Answer **: I slightly modify your code, using HuggingFaceEmbeddings instead of SentenceTransformerEmbeddings
```
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="/tmp/chroma_db"
)
db.persist()

---

** Question **: {question}

** Reasoning **:
"""


DECOMPOSITIONAL_QUERIES_PROMPT = """
Given the following long-description of a question, please decompose the question into sub-questions using the non-trivial information present to provide an answer.
The intent is to help retrieval-augmented answering (RAG) systems ask the right and important sub-questions. The questions can be either atomic texts, code snippets or commands.
Conduct a step-by-step reasoning whether each sub-question is an important piece of information in either highlighting the issue asked in the question.
After reasoning, provide the necessary and important sub-questions in this format: (1) sub-question, (2) sub-question, etc.

---

** Question **: Chromadb from_documents function giving error: The following function was working till a few days ago but now gives this error:
ValueError: Expected EmbeddingFunction._call_ to have the following signature: odict_keys(['self', 'input']), got odict_keys(['args', 'kwargs']) 
Please see https://docs.trychroma.com/embeddings for details of the EmbeddingFunction interface. 
Please note the recent change to the EmbeddingFunction interface: https://docs.trychroma.com/migration#migration-to-0416---november-7-2023
I am not sure what changes are necessary to work with this.
``` 
def create_chromadb(link): 
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = TextLoader(link)
    documents = loader.load()
    
    # Split the documents into chunks (no changes needed here)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    
    # Update for new EmbeddingFunction definition
    # D is set to the type of documents (Text in this case)
    D = Union[str, List[str]]  # Adjust based on your document format (single string or list of strings)
    embedding_function: EmbeddingFunction[D] = embedding_function
    
    # Initialize Chroma with the embedding function and persist the database
    db = Chroma.from_documents(chunks, embedding_function, ids=None, collection_name="langchain", persist_directory="./chroma_db")
    db.persist()
    return db
```

** Reasoning **: The user is facing an issue with the ChromaDB function `from_documents` due to a change in the EmbeddingFunction interface.
The user is unsure about the necessary changes to resolve the error. The user has provided links to the documentation for EmbeddingFunction and the recent changes.
The user is using the `SentenceTransformerEmbeddings` function to create the embedding function. The user is splitting the documents into chunks and initializing ChromaDB with the embedding function.

** Sub-Questions **: (1) What changes have been made to the SentenceTransformerEmbeddings interface according to the provided documentation?
(2) How is the EmbeddingFunction currently defined and used in the `create_chromadb` function?
(3) What should the type annotation 'D' be in the context of the new EmbeddingFunction?
(4) How should the `Chroma.from_documents` function be called with the updated EmbeddingFunction to avoid the error?

---

** Question **: {question}

** Reasoning **:
"""