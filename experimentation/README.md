# FreshStack Refresh and Contamination Analysis

This repo serves to evaluate the quality of RAG pipeline on a non-static knowledge base. RAG benchmark focus on using static corpora. However, that poorly reflect the constant evolution of knowledge online for example new Wikipedia versions and updated code on github.


The question this repo aims to answer is the following: Do retrieval scores change across the same corpus but at different temporal snapshots?

The Code will aim to compare the Retrieval performance of the langchain repository at different points in time.


## Pre-Requisite

To run the conde below you must have the following 
* `git`
* `python 3.11`
* `conda`
* `pip`
* `java 21` (according to pyserini/anserini documentation exactly that version)

## Setup

1. Clone the repo
```bash
https://github.com/suraj-subrahmanyan/RAG-decay.git
cd RAG-decay/experimentation
```

2. Create enviroment and install requirements

```bash
conda create python=3.11 -n freshstack -y
conda activate freshstack
pip install -r requirements.txt
```


## Code Overview

1. Config 
The folder contains all the configuration folder required to run various scripts.
the file `experimentation/config/repo_config.yaml` is meant to set which version of a repository to explore for the purposes of using as our corpus

2. Data Ingestion
This is where the process of finding the repository starts. Here are the files you need to run to create the knowledge base
```bash
python data_ingestion/find_commit_hashes.py --config config/repo_config.yaml  --update
python data_ingestion/chunk_repos_commit.py --config config/repo_config_updated.yaml 
python data_ingestion/concatenate_corpus.py --input-dir dataset/langchain/
```
The file `commit_chunker.py` is based on a few files present in freshstack. The problem with the files in freshstack is that they download
only the most recent versions of a repository. However, for our purposes, the ability to switch between multiple versions (time periods) of a repository is important. Therefore, The code present in this file allows to specify which version of a repo to access before chunking it.
**Note:** The chunking mechanism is the same as the one present in the orginal freshstack documentation. 

3. Analysis (Optional)

Methodology use to analyze the shift between versions

* Structural Volatility (File Churn)

    This analysis quantifies the stability of the repository's file organization over time. Since many retrieval systems rely on file paths for metadata filtering or logical grouping, high volatility can break existing filters.

    What it measures: The rate at which files are created, deleted, or renamed between the two corpus versions.

    Key Metric: File Retention Rate, calculated using the Jaccard Index of the unique file paths present in both versions. It also explicitly counts "Added Files" (new knowledge) and "Deleted Files" (lost or moved knowledge).

* Chunking Artifacts (Segmentation Stability)

    This dimension analyzes how the physical segmentation of text changes, which directly impacts the context window a retrieval model sees. Even if the content remains similar, changes in chunking can alter retrieval performance.

    What it measures: The distribution of text segment lengths and the density of chunks per file.

    Key Metrics:

    Chunk Length Distribution: A histogram comparing the character counts of chunks in V1 vs. V2 to detect if the documentation is becoming more fragmented (shorter chunks) or verbose (longer chunks).

    Chunk Count Distribution and Chunk Count Delta: The change in the number of chunks for the same file, indicating if a specific document has grown significantly in size or complexity.

* Semantic Drift (Meaning Shift)

    This is the core measure of "information rot." It determines if the actual meaning of a document has changed, which helps predict if a question that was valid for the old corpus will still be answered correctly by the new one.

    What it measures: The semantic distance between the 2024 and 2025 versions of the same file.

    Methodology: The text of a file from V1 is reconstructed and compared to its V2 counterpart.

    Primary Method: Cosine similarity of vector embeddings (using all-MiniLM-L6-v2) to capture conceptual changes.

    Interpretation: A low similarity score implies the file has been rewritten or refactored, posing a high risk for "hallucination" if the RAG system retrieves it for an older query.

* Technical Shift (Code-to-Prose Density)

    This analysis characterizes the stylistic evolution of the documentation. Technical libraries often shift from code-heavy "alpha" documentation to prose-heavy "mature" guides, or vice-versa.

    What it measures: The ratio of code-like characters (brackets, indentation, camelCase, keywords) to natural language prose in the corpus.

    Key Metric: Code Density Score (0.0 to 1.0).

    Implication: A significant shift suggests that the optimal retrieval algorithm may need to change. For example, a shift towards higher prose density favors dense vector retrievers, while a shift towards raw code favors sparse lexical retrievers (like BM25).

* Vocabulary & Deprecation Scan

    This dimension scans for explicit markers of obsolescence to quantify the "freshness risk" of the corpus.

    What it measures: The frequency of specific keywords that indicate breaking changes or outdated information.

    Key Markers: Words like "deprecated," "legacy," "migration," and "removed."

    Implication: A spike in these terms in the new corpus serves as a proxy for the number of "dead paths" or invalid functions that a RAG system might incorrectly recommend if it doesn't understand temporal context.

``` bash
python analysis/analyze_corpus_drift.py --v1 dataset/langchain/corpus_oct_2024.jsonl --v2 dataset/langchain/corpus_oct_2025.jsonl 
```

4. Indexing

Any RAG pipeline requires the usage of indexing to effeciently retrieval over the entire corpus. For the consistency with the original Freshstack paper, we use the following methods over the corpus being the chunked code files.

* BM25
* bge-gemma2
* e5-mistral
* Qwen3-Embedding


For indexing using BM25, use:
```bash
python indexing/build_bm25_index.py --config config/indexing_config.yaml 
```

To build the other supported dense indices, you may use
```bash
python indexing/build_dense_index.py --config config/indexing_config.yaml --model qwen
python indexing/build_dense_index.py --config config/indexing_config.yaml --model e5
python indexing/build_dense_index.py --config config/indexing_config.yaml --model bge
```


5. Query Preperation

The ultimate goal is to create Judgement Pools, which are necessary for evaluating the relevance of retrieved information to a given query. Since judging all queries against their potential answers is prohibitively expensive, we focus on retrieving a subset of potentially relevant queries to form these pools.

To find the potentially relevant queries, we employ two distinct retrieval settings: Oracle Retrieval and Inference Retrieval.

| Retrieval Setting     | Source Text for Retrieval                                                 | Purpose                                                                                                                                                  |
|-----------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Oracle Retrieval**  | Stack Overflow Answers and GPT-4o Generated Nuggets (per the previous paper). | Uses **known relevant** information (the “oracle” answer/factoids) to find queries that should be relevant.                                               |
| **Inference Retrieval** | Qwen-Generated Subquestions and Closed Book Answers.                       | Uses **synthetically generated** texts to simulate an inference step, finding queries relevant to intermediate or direct generated content.                |

The execution steps to make this work are

```bash
python query_preparation/load_queries_answers.py
python query_preparation/add_nuggets_to_queries.py
python query_preparation/add_qwen_generation.py
```

6. Retrieval

After indexing, retrieve documents using different methods and query formulations.

### Dense Retrieval (BGE, E5, Qwen)

Retrieve using dense embedding models:

```bash
# BGE retrieval
python retrieval/retrieval_dense.py --config config/config_retrieval.yaml --model bge --corpus-version oct_2024

# E5 retrieval
python retrieval/retrieval_dense.py --config config/config_retrieval.yaml --model e5 --corpus-version oct_2024

# Qwen retrieval
python retrieval/retrieval_dense.py --config config/config_retrieval.yaml --model qwen --corpus-version oct_2024
```

### BM25 Retrieval

Retrieve using sparse BM25 method:

```bash
# BM25 retrieval for oct_2024
python retrieval/retrieval_bm25.py --config config/config_retrieval.yaml --corpus-version oct_2024
```

### Fusion Retrieval

Combine results from multiple retrieval methods:

```bash
# Fuse all methods (bm25, bge, e5, qwen)
python retrieval/retrieval_fusion.py --config config/config_retrieval.yaml --corpus-version oct_2024

# Fuse specific methods only
python retrieval/retrieval_fusion.py --config config/config_retrieval.yaml --corpus-version oct_2024 --methods bge e5 qwen
```

### Query Fields

All retrieval methods support multiple query formulations:
- `answer`: Original Stack Overflow answer
- `nuggets`: GPT-4o generated nuggets
- `closed_book_answer`: Qwen-generated answer
- `subquestions`: Qwen-generated subquestions

Specify fields with `--query-fields`:
```bash
python retrieval/retrieval_dense.py --config config/config_retrieval.yaml --model bge --query-fields answer nuggets
```

### Output

Results are saved to `retrieval_results/{corpus_version}/{method}.jsonl`:
- `bge.jsonl`, `e5.jsonl`, `qwen.jsonl` - Dense retrieval results
- `bm25.jsonl` - BM25 retrieval results
- `fusion.jsonl` - Fused results from multiple methods

Each line contains:
```json
{
  "query_id": "75864073",
  "answer_id": "75864224",
  "doc_id": "langchain/docs/...",
  "score": 0.85,
  "rank": 1,
  "query_field": "answer"
}
```

### TREC Format Conversion

To convert the JSONL results into a single TREC-formatted file (useful for evaluation with tools like `trec_eval`), run the `convert_to_trec.py` script.

```bash
python retrieval/convert_to_trec.py --base-dir retrieval_results
```

This will create a file at `retrieval_results/trec_runs/all_runs.trec` containing aggregated results from both `oct_2024` and `oct_2025` for all retrieval methods and query fields.


7. Nugget-Level Relevance Assessment

Assess whether retrieved documents support the nuggets using a Qwen model (following the FreshStack paper methodology).

### Run Assessment

```bash
# Assess all methods for all corpus versions
python assessment/nugget_assessment.py --config config/config_assessment.yaml

# Assess specific corpus version
python assessment/nugget_assessment.py --corpus-version oct_2024

# Assess specific methods only
python assessment/nugget_assessment.py --methods fusion bge --corpus-version oct_2024
```

### How It Works

For each query:
1. Loads nuggets (factual statements from Stack Overflow)
2. Loads top-k retrieved documents
3. Uses Qwen to assess if each document supports any nugget
4. Adds binary `nugget_level_judgment` field (1=relevant, 0=not relevant)

### Output

Results saved to `assessment_results/{corpus_version}/{method}_assessed.jsonl`:

```json
{
  "query_id": "75864073",
  "answer_id": "75864224",
  "doc_id": "langchain/docs/...",
  "score": 0.85,
  "rank": 1,
  "query_field": "answer",
  "nugget_level_judgment": 1
}
```

### Configuration

Edit `config/config_assessment.yaml` to configure:
- Model settings (Qwen model, device, dtype)
- Generation parameters (temperature, max_tokens)
- Assessment parameters (top_k documents to assess)
