# FreshStack Refresh and Contamination Analysis

This repo serves to evaluate the quality of RAG pipeline on a non-static knowledge base. RAG benchmark focus on using static corpora. However, that poorly reflect the constant evolution of knowledge online for example new Wikipedia versions and updated code on github.


The question this repo aims to answer is the following: Do retrieval scores change across the same corpus but at different temporal snapshots?

The Code will aim to compare the Retrieval performance of the langchain repository at different points in time.


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

3. Indexing

Any RAG pipeline requires the usage of indexing to effeciently retrieve over the entire corpus. For the consistency with the original Freshstack paper, 

BM25
bge-gemma2
e5-mistral
Qwen3-Embedding

4. Retrieval

Uses 
Answer from stack overflow
Nuggets from GPT-4O
Answers Generated from (Qwen 2.5)



5. Judment pool






