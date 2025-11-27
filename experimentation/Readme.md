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
python data_ingestion/concatenate_corpus.py --input-dir dataset
```
The file `commit_chunker.py` is based on a few files present in freshstack. The problem with the files in freshstack is that they download
only the most recent versions of a repository. However, for our purposes, the ability to switch between multiple versions of a repository is important. Therefore, The code present in the file allows to specify which version of a repo to access before chunking it.
**Note:** The chunking mechanism is the same as the one present in the orginal freshstack documentation. 

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






