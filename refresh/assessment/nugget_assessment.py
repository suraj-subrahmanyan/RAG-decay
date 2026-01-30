"""
Nugget-Level Listwise Assessment using LLM-as-Judge

This script performs nugget-level relevance assessment where an LLM judges
which retrieved documents provide sufficient support for atomic facts (nuggets)
decomposed from reference answers.

Implementation uses native Transformers with text-based output parsing following
the FreshStack methodology for decompositional fact verification.
"""

import argparse
import json
import logging
import re
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Prompt template for nugget-level assessment
FRESHRAG_LISTWISE_NUGGET_V4 = """In this task, you will provide the exhaustive list of documents (either a code snippet or documentation) which sufficiently supports a decompositional fact.

You will first read the question and the answer, next read each decompositional fact carefully one by one which will be provided to you. You must carefully analyze every one of {count} documents provided to you and judge each document and whether it sufficiently supports or does not support the decompositional fact. Read every fact and document pair carefully as you would when proofreading.

It may be helpful to ask yourself, "Which list of documents can provide sufficient evidence required to support the decompositional fact?" Be sure to check all of the information in the document. You will be given two options to choose from:

---
- "Full Support": The document is sufficient in supporting the answer for the question and entailing *all* necessary parts of the decompositional fact.
- "No Support": The document does not support the answer for the question and *does not* provide information in entailing the decompositional fact.

You should provide your reasoning as bullet points on *why* and *how* every fully supported document sufficiently supports the decompositional fact and provide the list of fully supported documents after "VERDICT:" and output [] if none of the documents supports it. Follow the format below:

Fact (1): REASON: output your reasoning here. VERDICT: ## Full Support: [Doc (1), Doc (4)...], 
Fact (2): REASON: output your reasoning here. VERDICT: ## Full Support: [],
for all decompositional facts., etc.,

---

** Question **: {question}

** Answer **: {answer}

** Decompositional Facts **: 
{nuggets}

** Documents **: 
{context}

** Output **:
"""


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_queries(queries_file: Path) -> List[dict]:
    """Load queries from JSONL file."""
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_retrieval_results(retrieval_file: Path) -> Dict[str, List[dict]]:
    """Load retrieval results, grouped by query_id."""
    results = defaultdict(list)
    with open(retrieval_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results[result['query_id']].append(result)
    return results


def load_corpus(corpus_file: Path) -> Dict[str, dict]:
    """Load corpus documents."""
    corpus = {}
    if corpus_file.exists():
        with open(corpus_file, 'r') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    corpus[doc['_id']] = doc
    return corpus


class QwenAssessor:
    """Qwen model for nugget-level relevance assessment (Native Transformers)."""
    
    def __init__(self, config: dict):
        """Initialize Qwen model WITHOUT Outlines."""
        model_config = config['model']
        self.gen_config = config['generation']
        
        logger.info(f"Loading model: {model_config['name']}")
        logger.info("Using NATIVE Transformers (Outlines removed)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            trust_remote_code=model_config.get('trust_remote_code', True)
        )
        
        # Determine dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        torch_dtype = dtype_map.get(model_config['torch_dtype'], torch.bfloat16)
        
        # Load model WITHOUT flash_attention_2 (causes compatibility issues)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=model_config.get('trust_remote_code', True),
            attn_implementation="eager"  # Use eager instead of flash_attention_2
        )
        
        self.model.eval()
        logger.info("Model loaded successfully (eager attention)")
    
    def format_nuggets(self, nuggets: List[str]) -> str:
        """
        Format nuggets using Nandan's EXACT specification.
        
        Format:
          Fact (1): nugget text
          ---------------------------
          Fact (2): nugget text
          ---------------------------
        """
        formatted = []
        for i, nugget in enumerate(nuggets, 1):
            formatted.append(f"Fact ({i}): {nugget}")
            formatted.append("---------------------------")
        return "\n".join(formatted)
    
    def format_documents(self, documents: List[Dict[str, str]], truncation: int = 800) -> str:
        """
        Format documents using Nandan's EXACT specification.
        
        Format:
          Doc (1): document text
          ---------------------------
          Doc (2): document text
          ---------------------------
        
        Args:
            truncation: Number of characters to keep per document
        """
        formatted = []
        for i, doc in enumerate(documents, 1):
            # Truncate long documents to avoid context overflow
            text = doc.get('text', '')[:truncation].replace('\n', ' ')
            formatted.append(f"Doc ({i}): {text}")
            formatted.append("---------------------------")
        return "\n".join(formatted)
    
    def parse_verdict_output(self, text: str, num_nuggets: int) -> Dict[int, List[int]]:
        """
        Parse LLM output in REASON/VERDICT format.
        
        Expected format:
          Fact (1): REASON: ... VERDICT: ## Full Support: [Doc (1), Doc (4)]
          Fact (2): REASON: ... VERDICT: ## Full Support: []
        
        Returns:
          {nugget_index: [doc_indices]}
        """
        judgments = {}
        
        # Split by "Fact (N):" pattern
        fact_pattern = r'Fact \((\d+)\):'
        fact_chunks = re.split(fact_pattern, text)
        
        # Process chunks (pairs of [nugget_index, content])
        for i in range(1, len(fact_chunks), 2):
            if i + 1 >= len(fact_chunks):
                break
                
            nugget_idx = int(fact_chunks[i])
            content = fact_chunks[i + 1]
            
            # Extract VERDICT line
            verdict_match = re.search(
                r'VERDICT:\s*##\s*Full Support:\s*\[(.*?)\]',
                content,
                re.IGNORECASE | re.DOTALL
            )
            
            supported_docs = []
            if verdict_match:
                docs_str = verdict_match.group(1)
                # Extract "Doc (X)" indices
                doc_matches = re.findall(r'Doc \((\d+)\)', docs_str)
                supported_docs = [int(d) for d in doc_matches]
            
            judgments[nugget_idx] = supported_docs
        
        # Fill in missing nuggets with empty lists
        for i in range(1, num_nuggets + 1):
            if i not in judgments:
                judgments[i] = []
                logger.warning(f"No verdict found for Fact ({i}), assuming no support")
        
        return judgments
    
    def assess_nugget_support(
        self,
        question: str,
        answer: str,
        nuggets: List[str],
        documents: List[Dict[str, str]],
        doc_truncation: int = 800
    ) -> Dict[int, List[int]]:
        """
        Assess which documents support which nuggets.
        
        Args:
            doc_truncation: Number of characters to keep per document
        
        Returns:
          {nugget_index: [doc_indices]}
        """
        if not documents or not nuggets:
            return {}
        
        # Format inputs using Nandan's specification
        nuggets_text = self.format_nuggets(nuggets)
        docs_text = self.format_documents(documents, truncation=doc_truncation)
        
        # Create prompt
        prompt = FRESHRAG_LISTWISE_NUGGET_V4.format(
            count=len(documents),
            question=question,
            answer=answer,
            nuggets=nuggets_text,
            context=docs_text
        )
        
        # Generate
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.gen_config.get('max_new_tokens', 2048),
                    temperature=self.gen_config.get('temperature', 0.1),
                    top_p=self.gen_config.get('top_p', 0.9),
                    do_sample=self.gen_config.get('do_sample', True)
                )
            
            # Decode output (skip input prompt)
            output_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse text output
            judgments = self.parse_verdict_output(output_text, len(nuggets))
            return judgments
            
        except Exception as e:
            logger.error(f"Error in assessment: {e}")
            # Return empty judgments for all nuggets
            return {i: [] for i in range(1, len(nuggets) + 1)}


def process_method(
    config: dict,
    assessor: QwenAssessor,
    queries: List[dict],
    retrieval_results: Dict[str, List[dict]],
    corpus: Dict[str, dict],
    method: str,
    corpus_version: str,
    output_dir: Path
) -> None:
    """Process a single retrieval method."""
    logger.info(f"Processing {method} for {corpus_version}")
    
    output_file = output_dir / f"{method}_assessed.jsonl"
    
    # Check if already exists and if we should resume
    existing_queries = set()
    if output_file.exists():
        logger.info(f"Found existing file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_queries.add(data['query_id'])
        logger.info(f"Found {len(existing_queries)} existing queries, will skip them")
    
    # Filter queries
    queries_to_process = [q for q in queries if q['query_id'] not in existing_queries]
    logger.info(f"Assessing {len(queries_to_process)} queries (skipped {len(existing_queries)})")
    
    # Open output file in append mode
    with open(output_file, 'a') as f:
        for query in tqdm(queries_to_process, desc=f"Assessing {method}"):
            query_id = query['query_id']
            nuggets = query.get('nuggets', [])
            
            if not nuggets:
                logger.warning(f"Query {query_id} has no nuggets, skipping")
                continue
            
            # Get retrieval results for this query
            if query_id not in retrieval_results:
                logger.warning(f"No retrieval results for query {query_id}")
                continue
            
            # Get top documents (sorted by rank)
            top_results = sorted(
                retrieval_results[query_id],
                key=lambda x: x['rank']
            )[:config['assessment_parameters']['top_k']]
            
            # Hydrate with corpus text
            documents = []
            for result in top_results:
                doc_id = result['doc_id']
                if doc_id in corpus:
                    documents.append({
                        'doc_id': doc_id,
                        'text': corpus[doc_id].get('text', '')
                    })
                else:
                    logger.warning(f"Document {doc_id} not found in corpus")
            
            if not documents:
                logger.warning(f"No documents found for query {query_id}")
                continue
            
            # Assess nugget support
            doc_truncation = config['assessment_parameters'].get('doc_truncation', 800)
            judgments = assessor.assess_nugget_support(
                question=query['question'],
                answer=query['answer'],
                nuggets=nuggets,
                documents=documents,
                doc_truncation=doc_truncation
            )
            
            # Write results (one line per document)
            for result in top_results:
                doc_id = result['doc_id']
                
                # Find which nuggets this document supports
                supported_nuggets = []
                for nugget_idx, doc_indices in judgments.items():
                    # Map doc_id to index in documents list
                    try:
                        doc_idx_in_list = [d['doc_id'] for d in documents].index(doc_id) + 1
                        if doc_idx_in_list in doc_indices:
                            supported_nuggets.append(nugget_idx)
                    except ValueError:
                        pass
                
                # Calculate nugget-level judgment (binary: 1 if any nugget supported, 0 otherwise)
                nugget_level_judgment = 1 if supported_nuggets else 0
                
                # Create nugget support details
                nugget_support_details = [
                    {"nugget_index": nug_idx, "support": "Full Support"}
                    for nug_idx in supported_nuggets
                ]
                
                # Write output (matching Nathan's format for compatibility)
                output = {
                    "query_id": query_id,
                    "answer_id": query.get('answer_id', ''),
                    "doc_id": doc_id,
                    "score": result.get('score', 0.0),
                    "rank": result['rank'],
                    "query_field": result.get('query_field', 'question'),
                    "nugget_level_judgment": nugget_level_judgment,
                    "nugget_support_details": nugget_support_details
                }
                
                f.write(json.dumps(output) + '\n')
                f.flush()
    
    logger.info(f"Completed {method} assessment")


def main():
    parser = argparse.ArgumentParser(
        description="Nugget-level assessment (Native Transformers)"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--corpus-version", required=True, help="Corpus version (oct_2024, oct_2025, or 'all')")
    parser.add_argument("--methods", nargs='+', default=['fusion'], help="Methods to assess")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine corpus versions to process
    if args.corpus_version == 'all':
        corpus_versions = config['corpus_versions']
    else:
        corpus_versions = [args.corpus_version]
    
    # Initialize model (once)
    assessor = QwenAssessor(config)
    
    # Process each corpus version
    for corpus_version in corpus_versions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing corpus: {corpus_version}")
        logger.info(f"{'='*60}\n")
        
        # Define paths
        base_dir = Path("/workspace/RAG-decay")
        queries_file = base_dir / config['queries_file']
        corpus_file = base_dir / config['corpus_base_dir'] / f"{corpus_version}" / "corpus.jsonl"
        retrieval_dir = base_dir / config['retrieval_results_dir'] / corpus_version
        output_dir = base_dir / config['output_dir'] / corpus_version
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading queries...")
        queries = load_queries(queries_file)
        logger.info(f"Loaded {len(queries)} queries")
        
        logger.info("Loading corpus...")
        corpus = load_corpus(corpus_file)
        logger.info(f"Loaded {len(corpus)} documents")
        
        # Process each method
        for method in args.methods:
            retrieval_file = retrieval_dir / f"{method}.jsonl"
            
            if not retrieval_file.exists():
                logger.warning(f"Retrieval file not found: {retrieval_file}")
                continue
            
            logger.info(f"Loading retrieval results from {retrieval_file}")
            retrieval_results = load_retrieval_results(retrieval_file)
            logger.info(f"Loaded results for {len(retrieval_results)} queries")
            
            # Process this method
            process_method(
                config=config,
                assessor=assessor,
                queries=queries,
                retrieval_results=retrieval_results,
                corpus=corpus,
                method=method,
                corpus_version=corpus_version,
                output_dir=output_dir
            )
    
    logger.info("\nAssessment complete")


if __name__ == "__main__":
    main()
