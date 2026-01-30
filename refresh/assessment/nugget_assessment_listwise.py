"""
Listwise nugget-level assessment using LLM-as-judge.

Judges which retrieved documents fully support each atomic fact (nugget)
decomposed from reference answers. Implements FreshStack listwise methodology.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# FreshStack listwise assessment prompt template
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

** Decompositional Facts **: {nugget}

** Documents **: {context}

** Output **:
"""


class QwenListwiseAssessor:
    """Qwen model for listwise nugget-level assessment."""
    
    def __init__(self, config: dict):
        """Initialize Qwen model."""
        model_config = config['model']
        gen_config = config['generation']
        
        logger.info(f"Loading model: {model_config['name']}")
        
        # Determine dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        torch_dtype = dtype_map.get(model_config['torch_dtype'], torch.bfloat16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            trust_remote_code=model_config.get('trust_remote_code', True)
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            torch_dtype=torch_dtype,
            trust_remote_code=model_config.get('trust_remote_code', True)
        ).to(model_config['device'])
        
        self.model.eval()
        self.gen_config = gen_config
        
        logger.info("Model loaded")
    
    def assess_listwise(
        self,
        question: str,
        answer: str,
        nuggets: List[str],
        documents: List[Dict[str, str]],
    ) -> Dict:
        """
        Assess which documents support which nuggets (listwise).
        
        Returns dict mapping nugget_idx -> [list of supporting doc indices]
        """
        # Format nuggets
        nuggets_text = "\n".join([
            f"Fact ({i+1}): {nugget}" 
            for i, nugget in enumerate(nuggets)
        ])
        
        # Format documents (truncate to avoid context overflow)
        docs_text = "\n\n".join([
            f"Doc ({i+1}) [ID: {doc['doc_id']}]:\n{doc['text'][:1500]}"
            for i, doc in enumerate(documents)
        ])
        
        # Create prompt
        prompt = FRESHRAG_LISTWISE_NUGGET_V4.format(
            count=len(documents),
            question=question,
            answer=answer,
            nugget=nuggets_text,
            context=docs_text
        )
        
        # Generate
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.gen_config['max_new_tokens'],
                    temperature=self.gen_config['temperature'],
                    top_p=self.gen_config['top_p'],
                    do_sample=self.gen_config['do_sample'],
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response
            return self.parse_listwise_response(response, nuggets, documents)
            
        except Exception as e:
            logger.error(f"Error in assessment: {e}")
            return {i: [] for i in range(len(nuggets))}
    
    def parse_listwise_response(self, response: str, nuggets: List[str], documents: List[Dict]) -> Dict:
        """
        Parse the listwise response with multiple fallback strategies.
        
        Tries in order:
        1. Structured parsing (expected format)
        2. Regex-based extraction (flexible)
        3. Conservative fallback (empty support)
        """
        import re
        
        # Strategy 1: Try structured parsing
        try:
            result = self._parse_structured(response, len(nuggets))
            if result is not None and len(result) == len(nuggets):
                logger.debug(f"Structured parsing successful")
                return result
        except Exception as e:
            logger.debug(f"Structured parsing failed: {e}")
        
        # Strategy 2: Try regex-based extraction
        try:
            result = self._parse_regex(response, len(nuggets))
            if result is not None and len(result) == len(nuggets):
                logger.warning(f"Fell back to regex parsing")
                return result
        except Exception as e:
            logger.debug(f"Regex parsing failed: {e}")
        
        # Strategy 3: Conservative fallback
        logger.error(f"All parsing strategies failed. Response preview:\n{response[:500]}")
        logger.warning("Using conservative fallback (all empty support)")
        return {i: [] for i in range(len(nuggets))}
    
    def _parse_structured(self, response: str, num_nuggets: int) -> Dict:
        """
        Parse structured format: Fact (N): ... VERDICT: ## Full Support: [...]
        
        Expected format:
        Fact (1): REASON: ... VERDICT: ## Full Support: [Doc (1), Doc (4)]
        Fact (2): REASON: ... VERDICT: ## Full Support: []
        """
        import re
        nugget_support = {}
        
        # Split by lines but also handle multi-line facts
        lines = response.split('\n')
        current_fact_idx = None
        current_text = ""
        
        for line in lines:
            # Check for Fact (N):
            if line.strip().startswith('Fact ('):
                try:
                    fact_num_str = line.split('Fact (')[1].split(')')[0]
                    current_fact_idx = int(fact_num_str) - 1  # 0-indexed
                    current_text = line
                except Exception as e:
                    logger.debug(f"Failed to parse fact number from: {line[:100]}, error: {e}")
                    continue
            elif current_fact_idx is not None:
                # Continue accumulating text for current fact
                current_text += " " + line
            
            # Check for VERDICT (might be on same or different line)
            if current_fact_idx is not None and 'VERDICT:' in current_text:
                # Look for Full Support marker
                if '## Full Support:' in current_text:
                    support_part = current_text.split('## Full Support:')[1]
                    
                    # Extract doc indices - be flexible with format
                    supported_docs = []
                    
                    # Try to find bracketed content
                    bracket_match = re.search(r'\[(.*?)\]', support_part)
                    if bracket_match:
                        bracket_content = bracket_match.group(1)
                        
                        # Extract Doc (N) patterns with flexible spacing
                        doc_matches = re.findall(r'Doc\s*\((\d+)\)', bracket_content)
                        if doc_matches:
                            supported_docs = [int(d) - 1 for d in doc_matches]  # 0-indexed
                    
                    nugget_support[current_fact_idx] = supported_docs
                    current_fact_idx = None
                    current_text = ""
        
        # Fill in missing nuggets
        for i in range(num_nuggets):
            if i not in nugget_support:
                nugget_support[i] = []
        
        # Validate we got all nuggets
        if len(nugget_support) != num_nuggets:
            raise ValueError(f"Expected {num_nuggets} nuggets, got {len(nugget_support)}")
        
        return nugget_support
    
    def _parse_regex(self, response: str, num_nuggets: int) -> Dict:
        """
        Regex-based extraction for varied formats.
        
        Handles:
        - "Fact 1:", "Fact (1):", "1.", etc.
        - "Full Support: [Doc 1, Doc 2]", "Documents: (1), (4)", etc.
        """
        import re
        nugget_support = {i: [] for i in range(num_nuggets)}
        
        # Try to find fact blocks with various patterns
        # Pattern 1: Fact (N): ... [Doc (1), Doc (2)]
        fact_blocks = re.findall(
            r'Fact\s*\(?(\d+)\)?:.*?(?:\[([^\]]*)\]|Documents?[:\s]+([^\n]*))',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        for match in fact_blocks:
            try:
                fact_num = int(match[0]) - 1  # 0-indexed
                if 0 <= fact_num < num_nuggets:
                    # Check both capture groups for doc references
                    doc_text = match[1] or match[2] or ""
                    
                    # Extract all digit sequences that might be doc IDs
                    doc_nums = re.findall(r'\((\d+)\)|\b(\d+)\b', doc_text)
                    # Flatten and convert to ints
                    supported = []
                    for num_tuple in doc_nums:
                        for num_str in num_tuple:
                            if num_str:
                                doc_idx = int(num_str) - 1  # 0-indexed
                                if 0 <= doc_idx < len(nugget_support):
                                    supported.append(doc_idx)
                    
                    nugget_support[fact_num] = supported
            except Exception as e:
                logger.debug(f"Regex parsing error for match {match}: {e}")
                continue
        
        return nugget_support


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_corpus_documents(corpus_file: Path) -> Dict[str, Dict]:
    """Load corpus documents."""
    logger.info(f"Loading corpus from {corpus_file}")
    documents = {}
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                documents[doc['_id']] = doc
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def load_queries(queries_file: Path) -> List[dict]:
    """Load queries with nuggets."""
    logger.info(f"Loading queries from {queries_file}")
    queries = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_retrieval_results(results_file: Path, top_k: int) -> Dict[str, List[Dict]]:
    """Load retrieval results grouped by query_id."""
    logger.info(f"Loading retrieval results from {results_file}")
    results_by_query = defaultdict(list)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                query_id = result['query_id']
                results_by_query[query_id].append(result)
    
    # Sort by rank and limit to top_k
    for query_id in results_by_query:
        results_by_query[query_id] = sorted(
            results_by_query[query_id],
            key=lambda x: x['rank']
        )[:top_k]
    
    logger.info(f"Loaded results for {len(results_by_query)} queries")
    return dict(results_by_query)


def assess_query_listwise(
    query_data: dict,
    retrieval_results: List[Dict],
    corpus_docs: Dict[str, Dict],
    assessor: QwenListwiseAssessor,
    doc_batch_size: int = 20
) -> List[Dict]:
    """
    Assess nugget support for a single query using listwise evaluation.
    
    Returns retrieval results with:
    - nugget_level_judgment: 1 if doc supports ANY nugget, 0 otherwise
    - supported_nuggets: list of nugget indices this doc supports
    """
    query_id = query_data['query_id']
    question = query_data.get('query', query_data.get('query_text', ''))
    answer = query_data.get('answer', query_data.get('answer_text', ''))
    nuggets = query_data.get('nuggets', [])
    
    if not nuggets:
        logger.warning(f"Query {query_id} has no nuggets")
        for result in retrieval_results:
            result['nugget_level_judgment'] = 0
            result['supported_nuggets'] = []
        return retrieval_results
    
    # Deduplicate documents
    unique_doc_ids = list(set(r['doc_id'] for r in retrieval_results))
    doc_to_nuggets = {}  # doc_id -> list of supported nugget indices
    
    # Process in batches
    for i in range(0, len(unique_doc_ids), doc_batch_size):
        batch_ids = unique_doc_ids[i:i + doc_batch_size]
        
        # Prepare batch documents
        batch_documents = []
        for doc_id in batch_ids:
            if doc_id in corpus_docs:
                doc = corpus_docs[doc_id]
                batch_documents.append({
                    'doc_id': doc_id,
                    'text': doc.get('text', '')
                })
            else:
                logger.warning(f"Document {doc_id} not found in corpus")
                batch_documents.append({
                    'doc_id': doc_id,
                    'text': ''
                })
        
        # Assess batch (listwise)
        nugget_support = assessor.assess_listwise(
            question=question,
            answer=answer,
            nuggets=nuggets,
            documents=batch_documents
        )
        
        # Invert: doc_id -> [nugget_indices]
        for doc_idx, doc_id in enumerate(batch_ids):
            supported_nuggets = []
            for nugget_idx, supporting_docs in nugget_support.items():
                if doc_idx in supporting_docs:
                    supported_nuggets.append(nugget_idx)
            doc_to_nuggets[doc_id] = supported_nuggets
    
    # Add judgments to results
    for result in retrieval_results:
        doc_id = result['doc_id']
        supported_nuggets = doc_to_nuggets.get(doc_id, [])
        
        result['nugget_level_judgment'] = 1 if supported_nuggets else 0
        result['supported_nuggets'] = supported_nuggets
    
    return retrieval_results


def process_method(
    config: dict,
    corpus_version: str,
    method: str,
    queries: List[dict],
    corpus_docs: Dict[str, Dict],
    assessor: QwenListwiseAssessor
):
    """Process assessment for a specific retrieval method."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {method} for {corpus_version}")
    logger.info(f"{'='*60}")
    
    # Load retrieval results
    results_dir = Path(config['retrieval_results_dir']) / corpus_version
    results_file = results_dir / f"{method}.jsonl"
    
    if not results_file.exists():
        logger.warning(f"Results file not found: {results_file}")
        return
    
    retrieval_results = load_retrieval_results(results_file, config['top_k'])
    
    # Setup output
    output_dir = Path(config['output_dir']) / corpus_version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{method}_assessed_listwise.jsonl"
    
    # Process each query
    logger.info(f"Assessing {len(queries)} queries (listwise)...")
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for query in tqdm(queries, desc=f"Assessing {method}"):
            query_id = query['query_id']
            
            if query_id not in retrieval_results:
                logger.warning(f"No retrieval results for query {query_id}")
                continue
            
            # Assess this query (listwise)
            assessed_results = assess_query_listwise(
                query,
                retrieval_results[query_id],
                corpus_docs,
                assessor,
                config.get('doc_batch_size', 20)
            )
            
            # Write results
            for result in assessed_results:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Listwise nugget-level assessment with Qwen (Mentor's enhanced prompt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_assessment.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--corpus-version",
        type=str,
        choices=["oct_2024", "oct_2025", "all"],
        default="all",
        help="Corpus version to process"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["fusion", "bge"],
        help="Retrieval methods to assess"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine versions and methods
    if args.corpus_version == "all":
        versions = config.get('corpus_versions', ['oct_2024', 'oct_2025'])
    else:
        versions = [args.corpus_version]
    
    methods = args.methods if args.methods else config.get('methods', ['fusion', 'bge'])
    
    logger.info(f"Listwise Nugget-Level Assessment (Mentor's Enhanced Prompt)")
    logger.info(f"Corpus versions: {versions}")
    logger.info(f"Methods: {methods}")
    
    # Initialize assessor
    assessor = QwenListwiseAssessor(config)
    
    # Load queries
    queries = load_queries(Path(config['queries_file']))
    
    # Process each version
    for version in versions:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {version}")
        logger.info(f"{'='*70}")
        
        # Load corpus for this version
        corpus_file = Path(config['corpus_base_dir']) / f"corpus_{version}.jsonl"
        corpus_docs = load_corpus_documents(corpus_file)
        
        # Process each method
        for method in methods:
            process_method(
                config,
                version,
                method,
                queries,
                corpus_docs,
                assessor
            )
    
    logger.info("\nListwise assessment complete")


if __name__ == "__main__":
    main()
