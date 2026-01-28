import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm
from datasets import Dataset

from pydantic import BaseModel
import outlines
from outlines import models
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import FRESHRAG_LISTWISE_NUGGET_V4

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SupportedDocument(BaseModel):
    doc_index: int
    reasoning: str


class NuggetJudgment(BaseModel):
    nugget_index: int
    supported_documents: List[SupportedDocument]


class AssessmentResponse(BaseModel):
    assessments: List[NuggetJudgment]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class QwenAssessor:
    """Qwen model for nugget-level relevance assessment."""
    
    def __init__(self, config: dict):
        """Initialize Qwen model with Outlines."""
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

        # Initialize Outlines model
        hf_tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_config['name'], 
            dtype=torch_dtype,
            trust_remote_code=model_config.get('trust_remote_code', True),
            attn_implementation="flash_attention_2"
        ).to(model_config['device'])
        
        self.model = outlines.from_transformers(hf_model, hf_tokenizer)
        
        logger.info("Outlines generator initialized")
    
    def assess_nugget_support(
        self,
        question: str,
        answer: str,
        nuggets: List[str],
        documents: List[Dict[str, str]]
    ) -> AssessmentResponse:
        """
        Assess which documents support which nuggets.
        """
        if not documents:
            return {"assessments": []}

        # Format nuggets
        nuggets_text = "\n".join([f"{i+1}. {nugget}" for i, nugget in enumerate(nuggets)])
        
        # Format documents
        docs_text = "\n\n".join([
            f"Document {i+1} (ID: {doc['doc_id']}):\n{doc['text'][:2000]}"  # Truncate long docs
            for i, doc in enumerate(documents)
        ])
        
        try:
            # Format prompt explicitly since we are not using LangChain chain
            prompt = FRESHRAG_LISTWISE_NUGGET_V4.format(
                question=question,
                answer=answer,
                nuggets=nuggets_text,
                context=docs_text,
                count=len(documents)
            )
            #logging.info(prompt)

            result_pydantic = self.model(
                prompt,
                AssessmentResponse,
                max_new_tokens=4096,
                temperature=0.1
            )
            #logging.info(type(result_pydantic))
            #logging.info(type(AssessmentResponse.model_validate_json(result_pydantic)))
            #logging.info(AssessmentResponse.model_validate_json(result_pydantic))
            
            # Return Pydantic object directly
            return AssessmentResponse.model_validate_json(result_pydantic)
        except Exception as e:
            logger.error(f"Error in assessment: {e}")
            # Return empty judgments on error
            return AssessmentResponse(
                assessments=[
                    NuggetJudgment(
                        nugget_index=i+1,
                        supported_documents=[]
                    )
                    for i in range(len(nuggets))
                ]
            )


def load_corpus_documents(corpus_file: Path) -> Dict[str, Dict]:
    """
    Load all documents from a single corpus file.
    
    Args:
        corpus_file: Path to corpus_{version}.jsonl file
        
    Returns:
        Dict mapping doc_id to document data
    """
    logger.info(f"Loading corpus from {corpus_file}")
    
    documents = {}
    
    if not corpus_file.exists():
        logger.error(f"Corpus file not found: {corpus_file}")
        return documents
        
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


def load_retrieval_results(
    results_file: Path,
    top_k: int
) -> Dict[str, List[Dict]]:
    """
    Load retrieval results grouped by query_id.
    
    Args:
        results_file: Path to retrieval results JSONL
        top_k: Maximum number of results per query
        
    Returns:
        Dict mapping query_id to list of retrieval results
    """
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


def assess_query(
    query_data: dict,
    retrieval_results: List[Dict],
    corpus_docs: Dict[str, Dict],
    assessor: QwenAssessor,
    doc_batch_size: int = 20
) -> List[Dict]:
    """
    Assess nugget support for a single query.
    
    Args:
        query_data: Query with nuggets
        retrieval_results: Retrieved documents for this query
        corpus_docs: All corpus documents
        assessor: QwenAssessor instance
        
    Returns:
        List of retrieval results with nugget_level_judgment added
    """
    nuggets = query_data.get('nuggets', [])
    
    if not nuggets:
        logger.warning(f"Query {query_data['query_id']} has no nuggets")
        # Add 0 judgment to all results
        for result in retrieval_results:
            result['nugget_level_judgment'] = 0
        return retrieval_results
    
    # Deduplicate documents for assessment
    unique_doc_ids = list(set(r['doc_id'] for r in retrieval_results))
    doc_judgments = {}
    
    # Process in batches
    for i in tqdm(range(0, len(unique_doc_ids), doc_batch_size), leave=False, desc="Assessing nugget support"):
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
        
        # Assess batch
        assessment = assessor.assess_nugget_support(
            query_data.get('question', ''),
            query_data.get('answer', ''),
            nuggets,
            batch_documents
        )
        
        # Handle AssessmentResponse object
        if isinstance(assessment, AssessmentResponse):
            assessments_list = assessment.assessments
        else:
            logger.error(f"Assessment result is not an AssessmentResponse: {type(assessment)}")
            continue
            
        # Initialize judgments for this batch
        # Map 1-based index to doc_id
        index_to_doc_id = {j+1: doc['doc_id'] for j, doc in enumerate(batch_documents)}
        
        # Initialize batch doc judgments if not exists
        for doc_id in batch_ids:
            if doc_id not in doc_judgments:
                doc_judgments[doc_id] = {'score': 0, 'details': []}

        # Collect judgments
        for item in assessments_list:
            # item is NuggetJudgment
            nugget_index = item.nugget_index
            supported_docs = item.supported_documents
            
            for doc_support in supported_docs:
                doc_idx = doc_support.doc_index
                reasoning = doc_support.reasoning
                
                if doc_idx in index_to_doc_id:
                    supported_doc_id = index_to_doc_id[doc_idx]
                    doc_judgments[supported_doc_id]['score'] = 1
                    doc_judgments[supported_doc_id]['details'].append({
                        'nugget_index': nugget_index,
                        'reasoning': reasoning
                    })

    # Map judgments back to all results
    for result in retrieval_results:
        judgment_info = doc_judgments.get(result['doc_id'], {'score': 0, 'details': []})
        result['nugget_level_judgment'] = judgment_info['score']
        result['nugget_support_details'] = judgment_info['details']
    
    return retrieval_results


def process_method(
    config: dict,
    corpus_version: str,
    method: str,
    queries: List[dict],
    corpus_docs: Dict[str, Dict],
    assessor: QwenAssessor
):
    """Process assessment for a specific retrieval method."""
    logger.info(f"Processing {method} for {corpus_version}")
    
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
    output_file = output_dir / f"{method}_assessed.jsonl"
    
    processed_queries = set()
    write_mode = 'w'
    
    if output_file.exists() and not config.get('overwrite', False):
        logger.info(f"Checking for existing results in {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            result = json.loads(line)
                            processed_queries.add(result['query_id'])
                        except json.JSONDecodeError:
                            continue
            logger.info(f"Found {len(processed_queries)} already processed queries. Resuming...")
            write_mode = 'a'
        except Exception as e:
            logger.warning(f"Error reading existing results: {e}. Starting fresh.")
            write_mode = 'w'
    else:
        logger.info("Starting fresh assessment (overwrite or new file)")
        
    # Filter queries
    queries_to_process = [q for q in queries if q['query_id'] not in processed_queries]
    
    if not queries_to_process:
        logger.info(f"All {len(queries)} queries already assessed for {method}. Skipping.")
        return

    # Process each query
    logger.info(f"Assessing {len(queries_to_process)} queries (skipped {len(processed_queries)})...")
    
    with open(output_file, write_mode, encoding='utf-8') as fout:
        for query in tqdm(queries_to_process, desc=f"Assessing {method}"):
            query_id = query['query_id']
            
            if query_id not in retrieval_results:
                logger.warning(f"No retrieval results for query {query_id}")
                continue
            
            # Assess this query
            assessed_results = assess_query(
                query,
                retrieval_results[query_id],
                corpus_docs,
                assessor,
                config.get('doc_batch_size', 20)
            )
            
            # Write results
            for result in assessed_results:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
    
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Nugget-level relevance assessment with Qwen"
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
        default="oct_2025",
        help="Corpus version to process"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["fusion"], # ["bge", "qwen", "e5", "fusion", "bm25"]
        help="Retrieval methods to assess (default: all from config)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['overwrite'] = args.overwrite
    
    # Determine versions and methods
    if args.corpus_version == "all":
        versions = config['corpus_versions']
    else:
        versions = [args.corpus_version]
    
    if args.methods and "all" not in args.methods:
        methods = args.methods
    else:
        methods = config['methods']
    
    logger.info(f"Nugget-Level Relevance Assessment")
    logger.info(f"Corpus versions: {versions}")
    logger.info(f"Methods: {methods}")
    
    # Initialize assessor
    assessor = QwenAssessor(config)
    
    # Load queries
    queries = load_queries(Path(config['queries_file']))
    
    # Process each version
    for version in versions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {version}")
        logger.info(f"{'='*60}")
        
        # Load corpus for this version
        corpus_file = Path(config['corpus_base_dir']) / f"corpus_{version}.jsonl"
        corpus_docs = load_corpus_documents(corpus_file)
        logger.info(f"Loaded {len(corpus_docs)} documents")
        
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
    
    logger.info("\nASSESSMENT COMPLETE")


if __name__ == "__main__":
    main()
