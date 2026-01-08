import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

import torch
import yaml
from tqdm import tqdm

from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompts import DECOMPOSITIONAL_QUERIES_PROMPT_TEMPLATE, DIRECT_ANSWER_PROMPT_TEMPLATE

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class QwenLangChainGenerator:
    """Generator for Qwen model inference using LangChain."""
    
    def __init__(self, config: dict):
        """
        Initialize Qwen model with LangChain.
        
        Args:
            config: Configuration dictionary
        """
        model_config = config['model']
        gen_config = config['generation']
        
        logger.info(f"Loading model: {model_config['name']}")
        logger.info(f"Device: {model_config['device']}")
        
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
            dtype=torch_dtype,
            trust_remote_code=model_config.get('trust_remote_code', True)
        ).to(model_config['device'])
        self.model.eval()
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p'],
            do_sample=gen_config['do_sample'],
            return_full_text=False, # LangChain expects generation only
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Define parsers
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser() # Fallback or for simple text
        
        # Define chains
        self.subq_chain = (
            DECOMPOSITIONAL_QUERIES_PROMPT_TEMPLATE 
            | self.llm 
            | self.json_parser
        )
        
        self.answer_chain = (
            DIRECT_ANSWER_PROMPT_TEMPLATE 
            | self.llm 
            | self.json_parser
        )
        
        logger.info("LangChain chains initialized successfully")
    
    def generate_subquestions(self, question: str) -> List[str]:
        """Generate subquestions using LangChain."""
        try:
            result = self.subq_chain.invoke({"question": question})
            # Expecting {"subquestions": ["q1", "q2"]}
            if isinstance(result, dict) and "subquestions" in result:
                return result["subquestions"]
            elif isinstance(result, list): # fallback if parser returns list directly
                return result
            else:
                logger.warning(f"Unexpected subquestion format: {result}")
                return []
        except Exception as e:
            logger.error(f"Error generating subquestions: {e}")
            return []

    def generate_answer(self, question: str) -> str:
        """Generate answer using LangChain."""
        try:
            result = self.answer_chain.invoke({"question": question})
            # Expecting {"answer": "text"}
            if isinstance(result, dict) and "answer" in result:
                return result["answer"]
            elif isinstance(result, str):
                return result
            else:
                logger.warning(f"Unexpected answer format: {result}")
                return str(result)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ""


def load_queries(queries_file: Path) -> List[dict]:
    """Load queries from JSONL file."""
    logger.info(f"Loading queries from {queries_file}")
    
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def generate_for_queries(
    queries: List[dict],
    generator: QwenLangChainGenerator,
) -> List[dict]:
    """
    Generate closed-book answers and subquestions for all queries.
    
    Args:
        queries: List of query dictionaries
        generator: QwenLangChainGenerator instance
        
    Returns:
        Enriched queries
    """
    logger.info("Generating closed-book answers and subquestions...")
    
    enriched_queries = []
    
    for query in tqdm(queries, desc="Processing queries"):
        question = query.get('question', '')
        
        if not question:
            logger.warning(f"Query {query.get('query_id')} has no question, skipping")
            query['closed_book_answer'] = ""
            query['subquestions'] = []
            enriched_queries.append(query)
            continue
        
        # Generate closed-book answer
        closed_book_answer = generator.generate_answer(question)
        
        # Generate subquestions
        subquestions = generator.generate_subquestions(question)
        
        # Add to query
        query['closed_book_answer'] = closed_book_answer
        query['subquestions'] = subquestions
            
        enriched_queries.append(query)

    
    logger.info("Generation complete")
    
    # Statistics
    with_answers = sum(1 for q in enriched_queries if q.get('closed_book_answer'))
    with_subq = sum(1 for q in enriched_queries if q.get('subquestions'))
    avg_subq = sum(len(q.get('subquestions', [])) for q in enriched_queries) / len(enriched_queries) if enriched_queries else 0
    
    logger.info(f"Queries with answers: {with_answers}/{len(enriched_queries)}")
    logger.info(f"Queries with subquestions: {with_subq}/{len(enriched_queries)}")
    logger.info(f"Average subquestions per query: {avg_subq:.1f}")
    
    return enriched_queries


def save_queries(queries: List[dict], output_file: Path, backup_suffix: str):
    """
    Save enriched queries to JSONL file.
    
    Args:
        queries: List of enriched query dictionaries
        output_file: Path to output file
        backup_suffix: Suffix for backup file
    """
    # Create backup if file exists
    if output_file.exists():
        backup_file = output_file.with_suffix(backup_suffix)
        logger.info(f"Creating backup: {backup_file}")
        shutil.copy2(output_file, backup_file)
    
    # Write enriched queries
    logger.info(f"Writing enriched queries to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(queries)} enriched queries")


def main():
    parser = argparse.ArgumentParser(
        description="Add Qwen-generated answers and subquestions to queries using LangChain"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/query_preperation_config.yaml",
        help="Path to configuration YAML file (default: config_qwen.yaml)"
    )
    
    args = parser.parse_args()
    
    logger.info("Add Qwen Generations to Queries (LangChain)")
    # Load config
    config = load_config(args.config)
    
    # Initialize generator
    try:
        generator = QwenLangChainGenerator(config)
    except ImportError as e:
        logger.error(f"Failed to import necessary libraries: {e}")
        logger.error("Please ensure langchain and langchain-huggingface are installed.")
        return
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return
    
    # Load queries
    queries_file = Path(config['queries_file'])
    if not queries_file.exists():
        logger.error(f"Queries file not found: {queries_file}")
        exit(1)
    
    queries = load_queries(queries_file)
    
    # Generate
    enriched_queries = generate_for_queries(queries, generator)
    
    # Save
    save_queries(enriched_queries, queries_file, config['backup_suffix'])
    
    # Summary
    logger.info("SUMMARY")
    logger.info(f"Successfully enriched {len(enriched_queries)} queries")
    logger.info(f"Output: {queries_file}")
    logger.info(f"Backup: {queries_file.with_suffix(config['backup_suffix'])}")


if __name__ == "__main__":
    main()