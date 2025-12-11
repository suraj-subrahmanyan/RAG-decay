import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import List

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import DECOMPOSITIONAL_QUERIES_PROMPT, DIRECT_ANSWER_PROMPT

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


class QwenGenerator:
    """Generator for Qwen model inference."""
    
    def __init__(self, config: dict):
        """
        Initialize Qwen model.
        
        Args:
            config: Configuration dictionary
        """
        model_config = config['model']
        self.gen_config = config['generation']
        
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
        
        self.device = model_config['device']
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config['max_new_tokens'],
                temperature=self.gen_config['temperature'],
                top_p=self.gen_config['top_p'],
                do_sample=self.gen_config['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],  # Skip input prompt
            skip_special_tokens=True
        )
        
        return generated_text.strip()


def parse_subquestions(generated_text: str) -> List[str]:
    """
    Parse subquestions from generated text.
    
    Expected format: (1) question (2) question (3) question
    
    Args:
        generated_text: Generated text containing subquestions
        
    Returns:
        List of subquestions
    """
    # Look for "Sub-Questions" section
    if "** Sub-Questions **:" in generated_text:
        subq_section = generated_text.split("** Sub-Questions **:")[-1]
    else:
        subq_section = generated_text
    
    # Extract numbered questions
    # Pattern: (1) text (2) text (3) text
    pattern = r'\(\d+\)\s*([^(]+?)(?=\(\d+\)|$)'
    matches = re.findall(pattern, subq_section, re.DOTALL)
    
    # Clean up matches
    subquestions = [match.strip() for match in matches if match.strip()]
    
    return subquestions


def parse_answer(generated_text: str) -> str:
    """
    Parse answer from generated text.
    
    Expected format: ** Reasoning **: ... ** Answer **: ...
    
    Args:
        generated_text: Generated text containing answer
        
    Returns:
        Extracted answer
    """
    # Look for "Answer" section
    if "** Answer **:" in generated_text:
        answer_section = generated_text.split("** Answer **:")[-1]
        return answer_section.strip()
    
    # Fallback: return everything after reasoning
    if "** Reasoning **:" in generated_text:
        parts = generated_text.split("** Reasoning **:")
        if len(parts) > 1:
            return parts[-1].strip()
    
    # Last resort: return as-is
    return generated_text.strip()


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
    generator: QwenGenerator,
) -> List[dict]:
    """
    Generate closed-book answers and subquestions for all queries.
    
    Args:
        queries: List of query dictionaries
        generator: QwenGenerator instance
        
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
        
        try:
            # Generate closed-book answer
            answer_prompt = DIRECT_ANSWER_PROMPT.format(question=question)
            answer_output = generator.generate(answer_prompt)
            closed_book_answer = parse_answer(answer_output)
            
            # Generate subquestions
            subq_prompt = DECOMPOSITIONAL_QUERIES_PROMPT.format(question=question)
            subq_output = generator.generate(subq_prompt)
            subquestions = parse_subquestions(subq_output)
            
            # Add to query
            query['closed_book_answer'] = closed_book_answer
            query['subquestions'] = subquestions
            
        except Exception as e:
            logger.error(f"Error generating for query {query.get('query_id')}: {e}")
            query['closed_book_answer'] = ""
            query['subquestions'] = []
        
        enriched_queries.append(query)

    
    logger.info("Generation complete")
    
    # Statistics
    with_answers = sum(1 for q in enriched_queries if q.get('closed_book_answer'))
    with_subq = sum(1 for q in enriched_queries if q.get('subquestions'))
    avg_subq = sum(len(q.get('subquestions', [])) for q in enriched_queries) / len(enriched_queries)
    
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
        description="Add Qwen-generated answers and subquestions to queries"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/query_preperation_config.yaml",
        help="Path to configuration YAML file (default: config_qwen.yaml)"
    )
    
    args = parser.parse_args()
    

    logger.info("Add Qwen Generations to Queries")
    # Load config
    config = load_config(args.config)
    
    
    # Initialize generator
    generator = QwenGenerator(config)
    
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