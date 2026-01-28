
import argparse
import json
import logging
import sys
from pathlib import Path

# Add directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

from nugget_assessment import QwenAssessor, load_config, load_queries, load_retrieval_results
from prompts import FRESHRAG_LISTWISE_NUGGET_V4

from nugget_assessment import load_corpus_documents

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test assessment logic for a single query")
    parser.add_argument("--results-file", type=str, required=True, help="Path to retrieval results jsonl")
    parser.add_argument("--query-id", type=str, required=True, help="Query ID to assess")
    parser.add_argument("--config", type=str, default="config/config_assessment.yaml", help="Path to config file")
    parser.add_argument("--corpus-dir", type=str, default="dataset/langchain/", help="Path to corpus directory")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to assessment dir
        config_path = Path(__file__).parent / args.config
        
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        return

    logger.info(f"Loading config from {config_path}")
    config = load_config(str(config_path))
    
    # Load assessor
    logger.info("Initializing assessor...")
    assessor = QwenAssessor(config)
    
    # Load query info
    queries_file = Path(config['queries_file'])
    logger.info(f"Loading queries from {queries_file}")
    queries = load_queries(queries_file)
    query_data = next((q for q in queries if q['query_id'] == args.query_id), None)
    
    if not query_data:
        logger.error(f"Query {args.query_id} not found in queries file")
        return

    logger.info(f"Structure for Query {args.query_id} found.")
    logger.info(f"Question: {query_data.get('question', '')[:100]}...")
    logger.info(f"Nuggets: {len(query_data.get('nuggets', []))}")

    # Load results to get documents
    results_file = Path(args.results_file)
            
    if not results_file.exists():
        logger.error(f"Results file not found: {args.results_file}")
        return

    logger.info(f"Loading results from {results_file}")
    # Load all results first
    all_results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    all_results.append(json.loads(line))
                except:
                    pass
    
    query_results = [r for r in all_results if r['query_id'] == args.query_id]
    
    # Sort by rank
    query_results.sort(key=lambda x: x['rank'])
    
    # Take top K (e.g. 5 for testing)
    top_k = 5
    query_results = query_results[:top_k]
    logger.info(f"Testing with top {len(query_results)} documents")
    
    documents = []
    corpus_docs = {}
    
    # Cache for loaded corpora: topic -> {doc_id: doc_data}
    loaded_corpora = {}
    
    if query_results:
        year_dir = results_file.parent.name
        base_corpus_dir = Path(args.corpus_dir)
        
        # Pre-load corpora for all documents in results to be tested
        for res in query_results:
            doc_id = res['doc_id']
            # Parse topic from doc_id (e.g., langchain/docs/...)
            parts = doc_id.split('/')
            if not parts:
                continue
            topic = parts[0]
            
            if topic not in loaded_corpora:
                corpus_filename = f"corpus.{topic}.jsonl"
                corpus_path = base_corpus_dir / year_dir / corpus_filename
                   
                if corpus_path.exists():
                    logger.info(f"Loading corpus for topic '{topic}' from {corpus_path}")
                    loaded_docs = load_corpus_documents(corpus_path)
                    loaded_corpora[topic] = loaded_docs
                else:
                    logger.warning(f"Corpus file not found for topic '{topic}': {corpus_path}")
                    loaded_corpora[topic] = {} # Mark as tried/empty

    for i, res in enumerate(query_results):
        doc_id = res['doc_id']
        text = ''
        
        # Find in loaded corpora
        parts = doc_id.split('/')
        if parts:
            topic = parts[0]
            real_doc_id = "/".join(parts[1:])
            if topic in loaded_corpora and real_doc_id in loaded_corpora[topic]:
                text = loaded_corpora[topic][real_doc_id].get('text', '')
        
        if not text:
            text = f"[Mock text for document {doc_id} - text not found in corpus]"
            logger.warning(f"Text not found for document {doc_id}")
            
        documents.append({
            'doc_id': doc_id,
            'text': text
        })

    # Run assessment
    logger.info("Running assessment...")
    result = assessor.assess_nugget_support(
        query_data['question'],
        query_data['answer'],
        query_data['nuggets'],
        documents
    )
    
    print("\n" + "="*50)
    print("ASSESSMENT RESULT (JSON)")
    print("="*50)
    
    # Manually print the result structure nicely since it's a dict now, or just json dump it
    
    print("\nDETAILS PER DOCUMENT:")
    print(result.model_dump_json(indent=2, ensure_ascii=False))

    print("="*50)
    
if __name__ == "__main__":
    main()
