"""
Convert LLM assessment results to qrels format for evaluation.

Input: Assessment JSONL (query_id, doc_id, nugget_support_details)
Output: qrels JSON ({query_id_nuggetidx: {doc_id: relevance}})
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def convert_assessments_to_qrels(assessment_file: Path) -> dict:
    """
    Convert assessment results to qrels format.
    
    Assessment format:
    {
      "query_id": "...",
      "doc_id": "...",
      "nugget_support_details": [
        {"nugget_index": 1, "support": "Full Support"},
        ...
      ]
    }
    
    qrels format:
    {
      "qrels_nuggets": {
        "query_id_nuggetindex": {
          "doc_id": 1 or 0
        }
      }
    }
    """
    
    # Structure: qrels[query_nugget_key][doc_id] = relevance
    qrels = defaultdict(dict)
    
    # Track all query-doc pairs to ensure we have 0s for non-relevant
    query_docs = defaultdict(set)
    query_nuggets = defaultdict(set)
    
    print(f"Reading {assessment_file}...")
    
    with open(assessment_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}")
                continue
            
            query_id = str(data['query_id'])
            doc_id = data['doc_id']
            nugget_details = data.get('nugget_support_details', [])
            
            # Track this doc for this query
            query_docs[query_id].add(doc_id)
            
            # Process each nugget's support
            for detail in nugget_details:
                nugget_idx = detail['nugget_index']
                support = detail.get('support', '')
                
                # Track that this query has this nugget
                query_nuggets[query_id].add(nugget_idx)
                
                # Create qrels key
                qrels_key = f"{query_id}_{nugget_idx}"
                
                # Mark as relevant (1) if Full Support, otherwise 0
                relevance = 1 if support == "Full Support" else 0
                qrels[qrels_key][doc_id] = relevance
    
    # Ensure all docs have labels for all nuggets (fill in missing 0s)
    print("Filling in missing judgments...")
    for query_id in query_nuggets:
        docs = query_docs[query_id]
        for nugget_idx in query_nuggets[query_id]:
            qrels_key = f"{query_id}_{nugget_idx}"
            for doc_id in docs:
                if doc_id not in qrels[qrels_key]:
                    qrels[qrels_key][doc_id] = 0
    
    # Convert defaultdict to regular dict
    qrels_dict = {
        "qrels_nuggets": {k: dict(v) for k, v in qrels.items()}
    }
    
    print(f"Created qrels for {len(qrels)} query-nugget pairs")
    
    return qrels_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", required=True, help="Assessment JSONL file")
    parser.add_argument("--output", required=True, help="Output qrels JSON file")
    args = parser.parse_args()
    
    assessment_file = Path(args.assessment)
    output_file = Path(args.output)
    
    # Convert
    qrels = convert_assessments_to_qrels(assessment_file)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(qrels, f, indent=2)
    
    print(f"Saved qrels to {output_file}")
    
    # Print stats
    total_judgments = sum(len(docs) for docs in qrels["qrels_nuggets"].values())
    relevant_judgments = sum(
        sum(1 for v in docs.values() if v == 1)
        for docs in qrels["qrels_nuggets"].values()
    )
    
    print(f"\nStatistics:")
    print(f"  Query-nugget pairs: {len(qrels['qrels_nuggets'])}")
    print(f"  Total judgments: {total_judgments}")
    print(f"  Relevant (1): {relevant_judgments} ({100*relevant_judgments/total_judgments:.1f}%)")
    print(f"  Not relevant (0): {total_judgments - relevant_judgments} ({100*(total_judgments-relevant_judgments)/total_judgments:.1f}%)")


if __name__ == "__main__":
    main()
