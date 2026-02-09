#!/usr/bin/env python3
"""
Summarize evaluation results across all methods, query fields, and corpus versions.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_all_results(results_dir: Path):
    """Load all evaluation results."""
    all_results = []
    
    for corpus in ['oct_2024', 'oct_2025']:
        corpus_dir = results_dir / corpus
        if not corpus_dir.exists():
            continue
        
        for result_file in corpus_dir.glob('*_eval.json'):
            # Parse filename: {method}_{query_field}_eval.json
            parts = result_file.stem.replace('_eval', '').split('_')
            method = parts[0]
            query_field = '_'.join(parts[1:])
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            overall = data.get('overall_metrics', {})
            
            if overall:
                all_results.append({
                    'corpus': corpus,
                    'method': method,
                    'query_field': query_field,
                    'alpha_ndcg@10_mean': overall.get('alpha_ndcg@10', {}).get('mean', None),
                    'alpha_ndcg@10_median': overall.get('alpha_ndcg@10', {}).get('median', None),
                    'coverage@20_mean': overall.get('coverage@20', {}).get('mean', None),
                    'coverage@20_median': overall.get('coverage@20', {}).get('median', None),
                    'recall@50_mean': overall.get('recall@50', {}).get('mean', None),
                    'recall@50_median': overall.get('recall@50', {}).get('median', None),
                    'count': overall.get('alpha_ndcg@10', {}).get('count', 0),
                })
    
    return pd.DataFrame(all_results)

def main():
    results_dir = Path('evaluation_results')
    df = load_all_results(results_dir)
    
    if df.empty:
        print("No results found!")
        return
    
    print("="*80)
    print("FRESHSTACK EVALUATION RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Summary by method (averaged across query fields)
    print("="*80)
    print("1. SUMMARY BY METHOD (Oct 2024 vs Oct 2025)")
    print("="*80)
    print()
    
    for method in ['bge', 'e5', 'qwen', 'bm25', 'fusion']:
        method_df = df[df['method'] == method]
        if method_df.empty:
            continue
        
        print(f"\n{method.upper()}")
        print("-" * 60)
        
        for corpus in ['oct_2024', 'oct_2025']:
            corpus_df = method_df[method_df['corpus'] == corpus]
            if corpus_df.empty:
                continue
            
            ndcg_mean = corpus_df['alpha_ndcg@10_mean'].mean()
            cov_mean = corpus_df['coverage@20_mean'].mean()
            rec_mean = corpus_df['recall@50_mean'].mean()
            
            print(f"  {corpus:12} | α-nDCG@10: {ndcg_mean:7.4f} | Cov@20: {cov_mean:.4f} | Rec@50: {rec_mean:.4f}")
    
    # Detailed breakdown by query field
    print("\n\n" + "="*80)
    print("2. DETAILED BREAKDOWN BY QUERY FIELD")
    print("="*80)
    
    for query_field in ['question', 'closed_book_answer', 'subquestions', 'answer', 'nuggets']:
        field_df = df[df['query_field'] == query_field]
        if field_df.empty:
            continue
        
        print(f"\n\nQuery Field: {query_field.upper()}")
        print("=" * 80)
        
        pivot = field_df.pivot_table(
            index='method',
            columns='corpus',
            values=['alpha_ndcg@10_mean', 'coverage@20_mean', 'recall@50_mean'],
            aggfunc='mean'
        )
        
        print("\nα-nDCG@10 (Mean):")
        if 'alpha_ndcg@10_mean' in pivot:
            print(pivot['alpha_ndcg@10_mean'].to_string())
        
        print("\nCoverage@20 (Mean):")
        if 'coverage@20_mean' in pivot:
            print(pivot['coverage@20_mean'].to_string())
        
        print("\nRecall@50 (Mean):")
        if 'recall@50_mean' in pivot:
            print(pivot['recall@50_mean'].to_string())
    
    # Temporal degradation analysis
    print("\n\n" + "="*80)
    print("3. TEMPORAL DEGRADATION ANALYSIS (Oct 2024 → Oct 2025)")
    print("="*80)
    print()
    
    degradation = []
    for method in df['method'].unique():
        for field in df['query_field'].unique():
            oct24 = df[(df['method'] == method) & (df['query_field'] == field) & (df['corpus'] == 'oct_2024')]
            oct25 = df[(df['method'] == method) & (df['query_field'] == field) & (df['corpus'] == 'oct_2025')]
            
            if len(oct24) == 1 and len(oct25) == 1:
                ndcg_drop = oct24['alpha_ndcg@10_mean'].values[0] - oct25['alpha_ndcg@10_mean'].values[0]
                cov_drop = oct24['coverage@20_mean'].values[0] - oct25['coverage@20_mean'].values[0]
                rec_drop = oct24['recall@50_mean'].values[0] - oct25['recall@50_mean'].values[0]
                
                degradation.append({
                    'method': method,
                    'query_field': field,
                    'ndcg_drop': ndcg_drop,
                    'cov_drop': cov_drop,
                    'rec_drop': rec_drop
                })
    
    if degradation:
        deg_df = pd.DataFrame(degradation)
        
        print("\nAverage Degradation by Method:")
        print("-" * 60)
        method_deg = deg_df.groupby('method')[['ndcg_drop', 'cov_drop', 'rec_drop']].mean()
        print(method_deg.to_string())
        
        print("\n\nTop 10 Worst Degradations (by Coverage@20):")
        print("-" * 60)
        worst = deg_df.nlargest(10, 'cov_drop')[['method', 'query_field', 'ndcg_drop', 'cov_drop', 'rec_drop']]
        print(worst.to_string())
    
    # Save full results to CSV
    output_csv = 'evaluation_results_summary.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n\nFull results saved to: {output_csv}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
