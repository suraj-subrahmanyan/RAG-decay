"""
Dense Index Builder for Sentence Transformers

Builds FAISS indices from document corpora using dense embedding models.
Supports configurable sequence length, batch processing, and Flash Attention 2.
"""

import os
import yaml
import torch
import json
import argparse
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_config: Dict, device: str):
    model_name = model_config['model_name']
    dtype_str = model_config.get('dtype', 'float16')
    # CRITICAL: Get max_length from config, default to 4096 if missing
    max_length = model_config.get('max_length', 4096)
    
    print(f"Loading model: {model_name}")
    print(f"Requested dtype: {dtype_str}")
    
    model_kwargs = {}
    
    # Set Data Type
    if dtype_str == 'bfloat16' and torch.cuda.is_bf16_supported():
        model_kwargs['torch_dtype'] = torch.bfloat16
        print("Using BFloat16 (Optimized for Ampere)")
    else:
        model_kwargs['torch_dtype'] = torch.float16
        print("Using Float16")
    
    # Enable Flash Attention 2
    try:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            trust_remote_code=True
        )
        print("Flash Attention 2 enabled")
    except Exception as e:
        print(f"Flash Attention 2 not available: {e}")
        model_kwargs.pop('attn_implementation', None)
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            trust_remote_code=True
        )
    
    # Enforce max sequence length from config
    # Some models default to very long contexts which can slow processing
    model.max_seq_length = max_length
    print(f"Max sequence length set to: {model.max_seq_length}")

    return model

def build_index(config: Dict, model_key: str, corpus_version: str, output_dir: str, batch_size_override: int = None, device_override: str = None):
    model_conf = config['dense_models'][model_key]
    device = device_override or config['device']
    batch_size = batch_size_override or model_conf.get('batch_size', 32)
    corpus_path = config['corpus_files'][corpus_version]
    
    print(f"Starting indexing: {model_key.upper()} ({corpus_version})")
    print(f"   Batch Size: {batch_size}")
    print(f"   Output: {output_dir}")

    # Load Model
    model = load_model(model_conf, device)
    
    # Create Output Dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Corpus
    print("Reading corpus...")
    texts = []
    docids = []
    
    with open(corpus_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get('text') or item.get('page_content') or ""
            texts.append(text)
            doc_id = item.get('id') or item.get('_id') or item.get('doc_id')
            docids.append(doc_id)

    print(f"Loaded {len(texts):,} documents.")

    # Initialize FAISS
    dimension = model_conf['embedding_dim']
    index = faiss.IndexFlatIP(dimension)
    
    print(f"Encoding {len(texts)} documents with Smart Batching...")
    
    # Smart Batching: Pass all texts at once
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )
    
    # Cast to float32 for FAISS
    if embeddings.dtype == np.float16 or embeddings.dtype == torch.bfloat16:
        embeddings = embeddings.astype(np.float32)
    
    print(f"\nFinal embeddings shape: {embeddings.shape}")
    print(f"Adding to FAISS index...")
    index.add(embeddings)

    # Save
    print(f"Saving artifacts to {output_dir}...")
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
    
    with open(os.path.join(output_dir, "docids.pkl"), "wb") as f:
        pickle.dump(docids, f)
    
    idx_size = os.path.getsize(os.path.join(output_dir, "index.faiss")) / (1024*1024)
    print(f"Indexing complete. Index size: {idx_size:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    build_index(config, args.model, args.corpus_version, args.output_dir, args.batch_size, args.device)
