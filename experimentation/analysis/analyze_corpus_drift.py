import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
SEMANTIC_AVAILABLE = True

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)




class CorpusDriftAnalyzer:
    def __init__(self, v1_path: Path, v2_path: Path, output_dir: Path):
        self.v1_path = v1_path
        self.v2_path = v2_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.v1_data = []
        self.v2_data = []
        self.v1_files = defaultdict(list)  # map file_path -> list of chunk texts
        self.v2_files = defaultdict(list)  # map file_path -> list of chunk texts
        
        self.results = {}

    def load_data(self):
        """Loads JSONL data and organizes it by file_path."""
        logger.info(f"Loading V1: {self.v1_path}")
        self.v1_data, self.v1_files = self._load_jsonl(self.v1_path)
        
        logger.info(f"Loading V2: {self.v2_path}")
        self.v2_data, self.v2_files = self._load_jsonl(self.v2_path)

    def _load_jsonl(self, path: Path) -> Tuple[List[Dict], Dict[str, List[str]]]:
        data = []
        files = defaultdict(list)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    doc = json.loads(line)
                    data.append(doc)
                    
                    # Group by file_path for file-level analysis (not chunk-level)
                    fpath = doc.get("metadata", {}).get("file_path", "unknown")
                    files[fpath].append(doc)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            
        # Sort chunks by start_byte to reconstruct entire file text
        final_files = {}
        for fpath, chunks in files.items():
            chunks.sort(key=lambda x: x.get("metadata", {}).get("start_byte", 0))
            final_files[fpath] = [c.get("text", "") for c in chunks]
            
        return data, final_files

    # Structural Volatility (File Churn)
    def analyze_structural_volatility(self):
        logger.info("Analyzing Structural Volatility...")
        
        # get file path
        set_v1 = set(self.v1_files.keys())
        set_v2 = set(self.v2_files.keys())
        
        intersection = set_v1.intersection(set_v2)
        added = set_v2 - set_v1
        deleted = set_v1 - set_v2
        
        jaccard_index = len(intersection) / len(set_v1.union(set_v2)) if set_v1 or set_v2 else 0
        
        self.results["structural"] = {
            "v1_file_count": len(set_v1),
            "v2_file_count": len(set_v2),
            "common_files": len(intersection),
            "files_added": len(added),
            "files_deleted": len(deleted),
            "jaccard_index": jaccard_index,
            "churn_rate": (len(added) + len(deleted)) / len(set_v1) if set_v1 else 0
        }

    # Chunking Artifacts (Sliding Window & Length)
    def analyze_chunk_artifacts(self):
        logger.info("Analyzing Chunk Artifacts...")
        
        # 1. Chunk Length Distribution
        v1_lengths = [len(d.get("text", "")) for d in self.v1_data]
        v2_lengths = [len(d.get("text", "")) for d in self.v2_data]
        
        self.results["chunking"] = {
            "v1_avg_len": np.mean(v1_lengths),
            "v2_avg_len": np.mean(v2_lengths),
            "v1_total_chunks": len(self.v1_data),
            "v2_total_chunks": len(self.v2_data)
        }
        
        # Plot Histograms
        plt.figure(figsize=(10, 6))
        plt.hist(v1_lengths, bins=50, alpha=0.5, label='Version 1 (Old)', color='blue', density=True)
        plt.hist(v2_lengths, bins=50, alpha=0.5, label='Version 2 (New)', color='orange', density=True)
        plt.xlabel('Chunk Length (characters)')
        plt.ylabel('Density')
        plt.title('Chunk Length Distribution Shift')
        plt.legend()
        plt.savefig(self.plot_dir / "chunk_length_dist.png")
        plt.close()
        
        # 2. Chunk Count Delta for Common Files
        # Does the same file now split into more chunks in the new version?
        common_files = set(self.v1_files.keys()).intersection(set(self.v2_files.keys()))
        deltas = []
        for f in common_files:
            c1 = len(self.v1_files[f])
            c2 = len(self.v2_files[f])
            deltas.append(c2 - c1)
            
        # plot a histogram of deltas
        plt.figure(figsize=(10, 6))
        plt.hist(deltas, color='green', alpha=0.7)
        plt.xlabel('Chunk Count Delta (V2 - V1)')
        plt.ylabel('File Count')
        plt.title('Chunk Count Change for Common Files')
        plt.savefig(self.plot_dir / "chunk_count_delta.png")
        plt.close()
            
        self.results["chunking"]["avg_chunk_count_delta"] = np.mean(deltas)

    # Semantic Drift (Meaning Shift)
    def analyze_semantic_drift(self):
        logger.info("Analyzing Semantic Drift...")
        common_files = list(set(self.v1_files.keys()).intersection(set(self.v2_files.keys())))
        #common_files = common_files[:1000] # first 1000 for testing
        scores = []
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for fpath in tqdm(common_files, desc="Calculating Embeddings"):
            # Reconstruct full file text
            text_v1 = "\n".join(self.v1_files[fpath])
            text_v2 = "\n".join(self.v2_files[fpath])
            
            # note: there is a limit to the length of text that can be embedded
            emb = model.encode([text_v1, text_v2], show_progress_bar=False)
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
            scores.append(sim)

        avg_drift = np.mean(scores) if scores else 0
        self.results["semantic"] = {
            "average_similarity": float(avg_drift),
            "drift_score": 1.0 - float(avg_drift), # Higher is more drift
            "method": "embedding_cosine"
        }
        
        # Plot Drift Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, color='purple', alpha=0.7)
        plt.xlabel('Similarity Score (0=Changed, 1=Identical)')
        plt.ylabel('File Count')
        plt.title('Semantic Stability of Common Files')
        plt.savefig(self.plot_dir / "semantic_stability.png")
        plt.close()

    # Technical Shift (Code vs Prose Density)
    def _calculate_code_density(self, text: str) -> float:
        """Heuristic to determine percentage of 'code-like' characters."""
        if not text: return 0.0
        code_chars = set("{}[]()=;_<>/")
        code_keywords = ["def ", "class ", "import ", "return ", "var ", "const ", "public ", "private ", "function ", "if ", "else ", "for ", "while ", "#include ", "using "
                         ,"switch ", "case ", "try ", "catch ", "async ", "await ", "let ", "console.log", "System.out.println", "printf(", "println(", "std::", "->", "::",
                         "template ", "typename ", "typedef ", "struct ", "enum ", "namespace ", "final ", "static ", "void ", "int ", "float ", "double ", "string ", "char ", "bool ", "boolean ",
                         "and", "or", "not", "null", "nullptr", "this", "self", "super", "new ", "delete ", "throw ", "throws ", "yield ", "with ", "as ", "from ", "import ", "export ",
                         "try", "except", "finally", "lambda ", "await ", "async ", "global ", "nonlocal ", "pass", "break", "continue", "raise ", "assert ", "defun ", "let*", "cond ", "car "]
        
        char_score = sum(1 for c in text if c in code_chars) / len(text)
        keyword_score = sum(1 for k in code_keywords if k in text)
        
        # Normalize roughly (heuristics tailored for Python/JS)
        density = (char_score * 0.7) + (min(keyword_score, 5) * 0.06)
        return min(density, 1.0)

    def analyze_technical_shift(self):
        logger.info("Analyzing Technical Shift...")
        
        v1_densities = [self._calculate_code_density(d.get("text", "")) for d in self.v1_data]
        v2_densities = [self._calculate_code_density(d.get("text", "")) for d in self.v2_data]
        
        self.results["technical"] = {
            "v1_code_density": float(np.mean(v1_densities)),
            "v2_code_density": float(np.mean(v2_densities)),
            "shift_direction": "More Code" if np.mean(v2_densities) > np.mean(v1_densities) else "More Prose"
        }


    # Vocabulary "Deprecation" Scan
    def analyze_vocabulary(self):
        logger.info("Analyzing Vocabulary & Deprecation...")
        keywords = ["deprecated", "legacy", "migration", "removed", "v1", "v2"]
        
        v1_counts = {k: 0 for k in keywords}
        v2_counts = {k: 0 for k in keywords}
        
        # Simple scan
        for d in self.v1_data:
            txt = d.get("text", "").lower()
            for k in keywords:
                if k in txt: v1_counts[k] += 1
                
        for d in self.v2_data:
            txt = d.get("text", "").lower()
            for k in keywords:
                if k in txt: v2_counts[k] += 1
                
        self.results["vocabulary"] = {
            "v1_counts": v1_counts,
            "v2_counts": v2_counts
        }

    def generate_report(self):
        report_path = self.output_dir / "drift_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(" CORPUS DRIFT ANALYSIS REPORT")
        
        # Print Structural
        s = self.results["structural"]
        logger.info(f"[Structure] File Retention: {s['jaccard_index']:.2%}")
        logger.info(f"[Structure] Added: {s['files_added']} | Deleted: {s['files_deleted']}")
        
        # Print Semantic
        sem = self.results["semantic"]
        logger.info(f"[Semantic]  Avg File Similarity: {sem['average_similarity']:.3f} ({sem['method']})")
        
        # Print Technical
        tech = self.results["technical"]
        logger.info(f"[Style] V1 Density: {tech['v1_code_density']:.3f} -> V2 Density: {tech['v2_code_density']:.3f}")
        logger.info(f"[Style]Shift: {tech['shift_direction']}")
        
        logger.info(f"Plots saved to: {self.plot_dir}")
        logger.info(f"Full metrics: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze drift between two corpus versions.")
    parser.add_argument("--v1", type=str, required=True, help="Path to V1 corpus.jsonl")
    parser.add_argument("--v2", type=str, required=True, help="Path to V2 corpus.jsonl")
    parser.add_argument("--output", type=str, default="./analysis_results", help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = CorpusDriftAnalyzer(Path(args.v1), Path(args.v2), Path(args.output))
    analyzer.load_data()
    
    analyzer.analyze_structural_volatility()
    analyzer.analyze_chunk_artifacts()
    analyzer.analyze_semantic_drift()
    analyzer.analyze_technical_shift()
    analyzer.analyze_vocabulary()
    
    analyzer.generate_report()