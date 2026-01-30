# Retrieval Overlap Analysis: Justifying Hybrid Fusion

**Figure: Retrieval Method Overlap (Jaccard Similarity @ 20)**

## Motivation

RAG systems traditionally rely on a single retrieval method—either sparse lexical search (BM25) or dense semantic search (embedding-based). Recent work suggests these approaches access fundamentally different aspects of document relevance: BM25 excels at keyword matching and entity recognition, while dense retrievers capture semantic similarity and paraphrastic content.

We measured overlap between retrieval methods using Jaccard similarity on the top-20 retrieved documents for each query to test whether these methods retrieve substantially different documents, which would justify hybrid fusion.

**Jaccard Similarity:** `|A ∩ B| / |A ∪ B|`

Where A and B are the sets of top-20 document IDs retrieved by methods A and B for the same query.

## Methods Compared

- **BM25**: Sparse lexical retrieval (Pyserini/Lucene)
- **BGE**: Dense retrieval (bge-base-en-v1.5)
- **E5**: Instruction-tuned dense retrieval (e5-mistral-7b)
- **Qwen**: Dense retrieval (Qwen2.5-Coder-7B)

## Key Findings

### Sparse-Dense Orthogonality (4-5% Overlap)

- BM25 ↔ BGE: 4% overlap
- BM25 ↔ E5: 4% overlap
- BM25 ↔ Qwen: 5% overlap

Sparse and dense methods retrieve 95-96% different documents, confirming that lexical and semantic retrieval access fundamentally different semantic spaces.

**Example:** For a query about "parsing JSON with LangChain," BM25 retrieved documentation pages matching the keywords "JSON" and "LangChain," while BGE retrieved code implementations of JSON parsers and readers, with zero document overlap in the top-20.

### Dense Method Similarity (15-18% Overlap)

- BGE ↔ E5: 16% overlap
- BGE ↔ Qwen: 15% overlap
- E5 ↔ Qwen: 18% overlap

Dense embedding models show moderate similarity, suggesting they cluster within the same semantic space but differ in fine-grained ranking. This 15-18% overlap is significantly higher than the 4-5% overlap with BM25, confirming the lexical-semantic divide.

### Temporal Consistency

Overlap patterns remained stable across corpus versions (Oct 2024 vs Oct 2025), indicating that retrieval divergence is a fundamental characteristic of sparse-dense differences, not an artifact of specific corpus content.

## Implications for Fusion

Low Jaccard overlap provides a necessary condition for fusion effectiveness: methods must retrieve different documents for fusion to add value. Our 4-5% overlap confirms this.

**Fusion Strategy:** We employ Reciprocal Rank Fusion (RRF) to combine scores from BM25, BGE, E5, and Qwen without requiring score calibration. Given the low overlap, fusion is not simply re-ranking the same documents—it actively expands the candidate pool with documents missed by individual methods.

## Related Work

Retrieval method complementarity has been observed in prior work:

- **TREC experiments** (1990s-2000s) showed that combining multiple systems improved recall through document set diversity
- **Recent neural IR research** (Lin et al., 2021; Thakur et al., 2021) demonstrated that sparse-dense fusion improves zero-shot retrieval effectiveness, though most studies focused on static corpora

Our contribution extends this analysis to temporal evaluation, confirming that complementarity persists even as documentation evolves. This suggests fusion provides robust performance across corpus drift, a critical property for production RAG systems.

---

**Files:** `overlap_oct_2024.png`, `overlap_oct_2025.png`  
**Data:** Averaged across 812 queries (Oct 2024) and 798 queries (Oct 2025) using "answer" query field
