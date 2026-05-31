import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mental_health_chunks"
MODEL_NAME      = "all-mpnet-base-v2"
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT     = 0.3
CANDIDATE_POOL  = 100  # Increased from 50 to maximize base document recall
RERANK_POOL     = 35   # Expanded so the cross-encoder has a larger window to select from

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load everything once at import time ─────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer(MODEL_NAME)

print("Loading reranker...")
# UPGRADE: Replaced old MS-MARCO with a modern, high-performance contextual reranker
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

print("Connecting to Qdrant...")
qdrant_client = QdrantClient(
    url     = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY"),
)

print("Building BM25 index...")
_chunks_path     = os.path.join(_BASE_DIR, "data", "chunks.parquet")
_bm25_texts      = pd.read_parquet(_chunks_path)["text"].tolist()
_tokenized       = [text.lower().split() for text in _bm25_texts]
bm25_index       = BM25Okapi(_tokenized)
print(f"BM25 index built on {len(_bm25_texts):,} documents")


# ── Main retrieval function ──────────────────────────────────────────────────
def retrieve_chunks_hybrid(query_text: str, top_k: int = 5) -> list[str]:
    """
    Optimized Hybrid retrieval using Reciprocal Rank Fusion (RRF) + BGE Reranking
    
    Steps:
    1. Semantic  → Qdrant returns top CANDIDATE_POOL results
    2. BM25      → Local index scores all docs, take top CANDIDATE_POOL
    3. RRF       → Merges ranks without score scale distortion
    4. Payload   → Fetches top RERANK_POOL payloads from Qdrant
    5. Rerank    → Cross-encoder re-orders results to maximize Context Precision
    """

    # ── 1. Semantic search ──────────────────────────────────────────────────
    q_emb = embed_model.encode([query_text], normalize_embeddings=True)[0].tolist()

    semantic_hits = qdrant_client.query_points(
        collection_name = COLLECTION_NAME,
        query           = q_emb,
        limit           = CANDIDATE_POOL,
        with_payload    = False, # Saves bandwidth during initial stage ranking
    )
    # Map document ID to its 1-indexed rank position
    semantic_ranks = {r.id: i + 1 for i, r in enumerate(semantic_hits.points)}

    # ── 2. BM25 search ──────────────────────────────────────────────────────
    tokenized_query = query_text.lower().split()
    bm25_all_scores = bm25_index.get_scores(tokenized_query)

    bm25_top_ids = np.argsort(bm25_all_scores)[::-1][:CANDIDATE_POOL]
    # Map document ID to its 1-indexed rank position
    bm25_ranks = {int(i): rank + 1 for rank, i in enumerate(bm25_top_ids)}

    # ── 3. Reciprocal Rank Fusion (RRF) ─────────────────────────────────────
    all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
    
    combined = []
    RRF_CONSTANT = 60  # Standard smoothing constant to balance rank weights

    for cid in all_ids:
        sem_rank = semantic_ranks.get(cid)
        bm_rank  = bm25_ranks.get(cid)
        
        rrf_score = 0.0
        if sem_rank is not None:
            rrf_score += SEMANTIC_WEIGHT * (1.0 / (RRF_CONSTANT + sem_rank))
        if bm_rank is not None:
            rrf_score += BM25_WEIGHT * (1.0 / (RRF_CONSTANT + bm_rank))
            
        combined.append((cid, rrf_score))

    # Sort items based on their unified RRF position scores
    combined.sort(key=lambda x: x[1], reverse=True)
    candidate_ids = [cid for cid, _ in combined[:RERANK_POOL]]

    # ── 4. Fetch Payloads ───────────────────────────────────────────────────
    fetched = qdrant_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=candidate_ids,
        with_payload=True,
    )

    id_to_payload = {r.id: r.payload for r in fetched}

    candidate_texts = [
        (cid, id_to_payload[cid]["text"])
        for cid in candidate_ids
        if cid in id_to_payload
    ]

    # ── 5. Rerank with Cross-Encoder ────────────────────────────────────────
    pairs = [[query_text, text] for _, text in candidate_texts]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidate_texts, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [text for ((_, text), _score) in reranked[:top_k]]