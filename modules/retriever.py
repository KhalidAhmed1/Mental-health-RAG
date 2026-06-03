import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mental_health_rag"
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
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

print("Connecting to Qdrant...")
qdrant_client = QdrantClient(
    url     = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY"),
)

print("Building BM25 index...")
_chunks_path = os.path.join(_BASE_DIR, "data", "context_index.parquet")
# BM25 indexes the CONTEXT column — same space both retrievers operate in
_bm25_texts  = pd.read_parquet(_chunks_path)["context"].tolist()
_tokenized   = [text.lower().split() for text in _bm25_texts]
bm25_index   = BM25Okapi(_tokenized)
print(f"BM25 index built on {len(_bm25_texts):,} documents")


# ── Main retrieval function ──────────────────────────────────────────────────
def retrieve_responses_hybrid(query_text: str, top_k: int = 5) -> list[str]:
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF) + BGE Reranking.

    Steps:
    1. Semantic  → Qdrant returns top CANDIDATE_POOL results (by context embedding)
    2. BM25      → Local index scores all context docs, take top CANDIDATE_POOL
    3. RRF       → Merges ranks without score scale distortion
    4. Payload   → Fetches top RERANK_POOL payloads from Qdrant
    5. Rerank    → Cross-encoder scores (query, context) pairs — picks the most
                   relevant contexts; returns their stored original_response payloads
    """

    # ── 1. Semantic search ──────────────────────────────────────────────────
    q_emb = embed_model.encode([query_text], normalize_embeddings=True)[0].tolist()

    semantic_hits = qdrant_client.query_points(
        collection_name = COLLECTION_NAME,
        query           = q_emb,
        limit           = CANDIDATE_POOL,
        with_payload    = False,  # saves bandwidth during initial ranking stage
    )
    # Map document ID → 1-indexed rank position
    semantic_ranks = {r.id: i + 1 for i, r in enumerate(semantic_hits.points)}

    # ── 2. BM25 search ──────────────────────────────────────────────────────
    tokenized_query = query_text.lower().split()
    bm25_all_scores = bm25_index.get_scores(tokenized_query)

    bm25_top_ids = np.argsort(bm25_all_scores)[::-1][:CANDIDATE_POOL]
    # Map document ID → 1-indexed rank position
    bm25_ranks = {int(i): rank + 1 for rank, i in enumerate(bm25_top_ids)}

    # ── 3. Reciprocal Rank Fusion (RRF) ─────────────────────────────────────
    all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

    combined = []
    RRF_CONSTANT = 60  # standard smoothing constant to balance rank weights

    for cid in all_ids:
        sem_rank = semantic_ranks.get(cid)
        bm_rank  = bm25_ranks.get(cid)

        rrf_score = 0.0
        if sem_rank is not None:
            rrf_score += SEMANTIC_WEIGHT * (1.0 / (RRF_CONSTANT + sem_rank))
        if bm_rank is not None:
            rrf_score += BM25_WEIGHT * (1.0 / (RRF_CONSTANT + bm_rank))

        combined.append((cid, rrf_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    candidate_ids = [cid for cid, _ in combined[:RERANK_POOL]]

    # ── 4. Fetch payloads ───────────────────────────────────────────────────
    fetched = qdrant_client.retrieve(
        collection_name = COLLECTION_NAME,
        ids             = candidate_ids,
        with_payload    = True,
    )
    id_to_payload = {r.id: r.payload for r in fetched}

    # Reranker evaluates (query, context) pairs — the context is what determines
    # relevance. The original_response is the answer payload returned to the LLM.
    candidate_contexts = [
        (cid, id_to_payload[cid]["context"])
        for cid in candidate_ids
        if cid in id_to_payload
    ]

    # ── 5. Rerank on (query, context) — return original_response payload ────
    pairs         = [[query_text, ctx] for _, ctx in candidate_contexts]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidate_contexts, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Return the stored therapist responses for the top_k reranked contexts
    return [
        id_to_payload[cid]["original_response"]
        for (cid, _ctx), _score in reranked[:top_k]
    ]