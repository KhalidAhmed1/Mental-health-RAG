import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mental_health_chunks"
MODEL_NAME      = "all-MiniLM-L6-v2"
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT     = 0.3
CANDIDATE_POOL  = 50

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load everything once at import time ─────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer(MODEL_NAME)

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


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize(score_dict: dict) -> dict:
    """Min-max normalize a {id: score} dict to [0, 1]."""
    if not score_dict:
        return {}
    values = list(score_dict.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


# ── Main retrieval function ──────────────────────────────────────────────────
def retrieve_chunks_hybrid(query_text: str, top_k: int = 5) -> list[str]:
    """
    Hybrid retrieval: 0.7 * semantic + 0.3 * BM25

    Steps:
    1. Semantic  → Qdrant returns top CANDIDATE_POOL results with scores
    2. BM25      → local index scores all docs, take top CANDIDATE_POOL
    3. Union     → merge both candidate sets
    4. Normalize → scale each score set to [0, 1] independently
    5. Combine   → final = 0.7 * sem_norm + 0.3 * bm25_norm
    6. Return    → fetch payloads from Qdrant for the top_k winners
    """

    # ── 1. Semantic search ──────────────────────────────────────────────────
    q_emb = embed_model.encode([query_text], normalize_embeddings=True)[0].tolist()

    semantic_hits   = qdrant_client.query_points(
        collection_name = COLLECTION_NAME,
        query           = q_emb,
        limit           = CANDIDATE_POOL,
        with_payload    = True,
    )
    semantic_scores = {r.id: r.score for r in semantic_hits.points}

    # ── 2. BM25 search ──────────────────────────────────────────────────────
    tokenized_query = query_text.lower().split()
    bm25_all_scores = bm25_index.get_scores(tokenized_query)

    bm25_top_ids    = np.argsort(bm25_all_scores)[::-1][:CANDIDATE_POOL]
    bm25_scores     = {int(i): float(bm25_all_scores[i]) for i in bm25_top_ids}

    # ── 3. Union ────────────────────────────────────────────────────────────
    all_ids  = set(semantic_scores.keys()) | set(bm25_scores.keys())

    # ── 4. Normalize ────────────────────────────────────────────────────────
    sem_norm  = _normalize(semantic_scores)
    bm25_norm = _normalize(bm25_scores)

    # ── 5. Combine ──────────────────────────────────────────────────────────
    combined = []
    for cid in all_ids:
        sem   = sem_norm.get(cid,  0.0)
        bm25  = bm25_norm.get(cid, 0.0)
        score = SEMANTIC_WEIGHT * sem + BM25_WEIGHT * bm25
        combined.append((cid, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in combined[:top_k]]

    # ── 6. Fetch payloads ───────────────────────────────────────────────────
    fetched       = qdrant_client.retrieve(
        collection_name = COLLECTION_NAME,
        ids             = top_ids,
        with_payload    = True,
    )
    id_to_payload = {r.id: r.payload for r in fetched}

    return [id_to_payload[cid]["text"] for cid in top_ids if cid in id_to_payload]
