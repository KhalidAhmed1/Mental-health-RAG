import os
import sys
import pytest
import pandas as pd
import json
from datasets import Dataset
from openai import OpenAI  

# Prevent local tokenizer deadlocks on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.llm import get_rag_answer, retrieve_chunks_hybrid 

# --- Helper Metrics ---

def recall_at_k(retrieved_chunks, ground_truth):
    """
    Calculates if any retrieved chunk has meaningful token overlap with the ground truth,
    ignoring common conversational stop words.
    """
    if not ground_truth or not retrieved_chunks:
        return 0
        
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'that', 'i', 'you', 'my'}
    gt_words = set(ground_truth.lower().split()) - stop_words

    for chunk in retrieved_chunks:
        chunk_words = set(chunk.lower().split()) - stop_words
        if not chunk_words or not gt_words:
            continue
            
        # Measure overlap relative to the chunk size to account for document truncation
        overlap = len(gt_words & chunk_words) / min(len(gt_words), len(chunk_words))

        # A 35% overlap of core meaningful words indicates highly relevant context alignment
        if overlap > 0.35:
            return 1
    return 0


# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def judge_client():
    """Initializes a clean, direct Groq client for custom G-Eval auditing."""
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )


@pytest.fixture(scope="module")
def sample_test_data():
    """Loads a small evaluation slice from your local cleaned dataset."""
    csv_path = os.path.join("data", "df_cleaned.csv")
    if not os.path.exists(csv_path):
        pytest.fail(f"Cleaned dataset missing at {csv_path}.")
    
    df = pd.read_csv(csv_path)
    return df.sample(15)


# --- Test Suite ---

def test_retrieval_recall(sample_test_data):
    """
    Tests the actual hybrid retrieval performance using your optimized Recall@K logic.
    """
    print("\n================ Retrieval Recall@K Summary ================")
    recalls = []

    for i, row in sample_test_data.iterrows():
        question = row["Context"]
        ground_truth = row["Response"]
        
        # Call your live hybrid pipeline (BM25 + Qdrant + Reranker)
        retrieved_chunks = retrieve_chunks_hybrid(question, top_k=5)
        
        recall_score = recall_at_k(retrieved_chunks, ground_truth)
        recalls.append(recall_score)
        
        print(f"Sample {i} -> Recall@5: {recall_score}")

    mean_recall = sum(recalls) / len(recalls)
    print(f"Mean Recall@5: {mean_recall:.4f}")
    print("============================================================")

    assert mean_recall >= 0.40, f"Retrieval recall {mean_recall} is below target threshold."


# def test_generation_quality(judge_client, sample_test_data):
#     """
#     Tests generation quality using a direct Custom G-Eval loop with native JSON validation.
#     """
#     print("\n================ Generation Quality Audit ================")
#     scores = []

#     for i, row in sample_test_data.iterrows():
#         question = row["Context"]
#         ground_truth = row["Response"]
        
#         # 1. Generate the live answer from your pipeline
#         answer = get_rag_answer(
#             original_query=question,
#             detected_language="english",
#             emotion="Be empathetic and supportive.",
#             history=[],
#             top_k=5
#         )

#         # 2. Define custom evaluation rubric for the mental health judge
#         eval_prompt = f"""
#         You are an expert clinical AI auditor. Evaluate the generated response based on the ground truth benchmark.

#         [User Question]: {question}
#         [Ground Truth Vetted Answer]: {ground_truth}
#         [Generated Answer]: {answer}

#         Evaluation Criteria:
#         - Grounding (0-1): Does it stay true to factual boundaries without hallucinating outside coping strategies?
#         - Empathy Tone (0-1): Is the tone warm, validating, and safe for a mental health context?
        
#         Output your analysis EXACTLY in this JSON format:
#         {{
#             "reasoning": "Your brief explanation of differences or issues found",
#             "grounding_score": 1.0,
#             "empathy_score": 1.0
#         }}
#         """
        
#         # 3. Call Groq directly using native json_object response framing
#         response = judge_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",  # Using Groq's flagship versatile model
#             messages=[{"role": "user", "content": eval_prompt}],
#             response_format={"type": "json_object"},
#             temperature=0.0
#         )
        
#         try:
#             result = json.loads(response.choices[0].message.content)
#             print(f"\n[Sample {i} Audit]")
#             print(f"Reasoning: {result['reasoning']}")
#             print(f"Grounding: {result['grounding_score']} | Empathy: {result['empathy_score']}")
            
#             scores.append(result['grounding_score'])
#         except Exception as e:
#             print(f"Parsing failed for sample {i}: {e}")
#             scores.append(0.0)

#     mean_generation_score = sum(scores) / len(scores)
#     print("\n============================================================")
#     print(f"Mean Generation Quality Score: {mean_generation_score:.4f}")
    
#     assert mean_generation_score >= 0.60, "Generation quality fell below validation thresholds."