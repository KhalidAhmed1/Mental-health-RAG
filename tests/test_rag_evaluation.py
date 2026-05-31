import os
import sys
import pytest
import pandas as pd
from datasets import Dataset
from openai import OpenAI  
from ragas import evaluate
import numpy as np 

# Prevent local tokenizer deadlocks on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory  
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your live pipeline AND your retriever module
from modules.llm import get_rag_answer
from modules.retriever import retrieve_chunks_hybrid


@pytest.fixture(scope="module")
def evaluator_llm():
    """Initializes Groq model with expanded token headrooms to prevent JSON truncations."""
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return llm_factory(
        model="qwen/qwen3-32b",
        client=groq_client,
        max_tokens=4096
    )


@pytest.fixture(scope="module")
def evaluator_embeddings():
    """Initializes wrapped Hugging Face embeddings with native langchain attributes."""
    return HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2"
    )


@pytest.fixture(scope="module")
def sample_test_data():
    """Loads a small evaluation slice with true randomness on every execution pass."""
    csv_path = os.path.join("data", "df_cleaned.csv")
    if not os.path.exists(csv_path):
        pytest.fail(f"Cleaned dataset missing at {csv_path}.")
    
    df = pd.read_csv(csv_path)
    return df.sample(5, random_state=np.random.RandomState())


def test_rag_pipeline_accuracy(evaluator_llm, evaluator_embeddings, sample_test_data):
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    print("\nExecuting live evaluation loop across knowledge base...")

    for _, row in sample_test_data.iterrows():
        question = row["Context"]       # The user query
        ground_truth = row["Response"]  # The targeted golden human response
        
        # 1. Capture the REAL chunks from your database for Ragas tracking
        # (No patch mock used here)
        actual_live_chunks = retrieve_chunks_hybrid(question, top_k=5)
        
        # 2. Let your live pipeline process the answer naturally
        answer = get_rag_answer(
            original_query=question,
            detected_language="english",
            emotion="sadness", # Maps to your systemic EMOTION_TONE prompt instructions
            history=[],
            top_k=5
        )

        questions.append(question)
        answers.append(answer)
        contexts_list.append(actual_live_chunks) # Ragas now evaluates real context windows!
        ground_truths.append(ground_truth)

    # Wrap up elements into standard evaluation structures
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })

    f_metric = Faithfulness(llm=evaluator_llm)
    ar_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    cp_metric = ContextPrecision(llm=evaluator_llm)

    metrics = [f_metric, ar_metric, cp_metric]

    # Run your evaluation pass
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics
    )

    print("\n================ Ragas Evaluation Summary ================")
    df_results = results.to_pandas()
    print(df_results[['faithfulness', 'answer_relevancy', 'context_precision']])
    print("==========================================================")
    
    faithfulness_score = df_results['faithfulness'].mean()
    answer_relevancy_score = df_results['answer_relevancy'].mean()
    context_precision_score = df_results['context_precision'].mean()
    print(f"Calculated Faithfulness: {faithfulness_score:.4f}")
    print(f"Calculated Answer Relevancy: {answer_relevancy_score:.4f}")
    print(f"Calculated Context Precision: {context_precision_score:.4f}")

    # Adjusted baseline thresholds for live open-weights validation
    assert faithfulness_score >= 0.35, f"Faithfulness {faithfulness_score} below target boundaries."
    assert answer_relevancy_score >= 0.25, f"Relevancy {answer_relevancy_score} below target boundaries."