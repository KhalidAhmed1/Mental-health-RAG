import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch
from datasets import Dataset
from openai import OpenAI  
from ragas import evaluate

# Prevent local tokenizer deadlocks on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory  
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.llm import get_rag_answer


@pytest.fixture(scope="module")
def evaluator_llm():
    """Initializes Groq model with expanded token headrooms to prevent JSON truncations."""
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return llm_factory(
        model="llama-3.1-8b-instant",
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
    """Loads a small evaluation slice from your local cleaned dataset."""
    csv_path = os.path.join("data", "df_cleaned.csv")
    if not os.path.exists(csv_path):
        pytest.fail(f"Cleaned dataset missing at {csv_path}.")
    
    df = pd.read_csv(csv_path)
    return df.sample(5, random_state=42)


def test_rag_pipeline_accuracy(evaluator_llm, evaluator_embeddings, sample_test_data):
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for _, row in sample_test_data.iterrows():
        question = row["Context"]
        ground_truth = row["Response"]
        simulated_chunks = [ground_truth] 
        
        with patch("modules.llm.retrieve_chunks_hybrid") as mock_retrieve:
            mock_retrieve.return_value = simulated_chunks
            
            answer = get_rag_answer(
                original_query=question,
                detected_language="english",
                emotion="neutral",
                history=[],
                top_k=5  
            )

        questions.append(question)
        answers.append(answer)
        contexts_list.append(simulated_chunks)
        ground_truths.append(ground_truth)

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

    import numpy as np
    
    faithfulness_score = df_results['faithfulness'].mean()
    answer_relevancy_score = df_results['answer_relevancy'].mean()
    context_precision_score = df_results['context_precision'].mean()

    print(f"Calculated Faithfulness: {faithfulness_score:.4f}")
    print(f"Calculated Answer Relevancy: {answer_relevancy_score:.4f}")
    print(f"Calculated Context Precision: {context_precision_score:.4f}")

    # Threshold checks
    assert faithfulness_score >= 0.40, f"Faithfulness {faithfulness_score} below 0.40"
    assert answer_relevancy_score >= 0.25, f"Relevancy {answer_relevancy_score} below 0.25"