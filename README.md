# Mental-health-RAG

An emotion-aware Retrieval-Augmented Generation (RAG) chatbot that delivers empathetic, context-sensitive mental health support. The system combines multilingual understanding, real-time emotion detection, intent routing, and hybrid semantic search to generate grounded, compassionate responses.

---

## Features

- **Multilingual support** — Detects the user's language (20+ languages) and responds in kind, translating queries to English for retrieval before translating answers back.
- **Emotion-aware tone** — Classifies the emotional state of each message (sadness, joy, fear, anger, love, surprise) and adjusts the LLM's response tone accordingly.
- **Intent routing** — Routes greetings, farewells, gratitude, and out-of-scope questions directly without hitting the RAG pipeline, reserving retrieval for genuine mental health queries.
- **Hybrid retrieval** — Combines dense semantic search (Qdrant + `all-MiniLM-L6-v2`) with sparse BM25 scoring at a 0.7 / 0.3 blend for high-quality chunk retrieval.
- **Conversation history** — Injects prior turns into the LLM call for coherent multi-turn dialogue.
- **RAG evaluation suite** — Automated RAGAS evaluation of Faithfulness, Answer Relevancy, and Context Precision against a held-out slice of the cleaned dataset.

---

## Project Structure

```
Mental-health-RAG/
├── .env                        # API keys and service URLs (never commit)
├── .python-version             # Pinned Python version
├── app.py                      # Streamlit / FastAPI entry point
├── pipeline.py                 # Top-level orchestration pipeline
├── intentExamples.json         # Few-shot examples for the intent classifier
│
├── artifacts/
│   ├── chunk_embeddings.joblib # Precomputed chunk embeddings (cached)
│   └── index_metadata.joblib   # BM25 index metadata (cached)
│
├── data/
│   ├── chunks.parquet          # Chunked knowledge base (used for BM25)
│   
│
├── models/
│   ├── language_svc_model.pkl  # Trained SVC model for language detection
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer for language detection
│   ├── emotion.pkl    # trained model for emotion detection
│   
│
├── modules/
│   ├── emotion.py              # Emotion detection pipeline
│   ├── intent_classifier.py    # LLM-based intent classifier (LangChain + Groq)
│   ├── language.py             # Language detection (SVC + TF-IDF)
│   ├── llm.py                  # RAG answer generation and direct responses
│   └── retriever.py            # Hybrid BM25 + semantic retrieval (Qdrant)
│
└── tests/
    └── test_rag_evaluation.py  # RAGAS evaluation test suite
```

---

## Architecture

```
User Input
    │
    ▼
Language Detection (SVC)
    │
    ├─ Non-English ──► Translate to English (LLM)
    │
    ▼
Intent Classification (Groq LLM + few-shot)
    │
    ├─ greeting / goodbye / gratitude / out_of_scope
    │       └──► Direct Response (LLM)
    │
    └─ asking_mental_health_question
            │
            ▼
        Emotion Detection (HuggingFace)
            │
            ▼
        Hybrid Retrieval
        ├── Semantic Search  (Qdrant, weight=0.7)
        └── BM25 Search      (rank-bm25, weight=0.3)
            │
            ▼
        RAG Prompt Builder
        (tone-adjusted system message + retrieved context)
            │
            ▼
        LLM Generation (Groq)
            │
            ▼
        Response in User's Language
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key
- A [Qdrant](https://qdrant.tech) cluster (cloud or local)

### Installation

```bash
git clone https://github.com/KhalidAhmed1/Mental-health-RAG.git
cd Mental-health-RAG

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
Groq_API_KEY=your_groq_api_key_here
Groq_BASE_URL=https://api.groq.com/openai/v1
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key_here
```

> The `Groq_API_KEY` and `Groq_BASE_URL` variables point to the Groq endpoint, which exposes an OpenAI-compatible API.

### Data & Model Files

Before running, ensure the following files are present:

| Path | Description |
|---|---|
| `data/chunks.parquet` | Chunked knowledge base for BM25 and Qdrant ingestion |
| `data/df_cleaned.csv` | Cleaned Q&A dataset for evaluation |
| `models/language_svc_model.pkl` | Trained SVC language detector |
| `models/tfidf_vectorizer.pkl` | TF-IDF vectorizer paired with the SVC |
| `models/emotion.pkl`  | HuggingFace `text-classification` model directory |
| `intentExamples.json` | Few-shot examples for intent classification |

---

## Usage

### Run the App

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Pipeline (programmatic)

```python
from pipeline import run_pipeline

response = run_pipeline(
    user_input="أشعر بالقلق الشديد ولا أستطيع النوم",
    history=[]
)
print(response)
```

### Key Modules

**Language detection**
```python
from modules.language import predict_language

lang = predict_language("Je me sens très anxieux")
# → "French"
```

**Emotion detection**
```python
from modules.emotion import predict_emotion

emotion = predict_emotion("I feel so hopeless and alone")
# → "sadness"
```

**Intent classification**
```python
from modules.intent_classifier import IntentClassifier

clf = IntentClassifier()
intent = clf.predict("How do I cope with panic attacks?")
# → "asking_mental_health_question"
```

**RAG answer**
```python
from modules.llm import get_rag_answer

answer = get_rag_answer(
    original_query="How can I manage anxiety?",
    detected_language="English",
    emotion="fear",
    history=[],
    top_k=5
)
```

---

## Evaluation

The test suite uses [RAGAS](https://github.com/explodinggradients/ragas) to score the pipeline on three metrics against a 5-sample slice of `df_cleaned.csv`.

```bash
pytest tests/test_rag_evaluation.py -v
```

| Metric | Minimum threshold |
|---|---|
| Faithfulness | ≥ 0.40 |
| Answer Relevancy | ≥ 0.25 |
| Context Precision | measured, no hard threshold |

The evaluator uses `llama-3.1-8b-instant` via Groq and `all-MiniLM-L6-v2` embeddings from HuggingFace. Retrieval is mocked during evaluation so that ground-truth chunks are injected directly, isolating generation quality from retrieval noise.

---

## Models

| Component | Model | Source |
|---|---|---|
| Language detection | SVC + TF-IDF | Trained locally, saved as `.pkl` |
| Emotion classification | Fine-tuned transformer | HuggingFace (local `models/` dir) |
| Intent classification | `llama-3.3-70b-versatile` | Groq API |
| Translation | `llama-3.1-8b-instant` | Groq API |
| RAG generation | `llama-3.1-8b-instant` | Groq API |
| Embeddings (retrieval) | `all-MiniLM-L6-v2` | `sentence-transformers` |

---

## Supported Languages

Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Hindi, Italian, Japanese, Polish, Portuguese, Russian, Spanish, Swahili, Thai, Turkish, Urdu, Vietnamese.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow the existing module structure — new capabilities belong in `modules/`.
3. Add or update tests in `tests/` for any pipeline changes.
4. Open a pull request with a clear description of the change.

---

## Disclaimer

This chatbot is intended as a **supportive tool only** and is not a substitute for professional mental health care. If you or someone you know is in crisis, please contact a licensed mental health professional or a crisis helpline in your country.

---