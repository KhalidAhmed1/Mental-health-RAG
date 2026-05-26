# Mental Health RAG Chatbot

> **Note**: This is a development branch of the Mental Health RAG system.

## Executive Summary

An advanced Retrieval-Augmented Generation (RAG) system engineered to deliver empathetic, contextually-aware mental health support through multimodal natural language processing and machine learning techniques. The system integrates intent classification, multilingual support, and emotion-aware response generation to provide personalized therapeutic guidance.

## System Architecture

The platform comprises five integrated components:

1. **Data Pipeline**: Semantic clustering of counseling conversations for deduplication
2. **Embedding & Indexing**: Dense vector representations stored in Qdrant vector database
3. **Intent Classification**: LCEL-based routing using structured LLM outputs
4. **Retrieval Layer**: Hybrid semantic search with emotion-aware ranking
5. **Response Generation**: Context-aware LLM generation with emotional tone adaptation

## Technical Stack

- **NLP & Embeddings**: `sentence-transformers`, `NLTK`
- **Vector Database**: Qdrant with cosine similarity metrics
- **LLM Integration**: Groq API (llama-3.3-70b), OpenAI API
- **Framework**: LangChain (core, groq), Pydantic for structured outputs
- **Data Processing**: Pandas, NumPy, scikit-learn
- **ML Models**: SVC for language detection, Agglomerative Clustering for semantic deduplication

## Installation & Configuration

### Prerequisites
- Python 3.8+
- Qdrant instance (local or cloud)
- API keys for Groq and OpenAI services

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```
OPENAI_API_KEY=<your_openai_key>
QDRANT_URL=<your_qdrant_instance_url>
QDRANT_API_KEY=<your_qdrant_api_key>
```

### 3. Execute Pipeline
Launch `rag_pipeline.ipynb` in Jupyter or VS Code to execute the complete pipeline:
- Data ingestion and preprocessing
- Embedding generation (384-dim vectors)
- Vector indexing in Qdrant
- Pipeline validation and testing

## Project Structure

```
.
├── rag_pipeline.ipynb              Main execution notebook
├── requirements.txt                Python dependencies (15 packages)
├── intentExamples.json             Intent classifier training examples
├── data/
│   ├── df_cleaned.csv              Cleaned conversation dataset
│   └── semantic_clustered_rag.csv   Deduplicated responses
├── artifacts/
│   ├── chunk_embeddings.joblib     384-dim embeddings (joblib)
│   └── index_metadata.joblib       Chunk metadata records
└── models/                         Pre-trained language & emotion models
```

## Core Features

| Feature | Description |
|---------|------------|
| **Intent Routing** | 5-class classification (greeting, goodbye, gratitude, mental_health_q, out_of_scope) |
| **Multilingual Support** | 20+ languages with automatic translation to English |
| **Emotion Detection** | Real-time emotional tone classification (sadness, joy, fear, anger, neutral) |
| **Semantic Retrieval** | Hybrid BM25 + vector similarity with configurable top-k (default: 5) |
| **Context Awareness** | Conversation history integration for coherent multi-turn dialogues |
| **Response Personalization** | Emotion-aware system prompts with tone adaptation |

## Usage Example

```python
from rag_pipeline import get_rag_answer, predict_language

# Process user query
query = "I'm experiencing anxiety and insomnia"
language = predict_language(query)
response = get_rag_answer(
    original_query=query,
    detected_language=language,
    emotion="fear",
    top_k=5
)

print(response)
```

## Performance Considerations

- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Chunk Size**: 1500 characters (~375 tokens)
- **Chunk Overlap**: 150 characters for context continuity
- **Similarity Threshold**: 0.75 (semantic clustering)
- **Batch Processing**: 64 samples per inference batch