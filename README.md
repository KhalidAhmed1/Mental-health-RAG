# Mental-health-RAG
Emotion-aware RAG chatbot for empathetic, context-aware mental health support using NLP, embeddings, and LLMs.

---


# Mental-health-RAG: Intent Classifier

An intelligent Intent Classifier designed to route user queries efficiently within a Mental Health Retrieval-Augmented Generation (RAG) system. The project features a reusable classifier module with serialization support, local config storage, and an interactive CLI loop.

## 📁 Project Structure

```text
Mental-health-RAG/
├── .venv/                  # Virtual environment
├── saved/                  # Serialized models and configurations
│   └── classifier_config.json
├── .python-version         # Managed Python version
├── intentClassifier.ipynb  # Interactive development and testing notebook
├── intentExamples.json     # Intent training datasets/examples
├── main.py                 # Main execution script (Interactive CLI loop)
├── pyproject.toml          # Project metadata and dependencies (uv-managed)
├── uv.lock                 # Fast, reproducible dependency lockfile
└── README.md               # Project documentation