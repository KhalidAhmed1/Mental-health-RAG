
# Mental-health-RAG

Comprehensive yet lightweight Retrieval-Augmented Generation (RAG) pipeline focused on emotion-aware, context-sensitive mental-health assistance. The project demonstrates how to combine document embeddings, a retrieval index, and a large language model to produce grounded, empathetic responses.

Key features

- Emotion-aware retrieval and response generation.
- Modular components for retrieval, intent classification, and language generation.
- Reproducible notebook pipeline for indexing and evaluation.

Quick start

1. Requirements: Python 3.8+ and a virtual environment.
2. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies (create `requirements.txt` if needed):

```powershell
pip install -r requirements.txt
```

4. Run the example pipeline:

```powershell
jupyter notebook rag_pipeline.ipynb
```

Repository layout

- `artifacts/` — saved embeddings, index, and metadata
- `data/` — source CSVs and preprocessed datasets
- `models/` — model weights and tokenizer files
- `modules/` — core modules: `retriever.py`, `llm.py`, `emotion.py`, `intent_classifier.py`, `language.py`
- `tests/` — unit/integration tests (e.g., `test_rag_evaluation.py`)

Usage

- Reproduce the pipeline by running `rag_pipeline.ipynb`.
- Import modules from `modules/` to integrate the retriever and LLM into other applications.

Testing

- Run tests with `pytest`:

```powershell
pytest -q
```

Contributing

- Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests for changes.

License & contact

- This repository does not include an explicit license file. Add one if you plan to publish or share the project.
- For questions, provide contact details or open an issue in this repository.

If you want, I can add an explicit `requirements.txt`, CI configuration, or expand the Usage examples.
