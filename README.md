# CKD Information Chatbot

A retrieval-augmented chatbot that answers questions about chronic kidney disease (CKD), grounded in publicly available NHS content. Built as one component of a wider undergraduate dissertation project on a clinical companion app for CKD patients.

The chatbot is designed around three principles motivated by the healthcare context:

- **Privacy** — the language model runs locally via Ollama; patient questions are never sent to a third-party cloud API.
- **Safety** — responses are constrained to information retrieved from NHS source pages, and queries falling below a similarity threshold are rejected without invoking the LLM.
- **Transparency** — answers cite the specific NHS source pages used.

## Architecture

```
User question
     │
     ▼
Embedding (MiniLM) ──► FAISS retrieval ──► Similarity threshold check
                                                    │
                                  below threshold ──┴──► "out of scope" reply
                                                    │
                                  above threshold ──┴──► Llama 3.1 8B (via Ollama)
                                                              │
                                                              ▼
                                                  Answer + NHS source citations
```

## Prerequisites

- **Python 3.11** — [download](https://www.python.org/downloads/release/python-3119/)
- **Ollama** — [download](https://ollama.com/), with the `llama3.1:8b` model pulled:
  ```bash
  ollama pull llama3.1:8b
  ```
- Ollama must be running before the server is started; it listens on `localhost:11434` by default.

## Setup

```bash
git clone git@gitlab.aber.ac.uk:prs49/ckd-chatbot-dissertation.git
cd ckd-chatbot-dissertation

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate     # Windows (Git Bash)
# source venv/bin/activate       # macOS / Linux

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

```bash
uvicorn server:app --reload
```

The API will be available at `http://localhost:8000`. Endpoints:

- `GET  /health` — health check
- `POST /chat`   — submit a question, optionally with conversation history

## Testing

```bash
pytest
```

The test suite covers input validation, the greeting shortcut, off-topic fallback, the retrieval flow, conversation history handling, Ollama error handling, and CORS configuration.

## Project structure

```
.
├── data/
│   ├── raw/                       # Scraped NHS content
│   ├── processed/                 # Cleaned and chunked content
│   ├── index/                     # FAISS index and metadata
│   └── evaluation/                # Gold-standard test set + retrieval evaluation reports
├── scripts/
│   ├── scraper_nhs_ckd.py         # Scrapes NHS CKD pages
│   └── chunks_processor.py        # Processes raw content into retrieval chunks
├── tests/
│   └── test_server.py             # API integration tests
├── evaluate_retrieval1.py         # Retrieval evaluation harness
├── server.py                      # FastAPI application
└── requirements.txt
```

## Models

| Component  | Model                                       |
|------------|---------------------------------------------|
| LLM        | `llama3.1:8b` (served via Ollama)           |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`    |

## Context

This repository contains the chatbot subsystem of the dissertation project at Aberystwyth University. The wider clinical companion app (exercise tracking, mental wellbeing assessment, clinician–patient messaging) is maintained in a separate repository which has been submitted as a seperate folder in the zip file.