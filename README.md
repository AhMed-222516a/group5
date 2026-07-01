# Intelligent Support Ticket Classification with RAG

A **Retrieval-Augmented Generation (RAG)** system that classifies and resolves customer support tickets using semantic search (FAISS) and a Flan-T5 text generator.

## Architecture

```
User Query → Embedding → FAISS Search → Retrieved Tickets → Prompt → Flan-T5 → Response
```

| Component | Technology |
|-----------|-----------|
| **Vector Store** | FAISS (IndexFlatIP, cosine similarity) |
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) |
| **Generation Model** | `google/flan-t5-base` (fallback: `flan-t5-small`) |
| **API Framework** | FastAPI + Uvicorn |
| **Dataset** | 1,000 synthetic support tickets, 6 categories |

## Project Structure

```
├── src/                    # Application source code
│   ├── app.py              # FastAPI REST API
│   ├── config.py           # Central configuration
│   ├── model_loader.py     # Model loading (embedding + generation)
│   ├── rag_pipeline.py     # RAG orchestration
│   ├── retrieval.py        # FAISS retrieval module
│   └── utils.py            # Shared utilities
├── data/
│   ├── raw/                # Raw dataset CSVs
│   ├── processed/          # Intermediate files
│   └── models/faiss/       # Persisted FAISS index + metadata
├── notebooks/
│   ├── 01_generate_data.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_test_pipeline.ipynb
├── reports/figures/        # EDA and evaluation plots
├── scripts/
│   └── generate_data.py    # Synthetic data generator
├── docker/
│   └── Dockerfile          # Multi-stage Docker build
├── requirements/
│   ├── deployment.txt      # Production dependencies
│   └── full.txt            # All dependencies (incl. notebooks)
└── README.md
```

## Setup

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements/deployment.txt
```

### Running the API

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Using the RAG Pipeline

```python
from src.rag_pipeline import generate_response

response = generate_response("I cannot login to my account")
print(response.generated_response)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Liveness check |
| `GET` | `/health` | Detailed health probe |
| `POST` | `/predict` | RAG prediction |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"query": "I was charged twice for my subscription"}'
```

## Configuration

All settings are in `src/config.py`:

- **FAISS**: top_k=3, min_similarity=0.5, max_contexts=2
- **Generation**: flan-t5-base, temperature=0.3, max_new_tokens=128
- **Device**: auto-detected (CUDA or CPU)

## Docker Deployment

```bash
docker build -f docker/Dockerfile -t support-ticket-rag .
docker run -p 8000:8000 support-ticket-rag
```

## Dataset

1,000 synthetic support tickets across 6 categories:

| Category | Distribution |
|----------|-------------|
| Technical | 22% |
| Shipping | 18% |
| Billing | 16% |
| Account | 16% |
| Product | 15% |
| General | 13% |

## Improvements

- **Restructured** project into `src/`, `data/`, `notebooks/`, `reports/`
- **Optimised RAG prompt** for Flan-T5's 512-token limit — compact context format, removed few-shot example
- **Improved generation** parameters (lower temperature, fewer tokens) for focused answers
- **Reduced retrieval** top_k=3, max_contexts=2 for higher relevance
- **Consolidated** graphs and reports under `reports/figures/`
- **Cleaned up** unused directories, duplicated files, and temporary artifacts
