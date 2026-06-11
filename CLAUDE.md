# Fuzzy RAG — Biomedical QA

Hybrid retrieval-augmented generation system for biomedical QA using dense embeddings, BM25, and fuzzy metadata matching.

## Architecture

- **Backend**: Flask REST API (`backend/`)
- **Frontend**: Streamlit UI (`frontend/`)
- **Database**: PostgreSQL 16 + pgvector extension
- **Orchestration**: Docker Compose

## Quick Start

```bash
cp .env.example .env
# Optionally set RADQA_DATA_PATH; no API key needed (local model)
docker compose up -d
# Index datasets (PubMedQA + MedQA auto-download; RadQA requires local files)
docker compose exec backend python -c "from indexing.indexer import run_indexing; run_indexing()"
```

## Datasets

| Dataset | Source | Access |
|---------|--------|--------|
| PubMedQA | HuggingFace `qiaojin/PubMedQA` / `pqa_labeled` | Public |
| MedQA | HuggingFace `bigbio/med_qa` / `med_qa_en_bigbio_qa` | Public |
| RadQA | PhysioNet `radqa/1.0.0` | Credentialed — place files in `RADQA_DATA_PATH` |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | RAG query: `{"question": "...", "dataset_filter": null}` |
| POST | `/index` | Trigger indexing: `{"datasets": ["pubmedqa", "medqa", "radqa"]}` |
| GET | `/index/status` | Indexing progress |
| POST | `/eval` | Evaluate: `{"questions": [...], "references": [...]}` |

## Stack

- `BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` — 512-dim biomedical CLIP text embeddings (`open_clip`)
- `pgvector` HNSW index — approximate nearest-neighbour search
- PostgreSQL `tsvector` + GIN index — BM25 sparse retrieval
- `rapidfuzz` — fuzzy metadata matching
- RRF (k=60) — fuses ranked lists from all three retrievers
- `google/medgemma-4b-it` — answer generation (local HuggingFace model, 4-bit NF4 via `bitsandbytes`)
- Requires NVIDIA GPU + `nvidia-container-toolkit` for Docker GPU passthrough

## Environment Variables

See `.env.example` for all configuration options.
