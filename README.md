# Fuzzy RAG — Biomedical QA

Hybrid retrieval-augmented generation system for biomedical question answering. Combines dense vector search, BM25 sparse retrieval, and fuzzy metadata matching fused with Reciprocal Rank Fusion (RRF), with answer generation via a local quantized LLM.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit UI :8501                  │
└─────────────────────────┬────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼────────────────────────────┐
│                  Flask API :5000                     │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Dense   │  │  Sparse  │  │  Fuzzy Metadata  │   │
│  │  (HNSW)  │  │  (BM25)  │  │   (rapidfuzz)    │   │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       └─────────────┴──────────────────┘             │
│                   RRF Fusion (k=60)                  │
│                        │                             │
│               CheXagent-8b (4-bit NF4)               │
└──────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│           PostgreSQL 16 + pgvector :5432             │
└──────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker ≥ 24 with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with ≥ 6 GB VRAM (4-bit NF4) or ≥ 10 GB VRAM (8-bit)
- ~20 GB disk for model weights and dataset embeddings

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| Flask API | http://localhost:5000 |
| PostgreSQL | localhost:5432 |

Then index datasets (PubMedQA and MedQA download automatically; RadQA requires local files — see [RadQA Setup](#radqa-setup)):

```bash
# Trigger via API (runs in background)
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"datasets": ["pubmedqa", "medqa"]}'

# Or run directly in the backend container
docker compose exec backend python -c \
  "from indexing.indexer import run_indexing; run_indexing()"
```

## Datasets

| Dataset | Source | Notes |
|---------|--------|-------|
| PubMedQA | HuggingFace `qiaojin/PubMedQA` (`pqa_labeled`) | Downloaded automatically |
| MedQA | HuggingFace `bigbio/med_qa` (`med_qa_en_bigbio_qa`) | Downloaded automatically |
| RadQA | PhysioNet `radqa/1.0.0` | Requires credentialed access — see below |

### RadQA Setup

RadQA requires [PhysioNet credentialed access](https://physionet.org/content/radqa/1.0.0/).

1. Download the dataset and place the SQuAD-format JSON files in a local directory:
   ```
   data/radqa/
   ├── radqa_train.json
   ├── radqa_dev.json
   └── radqa_test.json
   ```
2. Set `RADQA_DATA_PATH` in `.env`:
   ```
   RADQA_DATA_PATH=./data/radqa
   ```

## Environment Variables

See `.env.example` for the full list. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `StanfordAIMI/CheXagent-8b` | HuggingFace model ID for generation |
| `LOAD_BITS` | `4` | Quantization: `4` (NF4, ~4–5 GB VRAM) or `8` (LLM.int8, ~8 GB VRAM) |
| `EMBEDDING_MODEL` | `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | BiomedCLIP text encoder |
| `VECTOR_DIM` | `512` | Embedding dimensionality |
| `RRF_K` | `60` | RRF smoothing constant |
| `TOP_K` | `5` | Number of contexts to retrieve |
| `RADQA_DATA_PATH` | `./data/radqa` | Local path to RadQA JSON files |

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns loaded model name |
| `POST` | `/query` | RAG query |
| `POST` | `/index` | Start background indexing |
| `GET` | `/index/status` | Indexing progress |
| `POST` | `/eval` | Batch BLEU / ROUGE-L / Token-F1 evaluation |

### POST /query

```json
{
  "question": "What are the symptoms of pneumonia?",
  "dataset_filter": "pubmedqa",
  "top_k": 5
}
```

`dataset_filter` is optional — omit or pass `null` to search all datasets.

```json
{
  "question": "...",
  "answer": "...",
  "contexts": [
    {"id": 1, "text": "...", "source": "pubmedqa", "rrf_score": 0.0328, "retriever": "dense"}
  ],
  "usage": {"model": "...", "input_tokens": 512, "output_tokens": 128}
}
```

### POST /eval

```json
{
  "questions": ["What causes X?", "How is Y treated?"],
  "references": ["X is caused by ...", "Y is treated with ..."]
}
```

```json
{"bleu": 0.12, "rouge_l": 0.35, "token_f1": 0.41, "n_samples": 2}
```

## Retrieval Pipeline

1. **Dense** — BiomedCLIP text embeddings (512-dim) stored in a `pgvector` HNSW index, cosine similarity
2. **Sparse** — PostgreSQL `tsvector` + GIN index, `plainto_tsquery` BM25 approximation
3. **Fuzzy metadata** — `rapidfuzz.process.extract` over pre-fetched candidate strings
4. **RRF fusion** — `score = Σ 1/(k + rank)` with k=60, deduplicates by document ID

## Project Structure

```
├── backend/
│   ├── app.py               # Flask routes
│   ├── config.py            # Config from environment variables
│   ├── database/            # PostgreSQL connection and schema init
│   ├── indexing/            # Dataset loading, embedding, and indexing
│   ├── retrieval/           # Dense, sparse, fuzzy, and RRF fusion
│   ├── generation/          # Prompt builder and LLM inference
│   └── eval/                # BLEU / ROUGE-L / Token-F1 metrics
├── frontend/
│   └── app.py               # Streamlit UI
├── scripts/
│   ├── download_datasets.py # Standalone dataset downloader
│   └── build_index.py       # Standalone indexer
├── docker-compose.yml
├── Dockerfile               # Backend image
└── .env.example
```
