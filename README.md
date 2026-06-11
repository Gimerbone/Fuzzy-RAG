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
│              MedGemma-4b-IT (4-bit NF4)               │
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

## Usage Guide

### 1. First-time setup

```bash
git clone <repo>
cd fuzzy-rag
cp .env.example .env
```

MedGemma is a gated model — you need to accept the licence on HuggingFace and add your token to `.env`:

```
HF_TOKEN=hf_...
```

Pass it through in `docker-compose.yml` under `backend.environment` if it isn't there already:

```yaml
HF_TOKEN: ${HF_TOKEN}
```

If you have RadQA files, set `RADQA_DATA_PATH` in `.env` now. Otherwise leave it — PubMedQA and MedQA download automatically.

---

### 2. Start the stack

```bash
docker compose up -d
```

All three services start in order: Postgres → backend → frontend. The backend runs a Postgres healthcheck before it starts, so the DB is always ready when the app connects.

Check everything came up:

```bash
docker compose ps
docker compose logs -f backend   # watch for "Running on http://0.0.0.0:5000"
```

The **first startup takes several minutes** — the backend downloads the BiomedCLIP embedding model and the MedGemma weights (~8 GB total) on first use. Subsequent starts load from the Docker layer / HuggingFace cache.

Verify the backend is healthy:

```bash
curl http://localhost:5000/health
# {"status": "ok", "model": "google/medgemma-4b-it"}
```

---

### 3. Index the datasets

Indexing downloads and embeds the datasets into PostgreSQL. It runs in the background inside the backend container.

**Via the API:**

```bash
# PubMedQA + MedQA (auto-download, takes ~10–20 min depending on hardware)
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"datasets": ["pubmedqa", "medqa"]}'

# All three (RadQA requires RADQA_DATA_PATH to be populated)
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"datasets": ["pubmedqa", "medqa", "radqa"]}'
```

**Poll progress:**

```bash
curl http://localhost:5000/index/status
# {"status": "running", "dataset": "pubmedqa", "indexed": 450, "total": 1000}
# ...
# {"status": "done"}
```

You can also trigger indexing from the **Streamlit sidebar** (Index Datasets → Start Indexing), and poll status with the "Check Index Status" button.

---

### 4. Ask a question — Streamlit UI

Open **http://localhost:8501** in a browser.

- **Dataset filter** (sidebar) — restrict retrieval to `pubmedqa`, `medqa`, or `radqa`. Leave as "All" to search across all indexed data.
- **Top-K contexts** (sidebar slider) — how many retrieved passages to send to the LLM (1–20, default 5).
- Type your question in the text area and click **Submit**.

The response panel shows:
- The generated answer
- **Retrieved Contexts** expander — ranked passages with source label and RRF score
- **Usage** expander — model name, input/output token counts

---

### 5. Query the API directly

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the common causes of pleural effusion?",
    "dataset_filter": "pubmedqa",
    "top_k": 5
  }'
```

Omit `dataset_filter` (or set it to `null`) to search all datasets. The response includes the answer, ranked contexts, and token usage.

---

### 6. Run an evaluation

The `/eval` endpoint runs the full RAG pipeline on a batch of questions and scores the generated answers against references using BLEU, ROUGE-L, and Token-F1.

```bash
curl -X POST http://localhost:5000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is the first-line treatment for community-acquired pneumonia?",
      "What imaging findings suggest pulmonary embolism?"
    ],
    "references": [
      "Amoxicillin or doxycycline are recommended first-line treatments.",
      "CT pulmonary angiography may show filling defects in the pulmonary arteries."
    ]
  }'
# {"bleu": 0.08, "rouge_l": 0.31, "token_f1": 0.38, "n_samples": 2}
```

The same form is available in the **Batch Evaluation** expander at the bottom of the Streamlit UI (paste one question per line, one reference per line).

---

### 7. Stopping and restarting

```bash
# Stop containers, keep all indexed data
docker compose down

# Full reset — stops containers and deletes the Postgres volume
docker compose down -v

# Restart a single service after a code change
docker compose build backend
docker compose up -d --no-deps backend
```

---

## Datasets

| Dataset | Source | Notes |
|---------|--------|-------|
| PubMedQA | HuggingFace `qiaojin/PubMedQA` (`pqa_labeled`) | Downloaded automatically |
| MedQA | HuggingFace `bigbio/med_qa` (`med_qa_en_bigbio_qa`) | Downloaded automatically |
| RadQA | PhysioNet `radqa/1.0.0` | Requires credentialed access — see below |

### RadQA Setup

RadQA requires [PhysioNet credentialed access](https://physionet.org/content/radqa/1.0.0/). There are two ways to provide the data:

**Option A — automatic download (recommended)**

Set your PhysioNet credentials in `.env` and the files are downloaded on first index run:

```
PHYSIONET_USER=your_username
PHYSIONET_PASSWORD=your_password
```

**Option B — manual placement**

Download the files yourself and place them at `RADQA_DATA_PATH`:

```
data/radqa/
├── radqa_train.json
├── radqa_dev.json
└── radqa_test.json
```

In both cases `RADQA_DATA_PATH` (default `./data/radqa`) controls where the files are stored.

## Environment Variables

See `.env.example` for the full list. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `google/medgemma-4b-it` | HuggingFace model ID for generation |
| `LOAD_BITS` | `4` | Quantization: `4` (NF4, ~3–4 GB VRAM) or `8` (LLM.int8, ~6 GB VRAM) |
| `EMBEDDING_MODEL` | `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | BiomedCLIP text encoder |
| `VECTOR_DIM` | `512` | Embedding dimensionality |
| `RRF_K` | `60` | RRF smoothing constant |
| `TOP_K` | `5` | Number of contexts to retrieve |
| `RADQA_DATA_PATH` | `./data/radqa` | Local path to RadQA JSON files |
| `PHYSIONET_USER` | — | PhysioNet username for auto-download |
| `PHYSIONET_PASSWORD` | — | PhysioNet password for auto-download |

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
