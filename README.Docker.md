# Docker Setup — Fuzzy RAG

## Requirements

- Docker ≥ 24
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough
- NVIDIA GPU with ≥ 6 GB VRAM (4-bit NF4 quantization)

## Running the Stack

```bash
cp .env.example .env
docker compose up -d
```

Three services start together:

| Service | Port | Image |
|---------|------|-------|
| `postgres` | 5432 | `pgvector/pgvector:pg16` |
| `backend` | 5000 | Built from `Dockerfile` |
| `frontend` | 8501 | Built from `frontend/Dockerfile` |

The backend waits for Postgres to pass its healthcheck before starting.

## Building Images

```bash
# Build and start everything
docker compose up --build -d

# Rebuild a single service without restarting others
docker compose build backend
docker compose up -d --no-deps backend
```

## GPU Access

The backend requests one NVIDIA GPU via Docker's `deploy.resources.reservations`. Verify the toolkit is working:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` fails inside the container, follow the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Data Volumes

PostgreSQL data is persisted in the named volume `pgdata`. RadQA files are bind-mounted read-only from `RADQA_DATA_PATH` on the host to `/data/radqa` in the backend container.

```bash
# Inspect volume location
docker volume inspect fuzzy-rag_pgdata

# Remove volume — destroys all indexed data
docker compose down -v
```

## Indexing Datasets

After the stack is running, trigger indexing:

```bash
# Via API (runs in the background)
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"datasets": ["pubmedqa", "medqa"]}'

# Check progress
curl http://localhost:5000/index/status

# Or run directly inside the backend container
docker compose exec backend python -c \
  "from indexing.indexer import run_indexing; run_indexing()"
```

PubMedQA and MedQA download automatically from HuggingFace. RadQA requires placing the SQuAD JSON files in `RADQA_DATA_PATH` first — see the [RadQA Setup](README.md#radqa-setup) section in the main README.

## Logs

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

## Stopping

```bash
# Stop containers, keep indexed data
docker compose down

# Stop and delete all indexed data
docker compose down -v
```

## Cross-Platform Builds

If deploying to a cloud host with a different CPU architecture than your dev machine:

```bash
docker build --platform=linux/amd64 -t fuzzyrag-backend .
docker build --platform=linux/amd64 -f frontend/Dockerfile -t fuzzyrag-frontend ./frontend
```
