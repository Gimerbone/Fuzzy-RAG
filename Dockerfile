# syntax=docker/dockerfile:1
# Backend — PyTorch + CUDA base so bitsandbytes 4-bit quantisation works.
# Build context must be the project root (docker build . or docker compose build).

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Keep the HuggingFace model cache inside the container at a known path.
# Mount a host volume here in docker-compose to persist downloads across rebuilds.
ENV HF_HOME=/cache/huggingface

WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Non-privileged user — create before any file copies so chown is cheap.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser \
    && mkdir -p /cache/huggingface /app/baselines \
    && chown appuser:appuser /cache/huggingface /app/baselines

# Leverage BuildKit cache mount — pip cache survives rebuilds without hitting PyPI.
# Bind-mount keeps requirements.txt out of the layer (only the installed packages land in the image).
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=backend/requirements.txt,target=requirements.txt \
    pip install -r requirements.txt

# NLTK punkt tokenizer needed by eval/metrics.py.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

USER appuser

# Copy only the backend package — frontend/, scripts/, data/ are excluded by .dockerignore.
COPY --chown=appuser:appuser backend/ .

EXPOSE 5000

CMD ["python", "app.py"]
