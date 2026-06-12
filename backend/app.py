import base64
import logging
import logging.config
import time

from flask import Flask, request, jsonify, g

from config import config
from database.connection import init_db

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    # Suppress noisy third-party loggers
    "loggers": {
        "werkzeug": {"level": "WARNING"},
        "urllib3": {"level": "WARNING"},
        "transformers": {"level": "WARNING"},
        "datasets": {"level": "WARNING"},
        "filelock": {"level": "WARNING"},
    },
})
from database.connection import get_source_counts
from indexing.embedder import embed_one, embed_image, model_health_check as embedder_health
from indexing.indexer import (
    run_indexing_async, get_status,
    run_download_async, get_download_status,
)
from retrieval.dense import dense_search
from retrieval.sparse import sparse_search
from retrieval.fuzzy_metadata import fuzzy_metadata_search
from retrieval.rrf_fusion import rrf_fusion
from generation.llm import generate_answer, model_health_check as llm_health
from eval.metrics import evaluate_batch, save_baseline, list_baselines

app = Flask(__name__)
log = logging.getLogger("fuzzyrag.api")


@app.before_request
def _start_timer():
    g.t0 = time.perf_counter()


@app.after_request
def _log_request(response):
    elapsed = round((time.perf_counter() - g.t0) * 1000)
    log.info("%s %s → %s  (%d ms)", request.method, request.path, response.status_code, elapsed)
    return response


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": config.LLM_MODEL})


@app.route("/health/models")
def health_models():
    """Run a real inference test on each model and return detailed status."""
    return jsonify({
        "embedding": embedder_health(),
        "llm": llm_health(),
    })


@app.route("/query", methods=["POST"])
def query():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    dataset_filter = body.get("dataset_filter") or None
    top_k = int(body.get("top_k", config.TOP_K))
    image_b64 = body.get("image") or None
    image_bytes = base64.b64decode(image_b64) if image_b64 else None

    log.info(
        "query | question=%r  top_k=%d  dataset=%s  image=%s",
        question[:80], top_k, dataset_filter or "all", "yes" if image_bytes else "no",
    )

    # Cross-modal retrieval: image embedding searches the text HNSW index directly
    # because BiomedCLIP aligns image and text in the same 512-dim space.
    query_vec = embed_image(image_bytes) if image_bytes else embed_one(question)

    dense_results = dense_search(query_vec, top_k=top_k, dataset_filter=dataset_filter)
    sparse_results = sparse_search(question, top_k=top_k, dataset_filter=dataset_filter)
    fuzzy_results = fuzzy_metadata_search(question, top_k=top_k, dataset_filter=dataset_filter)

    log.info(
        "retrieval | dense=%d  sparse=%d  fuzzy=%d",
        len(dense_results), len(sparse_results), len(fuzzy_results),
    )

    fused = rrf_fusion([dense_results, sparse_results, fuzzy_results])
    contexts = fused[:top_k]
    generation = generate_answer(question, contexts, image_bytes=image_bytes)

    log.info(
        "generation | input_tokens=%d  output_tokens=%d",
        generation["input_tokens"], generation["output_tokens"],
    )

    return jsonify({
        "question": question,
        "answer": generation["answer"],
        "contexts": [
            {
                "id": c["id"],
                "text": c["text"][:500],
                "source": c["source"],
                "rrf_score": c.get("rrf_score"),
                "retriever": c.get("retriever"),
            }
            for c in contexts
        ],
        "usage": {
            "model": generation["model"],
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
        },
        "image_used": image_bytes is not None,
    })


# ── Download ──────────────────────────────────────────────────────────────────

@app.route("/download", methods=["POST"])
def download():
    body = request.get_json(force=True) or {}
    datasets = body.get("datasets") or ["pubmedqa", "medqa", "radqa"]
    run_download_async(datasets)
    return jsonify({"status": "started", "datasets": datasets})


@app.route("/download/status")
def download_status():
    return jsonify(get_download_status())


@app.route("/upload/radqa", methods=["POST"])
def upload_radqa():
    """Accept manually-uploaded RadQA split files (SQuAD JSON format)."""
    import json
    from pathlib import Path

    data_path = Path(config.RADQA_DATA_PATH)
    try:
        data_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        return jsonify({"error": f"Cannot write to {data_path}: {exc}"}), 500

    split_map = {
        "train": "radqa_train.json",
        "dev":   "radqa_dev.json",
        "test":  "radqa_test.json",
    }

    saved, errors = [], []
    for split, local_fname in split_map.items():
        f = request.files.get(split)
        if f is None:
            continue
        try:
            content = f.read()
            parsed = json.loads(content)
            if "data" not in parsed:
                errors.append(f"{split}: missing 'data' key — not a SQuAD-format file")
                continue
            dest = data_path / local_fname
            dest.write_bytes(content)
            saved.append(local_fname)
            log.info("upload/radqa | saved %s (%d bytes)", local_fname, len(content))
        except json.JSONDecodeError as exc:
            errors.append(f"{split}: invalid JSON — {exc}")
        except Exception as exc:
            errors.append(f"{split}: {exc}")

    if not saved and not errors:
        return jsonify({"error": "No files received. Send 'train', 'dev', and/or 'test' fields."}), 400

    return jsonify({"saved": saved, "errors": errors})


# ── Index ─────────────────────────────────────────────────────────────────────

@app.route("/index", methods=["POST"])
def index():
    body = request.get_json(force=True) or {}
    datasets = body.get("datasets") or ["pubmedqa", "medqa", "radqa"]
    run_indexing_async(datasets)
    return jsonify({"status": "started", "datasets": datasets})


@app.route("/index/status")
def index_status():
    return jsonify(get_status())


@app.route("/index/stats")
def index_stats():
    """Per-dataset document counts — used by the frontend to detect already-indexed data."""
    try:
        return jsonify(get_source_counts())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Eval ──────────────────────────────────────────────────────────────────────

@app.route("/eval", methods=["POST"])
def eval_endpoint():
    body = request.get_json(force=True)
    questions = body.get("questions", [])
    references = body.get("references", [])
    dataset_filter = body.get("dataset_filter") or None
    save_as = body.get("save_as") or None

    if not questions or not references:
        return jsonify({"error": "questions and references are required"}), 400
    if len(questions) != len(references):
        return jsonify({"error": "questions and references must have equal length"}), 400

    predictions = []
    for question in questions:
        query_vec = embed_one(question)
        dense_results = dense_search(query_vec, dataset_filter=dataset_filter)
        sparse_results = sparse_search(question, dataset_filter=dataset_filter)
        fuzzy_results = fuzzy_metadata_search(question, dataset_filter=dataset_filter)
        fused = rrf_fusion([dense_results, sparse_results, fuzzy_results])
        contexts = fused[:config.TOP_K]
        gen = generate_answer(question, contexts)
        predictions.append(gen["answer"])

    metrics = evaluate_batch(predictions, references)

    saved = None
    if save_as:
        saved = save_baseline(
            name=save_as,
            metrics=metrics,
            model=config.LLM_MODEL,
            dataset_filter=dataset_filter,
        )

    return jsonify({**metrics, "saved": saved})


@app.route("/eval/compare", methods=["POST"])
def eval_compare():
    """Compare Fuzzy RAG (dense+sparse+fuzzy) vs Standard RAG (dense+sparse)."""
    body = request.get_json(force=True)
    questions = body.get("questions", [])
    references = body.get("references", [])
    dataset_filter = body.get("dataset_filter") or None

    if not questions or not references:
        return jsonify({"error": "questions and references are required"}), 400
    if len(questions) != len(references):
        return jsonify({"error": "questions and references must have equal length"}), 400

    log.info("eval/compare | %d question(s)  dataset=%s", len(questions), dataset_filter or "all")

    fuzzy_preds = []
    standard_preds = []

    for i, question in enumerate(questions):
        log.info("eval/compare | question %d/%d: %r", i + 1, len(questions), question[:60])
        query_vec = embed_one(question)
        dense = dense_search(query_vec, dataset_filter=dataset_filter)
        sparse = sparse_search(question, dataset_filter=dataset_filter)
        fuzzy = fuzzy_metadata_search(question, dataset_filter=dataset_filter)

        contexts_fuzzy = rrf_fusion([dense, sparse, fuzzy])[:config.TOP_K]
        fuzzy_preds.append(generate_answer(question, contexts_fuzzy)["answer"])

        contexts_standard = rrf_fusion([dense, sparse])[:config.TOP_K]
        standard_preds.append(generate_answer(question, contexts_standard)["answer"])

    fuzzy_metrics = evaluate_batch(fuzzy_preds, references)
    standard_metrics = evaluate_batch(standard_preds, references)

    log.info(
        "eval/compare | fuzzy BLEU=%.4f ROUGE-L=%.4f F1=%.4f | std BLEU=%.4f ROUGE-L=%.4f F1=%.4f",
        fuzzy_metrics["bleu"], fuzzy_metrics["rouge_l"], fuzzy_metrics["token_f1"],
        standard_metrics["bleu"], standard_metrics["rouge_l"], standard_metrics["token_f1"],
    )

    return jsonify({
        "fuzzy_rag": fuzzy_metrics,
        "standard_rag": standard_metrics,
        "delta": {
            "bleu": round(fuzzy_metrics["bleu"] - standard_metrics["bleu"], 4),
            "rouge_l": round(fuzzy_metrics["rouge_l"] - standard_metrics["rouge_l"], 4),
            "token_f1": round(fuzzy_metrics["token_f1"] - standard_metrics["token_f1"], 4),
        },
        "n_samples": len(questions),
    })


@app.route("/eval/baselines")
def eval_baselines():
    return jsonify(list_baselines())


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
