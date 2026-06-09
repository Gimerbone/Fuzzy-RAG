from flask import Flask, request, jsonify
from config import config
from database.connection import init_db
from indexing.embedder import embed_one
from indexing.indexer import run_indexing_async, get_status
from retrieval.dense import dense_search
from retrieval.sparse import sparse_search
from retrieval.fuzzy_metadata import fuzzy_metadata_search
from retrieval.rrf_fusion import rrf_fusion
from generation.llm import generate_answer
from eval.metrics import evaluate_batch

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": config.LLM_MODEL})


@app.route("/query", methods=["POST"])
def query():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    dataset_filter = body.get("dataset_filter") or None
    top_k = int(body.get("top_k", config.TOP_K))

    query_vec = embed_one(question)

    dense_results = dense_search(query_vec, top_k=top_k, dataset_filter=dataset_filter)
    sparse_results = sparse_search(question, top_k=top_k, dataset_filter=dataset_filter)
    fuzzy_results = fuzzy_metadata_search(question, top_k=top_k, dataset_filter=dataset_filter)

    fused = rrf_fusion([dense_results, sparse_results, fuzzy_results])
    contexts = fused[:top_k]

    generation = generate_answer(question, contexts)

    return jsonify(
        {
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
        }
    )


@app.route("/index", methods=["POST"])
def index():
    body = request.get_json(force=True) or {}
    datasets = body.get("datasets") or ["pubmedqa", "medqa", "radqa"]
    run_indexing_async(datasets)
    return jsonify({"status": "started", "datasets": datasets})


@app.route("/index/status")
def index_status():
    return jsonify(get_status())


@app.route("/eval", methods=["POST"])
def eval_endpoint():
    body = request.get_json(force=True)
    questions = body.get("questions", [])
    references = body.get("references", [])

    if not questions or not references:
        return jsonify({"error": "questions and references are required"}), 400
    if len(questions) != len(references):
        return jsonify({"error": "questions and references must have equal length"}), 400

    predictions = []
    for question in questions:
        query_vec = embed_one(question)
        dense_results = dense_search(query_vec)
        sparse_results = sparse_search(question)
        fuzzy_results = fuzzy_metadata_search(question)
        fused = rrf_fusion([dense_results, sparse_results, fuzzy_results])
        contexts = fused[: config.TOP_K]
        gen = generate_answer(question, contexts)
        predictions.append(gen["answer"])

    metrics = evaluate_batch(predictions, references)
    return jsonify(metrics)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
