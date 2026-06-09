import threading
from config import config
from database.connection import get_connection, init_db
from indexing.embedder import embed
from indexing.dataset_loader import load_pubmedqa, load_medqa, load_radqa

_status: dict = {"state": "idle", "indexed": 0, "errors": 0, "current_dataset": None}
_lock = threading.Lock()

BATCH_SIZE = 64


def get_status() -> dict:
    with _lock:
        return dict(_status)


def _set_status(**kwargs):
    with _lock:
        _status.update(kwargs)


def _index_records(records: list[dict], conn):
    texts = [r["text"] for r in records]
    vectors = embed(texts)

    with conn.cursor() as cur:
        for record, vector in zip(records, vectors):
            cur.execute(
                """
                INSERT INTO documents (text, source, question, answer, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    record["text"],
                    record["source"],
                    record.get("question"),
                    record.get("answer"),
                    __import__("json").dumps(record.get("metadata", {})),
                    str(vector),
                ),
            )
    conn.commit()


def run_indexing(datasets: list[str] | None = None):
    if datasets is None:
        datasets = ["pubmedqa", "medqa", "radqa"]

    _set_status(state="running", indexed=0, errors=0)
    init_db()

    loaders = {
        "pubmedqa": lambda: load_pubmedqa(),
        "medqa": lambda: load_medqa(),
        "radqa": lambda: load_radqa(config.RADQA_DATA_PATH),
    }

    try:
        with get_connection() as conn:
            for dataset in datasets:
                if dataset not in loaders:
                    continue
                _set_status(current_dataset=dataset)
                batch = []
                try:
                    for record in loaders[dataset]():
                        batch.append(record)
                        if len(batch) >= BATCH_SIZE:
                            _index_records(batch, conn)
                            with _lock:
                                _status["indexed"] += len(batch)
                            batch = []
                    if batch:
                        _index_records(batch, conn)
                        with _lock:
                            _status["indexed"] += len(batch)
                except Exception as exc:
                    with _lock:
                        _status["errors"] += 1
                    print(f"[indexer] Error in {dataset}: {exc}")

        _set_status(state="done", current_dataset=None)
    except Exception as exc:
        _set_status(state="error", current_dataset=None)
        raise


def run_indexing_async(datasets: list[str] | None = None):
    thread = threading.Thread(target=run_indexing, args=(datasets,), daemon=True)
    thread.start()
