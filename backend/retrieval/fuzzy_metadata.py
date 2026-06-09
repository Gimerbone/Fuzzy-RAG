from rapidfuzz import fuzz, process
from database.connection import get_connection
from config import config


def _fetch_metadata_candidates(dataset_filter: str | None = None) -> list[dict]:
    filter_clause = "WHERE source = %s" if dataset_filter else ""
    params = [dataset_filter] if dataset_filter else []

    sql = f"""
        SELECT id, text, source, question, answer, metadata
        FROM documents
        {filter_clause}
        LIMIT 5000
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "text": row[1],
            "source": row[2],
            "question": row[3],
            "answer": row[4],
            "metadata": row[5],
        }
        for row in rows
    ]


def fuzzy_metadata_search(
    query: str, top_k: int = None, dataset_filter: str | None = None
) -> list[dict]:
    top_k = top_k or config.TOP_K
    candidates = _fetch_metadata_candidates(dataset_filter)

    if not candidates:
        return []

    # Build searchable strings from question + metadata values
    search_strings = []
    for doc in candidates:
        parts = []
        if doc["question"]:
            parts.append(doc["question"])
        meta = doc["metadata"] or {}
        for v in meta.values():
            if isinstance(v, str) and v:
                parts.append(v)
        search_strings.append(" ".join(parts))

    results = process.extract(
        query,
        search_strings,
        scorer=fuzz.WRatio,
        limit=top_k,
        score_cutoff=40,
    )

    output = []
    for _, score, idx in results:
        doc = dict(candidates[idx])
        doc["score"] = score / 100.0
        doc["retriever"] = "fuzzy"
        output.append(doc)

    return output
