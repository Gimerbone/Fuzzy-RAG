from database.connection import get_connection
from config import config


def sparse_search(query: str, top_k: int = None, dataset_filter: str | None = None) -> list[dict]:
    top_k = top_k or config.TOP_K

    filter_clause = "AND source = %s" if dataset_filter else ""
    if dataset_filter:
        params = [query, query, dataset_filter, top_k]
    else:
        params = [query, query, top_k]

    sql = f"""
        SELECT id, text, source, question, answer, metadata,
               ts_rank(ts_vector, plainto_tsquery('english', %s)) AS score
        FROM documents
        WHERE ts_vector @@ plainto_tsquery('english', %s)
        {filter_clause}
        ORDER BY score DESC
        LIMIT %s
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
            "score": float(row[6]),
            "retriever": "sparse",
        }
        for row in rows
    ]
