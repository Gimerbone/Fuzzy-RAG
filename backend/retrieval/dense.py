from database.connection import get_connection
from config import config


def dense_search(query_vector: list[float], top_k: int = None, dataset_filter: str | None = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    filter_clause = "WHERE source = %s" if dataset_filter else ""
    params = [str(query_vector), top_k]
    if dataset_filter:
        params = [str(query_vector), dataset_filter, top_k]

    sql = f"""
        SELECT id, text, source, question, answer, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM documents
        {filter_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    if dataset_filter:
        sql = f"""
            SELECT id, text, source, question, answer, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM documents
            WHERE source = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [str(query_vector), dataset_filter, str(query_vector), top_k]
    else:
        params = [str(query_vector), str(query_vector), top_k]

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
            "retriever": "dense",
        }
        for row in rows
    ]
