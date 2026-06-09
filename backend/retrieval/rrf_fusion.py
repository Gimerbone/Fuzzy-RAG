from config import config


def rrf_fusion(ranked_lists: list[list[dict]], k: int = None) -> list[dict]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    Each list item must have an 'id' key.
    Returns merged list sorted by descending RRF score, with de-duplicated docs.
    """
    k = k or config.RRF_K
    scores: dict[int, float] = {}
    docs_by_id: dict[int, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = doc

    merged = []
    for doc_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(docs_by_id[doc_id])
        entry["rrf_score"] = rrf_score
        merged.append(entry)

    return merged
