from dataclasses import dataclass
from rapidfuzz import fuzz, process
from database.connection import get_connection
from config import config


# ---------- membership functions ----------

def _trimf(x: float, a: float, b: float, c: float) -> float:
    if x < a or x > c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


# combined_score MFs  (domain [0, 1])
_SCORE_MFS = {
    "low":  lambda s: _trimf(s, 0.0,  0.0,  0.55),
    "med":  lambda s: _trimf(s, 0.35, 0.60, 0.80),
    "high": lambda s: _trimf(s, 0.65, 1.0,  1.0),
}

# length_ratio MFs  (length_ratio = len(query) / len(candidate))
_LEN_MFS = {
    "very_short": lambda r: _trimf(r, 0.0,  0.0,  0.35),
    "short":      lambda r: _trimf(r, 0.20, 0.45, 0.65),
    "medium":     lambda r: _trimf(r, 0.50, 0.75, 0.90),
    "long":       lambda r: _trimf(r, 0.80, 0.95, 1.10),
    "very_long":  lambda r: _trimf(r, 1.00, 1.40, 1.40),
}


# ---------- rulebase ----------

@dataclass(frozen=True)
class FuzzyRule:
    antecedent_1: str  # combined_score label
    antecedent_2: str  # length_ratio label
    consequent: str    # Mamdani output label
    tsk_y: float       # TSK crisp output value


RULES: tuple[FuzzyRule, ...] = (
    # R01–R05: low combined_score → always reject; extreme length adds double penalty
    FuzzyRule("low",  "very_short", "reject",  15.0),
    FuzzyRule("low",  "short",      "reject",  20.0),
    FuzzyRule("low",  "medium",     "reject",  20.0),
    FuzzyRule("low",  "long",       "reject",  20.0),
    FuzzyRule("low",  "very_long",  "reject",  15.0),
    # R06–R10: med combined_score — length acts as tiebreaker
    FuzzyRule("med",  "very_short", "partial", 55.0),
    FuzzyRule("med",  "short",      "partial", 65.0),
    FuzzyRule("med",  "medium",     "accept",  80.0),
    FuzzyRule("med",  "long",       "partial", 65.0),
    FuzzyRule("med",  "very_long",  "partial", 55.0),
    # R11–R15: high combined_score — only extreme length gaps demote to partial
    FuzzyRule("high", "very_short", "partial", 70.0),
    FuzzyRule("high", "short",      "accept",  85.0),
    FuzzyRule("high", "medium",     "accept",  92.0),
    FuzzyRule("high", "long",       "accept",  88.0),
    FuzzyRule("high", "very_long",  "partial", 72.0),
)

_REJECT_THRESHOLD = 40.0


# ---------- TSK inference ----------

def _tsk_infer(combined_score: float, length_ratio: float) -> float:
    """TSK weighted-average inference. Returns a value in [0, 100]."""
    total_w = 0.0
    weighted_sum = 0.0
    for rule in RULES:
        w = min(
            _SCORE_MFS[rule.antecedent_1](combined_score),
            _LEN_MFS[rule.antecedent_2](length_ratio),
        )
        if w > 0.0:
            weighted_sum += w * rule.tsk_y
            total_w += w
    return weighted_sum / total_w if total_w > 0.0 else 0.0


# ---------- database ----------

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


# ---------- main search ----------

def fuzzy_metadata_search(
    query: str, top_k: int = None, dataset_filter: str | None = None
) -> list[dict]:
    top_k = top_k or config.TOP_K
    candidates = _fetch_metadata_candidates(dataset_filter)
    if not candidates:
        return []

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

    # Prefetch more than top_k so the fuzzy engine can re-rank by length_ratio
    prefetch = max(top_k * 5, 25)
    raw_results = process.extract(
        query,
        search_strings,
        scorer=fuzz.WRatio,
        limit=prefetch,
        score_cutoff=40,
    )

    query_len = len(query)
    fuzzy_scored = []
    for _, score, idx in raw_results:
        combined_score = score / 100.0
        candidate_len = len(search_strings[idx]) or 1
        length_ratio = query_len / candidate_len

        fuzzy_output = _tsk_infer(combined_score, length_ratio)
        if fuzzy_output < _REJECT_THRESHOLD:
            continue

        doc = dict(candidates[idx])
        doc["score"] = fuzzy_output / 100.0
        doc["retriever"] = "fuzzy"
        fuzzy_scored.append(doc)

    fuzzy_scored.sort(key=lambda d: d["score"], reverse=True)
    return fuzzy_scored[:top_k]
