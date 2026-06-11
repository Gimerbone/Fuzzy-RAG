import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = Counter(_tokenize(prediction))
    ref_tokens = Counter(_tokenize(reference))

    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def bleu(predictions: list[str], references: list[str]) -> float:
    smoothie = SmoothingFunction().method1
    refs = [[_tokenize(r)] for r in references]
    hyps = [_tokenize(p) for p in predictions]
    return corpus_bleu(refs, hyps, smoothing_function=smoothie)


def rouge_l(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score["rougeL"].fmeasure


def save_baseline(
    name: str,
    metrics: dict,
    model: str,
    dataset_filter: str | None = None,
) -> dict:
    from config import config

    baselines_dir = Path(config.BASELINES_PATH)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filename = f"{safe_name}_{ts}.json"

    entry = {
        "name": name,
        "filename": filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "dataset_filter": dataset_filter,
        "metrics": metrics,
    }
    with open(baselines_dir / filename, "w") as fh:
        json.dump(entry, fh, indent=2)
    return entry


def list_baselines() -> list[dict]:
    from config import config

    baselines_dir = Path(config.BASELINES_PATH)
    if not baselines_dir.exists():
        return []

    results = []
    for fpath in sorted(baselines_dir.glob("*.json"), reverse=True):
        try:
            with open(fpath) as fh:
                results.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def evaluate_batch(predictions: list[str], references: list[str]) -> dict:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")

    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    rouge_scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
    bleu_score = bleu(predictions, references)

    return {
        "bleu": round(bleu_score, 4),
        "rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4),
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "n_samples": len(predictions),
    }
