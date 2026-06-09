"""
Loaders for PubMedQA, MedQA, and RadQA datasets.

Each loader yields dicts with keys:
  text     – passage/context used for retrieval
  source   – dataset name
  question – original question (may be None)
  answer   – reference answer (may be None)
  metadata – dict of extra fields for fuzzy metadata matching
"""

import json
import os
from pathlib import Path
from typing import Iterator


def load_pubmedqa() -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)
    for row in ds:
        contexts = row.get("context", {})
        passages = contexts.get("contexts", []) if isinstance(contexts, dict) else []
        text = " ".join(passages) if passages else row.get("long_answer", "")
        if not text:
            continue
        yield {
            "text": text,
            "source": "pubmedqa",
            "question": row.get("question"),
            "answer": row.get("long_answer"),
            "metadata": {
                "pubid": str(row.get("pubid", "")),
                "final_decision": row.get("final_decision", ""),
                "labels": row.get("context", {}).get("labels", []) if isinstance(row.get("context"), dict) else [],
            },
        }


def load_medqa() -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa", split="train", trust_remote_code=True)
    for row in ds:
        choices = row.get("choices", [])
        choice_texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in choices]
        text = row.get("question", "")
        if choice_texts:
            text += " Options: " + " | ".join(choice_texts)
        if not text:
            continue
        answer_list = row.get("answer", [])
        answer = answer_list[0] if answer_list else None
        yield {
            "text": text,
            "source": "medqa",
            "question": row.get("question"),
            "answer": answer,
            "metadata": {
                "id": str(row.get("id", "")),
                "type": row.get("type", ""),
            },
        }


def load_radqa(data_path: str) -> Iterator[dict]:
    """
    RadQA requires PhysioNet credentialed access.
    Download from https://physionet.org/content/radqa/1.0.0/
    Place radqa_train.json (and optionally radqa_dev.json, radqa_test.json)
    in the directory pointed to by data_path / RADQA_DATA_PATH.
    """
    base = Path(data_path)
    if not base.exists():
        print(f"[radqa] Data path {base} not found — skipping RadQA.")
        return

    for fname in ("radqa_train.json", "radqa_dev.json", "radqa_test.json"):
        fpath = base / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            squad_data = json.load(f)

        for article in squad_data.get("data", []):
            title = article.get("title", "")
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                if not context:
                    continue
                qas = para.get("qas", [])
                if qas:
                    for qa in qas:
                        answers = qa.get("answers", [])
                        answer_text = answers[0]["text"] if answers else None
                        yield {
                            "text": context,
                            "source": "radqa",
                            "question": qa.get("question"),
                            "answer": answer_text,
                            "metadata": {
                                "title": title,
                                "qa_id": qa.get("id", ""),
                                "split": fname.replace("radqa_", "").replace(".json", ""),
                            },
                        }
                else:
                    yield {
                        "text": context,
                        "source": "radqa",
                        "question": None,
                        "answer": None,
                        "metadata": {"title": title, "split": fname.replace("radqa_", "").replace(".json", "")},
                    }
