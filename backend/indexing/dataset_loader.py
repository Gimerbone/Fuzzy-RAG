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
import logging
import os
from pathlib import Path
from typing import Iterator

log = logging.getLogger("fuzzyrag.dataset_loader")


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


_PHYSIONET_BASE = "https://physionet.org/files/radqa/1.0.0"
# (remote_filename, local_filename) — PhysioNet uses train/dev/test.json;
# we save with the radqa_ prefix to avoid ambiguity in the local data dir.
_RADQA_FILES = [
    ("train.json", "radqa_train.json"),
    ("dev.json",   "radqa_dev.json"),
    ("test.json",  "radqa_test.json"),
]


def _download_radqa(base: Path, user: str, password: str) -> None:
    import requests

    base.mkdir(parents=True, exist_ok=True)
    for remote_fname, local_fname in _RADQA_FILES:
        fpath = base / local_fname
        if fpath.exists():
            continue
        url = f"{_PHYSIONET_BASE}/{remote_fname}"
        log.info("radqa | downloading %s from PhysioNet …", remote_fname)
        resp = requests.get(url, auth=(user, password), stream=True, timeout=120)
        if resp.status_code == 401:
            raise PermissionError(
                "PhysioNet credentials rejected — check PHYSIONET_USER and PHYSIONET_PASSWORD."
            )
        resp.raise_for_status()
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        log.info("radqa | saved %s", local_fname)


def load_radqa(
    data_path: str,
    physionet_user: str | None = None,
    physionet_password: str | None = None,
) -> Iterator[dict]:
    base = Path(data_path)

    if physionet_user and physionet_password:
        _download_radqa(base, physionet_user, physionet_password)

    if not base.exists():
        log.warning("radqa | data path %s not found — skipping RadQA", base)
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
