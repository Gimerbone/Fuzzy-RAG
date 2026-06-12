import io
import logging
import threading
import time

import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image

from config import config

log = logging.getLogger("fuzzyrag.embedder")

_model = None
_tokenizer = None
_preprocess = None
_device = None
_load_lock = threading.Lock()


def _load():
    global _model, _tokenizer, _preprocess, _device
    if _model is not None:
        return
    with _load_lock:
        # Re-check inside the lock: another thread may have finished loading
        # while we were waiting to acquire it.
        if _model is not None:
            return
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading BiomedCLIP onto %s …", _device)
        t0 = time.perf_counter()
        model, _preprocess = create_model_from_pretrained(config.EMBEDDING_MODEL)
        _tokenizer = get_tokenizer(config.EMBEDDING_MODEL)
        model.to(_device)
        model.eval()
        # Publish the model only after it is fully initialised, so other
        # threads' `_model is not None` check never sees a half-built model.
        _model = model
        log.info("BiomedCLIP ready  (%.1f s)", time.perf_counter() - t0)


def embed(texts: list[str]) -> list[list[float]]:
    _load()
    tokens = _tokenizer(texts, context_length=256).to(_device)
    with torch.no_grad():
        features = _model.encode_text(tokens)
    # encode_text returns L2-normalised vectors — ideal for cosine similarity
    return features.cpu().float().numpy().tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def embed_image(image_bytes: bytes) -> list[float]:
    """Cross-modal retrieval: encode image with BiomedCLIP image encoder.
    Output is in the same 512-dim L2-normalised space as text embeddings."""
    _load()
    log.info("embed_image | encoding uploaded image (%d bytes)", len(image_bytes))
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = _model.encode_image(tensor)
        features = F.normalize(features, dim=-1)
    return features.cpu().float().numpy().tolist()[0]


def model_health_check() -> dict:
    log.info("health_check | testing BiomedCLIP …")
    try:
        t0 = time.time()
        _load()
        vec = embed_one("biomedical health check")
        latency_ms = round((time.time() - t0) * 1000, 1)
        log.info("health_check | BiomedCLIP OK  latency=%d ms", latency_ms)
        return {
            "status": "ok",
            "loaded": True,
            "device": str(_device),
            "output_dim": len(vec),
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        log.error("health_check | BiomedCLIP FAILED: %s", exc)
        return {"status": "error", "loaded": False, "error": str(exc)}
