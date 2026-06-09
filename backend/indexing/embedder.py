import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from config import config

_model = None
_tokenizer = None
_device = None


def _load():
    global _model, _tokenizer, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model, _ = create_model_from_pretrained(config.EMBEDDING_MODEL)
        _tokenizer = get_tokenizer(config.EMBEDDING_MODEL)
        _model.to(_device)
        _model.eval()


def embed(texts: list[str]) -> list[list[float]]:
    _load()
    tokens = _tokenizer(texts, context_length=256).to(_device)
    with torch.no_grad():
        features = _model.encode_text(tokens)
    # encode_text returns L2-normalised vectors — ideal for cosine similarity
    return features.cpu().float().numpy().tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
