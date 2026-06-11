import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from config import config
from generation.prompt_builder import build_prompt

_model = None
_processor = None

_SYSTEM = (
    "You are a biomedical expert. Answer the question using only the provided context passages. "
    "Cite passage numbers like [1], [2] when drawing on specific content. "
    "If the context is insufficient, say so clearly."
)


def _load():
    global _model, _processor
    if _model is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(config.LOAD_BITS == 4),
            load_in_8bit=(config.LOAD_BITS == 8),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _processor = AutoProcessor.from_pretrained(config.LLM_MODEL, token=config.HF_TOKEN)
        _model = AutoModelForImageTextToText.from_pretrained(
            config.LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            token=config.HF_TOKEN,
        )
        _model.eval()


def generate_answer(question: str, contexts: list[dict]) -> dict:
    _load()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": build_prompt(question, contexts)}]},
    ]
    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    new_tokens = output_ids[0][input_len:]
    answer = _processor.decode(new_tokens, skip_special_tokens=True)

    return {
        "answer": answer,
        "model": config.LLM_MODEL,
        "input_tokens": input_len,
        "output_tokens": len(new_tokens),
    }
