import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import config
from generation.prompt_builder import build_prompt

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(config.LOAD_BITS == 4),
            load_in_8bit=(config.LOAD_BITS == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        _model.eval()


def generate_answer(question: str, contexts: list[dict]) -> dict:
    _load()
    prompt = build_prompt(question, contexts)
    # CheXagent chat format
    query = f"USER: <s>{prompt} ASSISTANT: <s>"

    inputs = _tokenizer(query, return_tensors="pt").to(_model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    new_tokens = output_ids[0][input_len:]
    answer = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "answer": answer,
        "model": config.LLM_MODEL,
        "input_tokens": input_len,
        "output_tokens": len(new_tokens),
    }
