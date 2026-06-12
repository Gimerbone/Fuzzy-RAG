import io
import logging
import threading
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from config import config
from generation.prompt_builder import build_prompt

log = logging.getLogger("fuzzyrag.llm")

_model = None
_processor = None
_load_lock = threading.Lock()

_SYSTEM = (
    "You are a biomedical expert. Answer the question using only the provided context passages. "
    "Cite passage numbers like [1], [2] when drawing on specific content. "
    "If the context is insufficient, say so clearly."
)


def _load():
    global _model, _processor
    if _model is not None:
        return
    with _load_lock:
        # Re-check inside the lock: another thread may have finished loading
        # while we were waiting to acquire it.
        if _model is not None:
            return
        quant_label = f"{config.LOAD_BITS}-bit NF4" if config.LOAD_BITS in (4, 8) else "fp32"
        log.info("Loading MedGemma (%s, %s) …", config.LLM_MODEL, quant_label)
        t0 = time.perf_counter()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(config.LOAD_BITS == 4),
            load_in_8bit=(config.LOAD_BITS == 8),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # Allow layers that don't fit in VRAM to spill to CPU RAM instead of
            # aborting. The GPU is filled first; this is only a safety net.
            llm_int8_enable_fp32_cpu_offload=True,
        )

        # Optional explicit memory budget. If GPU_MAX_MEMORY is set we cap the
        # GPU and give CPU headroom; otherwise device_map="auto" decides.
        max_memory = None
        if config.GPU_MAX_MEMORY and torch.cuda.is_available():
            max_memory = {
                i: config.GPU_MAX_MEMORY for i in range(torch.cuda.device_count())
            }
            max_memory["cpu"] = config.CPU_MAX_MEMORY
            log.info("MedGemma | max_memory budget: %s", max_memory)

        _processor = AutoProcessor.from_pretrained(config.LLM_MODEL, token=config.HF_TOKEN)
        model = AutoModelForImageTextToText.from_pretrained(
            config.LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            token=config.HF_TOKEN,
        )
        model.eval()

        # Report where layers actually landed so CPU offload is visible in logs.
        device_map = getattr(model, "hf_device_map", {}) or {}
        offloaded = sorted({str(d) for d in device_map.values()
                            if str(d) in ("cpu", "disk") or d in (-1, "cpu", "disk")})
        if offloaded:
            log.warning(
                "MedGemma | %d module(s) offloaded to %s — GPU VRAM is tight, "
                "generation will be slower. Lower other GPU usage or set GPU_MAX_MEMORY.",
                sum(1 for d in device_map.values()
                    if str(d) in ("cpu", "disk")),
                ", ".join(offloaded),
            )
        else:
            log.info("MedGemma | fully on GPU (no CPU offload)")

        # Publish only after fully initialised so other threads never see a
        # half-built model via the `_model is not None` fast path.
        _model = model
        log.info("MedGemma ready  (%.1f s)", time.perf_counter() - t0)


def generate_answer(
    question: str,
    contexts: list[dict],
    image_bytes: bytes | None = None,
) -> dict:
    _load()
    log.info(
        "generate | contexts=%d  image=%s  question=%r",
        len(contexts), "yes" if image_bytes else "no", question[:60],
    )

    user_content = []
    if image_bytes is not None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        user_content.append({"type": "image", "image": pil_img})
    user_content.append({"type": "text", "text": build_prompt(question, contexts)})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user", "content": user_content},
    ]
    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    gen_s = time.perf_counter() - t0

    new_tokens = output_ids[0][input_len:]
    answer = _processor.decode(new_tokens, skip_special_tokens=True)

    log.info(
        "generate | done  input_tokens=%d  output_tokens=%d  time=%.1f s",
        input_len, len(new_tokens), gen_s,
    )

    return {
        "answer": answer,
        "model": config.LLM_MODEL,
        "input_tokens": input_len,
        "output_tokens": len(new_tokens),
    }


def model_health_check() -> dict:
    log.info("health_check | testing MedGemma …")
    try:
        t0 = time.time()
        _load()
        test_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ]
        inputs = _processor.apply_chat_template(
            test_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(_model.device)
        with torch.no_grad():
            _model.generate(**inputs, max_new_tokens=1, do_sample=False)
        latency_ms = round((time.time() - t0) * 1000, 1)

        device = str(next(_model.parameters()).device)
        quant = "4-bit NF4" if config.LOAD_BITS == 4 else "8-bit" if config.LOAD_BITS == 8 else "fp32"

        log.info("health_check | MedGemma OK  device=%s  latency=%d ms", device, latency_ms)
        return {
            "status": "ok",
            "loaded": True,
            "device": device,
            "quantization": quant,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        log.error("health_check | MedGemma FAILED: %s", exc)
        return {"status": "error", "loaded": False, "error": str(exc)}
