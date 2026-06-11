def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = ""
    for i, doc in enumerate(contexts, start=1):
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        context_block += f"[{i}] (source: {source})\n{text}\n\n"

    return (
        f"Question: {question}\n\n"
        f"Context:\n{context_block.strip()}"
    )
