def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = ""
    for i, doc in enumerate(contexts, start=1):
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        context_block += f"[{i}] (source: {source})\n{text}\n\n"

    return (
        "You are a biomedical expert. Answer the question below using only the provided context passages.\n"
        "Cite passage numbers like [1], [2] when drawing on specific content.\n"
        "If the context is insufficient, say so clearly.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_block.strip()}\n\n"
        "Answer:"
    )
