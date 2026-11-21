"""Sarvam chat rendering (empty-think or natural thinking) and answer parsing."""

import re

MODEL_ID = "sarvamai/sarvam-30b"
REVISION = "071ae95e933605ca1104a6b4524a36a98488efa4"


def encode(tokenizer, text, regime="empty_think"):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=(regime == "natural"),
    )
    if regime == "empty_think":
        rendered += "<think></think>\n"
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids or len(ids) > 1024:
        raise ValueError(f"Unsupported prompt length {len(ids)}")
    return ids, rendered


def parse_answer(raw, regime):
    """Empty-think: everything generated is the answer (stray think markers removed)."""
    if regime == "empty_think":
        text = re.sub(r"\s*</?think>\s*", "\n\n", raw).strip()
        return text, bool(text)
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1].strip(), True
    if "<think>" in raw:
        return "", False
    return raw.strip(), True


def quality(raw, answer, has_final, capped):
    return {"has_final_answer": has_final and bool(answer), "token_cap_reached": capped}
