"""Qwen3.5 chat rendering with thinking off (empty think block) and answer parsing."""

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
EMPTY_THINK = "<think>\n\n</think>\n\n"


def render(tok, prompt):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    return list(ids)


def parse(raw):
    """Returns dict(analysis, final, has_final, ended, has_think)."""
    text, ended = raw, False
    for stop in STOP_TOKENS:
        if stop in text:
            text, ended = text[: text.index(stop)], True
    has_think = "</think>" in text
    final = text.rsplit("</think>", 1)[-1] if has_think else text
    return {"analysis": "", "final": final.strip(), "has_final": True, "ended": ended, "has_think": has_think}
