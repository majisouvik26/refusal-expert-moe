"""Harmony prompt rendering (reasoning effort low) and final-channel parsing for GPT-OSS."""

import re

EFFORT = "low"
DATE = "2026-11-20"
STOP_TOKENS = ("<|return|>", "<|call|>")  # vLLM stop strings; ids are taken from the tokenizer at run time
_FINAL = re.compile(r"<\|channel\|>final(?:\s*<\|constrain\|>[^<]*)?<\|message\|>")
_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>")
_END = re.compile(r"<\|(?:end|return|call)\|>")


def encoding():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def render(enc, prompt, effort=EFFORT, date=DATE):
    from openai_harmony import Conversation, Message, ReasoningEffort, Role, SystemContent

    system = (
        SystemContent.new()
        .with_reasoning_effort(
            {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}[effort]
        )
        .with_conversation_start_date(date)
    )
    conv = Conversation.from_messages(
        [Message.from_role_and_content(Role.SYSTEM, system), Message.from_role_and_content(Role.USER, prompt)]
    )
    return list(enc.render_conversation_for_completion(conv, Role.ASSISTANT))


def parse(raw):
    """Split decoded completion text (special tokens kept) into reasoning and final answer.

    Returns dict(analysis, final, has_final, ended). 'ended' is True when the final message closed with
    <|return|> (or <|end|>); a final message cut by the token budget has has_final=True, ended=False and is kept.
    """
    finals = list(_FINAL.finditer(raw))
    first_final = finals[0].start() if finals else len(raw)
    analysis_parts = []
    for m in _ANALYSIS.finditer(raw[:first_final]):
        end = _END.search(raw, m.end())
        analysis_parts.append(raw[m.end() : end.start() if end and end.start() < first_final else first_final])
    if not finals:
        return {"analysis": "\n".join(analysis_parts).strip(), "final": "", "has_final": False, "ended": False}
    m = finals[-1]
    end = _END.search(raw, m.end())
    final = raw[m.end() : end.start() if end else len(raw)]
    return {
        "analysis": "\n".join(analysis_parts).strip(),
        "final": final.strip(),
        "has_final": True,
        "ended": end is not None,
    }
