"""GSM8K ability check: prompt template, gold answers and number matching."""

import math
import re

PROMPT = (
    "{question}\n\nSolve the problem. On the last line, write the final answer as a number "
    "in the form: Answer: <number>"
)


def gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def extract(text, which=-1):
    """Number after the last (which=-1) or first (which=0) 'Answer:', else the last number."""
    found = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not found:
        found = re.findall(r"-?\d[\d,]*\.?\d*", text)
        which = -1
    return found[which].replace(",", "").rstrip(".") if found else None


def same_number(a, b):
    try:
        return a is not None and math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return False
