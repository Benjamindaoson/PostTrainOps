from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_GSM8K_RE = re.compile(r"####\s*([^\n]+)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_RE = re.compile(r"final\s+answer\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def _canonicalize(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip().strip("`*_ ").rstrip(".。")
    value = value.replace("$", "").replace(",", "")
    value = value.strip()
    if not value:
        return None

    percent = value.endswith("%")
    if percent:
        value = value[:-1].strip()

    # Prefer an exact numeric canonical form when possible.
    try:
        number = Decimal(value)
        if number == number.to_integral():
            normalized = str(number.quantize(Decimal("1")))
        else:
            normalized = format(number.normalize(), "f")
        return normalized + ("%" if percent else "")
    except InvalidOperation:
        return value.lower()


def extract_gsm8k_gold(answer: str) -> str:
    match = _GSM8K_RE.search(answer)
    if not match:
        raise ValueError("GSM8K answer does not contain the expected '#### <answer>' marker")
    normalized = _canonicalize(match.group(1))
    if normalized is None:
        raise ValueError("Empty GSM8K gold answer")
    return normalized


def extract_prediction(text: str) -> tuple[str | None, bool, str]:
    """Return (normalized answer, explicit-format-valid, extraction method)."""
    matches = list(_BOXED_RE.finditer(text))
    if matches:
        return _canonicalize(matches[-1].group(1)), True, "boxed"

    matches = list(_FINAL_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip()
        number = _NUMBER_RE.search(candidate)
        return _canonicalize(number.group(0) if number else candidate), True, "final_answer"

    matches = list(_GSM8K_RE.finditer(text))
    if matches:
        return _canonicalize(matches[-1].group(1)), True, "gsm8k_marker"

    numbers = list(_NUMBER_RE.finditer(text))
    if numbers:
        return _canonicalize(numbers[-1].group(0)), False, "last_number"

    return None, False, "none"
