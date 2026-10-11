from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No prediction rows to evaluate")

    correct = [bool(row["is_correct"]) for row in rows]
    format_valid = [bool(row["format_valid"]) for row in rows]
    output_tokens = [int(row["output_tokens"]) for row in rows]
    methods = Counter(str(row["extraction_method"]) for row in rows)

    return {
        "num_samples": len(rows),
        "accuracy": mean(correct),
        "format_valid_rate": mean(format_valid),
        "avg_output_tokens": mean(output_tokens),
        "extraction_methods": dict(sorted(methods.items())),
    }
