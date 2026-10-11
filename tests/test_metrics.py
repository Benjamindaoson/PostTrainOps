from posttrain_stack.evaluation.metrics import compute_metrics


def test_compute_metrics():
    metrics = compute_metrics(
        [
            {
                "is_correct": True,
                "format_valid": True,
                "output_tokens": 10,
                "extraction_method": "final_answer",
            },
            {
                "is_correct": False,
                "format_valid": False,
                "output_tokens": 20,
                "extraction_method": "last_number",
            },
        ]
    )
    assert metrics["num_samples"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["format_valid_rate"] == 0.5
    assert metrics["avg_output_tokens"] == 15
