import pytest

from posttrain_stack.data.answers import extract_gsm8k_gold, extract_prediction


def test_extract_gsm8k_gold_integer_with_comma():
    assert extract_gsm8k_gold("reasoning\n#### 1,234") == "1234"


def test_extract_gsm8k_gold_decimal():
    assert extract_gsm8k_gold("reasoning\n#### 12.50") == "12.5"


def test_extract_gsm8k_gold_requires_marker():
    with pytest.raises(ValueError):
        extract_gsm8k_gold("answer is 4")


def test_prediction_prefers_explicit_final_answer():
    pred, valid, method = extract_prediction("2 + 3 = 5\nFinal answer: 5")
    assert pred == "5"
    assert valid is True
    assert method == "final_answer"


def test_prediction_boxed():
    pred, valid, method = extract_prediction(r"Therefore \\boxed{42}")
    assert pred == "42"
    assert valid is True
    assert method == "boxed"


def test_prediction_falls_back_to_last_number():
    pred, valid, method = extract_prediction("I think the result is 17")
    assert pred == "17"
    assert valid is False
    assert method == "last_number"
