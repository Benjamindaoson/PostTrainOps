import pytest

from posttrain_stack.evaluation.compare import pair_predictions, validate_protocol


def rows(flags):
    return [{'id': str(i), 'question': f'Q{i}', 'gold_answer': '1', 'response': 'text',
                 'prediction': '1' if flag else '2', 'is_correct': flag,
                 'format_valid': True, 'output_tokens': 10, 'extraction_method': 'final_answer'}
            for i, flag in enumerate(flags)]


def test_all_transitions():
    cases, counts = pair_predictions(rows([False, True, True, False]),
                                     rows([True, False, True, False]))
    assert [x['transition'] for x in cases] == [
        'improved', 'regressed', 'both_correct', 'both_wrong']
    assert set(counts.values()) == {1}


def test_invalid_pairing():
    with pytest.raises(ValueError):
        pair_predictions(rows([True]), rows([True, False]))
    with pytest.raises(ValueError):
        pair_predictions(rows([True]) * 2, rows([True]) * 2)


def test_protocol_mismatch():
    with pytest.raises(ValueError):
        validate_protocol({'seed': 1}, {'seed': 2})
