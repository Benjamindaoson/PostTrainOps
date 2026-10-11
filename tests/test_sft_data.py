import pytest

from posttrain_stack.data.sft import check_overlap, convert_sample, question_hash
from posttrain_stack.utils.io import sha256_file, write_jsonl


def test_conversion_and_hash():
    row = convert_sample({'question': 'How many?', 'answer': 'Work here\n#### 1,234.00'})
    assert row['messages'][1]['content'] == 'Work here\nFinal answer: 1234'
    assert row['messages'][0] == {'role': 'user', 'content': 'How many?'}
    assert question_hash('  HOW\n many? ') == question_hash('how many?')


def test_overlap_fails():
    with pytest.raises(ValueError, match='overlap'):
        check_overlap(['Hello  World'], [' hello world '])
    check_overlap(['one'], ['two'])


def test_stable_jsonl(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    write_jsonl(a, [convert_sample({'question': 'Q', 'answer': 'R\n#### 12.50'})])
    write_jsonl(b, [convert_sample({'question': 'Q', 'answer': 'R\n#### 12.50'})])
    assert sha256_file(a) == sha256_file(b)
