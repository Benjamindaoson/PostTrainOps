import json

import pytest
import yaml

from posttrain_stack.data.prepare import build_frozen_split
from posttrain_stack.utils.io import sha256_file, write_json, write_jsonl


def test_existing_frozen_is_verified_without_regeneration(tmp_path, monkeypatch):
    from posttrain_stack.data import prepare

    frozen, manifest, config = tmp_path/'test.jsonl', tmp_path/'manifest.json', tmp_path/'cfg.yaml'
    write_jsonl(frozen, [{'question': 'Q', 'gold_answer': '1'}])
    receipt = {'dataset': 'openai/gsm8k', 'dataset_config': 'main', 'source_split': 'test',
               'frozen_size': 1, 'seed': 42, 'frozen_file_sha256': sha256_file(frozen)}
    write_json(manifest, receipt)
    config.write_text(yaml.safe_dump({'project': {'seed': 42}, 'data': {
        'dataset': 'openai/gsm8k', 'config': 'main', 'source_split': 'test', 'frozen_size': 1,
        'output_path': str(frozen), 'manifest_path': str(manifest)}}))

    def forbidden(*a, **k):
        raise AssertionError('Must not regenerate frozen data')

    monkeypatch.setattr(prepare, 'load_dataset', forbidden)
    assert build_frozen_split(str(config)) == receipt
    frozen.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen'):
        build_frozen_split(str(config))
    assert json.loads(manifest.read_text(encoding='utf-8')) == receipt
