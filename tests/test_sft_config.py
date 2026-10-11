import pytest
import yaml

from posttrain_stack.training.sft import load_config


def test_configs():
    for path in ['configs/sft/smoke.yaml', 'configs/sft/full.yaml']:
        cfg = load_config(path)
        assert cfg['peft']['enabled'] is True
        assert cfg['dataset']['source_split'] == 'train'


def test_missing_config(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('model_name: x\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Missing'):
        load_config(path)


def test_invalid_boolean(tmp_path):
    cfg = load_config('configs/sft/smoke.yaml')
    cfg['peft']['enabled'] = 'false'
    path = tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='boolean'):
        load_config(path)
