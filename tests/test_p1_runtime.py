import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from posttrain_stack.evaluation.compare import compare
from posttrain_stack.evaluation.metrics import compute_metrics
from posttrain_stack.training.sft import discover_checkpoint, load_config, run_training
from posttrain_stack.utils.io import sha256_file, write_json, write_jsonl


def test_comparison_artifacts(tmp_path):
    from test_comparison import rows

    def save(name, flags, speed):
        prefix = tmp_path/name
        predictions = rows(flags)
        write_jsonl(f'{prefix}_predictions.jsonl', predictions)
        m = compute_metrics(predictions)
        m.update(frozen_file_sha256='frozen', seed=42, generation={'max_new_tokens': 256},
                 prompt={'system': 'same'}, tokenizer_sha256='tokenizer',
                 evaluator_sha256='evaluator', evaluation_limit=None,
                 predictions_sha256=sha256_file(f'{prefix}_predictions.jsonl'),
                 runtime={'samples_per_second': speed})
        write_json(f'{prefix}_metrics.json', m)
        return str(prefix)
    b, s = save('base', [False, True, True, False], 2), save('sft', [True, True, True, False], 3)
    result = compare(b, s, tmp_path/'out')
    assert result['metrics']['accuracy']['delta'] == 25
    assert result['metrics']['samples_per_second']['delta'] == 1
    assert (tmp_path/'out/analysis/base_vs_sft_cases.jsonl').is_file()
    path = Path(f'{s}_metrics.json')
    m = json.loads(path.read_text())
    m['accuracy'] = 1
    write_json(path, m)
    with pytest.raises(ValueError, match='inconsistent'):
        compare(b, s, tmp_path/'out')


def test_checkpoint_discovery(tmp_path):
    with pytest.raises(ValueError):
        discover_checkpoint(tmp_path)
    for name in ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json']:
        (tmp_path/name).write_text('{}')
    assert discover_checkpoint(tmp_path) == str(tmp_path.resolve())


@pytest.mark.parametrize('fail', [False, True])
def test_trainer_orchestration(tmp_path, monkeypatch, fail):
    import torch
    import transformers
    import trl

    from posttrain_stack.evaluation import run

    cfg = copy.deepcopy(load_config('configs/sft/smoke.yaml'))
    cfg['output_dir'] = str(tmp_path/'runs')
    cfg['dataset']['output_path'] = str(tmp_path/'train.jsonl')
    cfg['dataset']['manifest_path'] = str(tmp_path/'train.manifest.json')
    frozen = tmp_path/'frozen/test.jsonl'
    write_jsonl(frozen, [{'question': 'test question'}])
    ec = tmp_path/'eval.yaml'
    ec.write_text(yaml.safe_dump({'data': {'output_path': str(frozen)}}))
    cfg['evaluation_config'] = str(ec)
    cp = tmp_path/'train.yaml'
    cp.write_text(yaml.safe_dump(cfg))
    write_jsonl(cfg['dataset']['output_path'], [{'messages': [
        {'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'Final answer: 1'}]}])
    write_json(cfg['dataset']['manifest_path'], {
        'source_split': 'train', 'source_dataset': 'openai/gsm8k', 'num_samples': 1,
        'exact_overlap_count': 0, 'dataset_sha256': sha256_file(cfg['dataset']['output_path']),
        'frozen_file_sha256': sha256_file(frozen), 'dataset_fingerprint': 'fixture'})
    calls = []

    class Tokenizer:
        pad_token_id = 0

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path)/'tokenizer_config.json').write_text('{}')

    class Trainer:
        def __init__(self, **kwargs):
            calls.append('init')
            self.args = SimpleNamespace(to_dict=lambda: {'fixture': True})
            assert kwargs['peft_config'].r == 8
            assert kwargs['train_dataset'][0]['completion'][0]['role'] == 'assistant'

        def train(self):
            calls.append('train')
            if fail:
                raise RuntimeError('injected training failure')
            return SimpleNamespace(metrics={'train_loss': 1.0})

        def save_model(self, path):
            calls.append('save')
            Path(path).mkdir(parents=True)
            (Path(path)/'adapter_config.json').write_text('{}')
            (Path(path)/'adapter_model.safetensors').write_text('fixture')

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained',
                        lambda *a, **k: calls.append('reload'))
    monkeypatch.setattr(trl, 'SFTTrainer', Trainer)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(run, 'run_evaluation', lambda *a, **k: calls.append('evaluate'))
    if fail:
        with pytest.raises(RuntimeError, match='injected'):
            run_training(cp)
        root = next((tmp_path/'runs').glob('20*'))
        receipt = json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
        assert receipt['status'] == 'failed'
        assert receipt['final_checkpoint'] is None
        assert not (tmp_path/'runs/latest-smoke.json').exists()
        return
    receipt = run_training(cp)
    assert calls == ['init', 'train', 'save', 'reload', 'evaluate']
    assert receipt['status'] == 'completed'
    root = next((tmp_path/'runs').glob('20*'))
    assert (root/'training_metrics.json').is_file()
    assert json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))[
        'dataset_fingerprint'] == 'fixture'
