"""Single-device TRL SFT with explicit configuration and durable receipts."""
import argparse
import copy
import gc
import importlib.metadata
import json
import platform
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from posttrain_stack.utils.io import load_yaml, read_jsonl, sha256_file, write_json
from posttrain_stack.utils.seed import set_seed


def load_config(path):
    cfg = load_yaml(path)
    required = ('model_name', 'dataset', 'output_dir', 'seed', 'training', 'peft',
                'evaluation_config', 'smoke', 'evaluation_limit')
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f'Missing SFT configuration: {missing}')
    for section, keys in {
        'dataset': ('name', 'config', 'source_split', 'output_path', 'manifest_path', 'max_samples'),
        'training': ('learning_rate', 'num_train_epochs', 'per_device_train_batch_size',
                     'gradient_accumulation_steps', 'max_length', 'max_steps', 'bf16', 'fp16',
                     'gradient_checkpointing', 'logging_steps', 'save_strategy', 'save_steps',
                     'report_to', 'optim', 'lr_scheduler_type', 'warmup_steps', 'weight_decay',
                     'max_grad_norm', 'packing', 'completion_only_loss', 'eos_token'),
        'peft': ('enabled', 'r', 'lora_alpha', 'lora_dropout', 'target_modules', 'bias', 'task_type'),
    }.items():
        if not isinstance(cfg[section], dict) or any(k not in cfg[section] for k in keys):
            raise ValueError(f'Missing required {section} parameters')
    t = cfg['training']
    for section, key in [('peft', 'enabled'), *[('training', k) for k in (
            'bf16', 'fp16', 'gradient_checkpointing', 'packing', 'completion_only_loss')]]:
        if type(cfg[section][key]) is not bool:
            raise ValueError(f'{section}.{key} must be a boolean')
    if type(cfg['smoke']) is not bool or type(cfg['seed']) is not int:
        raise ValueError('smoke must be boolean and seed must be integer')
    p = cfg['peft']
    if (type(p['r']) is not int or p['r'] <= 0 or p['lora_alpha'] <= 0
            or not 0 <= p['lora_dropout'] < 1
            or not isinstance(p['target_modules'], list) or not p['target_modules']
            or p['task_type'] != 'CAUSAL_LM'):
        raise ValueError('Invalid LoRA configuration')
    if cfg['dataset']['source_split'] != 'train':
        raise ValueError('SFT source split must be train')
    if t['bf16'] and t['fp16']:
        raise ValueError('bf16 and fp16 are mutually exclusive')
    for key in ('learning_rate', 'num_train_epochs', 'per_device_train_batch_size',
                'gradient_accumulation_steps', 'max_length', 'logging_steps', 'save_steps'):
        if not isinstance(t[key], (int, float)) or t[key] <= 0:
            raise ValueError(f'Invalid training parameter: {key}')
    if cfg['smoke'] and (t['max_steps'] != 10 or cfg['evaluation_limit'] != 8):
        raise ValueError('Smoke requires 10 training steps and 8 evaluation examples')
    if not cfg['smoke'] and cfg['evaluation_limit'] is not None:
        raise ValueError('Formal SFT must evaluate the full frozen set')
    return cfg


def discover_checkpoint(path):
    path = Path(path)
    adapter = (path/'adapter_config.json').is_file() and (path/'adapter_model.safetensors').is_file()
    full = (path/'config.json').is_file() and any(path.glob('model*.safetensors'))
    if not (adapter or full) or not (path/'tokenizer_config.json').is_file():
        raise ValueError(f'Incomplete checkpoint: {path}')
    return str(path.resolve())


def run_training(config_path):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(config_path)
    dc, tc = cfg['dataset'], cfg['training']
    manifest = json.loads(Path(dc['manifest_path']).read_text(encoding='utf-8'))
    ec = load_yaml(cfg['evaluation_config'])
    if (manifest['source_split'] != 'train' or manifest['exact_overlap_count'] != 0
            or manifest['source_dataset'] != dc['name']
            or manifest['dataset_sha256'] != sha256_file(dc['output_path'])
            or manifest['frozen_file_sha256'] != sha256_file(ec['data']['output_path'])):
        raise ValueError('Dataset provenance/hash/leakage audit invalid')
    if Path(dc['output_path']).resolve().is_relative_to(
            Path(ec['data']['output_path']).resolve().parent):
        raise ValueError('Trainer must not consume frozen data')
    rows = read_jsonl(dc['output_path'])
    if not rows or len(rows) != manifest['num_samples']:
        raise ValueError('Training sample count mismatch')
    run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    root = Path(cfg['output_dir'])/run_id
    root.mkdir(parents=True, exist_ok=False)
    (root/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True), encoding='utf-8')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'trl', 'peft')}
    receipt = {'run_id': run_id, 'stage': 'sft', 'smoke': cfg['smoke'], 'status': 'running',
                   'model': cfg['model_name'], 'base_model': cfg['model_name'], 'dataset': manifest,
                   'dataset_sha256': manifest['dataset_sha256'],
                   'dataset_fingerprint': manifest['dataset_fingerprint'],
                   'frozen_file_sha256': manifest['frozen_file_sha256'], 'seed': cfg['seed'],
                   'git_commit': commit, 'python': platform.python_version(), 'versions': versions,
                   'cuda_available': torch.cuda.is_available(),
                   'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'training_config': tc, 'lora_config': cfg['peft'],
                   'start_time': datetime.now(UTC).isoformat(), 'end_time': None,
                   'training_duration': None, 'final_checkpoint': None}
    write_json(root/'run_manifest.json', receipt)
    start = time.perf_counter()
    try:
        if tc['bf16'] and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('Full bf16 config requires a supported CUDA device')
        if tc['fp16'] and not torch.cuda.is_available():
            raise RuntimeError('fp16 requires CUDA')
        set_seed(cfg['seed'])
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Conversational prompt/completion gives completion-only loss without changing templates.
        train = Dataset.from_list([{'prompt': r['messages'][:-1],
                                    'completion': r['messages'][-1:]} for r in rows])
        params = copy.deepcopy(tc)
        params.update(output_dir=str(root/'trainer'), seed=cfg['seed'], data_seed=cfg['seed'],
                      eval_strategy='no', load_best_model_at_end=False,
                      model_init_kwargs={'dtype': 'bfloat16' if tc['bf16'] else (
                          'float16' if tc['fp16'] else 'float32')})
        peft_params = {k: v for k, v in cfg['peft'].items() if k != 'enabled'}
        trainer = SFTTrainer(model=cfg['model_name'], args=SFTConfig(**params),
                             train_dataset=train, processing_class=tokenizer,
                             peft_config=LoraConfig(**peft_params) if cfg['peft']['enabled'] else None)
        receipt['resolved_training_config'] = trainer.args.to_dict()
        write_json(root/'run_manifest.json', receipt)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_started = time.perf_counter()
        metrics = trainer.train().metrics
        metrics['training_time'] = time.perf_counter()-train_started
        metrics['peak_vram_bytes'] = (torch.cuda.max_memory_allocated()
                                      if torch.cuda.is_available() else None)
        checkpoint = root/'checkpoint'
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        receipt['final_checkpoint'] = discover_checkpoint(checkpoint)
        write_json(root/'training_metrics.json', metrics)
        receipt['training_metrics'] = metrics
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Explicit reload validation, then the SAME evaluator for all inference and scoring.
        reloaded = AutoModelForCausalLM.from_pretrained(checkpoint)
        del reloaded
        gc.collect()
        from posttrain_stack.evaluation.run import run_evaluation
        prefix = str(root/'evaluation/sft')
        run_evaluation(cfg['evaluation_config'], limit=cfg['evaluation_limit'],
                       model_path=str(checkpoint), output_prefix=prefix)
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['end_time'] = datetime.now(UTC).isoformat()
        receipt['training_duration'] = receipt.get('training_metrics', {}).get('training_time')
        receipt['run_duration_seconds'] = time.perf_counter()-start
        write_json(root/'run_manifest.json', receipt)
    write_json(Path(cfg['output_dir'])/('latest-smoke.json' if cfg['smoke'] else 'latest.json'),
               {'run_dir': str(root.resolve()), 'checkpoint': receipt['final_checkpoint']})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--action', choices=['train', 'evaluate', 'compare'], default='train')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == 'train':
        from posttrain_stack.data.sft import prepare_data
        prepare_data(cfg)
        result = run_training(args.config)
    else:
        pointer = Path(cfg['output_dir'])/('latest-smoke.json' if cfg['smoke'] else 'latest.json')
        latest = json.loads(pointer.read_text(encoding='utf-8'))
        receipt = json.loads((Path(latest['run_dir'])/'run_manifest.json').read_text(encoding='utf-8'))
        if receipt['status'] != 'completed' or receipt['smoke'] != cfg['smoke']:
            raise ValueError('Latest run is not a completed matching run')
        prefix = str(Path(latest['run_dir'])/'evaluation/sft')
        if args.action == 'evaluate':
            from posttrain_stack.evaluation.run import run_evaluation
            result = run_evaluation(cfg['evaluation_config'], cfg['evaluation_limit'],
                                    latest['checkpoint'], prefix)
        else:
            from posttrain_stack.evaluation.compare import compare
            base = 'results/smoke/base' if cfg['smoke'] else 'results/base'
            result = compare(base, prefix,
                             'results/smoke' if cfg['smoke'] else 'results',
                             str(Path(latest['run_dir'])/'run_manifest.json'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
