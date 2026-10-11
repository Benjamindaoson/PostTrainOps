"""Real offline training/inference with a random tiny model, never a benchmark."""
import copy
import json

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from posttrain_stack.training.sft import load_config, run_training
from posttrain_stack.utils.io import sha256_file, write_json, write_jsonl


def test_real_offline_lora_train_save_reload_evaluate(tmp_path):
    model_dir = tmp_path/'tiny-model'
    vocab = {v: i for i, v in enumerate([
        '<unk>', '<pad>', '<|im_end|>', '<|im_start|>', 'user', 'assistant',
        'system', 'Question', 'Final', 'answer:', '1', '2', 'Reasoning'])}
    raw = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='<unk>',
                                       pad_token='<pad>', eos_token='<|im_end|>')
    tokenizer.chat_template = (
        '{% for message in messages %}{{ "<|im_start|> " + message["role"] + " " '
        '+ message["content"] + " <|im_end|> " }}{% endfor %}'
        '{% if add_generation_prompt %}{{ "<|im_start|> assistant " }}{% endif %}')
    tokenizer.save_pretrained(model_dir)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=16,
                                       intermediate_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=2,
                                       eos_token_id=2, pad_token_id=1))
    model.save_pretrained(model_dir)
    del model
    cfg = copy.deepcopy(load_config('configs/sft/smoke.yaml'))
    cfg['model_name'] = str(model_dir)
    cfg['output_dir'] = str(tmp_path/'runs')
    cfg['dataset'].update(name='fixture/synthetic', output_path=str(tmp_path/'train.jsonl'),
                          manifest_path=str(tmp_path/'train.manifest.json'))
    frozen = tmp_path/'frozen/test.jsonl'
    write_jsonl(frozen, [{'id': f'tiny-{i}', 'question': 'Question 2', 'gold_answer': '2'}
                        for i in range(8)])
    fm = tmp_path/'frozen/manifest.json'
    write_json(fm, {'frozen_file_sha256': sha256_file(frozen)})
    ec = {'project': {'seed': 42, 'name': 'OFFLINE_FIXTURE_NOT_BENCHMARK', 'stage': 'fixture'},
              'data': {'output_path': str(frozen), 'manifest_path': str(fm), 'dataset': 'fixture'},
              'model': {'name_or_path': str(model_dir), 'dtype': 'float32', 'device_map': 'cpu'},
              'prompt': {'system': 'Reasoning', 'user_template': '{question}'},
              'generation': {'batch_size': 8, 'max_new_tokens': 4, 'do_sample': False},
              'output': {'predictions_path': str(tmp_path/'predictions.jsonl'),
                      'metrics_path': str(tmp_path/'metrics.json')}}
    eval_config = tmp_path/'eval.yaml'
    eval_config.write_text(yaml.safe_dump(ec), encoding='utf-8')
    cfg['evaluation_config'] = str(eval_config)
    cp = tmp_path/'sft.yaml'
    cp.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    write_jsonl(cfg['dataset']['output_path'], [{'messages': [
        {'role': 'user', 'content': 'Question 1'},
        {'role': 'assistant', 'content': 'Reasoning Final answer: 1'}]}]*12)
    write_json(cfg['dataset']['manifest_path'], {
        'source_split': 'train', 'source_dataset': 'fixture/synthetic', 'num_samples': 12,
        'exact_overlap_count': 0, 'dataset_sha256': sha256_file(cfg['dataset']['output_path']),
        'frozen_file_sha256': sha256_file(frozen), 'dataset_fingerprint': 'synthetic-only'})
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        receipt = run_training(cp)
    finally:
        torch.set_num_threads(previous)
    assert receipt['status'] == 'completed'
    assert receipt['training_metrics']['train_loss'] > 0
    pointer = json.loads((tmp_path/'runs/latest-smoke.json').read_text(encoding='utf-8'))
    from pathlib import Path
    checkpoint = Path(pointer['checkpoint'])
    assert (checkpoint/'adapter_model.safetensors').is_file()
    metrics = json.loads((checkpoint.parent/'evaluation/sft_metrics.json').read_text(encoding='utf-8'))
    assert metrics['num_samples'] == 8
    assert metrics['dataset'] == 'fixture'
