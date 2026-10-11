"""Train-only data builder; frozen questions are read solely for leakage auditing."""
import argparse
import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from posttrain_stack.data.answers import extract_gsm8k_gold
from posttrain_stack.utils.io import load_yaml, read_jsonl, sha256_file, write_json, write_jsonl


def question_hash(question):
    normalized = ' '.join(unicodedata.normalize('NFKC', question).casefold().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def check_overlap(train_questions, frozen_questions):
    overlap = {question_hash(q) for q in train_questions} & {
        question_hash(q) for q in frozen_questions}
    if overlap:
        raise ValueError(f'Train/frozen question overlap: {len(overlap)}')


def convert_sample(item):
    question = item['question'].strip()
    if not question:
        raise ValueError('Empty training question')
    answer = extract_gsm8k_gold(item['answer'])
    reasoning = item['answer'].split('####', 1)[0].strip()
    return {'messages': [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': f'{reasoning}\nFinal answer: {answer}'.strip()},
    ]}


def prepare_data(config):
    from datasets import load_dataset

    cfg = load_yaml(config) if isinstance(config, (str, Path)) else config
    dc = cfg['dataset']
    if dc['source_split'] != 'train' or dc['name'] != 'openai/gsm8k':
        raise ValueError('P1 requires openai/gsm8k train')
    ec = load_yaml(cfg['evaluation_config'])
    frozen_path = ec['data']['output_path']
    frozen_manifest = json.loads(Path(ec['data']['manifest_path']).read_text(encoding='utf-8'))
    frozen_sha = sha256_file(frozen_path)
    if frozen_sha != frozen_manifest['frozen_file_sha256']:
        raise ValueError('Frozen SHA-256 mismatch')
    # Only questions leave this audit; frozen labels never enter converted rows.
    frozen_questions = [row['question'] for row in read_jsonl(frozen_path)]
    source = load_dataset(dc['name'], dc['config'], split='train')
    check_overlap(source['question'], frozen_questions)
    selected = source.shuffle(seed=cfg['seed'])
    size = dc['max_samples']
    if size is not None:
        if not 0 < size <= len(selected):
            raise ValueError('Invalid training subset size')
        selected = selected.select(range(size))
    rows = [convert_sample(item) for item in selected]
    if Path(dc['output_path']).resolve().is_relative_to(Path(frozen_path).resolve().parent):
        raise ValueError('Training data must not be stored under frozen directory')
    write_jsonl(dc['output_path'], rows)
    manifest = {'source_dataset': dc['name'], 'source_config': dc['config'], 'source_split': 'train',
                    'num_samples': len(rows), 'seed': cfg['seed'],
                    'dataset_sha256': sha256_file(dc['output_path']),
                    'dataset_fingerprint': source._fingerprint,
                    'created_at_utc': datetime.now(UTC).isoformat(),
                    'frozen_file_sha256': frozen_sha, 'exact_overlap_count': 0,
                    'audited_source_samples': len(source),
                    'question_normalization': 'NFKC-casefold-whitespace-v1'}
    write_json(dc['manifest_path'], manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build isolated GSM8K SFT data')
    parser.add_argument('--config', required=True)
    print(json.dumps(prepare_data(parser.parse_args().config), indent=2))


if __name__ == '__main__':
    main()
