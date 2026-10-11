"""Strict paired comparison of real artifacts from the common evaluator."""
import argparse
import json
import math
from pathlib import Path

from posttrain_stack.evaluation.metrics import compute_metrics
from posttrain_stack.utils.io import read_jsonl, sha256_file, write_json, write_jsonl


def validate_protocol(base, sft):
    for key in ('frozen_file_sha256', 'seed', 'generation', 'prompt',
                'tokenizer_sha256', 'evaluator_sha256', 'evaluation_limit'):
        if key not in base or key not in sft or base[key] != sft[key]:
            raise ValueError(f'Evaluation protocol mismatch or missing: {key}')


def pair_predictions(base, sft):
    def index(rows):
        result = {r['id']: r for r in rows}
        if not result or len(result) != len(rows):
            raise ValueError('Empty predictions or duplicate IDs')
        for r in rows:
            if type(r['is_correct']) is not bool or r['is_correct'] != (
                    r['prediction'] == r['gold_answer']):
                raise ValueError('Incorrect correctness flag')
        return result
    bi, si = index(base), index(sft)
    if bi.keys() != si.keys():
        raise ValueError('Prediction IDs differ')
    cases = []
    counts = dict.fromkeys(('improved', 'regressed', 'both_correct', 'both_wrong'), 0)
    for rid, b in bi.items():
        s = si[rid]
        if any(b[k] != s[k] for k in ('question', 'gold_answer')):
            raise ValueError(f'Question/gold mismatch: {rid}')
        transition = ('both_correct' if b['is_correct'] else 'improved') if s[
            'is_correct'] else ('regressed' if b['is_correct'] else 'both_wrong')
        counts[transition] += 1
        cases.append({'id': rid, 'question': b['question'], 'gold_answer': b['gold_answer'],
                          'base_output': b['response'], 'base_extracted_answer': b['prediction'],
                          'base_correct': b['is_correct'], 'sft_output': s['response'],
                          'sft_extracted_answer': s['prediction'], 'sft_correct': s['is_correct'],
                          'transition': transition})
    return cases, counts


def compare(base_prefix, sft_prefix, output_dir='results', training_manifest=None):
    def load(prefix):
        mp, pp = f'{prefix}_metrics.json', f'{prefix}_predictions.jsonl'
        metrics = json.loads(Path(mp).read_text(encoding='utf-8'))
        rows = read_jsonl(pp)
        if metrics.get('predictions_sha256') != sha256_file(pp):
            raise ValueError('Predictions hash mismatch or missing')
        for key, value in compute_metrics(rows).items():
            if metrics.get(key) != value:
                raise ValueError(f'Metrics inconsistent with predictions: {key}')
        return metrics, rows
    bm, br = load(base_prefix)
    sm, sr = load(sft_prefix)
    validate_protocol(bm, sm)
    cases, counts = pair_predictions(br, sr)
    table = {}
    for key in ('accuracy', 'format_valid_rate', 'avg_output_tokens', 'samples_per_second'):
        b = bm['runtime'][key] if key == 'samples_per_second' else bm[key]
        s = sm['runtime'][key] if key == 'samples_per_second' else sm[key]
        if b is None or s is None or not all(math.isfinite(x) for x in (b, s)):
            raise ValueError(f'Invalid metric: {key}')
        table[key] = {'base': b, 'sft': s, 'delta': s-b,
                          'delta_unit': 'percentage_points' if key in (
                              'accuracy', 'format_valid_rate') else 'absolute'}
        if key in ('accuracy', 'format_valid_rate'):
            table[key]['delta'] *= 100
    result = {'metrics': table, 'transitions': {f'num_{k}': v for k, v in counts.items()},
                  'num_samples': len(cases), 'evaluation_limit': bm['evaluation_limit'],
                  'protocol': {k: bm[k] for k in ('frozen_file_sha256', 'seed', 'generation',
                            'prompt', 'tokenizer_sha256', 'evaluator_sha256')},
                  'source_metrics': {'base': f'{base_prefix}_metrics.json',
                                      'sft': f'{sft_prefix}_metrics.json'}}
    if training_manifest:
        tm = json.loads(Path(training_manifest).read_text(encoding='utf-8'))
        if tm['status'] != 'completed' or tm['frozen_file_sha256'] != bm['frozen_file_sha256']:
            raise ValueError('Invalid training receipt')
        if str(Path(tm['final_checkpoint']).resolve()) != str(Path(sm['model']).resolve()):
            raise ValueError('Training checkpoint does not match SFT evaluation')
        result['training'] = tm.get('training_metrics')
    root = Path(output_dir)
    write_json(root/'comparisons/base_vs_sft.json', result)
    write_jsonl(root/'analysis/base_vs_sft_cases.jsonl', cases)
    lines = ['# Base vs SFT', '', f'Samples: {len(cases)}; limit: {bm["evaluation_limit"]}', '',
             '| Metric | Base | SFT | Delta |', '| --- | ---: | ---: | ---: |']
    for k, v in table.items():
        unit = ' pp' if v['delta_unit'] == 'percentage_points' else ''
        lines.append(f'| {k} | {v["base"]:.6f} | {v["sft"]:.6f} | {v["delta"]:+.6f}{unit} |')
    lines.extend(['', *[f'- {k}: {v}' for k, v in result['transitions'].items()]])
    (root/'comparisons/base_vs_sft.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-prefix', default='results/base')
    parser.add_argument('--sft-prefix', default='results/sft')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--training-manifest')
    args = parser.parse_args()
    print(json.dumps(compare(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
