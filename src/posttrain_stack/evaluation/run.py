from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain_stack.data.answers import extract_prediction
from posttrain_stack.evaluation.metrics import compute_metrics
from posttrain_stack.utils.io import load_yaml, read_jsonl, sha256_file, write_json, write_jsonl
from posttrain_stack.utils.seed import set_seed


def _render_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_evaluation(config_path: str, limit: int | None = None,
                   model_path: str | None = None, output_prefix: str | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    data_path = cfg["data"]["output_path"]
    frozen_manifest = json.loads(Path(cfg['data']['manifest_path']).read_text(encoding='utf-8'))
    if sha256_file(data_path) != frozen_manifest['frozen_file_sha256']:
        raise ValueError('Frozen file SHA-256 mismatch')
    if limit is not None and limit <= 0:
        raise ValueError('Evaluation limit must be positive')
    rows = read_jsonl(data_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Frozen evaluation set is empty")

    model_cfg = cfg["model"]
    model_name = model_path or model_cfg["name_or_path"]
    # Both models use the original base tokenizer and chat template.
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_cfg.get("dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
    )
    model.eval()

    system_prompt = cfg["prompt"]["system"]
    user_template = cfg["prompt"]["user_template"]
    gen_cfg = cfg["generation"]
    batch_size = int(gen_cfg["batch_size"])
    max_new_tokens = int(gen_cfg["max_new_tokens"])
    do_sample = bool(gen_cfg.get("do_sample", False))

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for batch in _batched(rows, batch_size):
        prompts = [
            _render_prompt(
                tokenizer,
                system_prompt,
                user_template.format(question=item["question"]),
            )
            for item in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        prompt_width = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_token_ids = generated[:, prompt_width:]
        texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for item, text, token_ids in zip(batch, texts, new_token_ids, strict=True):
            pred, format_valid, method = extract_prediction(text)
            predictions.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "prediction": pred,
                    "response": text,
                    "is_correct": pred == item["gold_answer"],
                    "format_valid": format_valid,
                    "extraction_method": method,
                    "output_tokens": int((token_ids != tokenizer.pad_token_id).sum().item()),
                }
            )

    elapsed = time.perf_counter() - started
    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "project": cfg["project"]["name"],
            "stage": cfg["project"]["stage"],
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "model": model_name,
            "dataset": cfg["data"]["dataset"],
            "frozen_file": data_path,
            "frozen_file_sha256": sha256_file(data_path),
            "seed": seed,
            "prompt": cfg['prompt'],
            "evaluation_limit": limit,
            "tokenizer_sha256": hashlib.sha256(json.dumps({
                'vocab': tokenizer.get_vocab(), 'template': tokenizer.chat_template,
                'special_tokens': tokenizer.special_tokens_map,
            }, sort_keys=True).encode()).hexdigest(),
            "evaluator_sha256": hashlib.sha256(''.join(sha256_file(p) for p in (
                Path(__file__), Path(__file__).with_name('metrics.py'),
                Path(__file__).parent.parent/'data/answers.py')).encode()).hexdigest(),
            "generation": {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "batch_size": batch_size,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "samples_per_second": len(predictions) / elapsed if elapsed else None,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
    )

    predictions_path = f'{output_prefix}_predictions.jsonl' if output_prefix else cfg[
        'output']['predictions_path']
    metrics_path = f'{output_prefix}_metrics.json' if output_prefix else cfg['output']['metrics_path']
    write_jsonl(predictions_path, predictions)
    metrics['predictions_sha256'] = sha256_file(predictions_path)
    write_json(metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base model on the frozen split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; do not use for final V0.")
    parser.add_argument('--model', help='Base model or local full/PEFT checkpoint')
    parser.add_argument('--output-prefix')
    args = parser.parse_args()
    metrics = run_evaluation(args.config, limit=args.limit, model_path=args.model,
                             output_prefix=args.output_prefix)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
