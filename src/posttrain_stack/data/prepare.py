from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from posttrain_stack.data.answers import extract_gsm8k_gold
from posttrain_stack.utils.io import load_yaml, sha256_file, write_json, write_jsonl


def build_frozen_split(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    seed = int(cfg["project"]["seed"])

    output_path = Path(data_cfg['output_path'])
    manifest_path = Path(data_cfg['manifest_path'])
    if output_path.exists() or manifest_path.exists():
        if not output_path.is_file() or not manifest_path.is_file():
            raise ValueError('Frozen artifact pair incomplete; refusing regeneration')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        expected = {'dataset': data_cfg['dataset'], 'dataset_config': data_cfg.get('config'),
                    'source_split': data_cfg['source_split'], 'seed': seed,
                    'frozen_size': int(data_cfg['frozen_size']),
                    'frozen_file_sha256': sha256_file(output_path)}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Frozen manifest/config/SHA-256 mismatch; refusing regeneration')
        return manifest

    dataset = load_dataset(
        data_cfg["dataset"],
        data_cfg.get("config"),
        split=data_cfg["source_split"],
    )
    frozen_size = int(data_cfg["frozen_size"])
    if frozen_size <= 0 or frozen_size > len(dataset):
        raise ValueError(f"frozen_size must be in [1, {len(dataset)}]")

    frozen = dataset.shuffle(seed=seed).select(range(frozen_size))
    rows = []
    for index, item in enumerate(frozen):
        rows.append(
            {
                "id": f"gsm8k-test-{index:04d}",
                "question": item["question"],
                "answer": item["answer"],
                "gold_answer": extract_gsm8k_gold(item["answer"]),
            }
        )

    output_path = Path(data_cfg["output_path"])
    write_jsonl(output_path, rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data_cfg["dataset"],
        "dataset_config": data_cfg.get("config"),
        "source_split": data_cfg["source_split"],
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "frozen_size": frozen_size,
        "seed": seed,
        "frozen_file": str(output_path),
        "frozen_file_sha256": sha256_file(output_path),
    }
    write_json(data_cfg["manifest_path"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the immutable V0 evaluation split.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = build_frozen_split(args.config)
    print(f"Prepared {manifest['frozen_size']} frozen examples")
    print(f"SHA256: {manifest['frozen_file_sha256']}")


if __name__ == "__main__":
    main()
