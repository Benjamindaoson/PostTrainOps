PYTHON ?= python
CONFIG ?= configs/base/qwen2_5_0_5b_gsm8k.yaml
SFT_CONFIG ?= configs/sft/full.yaml

.PHONY: install test lint data smoke eval-base clean-results

install:
	$(PYTHON) -m pip install -e ".[dev,sft]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

data:
	$(PYTHON) -m posttrain_stack.data.prepare --config $(CONFIG)

smoke:
	$(PYTHON) -m posttrain_stack.evaluation.run --config $(CONFIG) --limit 8 --output-prefix results/smoke/base

eval-base:
	$(PYTHON) -m posttrain_stack.evaluation.run --config $(CONFIG)

.PHONY: sft-data sft-smoke sft eval-sft compare-sft compare-smoke
sft-data:
	$(PYTHON) -m posttrain_stack.data.sft --config $(SFT_CONFIG)

sft-smoke:
	$(PYTHON) -m posttrain_stack.training.sft --config configs/sft/smoke.yaml

sft:
	$(PYTHON) -m posttrain_stack.training.sft --config $(SFT_CONFIG)

eval-sft:
	$(PYTHON) -m posttrain_stack.training.sft --config $(SFT_CONFIG) --action evaluate

compare-sft:
	$(PYTHON) -m posttrain_stack.training.sft --config $(SFT_CONFIG) --action compare

compare-smoke:
	$(PYTHON) -m posttrain_stack.training.sft --config configs/sft/smoke.yaml --action compare

clean-results:
	rm -f results/base_predictions.jsonl results/base_metrics.json
