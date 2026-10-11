# PostTrainStack

A reproducible LLM post-training engineering stack. V0 establishes the immutable evaluation boundary that every later SFT, DPO, Reward Model, and GRPO experiment must share.

## V0 scope

V0 intentionally does **not** train anything. It creates:

1. a deterministic frozen GSM8K evaluation subset;
2. a manifest with dataset fingerprint + SHA-256;
3. deterministic base-model generation;
4. robust answer extraction;
5. unified correctness / format / output-length metrics;
6. machine-readable predictions and metrics artifacts;
7. unit tests for the evaluator's critical parsing logic.

Default baseline:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `openai/gsm8k`, config `main`, source split `test`
- Frozen set: 500 examples, seed 42
- Generation: greedy, max 256 new tokens

## Why start here?

The evaluator is the contract for the entire repository. SFT, DPO, RM, and GRPO may change, but the frozen evaluation set and metric definitions do not change with the training method.

## Quick start

Python 3.11+ is recommended.

```bash
make install
make test
make data
make smoke      # 8 examples; pipeline check only
make eval-base  # full 500-example V0 baseline
```

Equivalent CLI commands:

```bash
pts-prepare --config configs/base/qwen2_5_0_5b_gsm8k.yaml
pts-eval --config configs/base/qwen2_5_0_5b_gsm8k.yaml --limit 8
pts-eval --config configs/base/qwen2_5_0_5b_gsm8k.yaml
```

## Outputs

After `make data`:

```text
data/frozen/gsm8k_test_500_seed42.jsonl
data/frozen/gsm8k_test_500_seed42.manifest.json
```

After `make eval-base`:

```text
results/base_predictions.jsonl
results/base_metrics.json
```

`base_metrics.json` contains accuracy, format validity, output length, throughput,
environment, seed, dataset hash and generation configuration. No formal model
benchmark has been produced in this workspace; unobserved metrics are N/A.

## V0 gates

- **G1 Installable:** editable install succeeds from a clean environment.
- **G2 Frozen data:** one command creates a deterministic evaluation split plus manifest/hash.
- **G3 Base inference:** one command runs the configured base model.
- **G4 Automatic evaluation:** correctness and format metrics are computed without manual grading.
- **G5 Artifact traceability:** predictions and metrics are saved in machine-readable files and tied to the frozen-set SHA-256.

Only after all five gates pass do we add SFT.

## Repository layout

```text
posttrain-stack/
├── configs/base/
├── data/frozen/                 # generated, immutable evaluation artifact
├── results/                     # generated predictions / metrics
├── src/posttrain_stack/
│   ├── data/
│   ├── evaluation/
│   └── utils/
├── tests/
├── Makefile
└── pyproject.toml
```

## Evaluation rule

Never train on `data/frozen/*`. Later stages must consume training data from separate train-derived artifacts. The frozen test file's SHA-256 is recorded with every evaluation result so a changed benchmark cannot silently look like a model improvement.

## P1 SFT

P1 adds GSM8K train preparation, TRL SFTTrainer, LoRA, unique run receipts and
paired comparison. It reuses `evaluation/run.py`; answer extraction, correctness,
the base YAML prompt and generation settings are preserved. Existing frozen
artifacts are verified and reused; `make data` refuses to overwrite mismatched
artifacts. Leakage auditing reads frozen questions; training consumes train-derived
messages, with conversational prompt/completion and completion-only loss.
There is no evaluation during optimization and no checkpoint selection using test accuracy.

Create a project environment before installing:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'
python -m pip install -e '.[dev,sft]'
python -m pytest -q
```

The local cache setting avoids the host's unavailable E: cache. It is a session
setting, not a change to global environment variables. `requirements-local.txt`
records the tested Windows CPU environment; use the appropriate PyTorch CUDA
wheel on a GPU host rather than treating the CPU environment as a GPU recipe.

With GNU Make available and the environment activated:

```bash
make data
make smoke          # 8 Base examples, saved under results/smoke/
make sft-smoke      # 32 train samples, 10 optimizer steps, save/reload/evaluate 8
make compare-smoke  # smoke results only, never a formal benchmark
make eval-base     # full frozen evaluation
make sft           # full train split, one epoch, then full frozen evaluation
make eval-sft      # reevaluate the last completed formal checkpoint
make compare-sft
```

This Windows host does not have GNU Make. The equivalent commands are:

```powershell
python -m posttrain_stack.data.prepare --config configs/base/qwen2_5_0_5b_gsm8k.yaml
python -m posttrain_stack.evaluation.run --config configs/base/qwen2_5_0_5b_gsm8k.yaml --limit 8 --output-prefix results/smoke/base
python -m posttrain_stack.training.sft --config configs/sft/smoke.yaml
python -m posttrain_stack.training.sft --config configs/sft/smoke.yaml --action compare
python -m posttrain_stack.evaluation.run --config configs/base/qwen2_5_0_5b_gsm8k.yaml
python -m posttrain_stack.training.sft --config configs/sft/full.yaml
python -m posttrain_stack.training.sft --config configs/sft/full.yaml --action evaluate
python -m posttrain_stack.training.sft --config configs/sft/full.yaml --action compare
```

Formal config uses bf16 and requires a supported CUDA GPU. Smoke uses float32 and
can run on CPU when weights are available. Both configs explicitly record LoRA
settings and training parameters. Config validation rejects missing parameters,
invalid booleans and unsupported smoke/formal evaluation limits.

### Artifacts and comparison

- `data/sft/{smoke,train}.jsonl` and their manifests: source, fingerprint, seed,
  count, SHA-256 and exact-overlap audit for the full source train split.
- `runs/<run_id>/`: original config, run manifest, resolved trainer config,
  training metrics, `checkpoint/`, and `evaluation/sft_{predictions,metrics}`.
- `runs/latest-smoke.json` and `runs/latest.json`: separate pointers, updated only
  after a successful train/save/reload/evaluation sequence.
- `results/comparisons/base_vs_sft.{json,md}`: paired metric deltas; accuracy and
  format deltas are percentage points.
- `results/analysis/base_vs_sft_cases.jsonl`: all paired cases and the four
  transitions improved, regressed, both_correct, both_wrong.
- Smoke comparisons stay under `results/smoke/`.

Comparison requires identical frozen SHA-256, prompts, decoding, seed, tokenizer,
evaluator source and evaluation limit, checks prediction hashes, recomputes metrics,
and rejects duplicate/missing IDs or changed questions/gold. Older V0 metrics
without protocol receipts must be regenerated using the same preserved protocol;
they cannot be silently accepted as comparable.

### Verification status

Read [P1 delivery evidence](docs/p1-delivery.md) for commands, results and blockers.
The test suite includes an offline **random tiny model** with real LoRA training,
checkpoint reload and common evaluation. This is integration evidence, not a
Qwen0.5B benchmark. Official model smoke and formal experiments remain gated by
model weights and hardware; no synthetic fixture metrics are reported as results.

P2 is not implemented. First obtain and audit a real Base/SFT paired result and
failure cases; only then consider preference data and DPO.
