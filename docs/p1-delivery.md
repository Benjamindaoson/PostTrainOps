# P1 SFT 交付与执行证据

日期：2026-10-01（Asia/Shanghai）。当前状态：P1 代码链路已实现并通过离线验证；正式模型实验受权重下载和 CUDA 环境阻塞，尚未满足完整实验 Definition of Done。

## VERIFIED

- 接管真实 PostTrainStack_V0.zip，解压到当前根目录；保留此前 docs/。
- 原始测试：7 passed；新增功能前独立执行。
- 最终测试：22 passed，1 warning（CPU 不使用 pin_memory），详见 p1-tests.log。
- 真正的离线极小随机 Qwen 模型：10 步 LoRA 训练、保存 adapter/tokenizer、重新加载、统一 evaluator 推理评测全部通过。测试为 test_local_training.py，不下载模型；数据与模型均为 synthetic fixture，不是正式 baseline。
- Mock 编排测试验证 trainer 初始化、保存、reload、调用统一 evaluator、失败 manifest 与成功指针隔离。
- 官方 GSM8K test 冻结 500 条；seed=42，原采样及答案规则保留。
- 官方 GSM8K train 全 7,473 条参与 question overlap 审计，重叠 0；生成 32 条 smoke 和 7,473 条 full SFT 数据。
- CLI help、Python compileall 与 pip dependency check 已执行。
- 新增 P1 模块及修改的 evaluator lint 通过；全仓 lint 仍有 5 条 V0 原有检查项，详见 lint.log。没有关闭规则或删除测试。

数据哈希：

| Artifact | SHA-256 |
| --- | --- |
| Frozen test | 6f10ab3cfb58393d2df503f3a1f640b85c056894f56610e65de256a7d40b8e2e |
| Smoke SFT data | d25a3ae763acea23c56f7946e957b62ee95b4b4bdabc72b8353e1cde7c017a54 |
| Full SFT data | 78f67bdbbeb08816c9f33b92a5041eb7b0eb07661db64704cff80a03b065bd4c |

训练来源 fingerprint 为 c6f812ae33c9159d；test 来源 fingerprint 以 frozen manifest 为准。

## IMPLEMENTED BUT NOT EXECUTED

- Qwen2.5-0.5B 的正式 500 条 Base evaluation。
- Qwen2.5-0.5B 的有效 10 步 smoke training 与正式 full SFT。
- 正式 checkpoint 的 frozen evaluation 与真实 Base/SFT comparison。
- comparison JSON、Markdown、四类逐题 failure cases 的真实正式输出；这些行为已用 fixture 自动验证，但未伪造正式输出文件。
- GNU Make targets 已实现；本 Windows 主机没有 make 可执行文件，实际使用等价 Python 模块命令。目标调用同一已验证模块，仍需在带 GNU Make 的环境直接运行。

## FAILED / 环境阻塞

1. 首次 datasets 导入遇到 pandas DLL 错误。修复为项目 .venv 的 numpy 2.2.6 / pandas 2.2.3，并记录依赖约束。修复后官方数据下载与准备通过。
2. 宿主 HF_HOME 指向不存在的 E:。运行会话改用项目 .cache/huggingface；没有修改全局配置。
3. 官方 Qwen 权重下载反复 read timeout，缓存权重保持 0 bytes；停止的是本次仍在下载的本地 Base smoke 进程，未进行模型推理。模型元数据和 tokenizer 已缓存，但它们不能替代权重。
4. 数据已缓存后以 offline 模式执行 SFT smoke，确认缺少权重会失败，并保存 failed run manifest；训练、保存和正式评测未发生。详见 sft-smoke-attempt.log 和 runs/ 的失败 receipt。
5. Formal 配置前置检查真实失败：本机 torch 2.14.1+cpu，cuda_available=false；bf16 配置要求支持的 CUDA GPU。详见 full-sft-attempt.log。不把 CPU tiny test 作为 GPU/VRAM 验证。
6. 全仓 lint：答案模块的 Decimal 写法、prepare 的 UTC 写法、io 的 import 与异常类型共 5 项。为保留原解析/IO 合同未做无关修订。

## 实际执行命令

全部在项目 .venv 中执行。安装过程先安装了原 V0 依赖，再增加 SFT extras。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e '.[dev]'
.\.venv\Scripts\python -m pytest -q tests/test_answers.py tests/test_metrics.py
.\.venv\Scripts\python -m pip install -e '.[dev,sft]'
# 项目会话中设置 HF_HOME 为 D:\01_work\PostTrainOps\.cache\huggingface
.\.venv\Scripts\python -m posttrain_stack.data.prepare --config configs/base/qwen2_5_0_5b_gsm8k.yaml
.\.venv\Scripts\python -m posttrain_stack.data.sft --config configs/sft/smoke.yaml
.\.venv\Scripts\python -m posttrain_stack.data.sft --config configs/sft/full.yaml
.\.venv\Scripts\python -m posttrain_stack.evaluation.run --config configs/base/qwen2_5_0_5b_gsm8k.yaml --limit 8 --output-prefix results/smoke/base
# 下载失败后使用 offline 模式检查缺失权重的失败路径
.\.venv\Scripts\python -m posttrain_stack.training.sft --config configs/sft/smoke.yaml
.\.venv\Scripts\python -c "from posttrain_stack.training.sft import run_training; run_training('configs/sft/full.yaml')"
.\.venv\Scripts\python -m pytest -q --basetemp .cache/verification-tests
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m pip wheel . --no-deps -w dist
```

原 V0 与交付代码的完整性比对见 evidence/v0-preservation.json；原答案解析、指标实现、Base YAML 和原测试文件保持字节一致。prepare 只增加已有冻结文件保护；evaluator 只增加通用模型/output 路径、冻结哈希验证和协议收据；Makefile 将 smoke 输出隔离，避免覆盖正式 Base 结果。

## 文件与架构变化

| 路径 | 作用 |
| --- | --- |
| src/posttrain_stack/data/sft.py | train 转换、规范化答案、重叠审计、数据 manifest |
| src/posttrain_stack/training/sft.py | 配置验证、TRL/LoRA、唯一 run、失败收据、reload、调用原 evaluator |
| src/posttrain_stack/evaluation/compare.py | 协议/预测一致性检查、指标差异、四类 transition |
| src/posttrain_stack/evaluation/run.py | 通用模型路径、output prefix、协议/预测 hash |
| src/posttrain_stack/data/prepare.py | 已有 frozen 文件验证复用，拒绝重建 |
| configs/sft/{smoke,full}.yaml | 两档显式训练和 LoRA 参数 |
| tests/test_sft_*.py、test_comparison.py、test_p1_runtime.py、test_local_training.py、test_frozen_guard.py | 数据、配置、比较、编排、真实离线集成、不可变性 |
| pyproject.toml、Makefile、.gitignore、requirements-local.txt | SFT extras、CLI/命令、产物边界、环境记录 |
| README.md、docs/、openspec/changes/p1-sft/ | 使用说明、来源与证据、阶段合同 |

运行主链：Train-only builder → leakage audit → TRL SFTTrainer + PEFT → checkpoint → 原 evaluation/run.py → strict paired comparison → failure cases。

官方 API 参考：[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)（conversational prompt/completion、completion-only loss），[PEFT Auto Classes](https://huggingface.co/docs/peft/package_reference/auto_class)。安装后的 SFTConfig 签名也已本地核验；使用 warmup_steps 而非该版本不支持的 warmup_ratio。

## RESULTS 与下一阶段

正式 Base Accuracy、SFT Accuracy、收益、吞吐与 GPU peak VRAM 全部 N/A。真实已产生的结果仅为数据/审计收据、软件测试与离线 tiny integration 证据。

下一步先在权重可下载且具备 CUDA 的环境运行官方 smoke，再执行完整 Base/SFT 配对评测、比较与代码/实验红队审计。P2 Preference Data + DPO 仅在上述结果经审计后进入；本轮未实现 P2。
