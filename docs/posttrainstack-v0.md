# PostTrainStack：项目规划与 V0 交接

整理日期：2026-10-01。来源：项目规划附件及本次对话中的 V0 交接说明。

## 核心目标

构建可复现的大模型后训练工程，在同一模型、同一数据和同一冻结评测集上比较 SFT、DPO、Reward Modeling、GRPO 的收益、成本和失败模式，作为求职工程项目，而非 Notebook 合集。

## 阶段边界

当前先做 V0：仓库骨架与 Base Evaluation Pipeline。SFT、DPO、RM、GRPO 尚不进入实施范围。Base 真实结果出来后，下一阶段只做 SFT，并使用同一冻结评测集比较。

规划中的 Reward Model 是独立评分模型；DPO 和 GRPO 可从同一 SFT checkpoint 出发，不应写成 DPO → RM → GRPO 的必经链路。模型规模升级、TRL 和后续分布式框架均属于规划，未验证实现。

## V0 交接范围

用户说明已生成 PostTrainStack_V0.zip，但本会话未收到压缩包或源码。

| Gate | 交接说明中的行为 | 当前证据状态 |
| --- | --- | --- |
| G1 标准 Python 工程 | pyproject.toml，支持 editable install 和 dev 依赖 | 未核验源码 |
| G2 冻结评测集 | GSM8K main/test，seed=42，确定性 shuffle 后取 500 条；manifest 保存 fingerprint、seed、样本数、SHA-256 | 未实际执行 |
| G3 Base 推理 | Qwen2.5-0.5B-Instruct，使用模型 chat template 生成 | 未实际执行 |
| G4 答案提取 | Final answer、boxed、####、最后一个数字 fallback；逗号和小数规范化；记录 extraction_method 与 format_valid | 未核验实现 |
| G5 统一结果 | predictions JSONL 与 metrics JSON，记录质量、性能、环境和配置 | 未生成结果 |

交接描述的目录：configs/base/、src/posttrain_stack/data/、src/posttrain_stack/evaluation/、src/posttrain_stack/utils/、tests/、data/frozen/、results/、Makefile、pyproject.toml、README.md。

## 数据隔离约束

训练阶段禁止读取 data/frozen/*。GSM8K train 用于未来训练数据；冻结 test 只用于评测。后续所有训练方法共用同一冻结评测集，不因结果调整抽样。

## 预期结果

- results/base_predictions.jsonl
- results/base_metrics.json
- data/frozen/frozen_test.jsonl 与 manifest.json（具体位置待源码核验）

指标包括 accuracy、format_valid_rate、avg_output_tokens、samples_per_second、elapsed_seconds，以及 CUDA device、PyTorch version、dataset SHA-256、seed、model、generation config。

## 运行与验收顺序

以下命令来自用户交接，尚未在当前工作区验证。先在项目根目录创建并使用 .venv；Windows 环境需核实 make 是否可用，或使用 Makefile 中对应命令。

```text
make install
make test
make data
make smoke
# 核对 8 条 smoke 的输出后再运行
make eval-base
```

Smoke 验收：模型加载、prompt 构造、生成、答案解析、指标计算、结果落盘均成功；正式 baseline 为 500 条。

用户报告已有 7 个测试通过，覆盖 gold 解析、逗号数字、小数规范化、Final Answer、boxed、fallback 和指标聚合。这是交接信息，不是本会话独立测试结果。

## 结果表

| 项目 | 当前值 |
| --- | --- |
| 拟用 Base Model | Qwen2.5-0.5B-Instruct |
| Accuracy | N/A |
| Format Valid Rate | N/A |
| Avg Output Tokens | N/A |
| Throughput | N/A |
| 正式评测完成状态 | 未执行 |

完整规划见 [原始附件](sources/2026-10-01-posttrainstack-project-plan.txt)。
