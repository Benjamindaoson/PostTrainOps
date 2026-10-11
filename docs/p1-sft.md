# P1 SFT：实施要求与启动审计

日期：2026-10-01（Asia/Shanghai）。用户已授权在现有 V0 上实施 P1，不另起仓库、不重建设计、不提前实施 P2。

## 目标与验收

GSM8K train → conversational SFT data → TRL SFTTrainer + LoRA → checkpoint → 原 V0 frozen evaluator → Base/SFT comparison 与 failure cases。

- 保留 frozen test、答案提取、正确性定义、prompt policy 与生成参数。
- SFT 数据包含原 reasoning 和标准化 Final answer；记录来源、split、fingerprint、seed、SHA-256、时间。
- 标准化 question hash 的 train/test overlap 非空时失败。
- smoke 配置为 10 steps，正式配置独立；LoRA 参数明确记录，支持 peft.enabled。
- 每次运行保存 runs/<run_id>/config.yaml、run_manifest.json、training_metrics.json、checkpoint/。
- 同一 evaluator 评测 Base 与 SFT；生成比较 JSON/Markdown，以及 improved、regressed、both_correct、both_wrong 案例和计数。
- Makefile 支持 test、data、eval-base、sft-smoke、sft、eval-sft、compare-sft。
- 新增数据、配置、训练编排及 comparison 测试；实际运行结果与 fixture、smoke、正式实验分开。
- 原测试通过，完成本地 smoke 与可执行部分验证后才报告完成；环境受限项须明确列出。

## 待实现时处理的边界

指令同时要求训练阶段不读取 data/frozen/ 和程序化 train/frozen overlap 检查。实施时应将 question-only 泄漏检查作为训练前独立审计；训练消费 train 数据及审计凭据，不消费 frozen labels。具体适配依赖原 V0 数据 schema，不在缺少源码时假定其格式。

## 启动审计结果

当前工作区 D:\01_work\PostTrainOps 只有 docs/。未发现 pyproject.toml、Makefile、src/、tests/、configs/、data/frozen/、模型或评测结果。

检查当前目录、用户 Downloads 目录及 D:\01_work 顶层，没有找到名称匹配 PostTrainStack/V0 的源码目录或压缩包。两份新附件均为文本，不含代码包。该检查不代表扫描了整台电脑。

阻塞：缺少现有 V0 源码及冻结评测资产，无法审计、运行原测试，或证明 evaluator 与 frozen 数据保持不变。为遵守用户禁止另建项目和重建设计的要求，本次没有创建替代 evaluator 或独立 SFT 项目。

## 执行记录

已执行：读取两份指令、目录检查、原文存档、本文记录。
未执行：依赖安装、代码实现、原测试、smoke training、full SFT、Base/SFT evaluation、真实比较。
指标：N/A；checkpoint：无。

下一步：将现有 V0 源码放入当前项目或提供其实际路径，随后直接审计并实现 P1。当前阻塞不是实现细节选择。

## 原始要求

- [完整开发指令](sources/2026-10-01-p1-sft-development-request.txt)
- [阶段计划及重复交接指令](sources/2026-10-01-p1-sft-phase-plan.txt)

## 后续实施记录

上述缺源码状态为启动时历史记录，现已解除。当前实现、测试与实验阻塞详见 [P1 交付证据](p1-delivery.md)。
