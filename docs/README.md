# PostTrainOps 文档库

整理日期：2026-10-01（Asia/Shanghai）。本目录保存讨论材料、阶段约束和证据状态，便于后续继续工作。

## 阅读入口

| 文档 | 内容 |
| --- | --- |
| [PostTrainStack 项目与 V0 交接](posttrainstack-v0.md) | 项目目标、V0 五个 Gate、运行顺序及验收边界 |
| [Agent 执行保障层：商业定位](agent-execution-assurance.md) | 客户痛点、产品闭环、价值指标与待验证假设 |
| [当前状态与待办](status.md) | 已保存内容、实际核验结果、待解决问题 |
| [PostTrainStack 原始规划](sources/2026-10-01-posttrainstack-project-plan.txt) | 第一份附件，原样保留 |
| [执行保障层原始论述](sources/2026-10-01-agent-execution-assurance-business.txt) | 第二份附件，原样保留 |

## 文档使用约定

- 两个方向分别记录；它们与 PostTrainOps 名称的关系尚未确定，不自动合并产品范围。
- 原文保存观点和来源；整理稿区分提议、用户提供的交接信息与独立验证结果。
- 附件中的引用占位符未恢复为可核验来源；市场数据、公司动态和示意收益不能作为已确认事实。
- 新的实验结果应记录命令、环境、数据版本、输出路径和验证证据；未实际运行的指标使用 N/A。

## P1 开发要求

[P1 SFT 实施要求与启动审计](p1-sft.md)：已收到实施授权，当前因缺少现有 V0 源码而阻塞。两份原始指令已存档到 sources/。

## P1 交付

[P1 交付与执行证据](p1-delivery.md)：实现清单、22 个测试、真实数据哈希、离线训练证据与正式实验阻塞。

## P1.5 正式实验与红队审计

[P1.5 执行记录与冻结顺序](p1.5-experiment.md)：当前 P1 OPEN；已保存执行前协议哈希与本机环境，待本项目 CUDA 主机。
