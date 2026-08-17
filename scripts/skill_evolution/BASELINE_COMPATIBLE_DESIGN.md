# Baseline-Compatible Skill Evolution（新分支实现）

本文件描述 `skill-evolution-baseline-compatible` 分支的实际实现。原
[`PIPELINE.md`](./PIPELINE.md) 按要求原样保留，用于记录上一版已废弃的强侵入设计；
其中的强制 skill protocol、phased VLA 和 `max_chunks <= 25` 不代表本分支行为。

## 设计边界

当前阶段只演化显式 skill，不进行 VLA 蒸馏。在线 execution stack 与 commit
`5750c8701a9f5ee684795d3a09b361928e53f4c8` 的 baseline 对齐：

- 相同 Qwen planner、system/user prompt、sampling 入口和最大轮数；
- 相同工具名称、JSON schema 和调用顺序自由度；
- 相同 Pi0.5 checkpoint、RPC 接口及 `pi0_pick/pi0_doubled` 能力；
- 不禁止长程 VLA，不强制 planner 使用新的 subtask/decision 工具；
- 同一 replay case 中，parent 与 candidate 唯一实验变量是 MEMORY 版本。

唯一的在线附加能力是 observer：它不阻止、不改写动作。

## 数据流

```text
reviewed resources/libero/memory
  -> exact snapshot libraries/S000/rendered_memory
  -> baseline planner reads resources/libero/memory/*
  -> runtime transparently serves S000/rendered_memory/*
  -> passive trace: memory read -> tool call/result -> official outcome
  -> OptimizerEvidence/v1 (seeds 0-2; structured trace, no console parsing)
  -> independent stronger multimodal skill optimizer
  -> no_patch or one exact MEMORY-index/leaf snippet patch
  -> candidate complete rendered-memory snapshot
  -> paired correction replay: parent vs candidate (seeds 3-5)
  -> paired preservation replay on baseline-proven cases
  -> accepted / rejected / pending
  -> accepted only: materialize next immutable libraries/SNNN
```

Planner 仍按 baseline prompt 先读完整 `MEMORY.md`，再读相关叶子 skill。读取叶子文件
被视为一次弱 activation；之后的工具调用继承当时已读 skill ID。该归因不等价于严格
因果归因，因此最终接纳仍必须依靠 parent/candidate 物理 replay，而不能只相信 LLM
解释或 planner 的 `finish(success)`。

## Outcome contract

控制循环在环境 `terminated` 或 `truncated` 时都会停止，但两者分开持久化：

- `libero_terminated=true`：唯一 benchmark success；
- `libero_truncated=true`：有效停止但不是成功；
- primitive `success` 与 planner `finish(status=success)` 仅作诊断。

## Patch 与接纳

`SkillPatch/v2` 每次只允许修改一个 Markdown 文件中的一个精确 snippet，且至少引用两条
discovery 轨迹。对 `MEMORY.md` 只允许替换 `Reusable manipulation patterns` 下指向目标
leaf 的一个 routing bullet；对 leaf 只允许修改
`activation/procedure/termination/recovery` 之一。目标 leaf 必须在有效 discovery 中真实
读过。novel task 若没有可归因 leaf，形成 `no_patch/insufficient_evidence`，不把错误归给
无关 skill。

Optimizer 的版本化约束位于
[`skill_optimizer/SKILL.md`](./skill_optimizer/SKILL.md)。它与 execution planner 使用独立
OpenAI-compatible 多模态服务。每条 rollout 从 `evolution_trace.jsonl`、transcript、
`states.json` 和最多六张精选图片生成 `optimizer_evidence.json`；thinking 与
`console.log` 不进入 optimizer 输入。

接纳条件：correction 成功数比 parent 至少增加 1、parent-success preservation case
零退化、目标 skill 在 candidate correction 中至少激活 2 次、无安全违规。运行或
planner 基础设施错误得到 `pending`，不被算作候选失败。

## 运行

编辑并执行：

```bash
bash scripts/skill_evolution/run_baseline_compatible_cycle.sh
```

脚本快速配置区暴露 suite/task、discovery/correction seeds、preservation cases、Qwen
模型和服务地址、独立 optimizer 的 URL/key/model/token/timeout/图片及 patch budget、
Pi0.5 checkpoint、GPU、VLA endpoint、token/turn/episode budget。每次启动只创建一个
新 `cycle_NNN`；它从编号最大的 accepted `libraries/SNNN` 开始。rejected candidate 和
旧 cycle 永不覆盖。

首次正式 evolve 前应先做 A/A 检查：同一组 baseline case 分别使用原始 MEMORY 与
`S000` memory view。二者允许存在模型采样和物理仿真的随机波动，但不应出现工具缺失、
VLA horizon 改变、prompt 缺失或系统性动作模式变化。

## 暂未实现

- 连续 control-step 级 VLA 蒸馏数据与训练；
- 视觉状态到具体 skill step 的强因果对齐；
- 自动挑选三个语义不同且已由 baseline 验证成功的 preservation cases；当前由脚本
  明确配置，避免系统悄悄使用失败 case；
- 自动从 baseline 结果库挑选 preservation cases；当前仍由脚本明确配置。
