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
  -> discovery evidence (seeds 0-2)
  -> isolated offline Qwen curator
  -> one exact Markdown snippet patch
  -> candidate complete rendered-memory snapshot
  -> paired correction replay: parent vs candidate (seeds 3-5)
  -> paired preservation replay on baseline-proven cases
  -> accepted / rejected / pending
  -> accepted only: materialize immutable libraries/S001
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

`SkillPatch/v1` 每次只允许修改一个 Markdown 文件中的一个精确 snippet，字段为
`activation/procedure/termination/recovery` 之一，且至少引用两条 discovery 轨迹。
当前自动 curator 主要面向已读取叶子 skill 的 replace；novel task 若没有可归因叶子，
应形成 evidence-insufficient 结果，而不是把错误归给无关 skill。

接纳条件：correction 成功数比 parent 至少增加 1、parent-success preservation case
零退化、目标 skill 在 candidate correction 中至少激活 2 次、无安全违规。运行或
planner 基础设施错误得到 `pending`，不被算作候选失败。

## 运行

编辑并执行：

```bash
bash scripts/skill_evolution/run_baseline_compatible_cycle.sh
```

脚本快速配置区暴露 suite/task、discovery/correction seeds、preservation cases、Qwen
模型和服务地址、Pi0.5 checkpoint、GPU、VLA endpoint、token/turn/episode budget。

首次正式 evolve 前应先做 A/A 检查：同一组 baseline case 分别使用原始 MEMORY 与
`S000` memory view。二者允许存在模型采样和物理仿真的随机波动，但不应出现工具缺失、
VLA horizon 改变、prompt 缺失或系统性动作模式变化。

## 暂未实现

- 连续 control-step 级 VLA 蒸馏数据与训练；
- 视觉状态到具体 skill step 的强因果对齐；
- 自动挑选三个语义不同且已由 baseline 验证成功的 preservation cases；当前由脚本
  明确配置，避免系统悄悄使用失败 case；
- 多轮 library chain 的自动调度；当前一键入口执行一个 S000→S001 cycle。
