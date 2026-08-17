# Skill Self-Evolution Related Work 与 RPent 方案分析

本文总结 `related_work/` 中与 Agent Skill 自演化相关的主要方法，并结合 RPent Harness
VLA、LIBERO 实验和三 seed case study，分析不同路线的适用性，给出当前阶段推荐的
skill-evolution 机制。

当前阶段只讨论显式 MEMORY/skill 的演化。VLA 权重更新、连续动作数据导出和蒸馏属于
后续阶段，本文不会将其描述为已经实现的能力。

相关项目文档：

- [Baseline-compatible 设计](./BASELINE_COMPATIBLE_DESIGN.md)
- [三 Seed MEMORY 路由 Case Study TODO](./CASE_STUDY_MEMORY_ROUTING_TODO.md)
- [长期研究方向](../../Self-evolving%20skills%20in%20Harness%20VLA.md)

## 1. 问题定义

最直接的 skill evolution 方案是：

```text
积累 rollout 日志
  -> 外接一个 LLM 阅读日志
  -> LLM 重写 skill/MEMORY
  -> 新版本在下一轮部署
```

该方案容易实现，也能利用大模型的归纳能力，但存在明显缺陷：

- 每轮都读取完整日志和 skill，分析延迟与 token 成本较高；
- 逐轨迹更新对日志顺序敏感，容易被偶然失败带偏；
- 全文重写可能删除已有有效经验，产生隐性回归；
- 长期 append 会造成 skill 膨胀、冲突和上下文污染；
- LLM 给出的失败解释未必是物理系统中的真实原因；
- “文本看起来更合理”不代表机器人执行成功率更高；
- 更新触发、目标 skill、修改字段和接纳标准如果都由同一个模型决定，演化难以归因。

因此，可靠的 self-evolution 不能只回答“用什么模型改写文本”，还必须回答：

1. 哪些 rollout evidence 有权触发更新？
2. 问题发生在 skill 检索、应用、执行还是恢复层？
3. 应该修改 activation、procedure、termination 还是 recovery？
4. 如何限制一次更新的范围？
5. 如何证明候选比父版本更好且没有破坏旧能力？

## 2. Related Work 方法分类

### 2.1 日志反思与直接文本更新

代表方法包括 AutoSkill、SkillForge、ReasoningBank 和部分 memory/self-reflection 系统。

典型流程：

```text
trajectory/dialogue
  -> LLM reflection/extraction
  -> reusable lesson or skill
  -> merge/rewrite existing artifact
  -> retrieval in future tasks
```

AutoSkill 将交互经验抽取为版本化 `SKILL.md`，通过 skill management 和 merge 模块持续
维护 skill bank。SkillForge 使用 Failure Analyzer、Skill Diagnostician 和 Skill
Optimizer 对批量失败进行诊断和重写。ReasoningBank 同时从成功和失败中提炼可检索的
reasoning strategy。

优点：

- 实现直接，适合快速验证“经验能否沉淀为文本知识”；
- 能表达复杂、开放式的操作知识；
- 不需要更新执行模型参数；
- 产物可读、可编辑、可版本化。

缺点：

- 对 curator 模型能力和 prompt 高度敏感；
- 开放式全文重写难以保证局部性；
- 单条失败容易诱发 anecdotal fix；
- 如果缺少执行验证，文本质量与任务成功率可能脱节。

对 RPent 的意义：适合用作受约束的候选生成器，不适合独自拥有更新决定权。

### 2.2 多轨迹并行 patch 与归纳合并

[Trace2Skill](../../related_work/latex-source/Trace2Skill/sections_v2/method.tex) 使用三阶段流程：

```text
冻结 agent 生成 success/failure trajectories
  -> success/error analysts 并行生成 trajectory-level patches
  -> hierarchical merge 去重、解决冲突并归纳共性
  -> 生成一个可移植的 consolidated skill
```

失败分析采用 agentic analyst，可以检查 artifact、验证假设并排除无法因果解释的失败；
成功分析则提取可复用行为。并行 many-to-one consolidation 相比逐条编辑降低了顺序依赖，
也能用重复出现的 patch 支持跨轨迹泛化。

优点：

- 多条证据共同决定更新，稳定性优于 sequential editing；
- success/failure 非对称处理符合实际证据特点；
- 并行分析降低串行 LLM 延迟；
- 适合从大量轨迹中压缩 recurring lessons。

缺点：

- 分析调用总量仍然较大；
- 合并后的 skill 可能包含未经过物理验证的组合；
- 对 RPent 当前的小规模三 seed MVP 偏重；
- 若每条轨迹都生成 patch，机器人低频但高成本的数据利用率不高。

对 RPent 的意义：适合后续多任务、多轮数据积累后的批量 procedure/recovery 深化，不适合
作为首版每个 episode 都运行的在线更新器。

### 2.3 有界文本优化与验证集接纳

[SkillOpt](../../related_work/latex-source/SkillOpt/sections/3_methods.tex) 将 skill 看作冻结 agent
的外部可训练状态：

```text
rollout batch = forward pass
minibatch reflection = textual gradient
bounded add/delete/replace = optimizer step
held-out evaluation = validation gate
```

它通过 edit budget 模拟 learning rate，限制每步允许修改的文本量；被拒绝的 edit 进入
rejected buffer，帮助后续优化避免重复失败；epoch-wise slow/meta update 用于保留跨轮次
的稳定经验。

优点：

- 不进行无界全文重写；
- 接纳标准基于执行分数，而不是 LLM 自评；
- rejected edit 形成负反馈；
- 部署时没有额外模型调用。

缺点：

- 需要多次 candidate evaluation；
- 在机器人场景中，物理 replay 成本远高于文本/软件任务；
- 单一全局 score 仍可能掩盖 retrieval 和 execution 的不同问题。

对 RPent 的意义：非常适合约束 patch 和保存 rejected candidate，但需要使用廉价 precheck
减少进入物理 replay 的候选数量。

### 2.4 结构化 Semantic Gradient

[Skill-Pro](../../related_work/latex-source/Skill-Pro/main.tex) 将 procedure skill 形式化为
activation/initiation、execution/policy 和 termination 三部分。Skill Doctor 根据执行历史
生成只针对必要组件的 semantic gradient，再由 Skill Evolver 应用聚合反馈。

典型更新不是“重写整个 skill”，而是：

```json
{
  "activation": "需要修改的触发边界，或空字符串",
  "procedure": "需要修改的策略，或空字符串",
  "termination": "需要修改的完成条件，或空字符串"
}
```

优点：

- 更新对象明确，可解释性强；
- 能区分“找不到 skill”“执行错误”和“过早结束”；
- 空字段机制避免无关部分被顺带修改；
- 与 RPent 的 activation/procedure/termination/recovery schema 高度兼容。

缺点：

- semantic gradient 仍可能来自错误诊断；
- 如果没有 attribution，可能修改一个仅被读取但未真正使用的 skill；
- surrogate/PPO gate 在机器人环境中仍需真实 replay 校准。

对 RPent 的意义：适合作为 patch 表达形式，但必须先接入 trajectory attribution 和
physical admission。

### 2.5 生命周期治理与 Subtask Attribution

[SkillsVote](../../related_work/latex-source/SkillsVote/sections/4_framework.tex) 强调 skill 的完整
生命周期：collection、recommendation、attribution 和 evolution。它不把完整 trajectory
或单个 tool call 直接作为更新单元，而是将轨迹拆成具有独立目标、主要评价信号和至多一个
主要 skill context 的 subtask。

其核心判断包括：

- skill 是否在执行前被暴露；
- 是否真正影响了动作；
- 成功来自 skill-guided execution 还是 agent independent exploration；
- 新发现是扩展现有 skill，还是应创建新 skill；
- 失败和弱证据是否只能保留作诊断，而不能直接授权更新。

优点：

- 能避免“所有读过的 skill 共同获得 success credit”；
- 把任务级稀疏 reward 与过细的 tool-call trace 连接起来；
- update/edit/create/skip 的边界清楚；
- 特别适合长时序 Harness。

缺点：

- attribution 本身需要可靠的结构化数据；
- 如果完全依赖 LLM 后验解释，仍可能产生伪因果；
- RPent 当前 baseline-compatible trace 只有 passive `skill_read`，还没有完整 subtask 标注。

对 RPent 的意义：这是当前最需要借鉴的前置层。没有 read/applied/effective 区分，后续任何
skill rewrite 都容易错误归因。

### 2.6 周期性 Hard-Case Buffer 与选择器/Skill 双重演化

[MemSkill](../../related_work/latex-source/MemSkill/2_Method.tex) 同时优化：

- controller：从可变 skill bank 中选择 Top-K skill；
- executor：按照选中的 skill 产生 memory update；
- designer：周期性查看 hard cases，修改旧 skill 或创建新 skill。

它不会让 designer 处理每个样本，而是维护 sliding hard-case buffer，将失败聚类后选择
高难度且具有代表性的样本。更新退化时回滚，同时短期提高新 skill 的探索概率。

SkillRL 也采用 hierarchical skill library、adaptive retrieval 和 recursive evolution，
并进一步让 skill library 与参数化 policy 在 RL 中共同演化。

优点：

- 周期性聚类降低 curator 调用频率；
- 重复失败和代表性失败比单一案例更可靠；
- 明确区分“skill 内容”和“如何选择 skill”；
- 能直接优化 retrieval bottleneck。

缺点：

- 训练 controller 或 policy 会引入新的实验变量；
- 首版若同时改变 planner、router 和 MEMORY，将难以证明收益来自 skill evolution；
- RL/learned controller 增加实现和调参成本。

对 RPent 的意义：双速 evolution 很适合长期方案，但第一版应先保持 planner/VLA 不变，
只通过版本化 MEMORY 文本验证 routing evolution。

### 2.7 Typed Artifact 与 Correction/Preservation Replay

[AutoRefine](../../related_work/latex-source/AutoRefine/section/3methodology.tex) 先从成功/失败对照中
构造 type-neutral intervention specification，再选择能够拥有所需状态和决策边界的最小
artifact：Rule、Skill 或 Subagent。候选首先通过 contract gate，再在 correction 与
preservation cases 上和父版本对跑。

优点：

- 先定义需要修复的行为，再决定用哪种 artifact 表达；
- 能避免把局部规则包装成过重 skill；
- correction 验证修复，preservation 防止遗忘；
- 和机器人环境的真实 terminated 信号契合。

缺点：

- 完整 typed artifact compiler 对当前 MVP 较重；
- subagent ownership 暂时不是本项目的首要问题；
- replay 保证只覆盖已测 cases，不是全局性能保证。

对 RPent 的意义：不必首版实现完整 artifact selection，但应保留 evidence-linked patch、
contract gate、correction replay 和 preservation replay。

### 2.8 程序搜索、迭代调试与 Verifier 共演化

Voyager、ASPIRE 和 CoEvoSkills 更偏向 generate--execute--verify--refine：

```text
生成一个或多个可执行候选
  -> 环境中执行
  -> 根据错误、视觉证据或 surrogate verifier 诊断
  -> 修改候选并再次验证
  -> 保存最佳 validated skill
```

ASPIRE 针对机器人任务维护 task analysis、候选 ledger、已排除方向和 replay 结果，并对困难
任务做多候选 evolutionary search。CoEvoSkills 让 Skill Generator 与 Surrogate Verifier
共同迭代，最后仍使用 oracle pass/fail 检查。

优点：

- 不要求一次 LLM 诊断就正确；
- 适合复杂接触、恢复和长程 procedure；
- 能发现日志归纳无法直接产生的新动作策略。

缺点：

- rollout 成本最高；
- 动态代码生成扩大安全和调试范围；
- surrogate verifier 也可能和真实物理 predicate 不一致；
- 对当前主要是 MEMORY routing 的问题过于复杂。

对 RPent 的意义：适合作为后续低成功率 hard-task mode，而不是第一版默认循环。

## 3. 方法横向比较

| 方法方向 | 额外 LLM 成本 | 物理验证成本 | 更新稳定性 | 适合当前 routing 问题 | 适合新物理 recovery |
|---|---:|---:|---:|---:|---:|
| 每条日志直接重写 | 高 | 可低但不可靠 | 低 | 中 | 中 |
| 多轨迹并行合并 | 高但可并行 | 中 | 中 | 中 | 高 |
| 有界文本优化 | 中 | 高 | 高 | 高 | 高 |
| Semantic gradient | 中 | 中 | 中到高 | 高 | 高 |
| Attribution-gated update | 低到中 | 中 | 高 | 很高 | 高 |
| Hard-case 周期更新 | 中、摊销后较低 | 中 | 高 | 高 | 高 |
| Router/skill 共同训练 | 训练成本高 | 中 | 中到高 | 很高 | 中 |
| 多候选程序搜索 | 很高 | 很高 | 验证后高 | 低 | 很高 |

没有一种方法同时达到零成本、零延迟和完全稳定。合理设计的目标是：

- 不在每个 episode 后调用大模型；
- 用结构化统计过滤不值得更新的轨迹；
- 让 LLM 只处理难以模板化的语义归纳；
- 用有界 patch 代替全文 rewrite；
- 用相同 case 的真实执行结果决定是否接纳。

## 4. RPent Case Study 暴露出的实际问题

实验目录：

```text
logs/skill_evolution/libero_object_t0_baseline_compatible_v2
```

任务：

```text
grab alphabet soup and put it into basket
```

### Seed 0

- 读取 `MEMORY.md`、目录和三份 guide；
- 没有打开任何 leaf skill；
- 大量 `back_project` 后使用 `pi0_pick` 和手工 carry/release；
- 40 turns 耗尽，`libero_terminated=false`。

### Seed 1

- 只读取 `MEMORY.md`；
- 没有打开任何 leaf skill；
- 多次局部 `pi0_pick` 后手工放置；
- 视觉上接近成功，但官方 predicate 未触发。

### Seed 2

- 读取 `can_and_small_package_basket_placement.md`；
- planner 明确引用其中的 `ZERO explicit localization`、`pi0_doubled`、verbatim task
  language 和 `max_chunks=80`；
- 实际工具调用匹配该 procedure；
- 28 chunks 后 `libero_terminated=true`。

这个 case study 暴露的首要问题不是 VLA 能力不足，而是：

1. 外层 MEMORY 对单物体 can→basket 的 routing 描述不充分；
2. 读取 MEMORY 后不保证继续读取 leaf；
3. 当前 trace 把 read 当作弱 activation，无法区分真正 applied；
4. task leaf 与 generic localization workflow 存在优先级冲突；
5. curator proposal 链存在响应解析和错误分类问题。

同时要严格注意：三个 seed 的初始布局不同，因此它们只能支持候选 hypothesis，不能单独
证明 skill 导致成功。因果结论必须来自相同 case 的 parent/candidate replay。

## 5. 对 RPent 的推荐方案：结构化归因驱动的分层双速演化

推荐组合不是完整照搬某一篇工作，而是：

```text
SkillsVote attribution
  + Skill-Pro typed semantic gradient
  + MemSkill periodic hard-case buffer
  + SkillOpt bounded text update
  + AutoRefine correction/preservation gate
```

### 5.1 L0：结构化 Evidence 与应用归因

首先将每个 task/skill 关系分成四层：

```text
indexed -> read -> applied -> effective
```

- `indexed`：skill 出现在本轮 planner 可见的 MEMORY index；
- `read`：planner 实际打开对应版本 leaf；
- `applied`：读取后的工具、参数和动作顺序匹配 skill execution signature；
- `effective`：应用后产生权威成功或有证据支持的局部物理进展。

只依赖 planner 文本声明不够。高置信 application 应由以下证据联合确定：

- skill read 的时间；
- planner 是否明确引用 skill；
- 后续 primary tool 是否匹配；
- prompt 构造和关键参数是否匹配；
- 是否出现和 skill 相反的动作；
- 环境 outcome 是否支持效果。

推荐指标：

```text
retrieval_rate = P(read correct skill | applicable task)
application_rate = P(applied | skill read)
effectiveness = P(authoritative success | skill applied)
```

### 5.2 快速层：MEMORY Routing Evolution

快速层只更新：

- activation；
- aliases；
- routing summary；
- index exposure/priority。

这类更新尽量不调用大模型。程序根据结构化证据和模板生成 candidate，leaf metadata 作为
语义源，再确定性重建 `MEMORY.md` index。禁止 curator 自由重写完整 MEMORY。

典型触发：

```text
同一 task pattern 中：
  applicable but unread failures
  + read/applied success outside documented scope
```

当前 seed 0/1/2 正好支持一个 routing hypothesis：

```text
将 can_and_small_package_basket_placement
从“双物体 can + package → basket”
扩展到“单个或多个 rigid can/package → basket”。
```

这类 update 可由固定 operator 表达：

```yaml
operator: expand_activation
target: can_and_small_package_basket_placement
add:
  cardinality: [single, multiple]
  object_types: [rigid_can, small_package]
  target_types: [basket]
  aliases:
    - grab alphabet soup and put it into basket
```

### 5.3 慢速层：Leaf Procedure/Recovery Evolution

慢速层处理：

- procedure；
- execution priority；
- failure signature；
- recovery；
- termination evidence。

它只在 hard-case buffer 达到触发条件时运行 curator。轨迹先按 skill、失败 primitive 和
observable failure signature 聚类，再分别从 success/failure 生成 semantic gradient。

模型不接收开放式“重写 skill”指令，而只接收：

- 目标 skill 和唯一允许修改的字段；
- compact evidence spans；
- 必须保留的有效行为；
- forbidden edits 和最大长度；
- rejected edit history；
- 期望修复的可观测结果。

### 5.4 Hard-Task 模式：受预算限制的多候选搜索

当同一个 skill 已高置信 applied、同类物理失败反复出现，且 bounded rewrite 多轮无提升时，
才进入 hard-task search：

- 一次生成少量互斥假设；
- 每个 candidate 明确预期失败模式；
- 在固定 debug cases 上运行；
- 保存 eliminated、blocked 和 best candidate；
- 未超过父版本则回退。

首版禁止动态生成无边界机器人代码；候选只能修改允许的 Markdown procedure/recovery，
并继续通过现有 primitive 接口执行。

## 6. 推荐 Evolution Operators

### Operator A：Routing Expansion

输入条件：

- task 在语义上适用，但正确 skill 没有被读取；
- 同一个 skill 在相邻的新 scope 中被读取、实际应用并成功。

动作：

- 扩展 activation/aliases；
- 自动重建 MEMORY index；
- 不修改 procedure。

### Operator B：Routing Narrowing

输入条件：

- skill 被频繁读取；
- 但 application 低或实际采用后持续有害；
- 问题来自适用范围过宽，而不是 procedure 缺陷。

动作：

- 收窄 activation；
- 添加 negative boundary；
- 必要时 quarantine。

### Operator C：Execution Clarification

输入条件：

- skill 已读取；
- planner 没有执行 primary procedure，或先执行了相反/冗余动作；
- skill 内容本身已有正确策略但显著性或优先级不足。

动作：

- 在 procedure 开头增加短 execution contract；
- 明确 first physical action；
- 明确 generic workflow 的可覆盖范围；
- 不在文末重复追加同义历史说明。

### Operator D：Novel-Scope Consolidation

输入条件：

- planner 将现有 skill 成功应用到原 activation 之外；
- 工具签名匹配且环境 outcome 权威成功。

动作：

- 将成功泛化写回 activation boundary；
- 保留来源 episode 和适用限制；
- 通过新 scope correction 和旧 scope preservation 后接纳。

### Operator E：Failure/Recovery Refinement

输入条件：

- `behavior_matched=true`；
- 出现当前 skill 未描述的可观测物理失败；
- 至少有成功/失败对比，或两个同类失败。

动作：

- 添加简洁的 failure condition；
- 添加可验证 recovery；
- 不把基础设施错误、路径错误和服务未配置写进任务 skill。

## 7. 更新触发与成本控制

推荐事件触发而不是固定每 episode 更新：

```text
episode end
  -> 本地结构化 attribution（无 LLM）
  -> 更新 task/skill statistics 和 hard-case buffer
  -> 检查 trigger
       routing operator 可由模板生成
       procedure/recovery 达到聚类阈值才调用 curator
  -> candidate precheck
  -> physical replay admission
```

降低成本的具体方式：

- curator 读取 compact trace，不读取完整 transcript；
- 只保留决策、工具参数/结果、前后状态、权威 outcome 和必要图像引用；
- 对 success 和 failure 使用不同 analyst prompt；
- 多个相似案例先聚类，再调用一次 curator；
- parent 相同 case 结果缓存复用；
- router-only 测试作为廉价 prefilter，但不能替代物理 success；
- 无法唯一归因的 evidence 不生成 edit；
- rejected candidate 和原因永久保存，避免重复尝试。

## 8. Candidate 验证与接纳

候选进入 active library 前依次经过：

### 8.1 Schema/Contract Gate

- 目标 skill 与字段合法；
- 一次只修改一个字段；
- 工具名称与参数有效；
- 不覆盖安全和 outcome contract；
- activation、procedure 和 recovery 有可观测边界；
- execution signature 可以由 runtime 检查。

### 8.2 Routing/Application Precheck

- candidate index 是否暴露了目标 skill；
- planner 是否更可能在首个 physical action 前读取它；
- 读取后动作是否匹配 execution signature；
- 该阶段只筛选候选，不宣称任务成功提升。

### 8.3 Correction Replay

parent/candidate 在相同未见 cases、相同 planner/VLA/settings/budget 下对跑，要求候选修复
目标问题。

### 8.4 Preservation Replay

在父 skill 已成功的旧范围 case 上对跑，任何 parent-success 回归都拒绝候选。

### 8.5 Decision

- `accepted`：correction 提升、preservation 零退化、目标 field 被实际使用、无安全违规；
- `rejected`：候选未改善或导致退化；
- `pending`：基础设施错误或有效 evidence 不足。

## 9. 第一版 MVP 建议

第一版只实现 routing evolution，不同时修改 procedure、planner 或 VLA：

1. 补齐 `indexed/read/applied/effective` summarization；
2. 为 can/basket leaf 定义 execution signature；
3. 生成单字段 activation/alias candidate；
4. 确定性重建 MEMORY index；
5. 使用相同 correction seeds 对跑 S000/S001；
6. 在原双物体 basket case 做 preservation；
7. 比较 retrieval、application、success 和成本指标。

首轮假设：

> 将成功轨迹中 planner 自发完成的“双物体 skill → 单物体任务”泛化写回 activation 和
> MEMORY routing description，能够提高未来任务中正确 leaf 的读取与实际应用概率。

首轮判据：

```text
candidate correct-skill read/application 至少提升一次
AND correction benchmark success 至少比 parent 多 1
AND parent-success preservation 零退化
AND 无新的安全或基础设施错误
```

第二个独立 candidate 再处理 seed 2 的冗余定位：在 procedure 顶部明确 primary
`pi0_doubled` 是第一项 physical action，并规定只有 primary call 失败后才允许 segment、
back-project 和 scripted fallback。activation 与 efficiency 必须分开验证，避免一次 patch
同时改变检索和执行而无法归因。

## 10. 暂不推荐的首版路线

### 每个 episode 后全文重写

成本高、顺序敏感、容易积累矛盾，只适合作为 naive append/rewrite baseline。

### 直接训练 learned router

虽然很可能改善 seed0/1 的问题，但会同时改变 planner stack 和 MEMORY，削弱首轮关于显式
skill evolution 的因果主张。应在文本路由 MVP 后作为独立增强。

### 大规模多候选程序进化

当前瓶颈主要是正确 skill 未被读取，而不是缺少复杂控制程序。先做程序搜索会把大量预算
花在次要问题上。

### 只保存成功轨迹

成功轨迹能扩展 activation，但无法暴露失败边界和 recovery；失败轨迹则必须经过 applied
归因后才能用于修改 procedure，二者用途不同。

### 仅依赖 planner/verifier 自报

视觉或语言判断只能作为诊断 evidence。LIBERO 的 `libero_terminated=true` 仍是唯一任务级
benchmark success。

## 11. 与长期 VLA Distillation 的关系

当前显式 skill evolution 的直接产物是：

- 更稳定的 skill routing；
- 更明确的 procedure application；
- 经物理验证的 failure/recovery；
- 带 skill 来源、版本和质量标签的成功轨迹。

只有候选被接纳为 `S{k+1}` 后，才应使用新 library 重新采样 teacher trajectories。后续
control-step trace 需要保存图像、proprioception、7D action、instruction、action source、
subtask/skill IDs，再导出给外部 VLA 训练。Discovery 中的探索轨迹和 rejected candidate
轨迹不能直接当作高质量 SFT 数据。

因此本阶段的目标不是证明 VLA 已经 self-evolve，而是先建立：

```text
可归因经验
  -> 受约束 skill candidate
  -> 物理 replay admission
  -> 更强且可审计的 Harness teacher
  -> 后续高质量 VLA distillation 数据
```

## 12. 结论

对于 RPent 当前阶段，最合理的 self-evolve skill 方案不是让一个外部 LLM 周期性自由重写
全部 MEMORY，而是：

> 使用结构化 attribution 确定问题层级，以事件触发的 typed semantic gradient 生成有界
> patch；routing 更新尽量由模板与确定性编译完成，procedure/recovery 更新才周期性调用
> curator；所有候选最终通过相同 case 的 correction/preservation physical replay 决定是否
> 进入下一版不可变 skill library。

这一设计能够分别衡量 routing、application 和 physical effectiveness，降低延迟与模型
成本，并把 skill evolution 的收益与 planner/VLA 能力变化解耦，适合作为本项目第一版
self-evolving skill 机制。
