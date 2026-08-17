# Harness VLA Skill Self-Evolution Pipeline（当前实现）

本文描述当前仓库已经实现的 skill evolution 数据流。VLA 权重蒸馏仍是下一阶段，
不在本轮代码中执行。

## 1. 系统组成与所有权

### TaskContext：任务输入，不是 skill

episode 初始化后，runtime 从可观察状态生成 `TaskContext/v1`：完整 task language、
初始 EEF 位置、可观察对象名称、camera 列表和 VLA control mode。它在首次
`search_skills` 前注入 planner，并写入 trace。对象名称只用于语义检索，实际位置
仍必须从当前图像定位。

### G0：不可演化 policy

G0 由 `rpent/evolution/policies/libero.json` 和 runtime guard 执行，不进入 BM25：

- 允许的工具和 `max_chunks <= 25`；
- 环境 termination 是唯一权威任务成功；
- subtask/decision attribution；
- 禁止隐藏 benchmark 信息、越权工具和未分解的整 episode VLA 接管；
- 长接触 OSC 等物理安全边界。

### G1：ToolCapability

toolkit 根据本次实际注册的工具生成 `ToolCapabilityRegistry/v1`。它说明参数、
副作用、可用性和 evidence 字段。phased-VLA 模式包含：

- `pi0_pick`：闭环 grasp phase；
- `pi0_place`：对已持物体执行闭环 place phase；
- `pi0_doubled`：knob/button/短 push 等 contact phase；
- scripted primitive、感知工具和 skill protocol 工具。

G1 是代码能力，不由 curator 修改。

### G2：immutable reference

G2 提供坐标语义、工具结果解释、fixture re-localization 和任务判定知识。可检索、
可附着到 subtask，但不接受 patch，也不承担主要执行 credit。

### G3：patchable heuristic advisor

G3 是跨任务微策略，例如低桌面 grocery 标定、held-object transport、对象顺序、
Pi0 起始位姿和失败恢复。每个 subtask 最多选择两个 advisor。它们可以被单字段
patch，但只有被 `record_decision.applied_skill_ids` 明确引用的动作才产生 procedure/
recovery 使用证据。

### G4：patchable primary procedure

G4 表示语义完整的任务/子任务流程。每个 subtask 最多一个 primary procedure。
当前 seed library 包含旧 MEMORY 的安全 procedure、8 个显式 phased-VLA 迁移版本，
以及 `single-object-basket-placement`。没有适用 G4 时使用 null，轨迹标记为 novel
scope，只能支持新增 procedure，不能把探索错误归给无关 skill。

### Planner、curator 与 VLA

- Execution planner：冻结 Qwen；只负责在线检索、分解、选择工具、验证和恢复。
- Offline curator：同一模型服务的隔离角色；只读取结构化 trace，不控制机器人。
- VLA：冻结 Pi0.5；在当前阶段只作为 G1 primitive，不更新权重。
- Environment：LIBERO；`libero_terminated` 提供权威任务结果。

## 2. 在线 rollout 数据流

```text
LIBERO reset/init
  -> TaskContext/v1
  -> execution planner
  -> search_skills(task/subgoal language)
       -> G4 procedure_candidates
       -> G3 heuristic_advisors
       -> G2 references
  -> begin_subtask(primary_procedure, advisors, references)
  -> view_driver_state / read_image / localization
  -> record_decision(evidence, intended_tool, applied_skill_ids, expected result)
  -> physical tool
       -> scripted primitive OR pi0_pick/pi0_place/pi0_doubled
  -> LIBERO observation + terminated/truncated
  -> TraceEvent/v2 + states.json + images + action video
  -> verify / recover / end_subtask
  -> episode result.json
```

每个 physical primitive 自动继承当前 subtask、G4、G3 和 G2 context。tool call 与
recorded decision 不一致会被 guard 阻止。primitive-local success 只用于诊断；只有
`libero_terminated=true` 会自动结束 subtask/episode 并产生 benchmark success。

## 3. Library 初始化数据流

```text
reviewed resources/libero/memory/*.md
  -> deterministic importer
       -> G0/G1 条目移出检索库
       -> MEMORY.md 低桌面表编译为 G3
       -> feedback 分类为 G2/G3
       -> task patterns 分类为 G4
       -> 旧长 VLA recipe 显式迁移为 phased-VLA G4
  -> immutable libraries/S000*/
       -> manifest.json
       -> policy.json
       -> skills/*.json
       -> quarantine/*.json
```

planner 不读取原始 MEMORY。旧文档只作为 import provenance，实际执行只消费当前
immutable library 中的结构化 artifact，因此不会绕过版本、检索和 attribution。

## 4. Skill evolution cycle

### Discovery

父库在 discovery seeds 0–2 上运行。输出完整 multimodal trace。至少两条有效
rollout 才进入 proposal；planner/infrastructure 错误不当作机器人失败证据。

### Proposal

offline curator 分三步：

1. success analyst 提取真正贡献成功的决策；
2. failure analyst 定位 subtask/primitive 和最小可复用修复；
3. consolidator 先产生 evidence-linked semantic gradient，再生成单字段 patch。

patch 只能新增一个 G3/G4，或修改一个 patchable artifact 的 activation、procedure、
termination、recovery 之一。证据不足时输出 hypothesis，不伪造候选。

### Candidate library

合法 patch 应用到 shadow candidate library。父库保持不可变；candidate 继承全部
未修改 artifact，并只包含一个候选 delta。

### Correction replay

父库和 candidate 使用相同 planner/VLA/settings，在 curator 未见过的 seeds 3–5 上
各运行一次，用于检验候选是否修复目标失败。G3 advisor 和 G4 procedure 都通过
trace 中的 `skill_ids`/`applied_skill_ids` 统计真实 activation 与 causal use。

### Preservation replay

选择三个父库已成功且语义不同的 case，父库与 candidate 对跑，检查候选是否破坏
旧能力。没有三个已验证父库成功 case 时 admission 为 pending，而不是降低标准。

### Admission

候选必须同时满足：

- correction success 比父库至少多 1；
- 三个 parent-success preservation case 零退化；
- candidate skill 至少在 2/3 correction case 激活；
- update patch 的被修改字段至少在 2/3 correction case 被因果使用；
- 无 safety violation。

基础设施错误产生 pending；未通过 gate 产生 rejected；通过后写入新的不可变
`S{k+1}`。accepted、rejected、pending 的 patch 和依据都会保留。

## 5. 主要产物与 lineage

```text
experiment/
├── resolved_config.json                 # task/planner/VLA/library 配置
├── libraries/S000*/                     # immutable parent
├── libraries/S001...                    # 仅 accepted 后生成
├── rollouts/<phase>/<role>/<case>/
│   ├── evolution_trace.jsonl             # attribution 主链
│   ├── states.json                       # primitive 端点状态
│   ├── images*/ + depths*/ + world*/
│   ├── episode.mp4
│   ├── tool_capabilities.json
│   ├── transcript_*.json                 # 含 debug reasoning，不给 curator
│   └── result.json                       # 权威 rollout 标签
└── cycle_001/
    ├── candidate.patch.json
    ├── candidate_library/
    ├── replay_plan.json
    ├── replay_manifest.json
    ├── validation.json
    ├── decisions/{accepted,rejected,pending}/
    └── cycle_result.json
```

核心 lineage 是：

```text
TaskContext
 -> retrieved G2/G3/G4 versions
 -> Subtask
 -> Decision(applied skills)
 -> Tool/VLA action
 -> Observation/evidence
 -> authoritative episode outcome
 -> semantic gradient
 -> patch
 -> parent/candidate replay
 -> admission decision
 -> next immutable library
```

## 6. 尚未实现的下一阶段

当前 trace 仍主要保存 primitive 端点，不足以直接训练连续动作 VLA。后续蒸馏阶段
需要为每个 env control step 保存图像、proprioception、7D action、instruction、
action source、subtask/skill ids，并在 `S{k+1}` 接纳后重新采样 teacher trajectory，
再导出模型无关 NPZ/manifest 给外部 RLinf 训练。不能把 discovery 中的探索/失败
轨迹直接当作高质量 VLA SFT 数据。
