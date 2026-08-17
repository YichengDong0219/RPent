# MEMORY 路由与 Skill 应用演化：三 Seed Case Study TODO

本文记录实验
`logs/skill_evolution/libero_object_t0_baseline_compatible_v2` 中 task 0、seed 0–2 的
case study 结论，并把结论转化为下一阶段可验证的实现与实验 TODO。

当前分支仍遵循 baseline-compatible 原则：execution planner、VLA checkpoint、工具能力和
prompt 主体与 baseline 对齐，parent/candidate replay 中只改变版本化 MEMORY。本文只规划
显式 skill evolution，不包含 VLA 权重蒸馏。

## 1. Case study 事实

三次 rollout 的任务都是：

```text
grab alphabet soup and put it into basket
```

| Seed | MEMORY/skill 读取 | 读取后的主要策略 | 权威结果 |
|---|---|---|---|
| 0 | 读取 `MEMORY.md`、目录和三份 guide；没有读取 leaf skill | 大量 `back_project`，随后 `pi0_pick`、`move_to`、`release`；40 turns 耗尽 | `libero_terminated=false` |
| 1 | 只读取 `MEMORY.md`；没有读取 leaf skill | 多次 `pi0_pick`，随后手工 carry/release；视觉上接近成功但 predicate 未触发 | `libero_terminated=false` |
| 2 | 读取 `MEMORY.md`、`can_and_small_package_basket_placement.md` 和 `tall_item_first_basket_placement.md` | 明确引用 can/basket skill，调用 `pi0_doubled(verbatim task, max_chunks=80)` | 28 chunks 后 `libero_terminated=true` |

seed 2 中存在一条高置信 skill 应用证据链：

```text
read can_and_small_package_basket_placement.md
  -> planner 明确引用该文件及 ZERO localization / pi0_doubled / verbatim prompt
  -> 实际调用 pi0_doubled(prompt=<完整任务语言>, max_chunks=80)
  -> libero_terminated=true
```

`tall_item_first_basket_placement.md` 虽然被读取，但其 tall-item ordering 和两次
`pi0_pick` procedure 没有被执行，因此只能标记为 `read`，不能获得执行 credit。

严格限制：三个 seed 的初始布局不同。上述对比可以生成演化 hypothesis，但不能单独证明
skill 导致成功；最终因果结论必须来自相同 case 的 parent/candidate replay。

## 2. 当前暴露出的核心问题

### 2.1 外层 MEMORY 路由召回不足

现有索引将 can/basket skill 描述为“rigid can and a small package”的双物体 recipe，
而本任务只有一个 alphabet soup can。seed 2 自发完成了“双物体 → 单物体”的语义泛化，
seed 0/1 没有稳定完成该泛化。

### 2.2 读取 MEMORY 不保证继续读取 leaf

当前 prompt 要求 planner 从 `MEMORY.md` 再打开 2–3 个相关 leaf，但 runtime 不验证该
步骤。seed 0 在读取大量 guide 后过早认为 memory workflow 已完成；seed 1 直接进入
感知和动作。对 9B planner 而言，纯自然语言约束的执行稳定性不足。

### 2.3 `skill_read` 不等于 `skill_applied`

当前 passive trace 将 leaf read 作为弱 activation。它无法区分：

- 只读未采用；
- planner 声称采用但动作不匹配；
- 工具、参数和顺序真正匹配 procedure；
- skill 被采用后是否产生有效结果。

### 2.4 Leaf 与全局 workflow 的优先级冲突

can/basket skill 已写明 primary recipe 不需要显式定位，但 seed 2 仍在 primary VLA 前
调用两次不可用的 `segment` 和四次 `back_project`。这说明问题不只是缺少强调语句，
还包括 generic localization workflow 与 task procedure 的优先级不清楚。

### 2.5 当前 proposal 链仍有工程阻塞

本轮 curator 响应解析出现 `NoneType` 异常，却被记录成 `insufficient_evidence`。在该问题
修复前，即使 discovery 已形成有效成功/失败对比，也无法稳定生成候选并保留完整审计证据。

## 3. 目标演化机制

把在线轨迹分为四个连续层级：

```text
indexed -> read -> applied -> effective
```

- `indexed`：skill 出现在本轮 planner 可见的 MEMORY index。
- `read`：planner 实际读取对应版本 leaf。
- `applied`：读取后的工具、参数和动作顺序匹配该 skill 的 execution signature。
- `effective`：应用后产生权威任务成功，或有证据支持的局部物理进展。

对应诊断指标：

```text
retrieval_rate = P(read correct skill | applicable task)
application_rate = P(applied | skill read)
effectiveness = P(authoritative success | skill applied)
```

不同失败层级应触发不同 patch：

| 轨迹模式 | 演化目标 | 允许修改 |
|---|---|---|
| applicable 但未读取 | 路由召回 | `activation`、aliases、routing summary |
| 已读取但未执行 | procedure 显著性/优先级 | default action、条件、execution contract |
| 在新范围应用并成功 | 能力边界积累 | 扩展可观测 activation scope |
| 已应用但出现未覆盖失败 | 执行与恢复 | failure signature、recovery |
| 已应用并成功但前置步骤冗余 | 效率 | 禁止不必要的 pre-actions |
| 已应用后稳定有害 | 适用范围治理 | 收窄 activation 或 quarantine |

## 4. TODO

### P0：先补齐可审计的应用归因

- [ ] 为 leaf skill 定义可选的机器可读 `execution_signature`：primary tool、prompt 来源、
  关键参数、允许的前置动作和禁止的替代路径。
- [ ] 在 rollout summarizer 中，按 `skill_read` 后的工具事件匹配 signature，输出
  `claimed_applied`、`behavior_matched`、`application_confidence` 和 `matched_event_ids`。
- [ ] 将“planner 明确引用 skill”的可见 assistant 文本作为辅助证据；不能仅凭该文本给
  execution credit，也不把隐藏 thinking 提供给 curator。
- [ ] 同时读取多个 skill 时，分别匹配动作 signature；无法唯一归因时标为 `ambiguous`，
  不把 success 同时归给全部已读 skill。
- [ ] 修复 curator 对空 `message.content`、tool-call arguments 和 reasoning/content 变体的
  解析；保存原始 request/response。
- [ ] 将 curator 协议/解析异常标记为 `curator_protocol_error`，不得伪装成
  `insufficient_evidence`。

### P1：实现可演化的 MEMORY 路由层

- [ ] 不允许 curator 自由重写整份 `MEMORY.md`；leaf metadata 是路由语义的唯一来源。
- [ ] 为 leaf 增加结构化 `activation`、`aliases` 和 `routing_summary`。
- [ ] 根据 leaf metadata 确定性生成 `MEMORY.md` 索引项；leaf activation patch 应自动
  生成对应的新 index，而不是形成第二个人工语义 patch。
- [ ] 在 trace 中记录 planner 看见的 `library_id`、index entry 和实际读取的 leaf version。
- [ ] 增加“首个 physical action 前未读取任何 leaf”的诊断事件，但 baseline-compatible
  阶段先只观察，不 hard-block planner。
- [ ] 分别统计 correct-skill index exposure、read rate 和 first-action 前 read rate。

### P2：积累 seed 2 的成功泛化

- [ ] 为 `can_and_small_package_basket_placement` 生成单字段 activation candidate，将范围从
  “双物体 can + package”扩展到“单个或多个 rigid can/package → basket”。
- [ ] 加入可观测 aliases，例如：
  `grab alphabet soup and put it into basket`、
  `pick up the can and place it in the basket`。
- [ ] 自动生成更清晰的 MEMORY routing summary，显式包含 single/multiple、can、basket 和
  full-task VLA 关键词。
- [ ] 不把 `tall_item_first_basket_placement` 作为本次 success contributor，也不根据该
  success 修改它。
- [ ] 使用未进入 curator context 的相同 correction cases 对跑 S000/S001，验证 candidate
  是否提高正确 leaf 的读取率和实际应用率。
- [ ] 使用原双物体 basket 成功 case 做 preservation，防止 activation/procedure 泛化破坏
  原有能力。

### P3：消除 skill 应用前的冗余步骤

- [ ] 单独生成 procedure/efficiency candidate，不能与 activation candidate 合并接纳。
- [ ] 将简短 execution contract 放在 leaf procedure 开头，而不是继续在文末追加历史文本。
- [ ] contract 明确默认第一项 physical action：
  `pi0_doubled(prompt=<verbatim task_language>, max_chunks=80)`。
- [ ] 明确 primary call 前不调用 `segment`、`back_project`、`move_to` 或 `pi0_pick`；只有
  primary call 未完成任务后才进入定位与 fallback。
- [ ] 在全局 workflow 中定义优先级：经过验证的 task procedure 可以覆盖 generic
  localization workflow，但不能覆盖安全边界和权威 outcome contract。
- [ ] 以 tool calls、planner turns、input/output tokens、wall time 和 success 为效率验收指标。

### P4：处理“正确应用后出现新失败”的未来轨迹

- [ ] 只有 `behavior_matched=true` 的失败才允许支持 procedure/recovery patch。
- [ ] 从动作前后图像、primitive result 和环境状态提取可观测 `failure_signature`，不采用
  纯 planner 自报诊断。
- [ ] 基础设施错误、服务未配置和路径错误进入 runtime/capability backlog，不写进任务 skill。
- [ ] 新 recovery 至少由成功/失败对比或两个相同失败模式支持；证据不足只保留 hypothesis。
- [ ] 将历史失败压缩成“条件 → 诊断 → recovery”表，避免无限 append 导致 skill 膨胀。
- [ ] 通过 correction 提升和 parent-success preservation 零退化后才接纳 recovery。

## 5. 第一个最小闭环实验

第一轮只验证路由演化，不同时修改 VLA、planner、procedure 和 recovery：

1. Parent 使用当前 S000。
2. Candidate 只扩展 can/basket leaf 的 activation/aliases；程序自动重建 index。
3. Parent 和 candidate 使用相同 planner、sampling、VLA checkpoint、task seed 和预算。
4. 在 correction seeds 上比较：
   - 是否在首个 physical action 前读取正确 leaf；
   - 是否执行匹配 signature 的 `pi0_doubled`；
   - 是否触发 `libero_terminated=true`。
5. 在原双物体 basket case 上做 preservation。

首轮关键判据：

```text
candidate correct-skill read/application 至少提升一次
AND correction benchmark success 至少比 parent 多 1
AND parent-success preservation 零退化
AND 无新的安全或基础设施错误
```

如果 retrieval 提升但 success 未提升，保留为 rejected/hypothesis，并继续诊断 VLA 执行或
场景难度；不能把 retrieval 指标改善直接宣称为任务能力提升。

## 6. 后续消融与研究主张

建议至少保留以下对照：

- 原始 S000；
- 只演化 index/activation；
- 只演化 leaf procedure；
- activation + procedure 分阶段演化；
- naive append 历史轨迹；
- 完整 parent/candidate replay admission。

本 case study 最适合支撑的近期主张是：

> 显式 MEMORY/skill 演化能够分别改善 Harness 中的 skill routing、procedure application
> 和 failure recovery，并通过物理 replay 将这些改进转化为带因果来源和质量标签的轨迹。

在当前阶段不能声称 VLA 权重已经 self-evolve；只有接纳后的 skill library 重新采样出的
高质量 control-step 轨迹，才会进入后续 VLA distillation 阶段。
