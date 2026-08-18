以下内容可直接复制给下一位 agent 作为交接提示。

---

# RPent Harness VLA Skill Self-Evolution 交接上下文

## 1. 项目与工作区

- 仓库：`/home/dongyicheng/rpent`
- LIBERO checkout：`/home/dongyicheng/LIBERO`
- 当前分支：`LLM`
- 当前提交：`703fb0d V1.1: LLM serves as skills-optimizer`
- 远程分支：`origin/LLM`
- 工作树当前干净
- baseline 参考提交：`5750c8701a9f5ee684795d3a09b361928e53f4c8`

每次工作前必须完整读取：

- `/home/dongyicheng/rpent/AGENTS.md`
- `/home/dongyicheng/rpent/Self-evolving skills in Harness VLA.md`

只做与当前改动相关的局部验证，不默认进行全量测试或哈希检查。

## 2. 用户的核心研究主张

用户的目标不是限制 VLA，而是：

1. 保留 Harness 相比 standalone VLA 更强的 planner、tools、memory、recovery 和长程 VLA 调用能力。
2. 从机器人执行轨迹中演化显式 skill：
   \[
   S_k \rightarrow S_{k+1}
   \]
3. 使用更新后的 Harness 重新采集高质量轨迹。
4. 最终将 Harness+skill 产生的更强能力蒸馏回 VLA：
   \[
   \pi_k \rightarrow \pi_{k+1}
   \]
5. 形成 skill 与 VLA 交替更新的双重 self-evolve。

当前阶段只实现显式 skill evolution，不实现 distillation。不要把后续 VLA 训练描述成已经完成。

## 3. 已废弃的方向

此前做过一版强侵入式 evolution runtime，包括：

- 强制 `search_skills/begin_subtask/record_decision/end_subtask`
- 限制长程 VLA
- 新的 skill 分级 protocol
- 强制 planner attribution

这导致 planner prompt、MEMORY、工具行为与 baseline 差异过大，Qwen3.5-9B 成功率明显下降。

当前共识是：从 baseline-compatible 实现重新出发。

在线执行必须尽量与 baseline 对齐：

- 相同 planner prompt
- 相同工具 schema
- 相同 Pi0.5 checkpoint
- 保留 `pi0_doubled` 长程能力
- 不强制新的 subtask protocol
- evolution 主要作为被动 observer 和离线 optimizer

## 4. 当前实现的整体 pipeline

```text
resources/libero/memory
    ↓ exact snapshot
libraries/S000/rendered_memory
    ↓ baseline planner 正常读取 MEMORY/leaf
baseline-compatible Harness execution
    ↓ passive observer
evolution_trace.jsonl
transcript_*.json
states.json
episode images
    ↓ build-evidence
OptimizerEvidence/v1
    ↓ independent multimodal skill_optimizer
no_patch 或一个 SkillPatch/v2
    ↓
S{next}-candidate
    ↓ paired replay
correction parent vs candidate
preservation parent vs candidate
    ↓ admission
accepted / rejected / pending
    ↓ accepted only
immutable libraries/SNNN
```

每次脚本只运行一个 `cycle_NNN`。再次启动时自动选择编号最大的 accepted `SNNN` 作为 parent。

Rejected candidate、旧 cycle 和旧 library 不覆盖。

## 5. 当前实现的关键文件

### Evidence 构建

`/home/dongyicheng/rpent/rpent/evolution/evidence.py`

从以下结构化产物提取 `OptimizerEvidence/v1`：

- `evolution_trace.jsonl`
- `transcript_*.json`
- `states.json`
- episode 图片

不解析 `console.log`。

主要字段：

- identity
- authoritative outcome
- MEMORY/leaf read 顺序
- leaf 是否在首个 physical action 前读取
- assistant 可见文本
- skill 显式引用
- transcript tool-use 与 trace event 对齐
- physical actions 和参数
- primitive diagnostics
- EEF/gripper state delta
- path/service/tool errors
- repeated calls
- cost/token/VLA chunks
- visual evidence
- provenance

Thinking block 不进入 optimizer evidence，但仍保留在原始 transcript。

每条 rollout 最多选择六张图片，优先：

1. initial agentview
2. 首个物理动作前图像
3. 首次失败后的 agentview/wrist
4. 最后失败或关键动作后的图像
5. terminated 后图像
6. final agentview

高分辨率不存在时回退普通图片。

### Optimizer

`/home/dongyicheng/rpent/rpent/evolution/optimizer.py`

功能：

- 独立 OpenAI-compatible `/chat/completions`
- 多模态图片转 base64 `image_url`
- `temperature=0`
- JSON response format
- 一次 schema repair retry
- API key/base64 不写入 request manifest
- 完整响应写入 `raw_response.json`

错误分类：

- HTTP/timeout/API：`optimizer_infrastructure_error`
- 两次 schema 错误：`optimizer_protocol_error`
- 静态 patch 违规：`invalid_patch`
- 合法 `no_patch`：正常结束，不 replay

### Optimizer 规则 skill

`/home/dongyicheng/rpent/scripts/skill_evolution/skill_optimizer/SKILL.md`

核心原则：

- 只承认 `libero_terminated=true`
- 区分 indexed/read/claimed-use/behavior/outcome
- 未读 leaf 的失败不能修改该 leaf procedure
- 基础设施和路径错误不得写入任务 skill
- 同时读多个 skill 时只给行为匹配者 credit
- 图片不能提供隐藏 GT 或 benchmark predicate
- 每轮只能修改一个 MEMORY bullet 或一个 leaf snippet
- 禁止全文重写、限制长程 VLA、删除 baseline 能力、增加未知工具
- 证据不足返回 `no_patch`

该 skill 已通过 skill-creator 的 `quick_validate.py`。

### Schema

`/home/dongyicheng/rpent/rpent/evolution/schemas.py`

包含：

- `SkillPatch/v2`
- `SkillOptimizationDecision/v1`
- evidence references

Patch 字段：

- `routing`
- `activation`
- `procedure`
- `termination`
- `recovery`

### Cycle runner

`/home/dongyicheng/rpent/scripts/skill_evolution/run_cycle.py`

实现：

- latest accepted `SNNN` 选择
- 新建不复用的 `cycle_NNN`
- discovery evidence
- optimizer 调用
- candidate materialization
- correction/preservation paired replay
- admission
- accepted 后生成下一个 `libraries/SNNN`
- resolved config 中 API key 会被 redacted

### 一键脚本

`/home/dongyicheng/rpent/scripts/skill_evolution/run_baseline_compatible_cycle.sh`

当前默认：

```text
EXPERIMENT_NAME=libero_object_t0_natural_language_optimizer_mvp_v1
LIBERO_CHECKOUT=/home/dongyicheng/LIBERO
LIBERO_TYPE=pro

EVAL_TASKS=libero_object_lan:0
DISCOVERY_SEEDS=0,1,2
CORRECTION_SEEDS=3,4,5

Planner:
QWEN_BASE_URL=http://114.212.227.193:8000/v1
PLANNER_MODEL=qwen-vl:Qwen3.5-9B

Optimizer:
SKILL_OPTIMIZER_BASE_URL=http://127.0.0.1:8001/v1
SKILL_OPTIMIZER_MODEL=Qwen3.6-27B

Pi0.5:
checkpoint=/home/dongyicheng/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
GPU=3
endpoint=http://127.0.0.1:18081
```

运行：

```bash
cd /home/dongyicheng/rpent
bash scripts/skill_evolution/run_baseline_compatible_cycle.sh
```

脚本启动顺序：

1. 检查 planner Qwen API，包括 tool calling。
2. 检查 optimizer models、图片输入和 JSON 输出，不要求 tool calling。
3. 启动共享 Pi0.5 RPC。
4. 开始新的 evolution cycle。

运行前必须确认本地 `127.0.0.1:8001` 确实部署了 served name 为 `Qwen3.6-27B` 的多模态模型，否则会在 optimizer health check 阶段退出。

## 6. Patch 静态约束

### MEMORY routing patch

只允许：

- target=`MEMORY.md`
- field=`routing`
- `old_text` 是一条完整、唯一 bullet
- bullet 必须位于 `## Reusable manipulation patterns`
- bullet 必须链接到 `<target_skill_id>.md`

禁止修改：

- calibration
- Core operating rules
- 其他全局规则或标题

### Leaf patch

只允许：

- target=`<target_skill_id>.md`
- target leaf 至少在一个有效 discovery rollout 中实际读取
- field 为 activation/procedure/termination/recovery
- exact unique snippet replacement

预算：

```text
MAX_PATCH_LINES=24
MAX_PATCH_NEW_CHARS=2000
MAX_PATCH_GROWTH_CHARS=1000
```

Patch 至少引用两个 discovery run ID。引用的 event/message/image ID 必须存在。

## 7. Admission gate

Candidate 接纳要求：

- correction candidate success 数至少比 parent 增加 1
- parent-success preservation case 零退化
- candidate correction 中目标 leaf 至少被读取 2 次
- 所有 replay 无安全违规
- 任一 replay 基础设施或 agent error → pending

对于 MEMORY routing patch，activation 按 `target_skill_id` 对应 leaf 的真实读取计算，不把 MEMORY 本身当作 skill。

Token、耗时和工具数只用于诊断，不能在成功率没有提高时接纳效率 patch。

## 8. 关键 case study

已有三 seed 实验：

```text
/home/dongyicheng/rpent/logs/skill_evolution/
libero_object_t0_baseline_compatible_v2/rollouts/discovery/parent/
```

### Seed 0

- benchmark failure
- 没有读取任何 leaf skill
- 直接使用手工定位、`move_to`、`pi0_pick` 和 release
- 轨迹冗长
- 核心问题更像 routing failure，不能据此修改未读 leaf 的 procedure

### Seed 1

- benchmark failure
- 同样没有读取 leaf
- 多次 `pi0_pick` 和手工 transport
- 最终未产生 official termination
- 也主要支持 routing 问题

### Seed 2

成功日志：

```text
/home/dongyicheng/rpent/logs/skill_evolution/
libero_object_t0_baseline_compatible_v2/rollouts/discovery/parent/
libero_object_lan__t000__s000002/attempts/attempt_01/
```

因果链：

```text
读取 MEMORY
→ 读取 tall_item_first_basket_placement
→ 读取 can_and_small_package_basket_placement
→ 显式引用 can/basket skill
→ 将双物体 skill 泛化到单个 alphabet soup
→ pi0_doubled(
      prompt="grab alphabet soup and put it into basket",
      max_chunks=80
  )
→ chunks_used=28
→ libero_terminated=true
```

这说明第一轮最合理候选是：

- 修改 MEMORY 中 `can_and_small_package_basket_placement` 的索引描述
- 明确它同样适用于单个 can/alphabet soup → basket
- 使 seed0/1 更容易在物理动作前读取正确 leaf

第一轮不要同时修改 routing 和 leaf procedure。Routing patch 接纳后，下一轮才能独立考虑 procedure/recovery patch。

## 9. 已完成验证

局部测试：

```text
tests/test_baseline_compatible_evolution.py
7 passed
```

已验证：

- schema
- exact snapshot/patch
- terminated/truncated 分离
- admission accept/reject/pending
- thinking 排除、visible text 保留
- MEMORY routing patch 保护
- 多模态 base64 request
- API key/base64 不落 manifest
- 一次 repair retry
- version-chain selection

也用真实 seed0/1/2 artifact 只读生成过 evidence：

- seed0：无 leaf，首个物理动作前未读 leaf
- seed1：无 leaf，首个物理动作前未读 leaf
- seed2：完整恢复正确 leaf、明确引用、`pi0_doubled` 和 terminated

尚未完成真实 optimizer+LIBERO 全闭环 smoke。

## 10. 下一位 agent 的优先任务

1. 确认 `Qwen3.6-27B` optimizer 服务在 `127.0.0.1:8001/v1` 可用。
2. 运行一轮 task0。
3. 检查：
   - `cycle_NNN/optimizer/evidence.json`
   - `image_manifest.json`
   - `request_manifest.json`
   - `raw_response.json`
   - `decision.json`
4. 如果 optimizer 返回 patch：
   - 检查是否只修改 can/basket 的 MEMORY routing bullet
   - 检查 evidence citation 是否真实
   - 观察 correction seeds 3–5 的 leaf read 时机和 success
   - 观察 preservation 是否退化
5. 如果报错，不要先放宽 patch contract；先区分：
   - optimizer API兼容问题
   - schema输出问题
   - evidence缺失
   - patch静态违规
   - robot replay基础设施问题
6. 根据第一轮真实结果再决定：
   - 改 optimizer prompt
   - 改 evidence 压缩方式
   - 改 patch validator
   - 或进入下一轮 leaf procedure/recovery evolution

## 11. 尚未实现

- VLA control-step 数据采集与蒸馏
- VLA SFT/RLinf exporter
- 视觉状态到具体 skill step 的强因果归因
- 自动选择 baseline-proven preservation cases
- 多候选 evolutionary search
- optimizer curator 的训练
- 跨 task consolidation
- statistical significance evaluation

不要把这些模块描述为已完成。