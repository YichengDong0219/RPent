# Self\-evolving skills in Harness VLA

# Self\-evolving skills

### ASPIRE：目前与机器人 Harness 最接近

[ASPIRE](https://arxiv.org/abs/2607.00272) 的关键不是“让 LLM 写代码”，而是它建立了一个适合机器人 self\-evolution 的信息闭环：

- 每次 perception、planning、grasp、control primitive 调用都记录输入、输出、返回状态和局部视觉证据；

- agent 根据细粒度 multimodal trace 定位失败 primitive；

- 生成多个 repair program；

- 在 debug seeds 上搜索，在独立 validation seeds 上验证；

- 将通过验证的局部修复抽象成可复用 skill；

- 后续任务通过 skill library 复用这些修复知识。

实验事实：ASPIRE 的 robot execution engine 将宏观平均成功率从 14% 提升到 62%，加入 evolutionary search 后达到 72%；其 skill library 还表现出跨任务和初步 sim\-to\-real 迁移。[论文](https://arxiv.org/html/2607.00272)

优点：

- 最贴合物理机器人和 Harness；

- 强调 per\-primitive trace，而不是只看 episode success；

- skill 来源是“经过重新执行验证的修复”；

- 支持 recovery、grasp、navigation、perception prompt 等异质 skill。

缺点：

- skill 主要是供 coding agent 使用的程序或文字修复；

- 没有把知识蒸馏回 VLA 权重；

- 默认允许生成新的控制程序，而当前 HarnessVLA 更接近固定 primitive 集合；

- 对 skill 自身的版本比较和 preservation gate 还不够严格。

综合判断：**它应当成为你的 trace infrastructure 和机器人修复搜索基础，但不能直接照搬其动态代码生成。**

---

### Uni\-Skill：最适合解决 skill coverage gap

[Uni\-Skill](https://arxiv.org/abs/2603.02623) 让 planner 先判断已有 skill 是否足够；不足时生成新的 skill 描述，再从大规模机器人视频构成的分层 SkillFolder 中检索演示，实现新 skill。

其 SkillFolder 从约 350 小时 DROID 视频中提取超过 10,000 个 skill segment，组织为 VerbNet 类别、动词实例、skill 描述、具体示例四层结构。论文报告，在超出基础 skill 集的 RLBench 任务上，平均成功率为 0\.41，而 MOKA 为 0\.10。[论文方法与结果](https://arxiv.org/html/2603.02623)

优点：

- 非常适合发现“当前 library 缺了什么”；

- 分层 skill taxonomy 对检索和 merge/split 很有价值；

- 能利用离线机器人数据补足覆盖范围。

缺点：

- 更接近“扩充 skill repository”，而不是迭代修正已有 skill；

- 依赖外部视频、轨迹对齐和 waypoint transfer；

- 对失败驱动的在线 self\-evolution 和防退化验证较弱。

综合判断：**适合作为 cold start 和新 skill discovery，不适合作为主要的在线更新算法。**

---

### SkillsVote：最适合解决你的 skill credit assignment

[SkillsVote](https://arxiv.org/abs/2605.18401) 指出，完整 trajectory 太粗、单个 tool call 又太碎，因此采用 **subtask\-level attribution**：

> 一个 subtask 是最小的语义完整单元，拥有一个独立目标、一个主要评价信号，并且最多关联一个 skill context。
> 
> 

它进一步区分：

- 结果是否由 skill\-guided execution 带来；

- 是否来自 agent 自己的探索；

- 是否是看到错误 skill 后进行的修正；

- 失败是否其实由环境或 evaluator 引起；

- 哪些知识是真正可复用的 delta。

只有“成功、证据充分、包含可复用探索”的 subtask 才能直接触发 skill evolution；失败主要作为诊断证据。[归因方法](https://arxiv.org/html/2605.18401)

优点：

- 正好弥补“按 skill 聚类轨迹”的归因缺陷；

- 能防止将 planner、自主探索或环境随机性的功劳错误归给 skill；

- 与 Harness 的 tool\-call trace 天然兼容；

- 支持 procedure、precondition、recovery pattern 等多种更新。

缺点：

- 当前主要在软件 agent 上验证；

- 失败不能直接触发更新的策略较保守；机器人中某些明确失败其实非常有价值；

- 本身不提供特别强的 candidate verification。

综合判断：**这是你方法中最应该吸收的模块之一。**

---

### Skill\-Pro：最适合定义细粒度 skill 的结构

[Skill\-Pro](https://arxiv.org/abs/2602.01869) 将一个 skill 定义为：

\\\[ \\omega=\\langle \\mathcal I\_\\omega,\\pi\_\\omega,\\beta\_\\omega\\rangle \\\]

分别表示：

- activation condition；

- execution procedure；

- termination condition。

它只分析该 skill 实际控制的 trajectory segment，再分别产生针对 activation、execution、termination 的 semantic gradient；多个轨迹的更新方向经过聚合，过滤单例和互相冲突的建议。[具体定义](https://arxiv.org/html/2602.01869)

它还利用 PPO\-style gate 验证候选 skill，并通过 online score 删除低贡献或重复 skill。去掉 semantic gradient、PPO gate 或 score pruning 都会显著降低性能和复用率。[消融实验](https://arxiv.org/html/2602.01869)

优点：

- 与你关心的 planner 调用、执行、terminate 完全对齐；

- 天然支持“按 skill 控制的 segment 聚类”；

- skill update 可以明确归因到三个字段之一；

- 具有 pool pruning，能控制 library 膨胀。

缺点：

- PPO gate 是基于已有轨迹的近似评价，不等于机器人重新执行；

- skill 结构仍缺 recovery、安全约束和版本 provenance；

- 在 LLM agent benchmark 上验证，不是物理控制。

综合判断：**适合作为你的 skill schema 基础，但 verification 应替换为真实环境 replay。**

---

### AutoRefine：最适合决定“什么知识应该写到哪里”

[AutoRefine](https://arxiv.org/abs/2601.22758) 不预先假设所有经验都必须成为 skill。它先比较成功与失败轨迹，形成与证据绑定的 intervention specification，再依照 ownership boundary，选择第一个能完整拥有相关观察、状态、决策和完成条件的 artifact：

- Rule；

- Skill；

- bounded Subagent。

候选必须同时通过：

- type\-specific contract gate；

- correction replay；

- preservation replay。

任何 preservation case 退化都会拒绝更新。去掉 boundary closure 或 replay gate 分别造成 15\.00 和 16\.11 个百分点下降。[论文证据](https://arxiv.org/html/2601.22758)

优点：

- 能避免“所有知识都硬塞进 skill”；

- preservation cases 非常适合机器人防退化；

- 对 termination、跨步骤状态和工具所有权的判断很有帮助。

缺点：

- artifact 类型较多，会增加实现复杂度；

- 仍以软件 agent 为主；

- 其 replay guarantee 只覆盖测试过的 case，不是全局保证。

综合判断：**你的第一版可以只保留 Rule/Skill 两类，但一定要吸收 ownership closure 和 correction/preservation gate。**

---

### SkillOpt 与 Trace2Skill：适合候选编辑

[SkillOpt](https://arxiv.org/abs/2605.23904) 强调：

- batch rollout；

- 反思后执行有限的 add/delete/replace edit；

- held\-out validation；

- 退化更新拒绝；

- 保存 rejected edits，避免重复探索错误方向。

[Trace2Skill](https://arxiv.org/abs/2603.25158) 则更擅长：

- 每条轨迹独立分析；

- success analyst 与 failure analyst 分开；

- 并行提出局部 patch；

- 再进行分层 consolidation。

优点：

- SkillOpt 的 bounded edit 和 validation 非常稳；

- Trace2Skill 适合从大量异质轨迹产生候选；

- 两者组合后可实现“发散生成、保守接纳”。

缺点：

- SkillOpt 偏向优化单个较大 skill document；

- Trace2Skill 默认接受机制偏弱；

- 都缺乏机器人级 per\-primitive attribution。

综合判断：**Trace2Skill 用于 proposal，SkillOpt 用于 edit discipline，AutoRefine 用于 admission。**

---

### SkillOS：如果以后想训练 curator，这是最合适的方向

[SkillOS](https://arxiv.org/abs/2605.06614) 保持 executor 冻结，单独训练 skill curator 对 SkillRepo 执行 insert/update/delete。其关键设计是把相关任务组成 task stream：

- 前面的任务产生轨迹并更新 repository；

- 后面的相关任务检验这次更新是否真正有长期价值；

- curator 的奖励包含未来任务成功率、内容质量、合法操作和 repository 简洁性。

消融显示，随机任务序列会使成功率从 61\.2 降到 57\.3；训练过程中 curator 会从主要执行 insert，逐渐转向 update 和 consolidation。[方法与消融](https://arxiv.org/html/2605.06614)

优点：

- curator 学的是 skill 的长期 downstream utility；

- executor 可以保持冻结；

- insert/update/delete 比固定 prompt heuristic 更灵活。

缺点：

- 等于重新引入一个需要训练的 policy；

- RL 成本高，而且机器人 reward 稀疏；

- 当前 skill 是扁平 Markdown，尚不支持多文件、可执行和层级 skill；

- 会增加与你已经放弃的 handoff policy 类似的训练复杂度。

综合判断：**不建议第一阶段使用。等你积累大量 skill evolution history 后，再把已有编辑日志用于训练 curator。**

---

### SkillRL / Skill1：最接近“skill 与模型共同提升”

[SkillRL](https://arxiv.org/abs/2602.08234) 从成功和失败经验构建分层 SkillBank，先用 skill\-augmented trajectories 做 SFT，再通过 GRPO 更新 policy；每个 validation epoch 又分析失败轨迹，补充或修正 skill，形成递归共演化。[算法](https://arxiv.org/html/2602.08234)

[Skill1](https://arxiv.org/abs/2605.06130) 更进一步，用同一个任务结果信号同时训练 skill selection、utilization 和 distillation，并将 reward 的低频趋势分给 skill utility，高频偏差分给新 skill 的边际贡献。[论文](https://arxiv.org/html/2605.06130)

优点：

- 是你“双循环”最直接的算法先例；

- 已经包含 skill\-augmented SFT/RL；

- 明确意识到模型能力变化后 skill 也必须继续演化。

缺点：

- 面向 LLM agent，不处理连续动作和多模态物理失败；

- 使用 episode\-level reward，skill credit 仍偏粗；

- 同时更新 skill 与 policy 容易形成非平稳闭环；

- 很难区分 skill 提升、policy 提升和 retrieval 提升。

综合判断：**可以作为论文定位的相关工作，但不建议直接用联合 RL；你的交替冻结方案更容易分析。**

---

### MERA：与你的 Skill→模型蒸馏顺序最接近

最新的 [MERA](https://arxiv.org/abs/2608.10333) 每轮：

1. 重放 small model 的失败调用；

2. 获得 verifier\-backed teacher demonstration；

3. 更新 SkillBook；

4. 用当前 SkillBook 指导的数据训练 LoRA；

5. 最后重新训练 router；

6. 对 SkillBook、adapter 和 router 的联合状态做 replay admission。

它明确采用：

\\\[ \\text\{Skill\}\\rightarrow\\text\{Model\}\\rightarrow\\text\{Router\} \\\]

因为如果 model 和 SkillBook 同时更新，model 会学到过期 skill。其多轮训练将 1\.5B 模型在 HumanEval\+MBPP 上从 28\.7% 提升到 49\.7%，但主要证据仍来自代码任务。[论文结果](https://arxiv.org/html/2608.10333)

优点：

- 几乎直接验证了你提出的“先优化显式知识，再蒸馏到参数”；

- 强调 shared trace、版本依赖和 joint replay；

- 清楚区分 standalone model improvement 与系统 fallback improvement。

缺点：

- SkillBook 粒度较粗，主要按 prompt signature 聚合；

- 软件 verifier 比机器人 verifier 容易；

- 没有细粒度 skill activation/termination；

- 论文很新，机器人和长时序证据不足。

综合判断：**它是你需要重点讨论的最近邻工作；你的创新不能只写成“skill 生成高质量轨迹，再蒸馏模型”。**

---

### Skill\-to\-LoRA：证明 skill 行为可以参数化

[Skill\-to\-LoRA](https://arxiv.org/abs/2606.16769) 先使用完整 skill document 生成 skill\-guided demonstrations，再训练 skill\-specific LoRA；推理时不再注入完整 skill 文本。错误加载其他 LoRA 或将多个 skill 混为一个 adapter 都会降低性能。[论文](https://arxiv.org/html/2606.16769)

优点：

- 直接支持“将显式 skill 行为蒸馏进参数”；

- 能减少推理上下文；

- 证明不同 skill 的行为对齐很重要。

缺点：

- 每个 skill 一个 adapter 会造成库和切换复杂度；

- 不适合你希望得到单一增强 VLA 的目标；

- 当前只在代码 agent 上验证。

综合判断：**你的主线应蒸馏进统一 VLA，而不是为每个 skill 维护 LoRA；但它可以作为 skill\-conditioned distillation 的理论先例。**

---

### SIL\-C：解决 VLA 更新后 skill 接口漂移

[SIL\-C](https://arxiv.org/abs/2509.20612) 研究 skill 更新后，旧的 downstream policy 是否还能正确调用它。它区分：

- Forward Skill Compatibility：未来 policy 能利用旧 skill；

- Backward Skill Compatibility：旧 policy 能利用更新后的 skill。

其方法是在 subtask space 和 skill space 之间建立基于 trajectory distribution similarity 的 lazy interface，发现不匹配时重新 hooking 到合适 skill。[论文方法](https://arxiv.org/html/2509.20612)

优点：

- 点出了你双循环里很容易忽视的问题：VLA 更新会改变 skill 的实际执行分布；

- 可以避免 skill name/interface 不变但行为语义已经漂移；

- 提供了显式 backward/forward compatibility 指标。

缺点：

- 假设的是 latent skill decoder 和高层 policy；

- 当前 Harness 的 skill 主要指导 planner，不完全是低层 policy option；

- 实现其完整 Gaussian prototype interface 可能过重。

综合判断：**第一版不用完整实现 SIL\-C，但每轮 VLA 更新后必须做 compatibility recalibration。**

# Pipeline

## Stage 0：初始化

系统由四部分组成：

1. 预训练 VLA；

2. frozen planner；

3. Harness，包括工具、primitive、反馈和执行监控；

4. 初始 skill library。

这里的 skill 不直接充当低层控制器，而是增强 Harness 的：

- 任务规划；

- subgoal decomposition；

- primitive/tool selection；

- 参数和约束选择；

- 失败恢复；

- 终止判断。

因此 Harness 整体是一个能力强于原始 VLA 的 teacher system。

---

## Stage 1：Skill\-Augmented Execution

对于每个任务，planner 检索适用 skill，并通过 Harness 在 VLA、解析式 primitive 和工具之间组织执行。

这一阶段的输出不是只有成功/失败，而是完整的系统轨迹：

\\\[ \\tau= \(o\_t,\\text\{subgoal\}\_t,\\text\{skill\}\_t, \\text\{tool\}\_t,a\_t,f\_t\)\_\{t=1\}^\{T\} \\\]

其中 $f_t$ 是环境反馈、工具返回值或执行状态。

这使后续能够看到：

- planner 如何分解任务；

- 何时调用 VLA；

- 何时使用其他 primitive；

- 何时恢复或重新规划；

- 任务最终为何成功或失败。

---

## Stage 2：显式知识积累

系统从一批轨迹中提取可复用经验，使：

\\\[ S\_k\\rightarrow S\_\{k\+1\} \\\]

这里不限定 skill evolution 的具体方法，只规定其系统功能：

- 将多次交互中的知识压缩到 skill library；

- 让下一轮 Harness 不必重新发现同样的策略；

- 提升规划、工具使用、恢复和终止质量；

- 保留跨任务可复用的经验。

这个阶段对应一个较快的非参数学习循环：不更新 planner 或 VLA 权重，只改变系统的外部知识状态。

---

## Stage 3：更新后的 Harness 作为 Teacher 重新采样

这是一个必要但容易被忽略的步骤。

不应该直接把用于更新 skill 的旧轨迹全部蒸馏给 VLA，而应该使用 $S_{k+1}$ 再运行一次 Harness：

\\\[ D^\{teacher\}\_\{k\+1\} = \\operatorname\{Rollout\}\(H,P,S\_\{k\+1\},\\pi\_k\) \\\]

因为真正想蒸馏的是“改进后的系统行为”，而不是包含旧 skill 缺陷和探索噪声的原始轨迹。

此时 teacher 的能力来源包括：

- frozen planner 的任务推理；

- skill library 的跨轨迹经验；

- Harness 的工具和 primitive；

- 环境反馈和闭环恢复；

- VLA 原有的 visuomotor 能力。

---

## Stage 4：高质量轨迹构建

Teacher trajectories 被转成 VLA 可学习的数据。

这一步在概念上需要完成三件事：

1. 识别真正高质量或有训练价值的轨迹；

2. 将不同 primitive 产生的行为统一到 VLA action space；

3. 构造 VLA 的输入—输出训练样本。

训练数据可以包含：

- 完整任务指令；

- 图像和 proprioception；

- subgoal 或阶段标签；

- 最终执行动作；

- 成功、进度、恢复或终止标签。

但这些辅助信息不一定都要在 VLA 推理时提供，可以只作为训练期监督。

---

## Stage 5：VLA 参数内化

使用筛选后的 teacher trajectories 更新 VLA：

\\\[ \\pi\_k\\rightarrow\\pi\_\{k\+1\} \\\]

这一步的目标有两个层次。

### Harness\-compatible improvement

让 VLA 更好地执行 Harness 给出的 subgoal：

- 更精确；

- 更稳定；

- 更适应 Harness 的控制边界；

- 更少需要其他 primitive 接管；

- 更容易被 planner 正确调用。

### Standalone improvement

让 VLA 在没有 Harness 和 skill library 时，也能从完整任务指令直接完成更多任务。

这两个目标必须分别评测，因为：

> “在 Harness 中更好用”不自动等价于“独立 VLA 能力更强”。
> 
> 

---

## Stage 6：重新部署并进入下一轮

更新后的 $\pi_{k+1}$ 被放回 Harness，与 $S_{k+1}$ 共同运行。

由于 VLA 能力已经变化，它会产生新的状态分布和失败模式，进而给 skill library 提供新的经验：

\\\[ \(S\_k,\\pi\_k\) \\rightarrow \(S\_\{k\+1\},\\pi\_\{k\+1\}\) \\rightarrow \(S\_\{k\+2\},\\pi\_\{k\+2\}\) \\\]

因此不是单向的“skill 教 VLA”，而是：

- 更好的 skill 产生更好的训练数据；

- 更好的 VLA 扩展 Harness 能够解决的任务和状态；

- 新状态和新失败进一步推动 skill evolution。

# Related work

## ASPIRE：Skill\-enhanced robot teacher

[ASPIRE](https://arxiv.org/abs/2607.00272) 使用机器人执行轨迹发现和积累可复用修复知识，使 coding agent 随经验增长而更擅长编写机器人程序。

它对应你的：

\\\[ D\_k\\rightarrow S\_\{k\+1\}\\rightarrow \\text\{better Harness execution\} \\\]

作者主张：细粒度 multimodal execution traces、持续扩展的 skill library 和 evolutionary program search 可以构成机器人 continual learning loop。

缺失部分：

- 没有把 skill\-enhanced trajectories 蒸馏进 VLA；

- 改进停留在外部程序和 skill library；

- end\-to\-end policy 不随之升级。

因此，ASPIRE 可以视为你 pipeline 的“左半部分”。

---

## RLDG：最接近 Harness teacher → VLA distillation

[RLDG](https://arxiv.org/abs/2412.09858) 先训练任务专用的 RL specialist policy，让 specialist 产生高质量轨迹，再使用这些轨迹微调 OpenVLA 或 Octo：

\\\[ \\text\{Specialist RL\} \\rightarrow \\text\{High\-quality trajectories\} \\rightarrow \\text\{Generalist VLA\} \\\]

实验事实：

- OpenVLA 在 FMB insertion 和 connector insertion 上分别比人类演示微调高 33 和 23 个百分点；

- VGA insertion 达到 100% 时，RL 数据只需 45 条，而人类演示需要 300 条；

- 长时序 assembly 中，只在精密 insertion bottleneck 使用 RL 数据，OpenVLA 从 12/20 提升到 20/20；

- 蒸馏后的 generalist 在未见对象上比产生数据的 specialist RL policy 泛化更好。[结果](https://arxiv.org/html/2412.09858)

它和你的区别是：

- RLDG 的 teacher 是每个任务单独训练的 RL policy；

- 你的 teacher 是共享 Planner、Skill、Tools 和 VLA 的 Harness system；

- RLDG 没有跨轨迹演化的显式知识库；

- 你的 teacher 能从过去任务持续积累知识，并可能覆盖规划、恢复和工具调用。

综合判断：**RLDG 是你的 VLA 蒸馏部分最重要的直接先例。**

---

## RoboCat：self\-generated data → generalist retraining

[RoboCat](https://arxiv.org/abs/2306.11706) 的循环是：

1. 用少量演示把 generalist 适配成 task specialist；

2. task specialist 自主生成更多轨迹；

3. 将这些轨迹加入总训练集；

4. 重新训练新的 generalist。

它已经证明：

\\\[ \\text\{specialized agent\} \\rightarrow \\text\{self\-generated data\} \\rightarrow \\text\{stronger generalist\} \\\]

实验事实：加入 self\-generated data 的模型在四个隔离测试任务上都优于只加入少量原始演示的模型；论文展示了一轮完整 self\-improvement。[论文第 2\.1\.2、5\.3 节](https://arxiv.org/html/2306.11706)

与当前想法的区别：

- RoboCat 的知识只保存在模型参数和轨迹数据中；

- 没有持续存在的显式 skill memory；

- 生成数据的 specialist 仍需单独 fine\-tune；

- 没有 Harness 的工具、planner 和恢复能力。

---

## π\*0\.6 与 LWD：部署—学习—再部署的数据飞轮

[\\\(\\pi^\{\*\}\_\{0\.6\}\\\)](https://arxiv.org/abs/2511.14759) 使用自主 rollout、成功/失败和人工 correction 进行多轮 VLA post\-training。实验中，laundry throughput 累计提升约 50%，box assembly 在两轮后约提升 2 倍。[迭代结果](https://arxiv.org/html/2511.14759)

[Learning While Deploying](https://arxiv.org/abs/2605.00416) 则在 16 台双臂机器人上构建 fleet\-scale data flywheel：

\\\[ \\text\{Deployment\} \\rightarrow \\text\{shared experience\} \\rightarrow \\text\{offline\-to\-online RL\} \\rightarrow \\text\{redeployment\} \\\]

论文在八个真实机器人任务上报告平均 95% 成功率，并强调成功、失败、恢复、partial progress 和人工 intervention 都可以进入 replay buffer。[论文](https://arxiv.org/html/2605.00416)

它们解决的是：

- 如何让 VLA 从部署经验持续更新；

- 如何使用成功和失败数据；

- 如何避免只做 filtered behavior cloning。

但它们没有显式 skill library，所以每一轮知识积累完全依赖参数更新和 replay buffer。

综合判断：**它们可以视为你 pipeline 的“参数学习循环”，而你的 skill library 为其增加了一个更新更快、可解释的外部学习层。**

---

## MimicGen、AutoRT、RoboGen：系统自动产生训练数据

[MimicGen](https://arxiv.org/abs/2310.17596) 从约 200 条人工演示自动生成超过 50,000 条机器人演示，再用这些数据训练闭环 visuomotor policy。它说明“生成轨迹的系统”和“最终部署 policy”可以是两个不同对象。[论文](https://arxiv.org/html/2310.17596)

[AutoRT](https://arxiv.org/abs/2401.12963) 用 VLM 理解场景、LLM 提出任务，再组织机器人自主或半自主收集 77,000 个真实 episode。

[RoboGen](https://arxiv.org/abs/2311.01455) 则采用：

\\\[ \\text\{Propose task\} \\rightarrow \\text\{Generate environment\} \\rightarrow \\text\{Generate supervision\} \\rightarrow \\text\{Learn policy\} \\\]

这些工作支持你的基本假设：

> 复杂的生成/规划系统可以只作为数据引擎存在，最终把行为能力交给更简洁的 policy。
> 
> 

但它们的 data engine 本身不会从执行经验中形成持续演化的显式 skill memory。

---

## SkillRL、Skill1、MERA：显式知识与模型共同更新

[SkillRL](https://arxiv.org/abs/2602.08234) 先从轨迹形成 SkillBank，再使用 skill\-conditioned SFT 和 RL 更新 agent policy；策略遇到新失败后，又继续扩展 SkillBank。

[Skill1](https://arxiv.org/abs/2605.06130) 将 skill selection、skill utilization、skill distillation 与 policy learning 统一到同一个任务结果信号下。

[MERA](https://arxiv.org/abs/2608.10333) 的更新顺序更加接近你的思路：

\\\[ \\text\{SkillBook\} \\rightarrow \\text\{Student model\} \\rightarrow \\text\{Router\} \\\]

它先更新 SkillBook，再用包含当前 SkillBook 行为的数据训练 student，避免 student 学到过期的显式知识。代码任务中，1\.5B student 的直接通过率从 28\.7% 提升到 49\.7%。[论文](https://arxiv.org/html/2608.10333)

这些工作在系统结构上与你最接近，但主要面向语言或软件 agent，没有处理：

- 连续动作；

- 物理接触；

- 多模态状态；

- 异构机器人 primitive；

- 真实机器人数据成本；

- VLA action\-space distillation。

---

## RISE、GRAPE、ForesightFlow：如何利用非完美轨迹

你的 Harness 产生的轨迹不会全部成功，因此 VLA 更新不一定只能使用成功 trajectory。

现有解决方法包括：

- [RISE](https://arxiv.org/abs/2602.11075)：用 world model 产生 imagined rollouts，再在 imagination 中更新 policy；

- [GRAPE](https://arxiv.org/abs/2411.19309)：使用成功和失败 trajectory 构建 preference optimization；

- [ForesightFlow](https://arxiv.org/abs/2606.04968)：同时利用成功、partial completion、recoverable mistake 和 failure，避免 full BC 模仿失败，也避免 filtered BC 丢失失败前的有用片段；

- [PAPO\-VLA](https://arxiv.org/abs/2605.19580)：从密集动作中识别对任务结果更重要的 planning actions，提高这些关键动作在 VLA 优化中的权重。

它们对应你 pipeline 中的：

\\\[ Q\(D^\{teacher\}\) \\rightarrow \\text\{VLA post\-training\} \\\]

也就是说，pipeline 可以保持不变，后续再选择 BC、preference learning、offline RL 或 advantage\-weighted learning 作为具体的 VLA 蒸馏方法。

