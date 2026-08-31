# Group-Contrastive Multimodal Skill OPVD：研究与实现计划

> 工作名称：Group-Contrastive Visual-SEED（GC-Visual-SEED）  
> 核心方法：Externally-Anchored Group-Contrastive On-Policy Visual Distillation  
> 状态：研究方案确定；训练尚未开始  
> 日期：2026-08-24

## 1. 结论与研究定位

本项目研究一种面向多模态 Chain-of-Thought 的 on-policy visual distillation 方法：当前策略先在原图上
采样一组纯文本 reasoning trajectories；固定的外部多模态 Analyzer 对比同一 Group 中成功与失败轨迹，
总结最能解释组内差异的视觉关注区域；Grounding DINO 将语言化关注指令执行为原图上的 bbox，并生成
Crop/Zoom 图像；该放大图与一条不泄露答案的视觉提示共同构成训练期 Teacher 的多模态 privileged skill。

Student 始终只读取原图与问题。Teacher 使用相同策略模型，但额外读取视觉 skill，并对 Student 已采样的
相同 token 重新打分。Crop 引起的 token-level probability shift 构成 OPVD 信号，并与 GRPO 联合训练。

该方案的核心不是让外部 Analyzer 直接给出答案，而是让它完成组内对比式视觉归因：

> 成功轨迹共同看对了什么，失败轨迹共同漏看、错看或被什么干扰；哪一个最小视觉区域最能解释这种差异？

Crop/Zoom 本身被视为多模态 hindsight skill，而不是普通的数据增强或部署时工具。

## 2. 立项依据

已有无训练 probe 已确认：高质量局部放大图能够给错误 Student rollout 带来强且可蒸馏的 Teacher 信号。
在当前 40 样本 dense probe 的错误 rollout 上：

| Teacher 证据 | `R_1` 前 gold 概率相对 dense blank | 逐 step 平均 gold 概率变化 |
|---|---:|---:|
| Random crop | +1.76 pp | +1.66 pp |
| CoFFT-stable crop | +1.62 pp | +0.34 pp |
| **Oracle crop** | **+14.31 pp** | **+7.92 pp** |

详见 [`phase1_dense_interleaved_probe_report.md`](phase1_dense_interleaved_probe_report.md)。这说明当前主要
瓶颈不是 Crop Teacher 是否存在信号，而是能否在不使用 V*Bench GT bbox 的情况下稳定得到接近 oracle
的紧致 Crop/Zoom。

原 attention/sliding-window 方案虽然好于随机定位，但窗口仍过大，且相对注意力存在小分母伪峰。新方案
将“理解应该看哪里”和“执行空间定位”拆开：

1. 外部 Analyzer 根据完整 on-policy Group 生成视觉归因与 grounding program；
2. Grounding DINO 只负责开放词汇定位；
3. 确定性 Crop/Zoom executor 从原始高分辨率图像生成 Teacher 证据。

## 3. 与 SEED 的关系

参考：

- [SEED 论文](https://arxiv.org/abs/2607.14777)
- [SEED HTML 全文](https://arxiv.org/html/2607.14777)
- [SEED 官方代码](https://github.com/jinyangwu/SEED)

### 3.1 保留的 SEED 核心

本方案保留以下训练结构：

1. 轨迹由当前冻结策略快照 on-policy 采样；
2. hindsight 信息只在完整 rollout 结束后构造；
3. Teacher 与 Student 对同一批 sampled tokens 做 paired contextual re-scoring；
4. Teacher 只比 Student 多看到训练期 privileged skill；
5. Teacher log-prob 与 gate stop-gradient，梯度只进入普通 Student 分支；
6. token-level OPVD 与 group-relative RL 联合优化；
7. 部署时移除 Analyzer、DINO、Crop 和视觉提示。

### 3.2 有意修改的部分

| SEED | GC-Visual-SEED |
|---|---|
| 每条 trajectory 独立生成自然语言 skill | 对同题整个 Group 做成功/失败对比，生成共享的多模态 skill |
| RL 阶段使用当前 policy checkpoint 作为 Analyzer | 主方案使用固定的外部强多模态 Analyzer |
| Skill 主要是自然语言 | Skill 是不泄露答案的 focus 文字 + DINO 生成的 Crop/Zoom 图像 |
| Analyzer 与 Actor 参数共同变化 | Analyzer 参数固定，但输入 trajectory Group、输出 focus 和 Crop 每轮变化 |

因此，本方案仍然是 on-policy、trajectory-adaptive 的 visual distillation，但不是严格意义上的
“Analyzer self-evolution”。更准确的定位是 externally anchored on-policy evolution：策略、轨迹分布、
视觉归因输出和 Crop 都随训练改变，外部 Analyzer 的能力本身保持稳定。

## 4. 问题定义

给定原图 \(I_q\) 和问题 \(Q_q\)，在第 \(k\) 个 outer update 开始时冻结当前策略：

\[
\pi_k=\pi_{\theta_k}.
\]

使用 \(\pi_k\) 为同一道题采样 \(N=8\) 条纯文本 reasoning trajectories：

\[
\mathcal G_q^{(k)}
=
\left\{
(\tau_{q}^{(n)},R_{q}^{(n)})
\right\}_{n=1}^{8},
\qquad
\tau_q^{(n)}\sim\pi_k(\cdot\mid I_q,Q_q).
\]

其中 \(R_q^{(n)}\) 是可验证的 trajectory-level outcome。Student rollout 不包含中间裁图，也不调用
Grounding DINO。

固定外部 Analyzer 一次读取完整 Group：

\[
z_q^{G,(k)}
=
A_{\mathrm{ext}}
\left(
I_q,Q_q,\mathcal G_q^{(k)}
\right).
\]

Grounding executor 将语言程序编译成 bbox 和 Crop：

\[
B_q^{(k)}
=
D_{\mathrm{DINO}}
\left(I_q,z_q^{G,(k)}\right),
\]

\[
C_q^{(k)}
=
\operatorname{CropZoom}
\left(I_q,B_q^{(k)},z_q^{G,(k)}\right).
\]

最终组级多模态 hindsight skill 为：

\[
S_q^{G,(k)}
=
\left(
u_q^{(k)},C_q^{(k)}
\right),
\]

其中 \(u_q^{(k)}\) 是 Teacher 可见但不包含答案的 focus instruction。

## 5. Group Analyzer

### 5.1 输入

单图 VQA 场景中，同一 Group 的原图和问题相同，因此只传一次原图和问题，再附加 8 条轨迹：

```text
[Original image]
[Question]

Trajectory 0
- reasoning: ...
- final answer: ...
- reward: 1

Trajectory 1
- reasoning: ...
- final answer: ...
- reward: 0

...

Trajectory 7
- reasoning: ...
- final answer: ...
- reward: 0
```

未来若扩展到多步视觉 Agent，每条 trajectory 应附带与其 observation/action 对应的状态图，而不能只给
最终帧。

### 5.2 分析目标

Analyzer 必须依次完成：

1. 找出成功轨迹中共同使用或共同需要的视觉证据；
2. 找出失败轨迹中的共同 distractor、遗漏区域或错误视觉假设；
3. 提炼能区分成功与失败的最小必要视觉区域；
4. 将语义目标拆成 DINO 可执行的 object query 和确定性 spatial selector；
5. 生成不泄露答案的 Teacher-visible focus instruction；
6. 对 grounding 可执行性和对比归因置信度做自评。

Analyzer 不负责生成 bbox，不负责输出 Crop，也不直接生成 Teacher reasoning。

### 5.3 输出协议

```json
{
  "group_summary": "成功与失败轨迹的关键差异",
  "success_common_evidence": "成功轨迹共同依赖的视觉证据",
  "failure_common_pattern": "失败轨迹共同遗漏或错看的内容",
  "visible_focus_instruction": "Teacher 可见、但不包含答案的操作性视觉提示",
  "grounding_queries": ["DINO 可执行的名词短语"],
  "spatial_selector": {
    "type": "ordinal_x | ordinal_y | nearest | overlap | union | none",
    "arguments": {}
  },
  "crop_policy": {
    "context_margin": 0.25,
    "max_crops": 1,
    "preserve_relation_context": true
  },
  "supporting_success_ids": [0, 3],
  "contrasting_failure_ids": [1, 2, 5],
  "confidence": 0.0
}
```

`grounding_queries` 和 `spatial_selector` 是工具私有字段，不能直接展示给 Teacher。否则“第二个目标”或
“左侧目标”等字符串可能直接泄露答案，使 OPVD 退化为语言提示蒸馏。

### 5.4 Group 特殊情况

- 同时存在成功和失败：使用完整 contrastive analysis，是主研究条件；
- 全部成功：总结成功轨迹的共同最小视觉证据，缺少 failure contrast；
- 全部失败：总结共同失败模式与应检查区域，信号可靠性单独报告；
- reward 全相同：GRPO 的 group-relative advantage 为零，但 OPVD 仍可能有 token-level signal；
- Analyzer 输出格式无效或 grounding 无有效框：该 Group 的 OPVD mask 置零，仍保留 GRPO。

最后一项只是一条必要的执行有效性检查，不设计复杂 learned gate。

## 6. Grounding DINO 与 Crop/Zoom Executor

### 6.1 职责边界

Grounding DINO 只执行语言目标到 bbox 的映射，不参与轨迹推理和 reward 判断。空间关系由确定性代码处理：

- `leftmost/rightmost`：按 bbox 中心横坐标排序；
- `top/bottom`：按纵坐标排序；
- `second_from_left`：检测同类全部实例后排序取第二个；
- `nearest(A,B)`：分别检测 A/B，按中心距离匹配；
- `overlap(A,B)`：按 bbox 或 mask 重叠匹配；
- 多目标比较：取 union bbox，或输出最多两张 Crop。

对于“船帆上的小图案”等小目标，允许 coarse-to-fine：先定位父物体并放大，再在局部图上进行第二次
grounding。第一阶段不要求 SAM 2；DINO bbox 已足够完成截图。SAM 2 只作为后续精细 mask 消融。

### 6.2 Crop 规则

1. 始终从原始高分辨率图像裁剪；
2. bbox 默认向外扩张 25%，保留必要上下文；
3. 关系题不得只保留单个 mask，必须保留关系参照物；
4. 最小边长和最大边长按目标相对面积约束，不再固定使用原图 40% 的窗口下限；
5. Crop resize 到模型支持的 canonical resolution；
6. 记录原始 bbox、扩张 bbox、DINO score、相对面积和执行失败原因。

## 7. Teacher 与 Student 上下文

本方案采用 episode-level prefix，不再逐句 interleave。

Student：

```text
[Original image]
[Question]
Please reason and answer.
```

Teacher：

```text
[Original image]
[Question]

Hindsight visual focus:
[visible_focus_instruction]

[Zoomed crop]

Please reason and answer.
```

Teacher 不生成一条新的轨迹，而是对每条 Student rollout 的相同 token 做 teacher forcing。Group 中 8 条
成功和失败轨迹全部保留，并共享该 Group 的 visual skill。

## 8. OPVD 与联合目标

对第 \(n\) 条轨迹中第 \(t\) 个 sampled token：

\[
\ell^{\mathrm{visual}}_{q,n,t}
=
\log\pi_\theta
\left(
y_{q,n,t}\mid \tilde h_{q,n},y_{q,n,<t}
\right),
\]

\[
\ell^{\mathrm{plain}}_{q,n,t}
=
\log\pi_\theta
\left(
y_{q,n,t}\mid h_{q,n},y_{q,n,<t}
\right).
\]

使用 SEED 式 detached visual shift：

\[
\Delta_{q,n,t}
=
\operatorname{sg}
\left[
\ell^{\mathrm{visual}}_{q,n,t}
-
\ell^{\mathrm{plain}}_{q,n,t}
\right],
\]

\[
g_{q,n,t}
=
\sigma
\left(
\beta_{\mathrm{opvd}}\Delta_{q,n,t}
\right).
\]

OPVD loss：

\[
\mathcal L_{\mathrm{opvd}}
=
\mathbb E
\left[
m_{q,n,t}g_{q,n,t}
\left(
\operatorname{sg}[\ell^{\mathrm{visual}}_{q,n,t}]
-
\ell^{\mathrm{plain}}_{q,n,t}
\right)
\right].
\]

联合目标：

\[
\mathcal L
=
\mathcal L_{\mathrm{GRPO}}
+
\lambda_{\mathrm{opvd}}
\mathcal L_{\mathrm{opvd}}.
\]

初始超参数沿用 SEED：

- rollout group size \(N=8\)；
- \(\beta_{\mathrm{opvd}}=5.0\)；
- \(\lambda_{\mathrm{opvd}}=0.01\)；
- KL coefficient `0.01`；
- learning rate `1e-6`。

实现时，冻结 \(\pi_{\theta_{\mathrm{old}}}\) 负责 rollout、Group 分析输入和 old log-prob；当前
\(\pi_\theta\) 同时计算 visual Teacher 与 plain Student 分支，Teacher log-prob 和 gate detached。

## 9. 为什么保留全部失败轨迹

失败 trajectory 可能包含大量局部正确 token，只在少数视觉判断处出错。高质量 Crop 可以：

1. 增强失败轨迹中仍受到视觉证据支持的局部推理；
2. 减弱与 Crop 证据冲突的 sampled token 的 OPVD 权重；
3. 与负 GRPO advantage 共同工作，让整体错误轨迹受到惩罚；
4. 在下一轮重新 rollout 后，逐步修正前缀分布，而不是要求一次 teacher forcing 改写整条轨迹。

需要注意：SEED 式 sampled-token OPVD 不直接提供“未采样的正确替代 token”。因此评估信号时不能只看
平均绝对 KL 或 \(|\Delta|\)，而要检查 visual shift 是否具有正确的相对区分结构。

## 10. 核心研究假设

### H1：Group 对比能改善视觉归因

相对于单 trajectory Analyzer，Group Analyzer 能通过成功/失败差异减少事后合理化，并生成更接近
oracle 的 grounding program 和 Crop。

### H2：多模态 Skill 比纯语言 Skill 提供更强的 OPVD

在控制插图格式和可见提示后，真实正确 Crop 能产生显著高于 text-only、random 和 shuffled Crop 的
token-level signal。

### H3：失败轨迹包含可利用的视觉蒸馏信号

同一个 Group Crop 能够保留失败轨迹中的局部正确 reasoning，并降低关键错误 span 的相对支持，而不必
删除失败轨迹。

### H4：固定外部 Analyzer 比共享 Actor Analyzer 更稳定

外部 Analyzer 能够避免 Actor RL 更新引起的分析格式漂移和视觉归因遗忘，同时仍通过读取当前 on-policy
Group 生成随策略分布变化的 visual skill。

### H5：视觉 Skill 能被无 Crop Student 内化

训练时只向 Teacher 提供 Crop；验证时移除 Analyzer、DINO 和 Crop 后，普通 Student 的准确率、推理质量
和视觉 grounding 能力仍然提高。

## 11. Phase A：完全不训练的信号验证

### 11.1 数据

继续使用 V*Bench 191 样本作为 probe/evaluation set，不用其 GT bbox 构造 Analyzer 输入或 Crop。GT bbox
只用于训练后不可见的定位评测与 oracle 上界。

为每个样本生成 8 条 on-policy rollout。主要分析同时含成功与失败的 mixed-outcome Group；all-correct 和
all-failed Group 单独报告。

### 11.2 条件

至少比较：

1. `plain`：原图，无额外提示和 Crop；
2. `text_only`：原图 + visible focus instruction；
3. `crop_only`：原图 + DINO Crop；
4. `text_crop`：原图 + visible focus instruction + DINO Crop；
5. `random_same_size`：同尺寸随机 Crop；
6. `shuffled_crop`：使用其他问题的 Group Crop；
7. `per_trajectory_analyzer`：外部 Analyzer 分别分析单条 trajectory；
8. `group_analyzer`：外部 Analyzer 对比完整 Group；
9. `oracle_crop`：仅用于上界，不进入真实 pipeline。

### 11.3 定位指标

- target center hit；
- bbox coverage / recall；
- Crop 相对面积；
- DINO 原始框与扩张框 IoU；
- 关系题是否同时保留参照物；
- Group Analyzer 相对 per-trajectory Analyzer 的定位增益；
- mixed、all-correct、all-failed Group 分层结果。

### 11.4 OPVD 信号指标

- visual Teacher 相对 plain Student 的 sampled-token \(\Delta\)；
- visual Teacher 相对 text-only 和 random/shuffled control 的直接 KL；
- OPVD gate 均值、方差和 active-token ratio；
- Student top-k support 中的 Teacher probability mass；
- 正确轨迹与失败轨迹的 \(\Delta\) 分布；
- reasoning span、answer span 和错误关键 span 的分层 \(\Delta\)；
- gold answer probability 的变化；
- wrong-to-correct 与 correct-to-wrong 翻转。

主要视觉区分度定义为：

\[
S_{\mathrm{visual}}
=
\mathbb E[\Delta\mid\text{success}]
-
\mathbb E[\Delta\mid\text{failure-critical-error}].
\]

### 11.5 进入训练的最低条件

满足以下条件才进入 Phase B：

1. Group Analyzer Crop 定位显著优于同尺寸随机与 shuffled Crop；
2. `text_crop` 的方向性 OPVD 显著优于 `text_only` 和 `crop_only` 中较强者；
3. 错误 rollout 上的 gold probability 增益显著为正；
4. 目标至少恢复 oracle 首步增益的 50%，即当前 probe 上约达到 `+7 pp`；
5. visible focus instruction 的 answer-leakage audit 通过；
6. 外部 Analyzer 的 JSON、DINO execution 和 Crop 生成成功率达到可训练水平。

## 12. Phase B：小规模联合训练

### 12.1 训练设置

- Backbone：Qwen3.5-4B，多模态输入，thinking 关闭；
- Group size：8；
- Actor rollout：原图 + 问题，仅生成纯文本 CoT；
- Analyzer：固定外部多模态模型，temperature 0；
- Grounder：固定 Grounding DINO；
- Teacher skill：一条 focus instruction + 一张 Crop，必要时最多两张；
- Loss：GRPO + sampled-token OPVD；
- 第一次训练只跑少量 update，保持与纯 GRPO 相同 rollout budget。

### 12.2 必要基线

1. `GRPO`；
2. `GRPO + text-only OPD`；
3. `GRPO + random-crop OPVD`；
4. `GRPO + external group visual OPVD`；
5. `GRPO + external per-trajectory visual OPVD`；
6. `GRPO + static offline Crop OPVD`；
7. `GRPO + current-policy Group Analyzer OPVD`，作为 SEED 式 analyzer 对照。

### 12.3 主要判断

- 是否优于 matched-budget GRPO；
- Group Analyzer 是否优于 per-trajectory Analyzer；
- 动态 on-policy Crop 是否优于 static Crop；
- 外部 Analyzer 是否比共享 policy Analyzer 更稳定；
- 去掉 Crop 后是否仍保留收益；
- 失败轨迹参与 OPVD 是否优于只蒸馏成功轨迹。

## 13. Phase C：规模化与 Self-Analyzer 扩展

只有外部 Analyzer 版本明确有效后，再研究：

1. 使用在线收集的 `(Group, external analysis)` 数据训练本地 Group Analyzer；
2. 将 Analyzer 放在独立 LoRA/adapter 中，避免 Actor RL 梯度破坏分析能力；
3. 周期性使用外部 Analyzer 做校准或 replay，而不是每轮调用；
4. 比较 fixed external、EMA analyzer、latest actor analyzer 和 hybrid analyzer；
5. 逐步减少外部调用比例，评估能否恢复更严格的 self-evolution；
6. 可选加入 SAM 2 精细 mask 或二阶段 coarse-to-fine grounding；
7. 可选从单个 episode-level Crop 扩展到少量 step-specific Crop，但不作为当前主线。

## 14. 关键风险与防护

### 14.1 答案泄漏

成功轨迹中的答案、序号或属性可能被 Analyzer 直接复述到 visible instruction。必须：

- 分离 Teacher-visible instruction 与 tool-private grounding query；
- 禁止 visible instruction 包含选项字母、最终答案、目标属性值和直接结论；
- 单独做 `text_only` 消融；
- 保存全部 Analyzer 原始输出以供审计。

### 14.2 Group 多数投票替代视觉归因

Analyzer 可能只根据 8 条答案的多数关系推测结论。Prompt 必须要求引用成功/失败 reasoning 中的视觉
差异，并输出可执行 object query，而不能只给答案一致性总结。

### 14.3 DINO 不擅长复杂关系

将名词检测和空间关系执行拆开；复杂 ordinal、nearest、overlap 和 union 由确定性代码处理。小目标使用
coarse-to-fine，不把一句复杂 referring expression 全部交给 DINO。

### 14.4 外部 Analyzer 成本与复现性

- temperature 固定为 0；
- 固定 system prompt、schema 和模型版本；
- 对完全相同的 Group 输入做内容哈希缓存；
- 记录 request、response、版本、时间、失败重试与成本；
- 不跨 policy iteration 复用旧 Group skill。

### 14.5 全失败 Group 的信号可靠性

全失败 Group 缺少正对照，必须单独报告。主实验保留这些轨迹，但不得把其结果与 mixed-outcome Group
混在一起得出视觉归因结论。

### 14.6 单 Crop 不足以支持比较题

第一主线仍以一个 episode-level visual skill 为单位，但该 skill 最多允许两张 Crop 或一张 union Crop。
是否需要多 Crop 由关系保持需求决定，而不是按 reasoning step 数量决定。

## 15. 日志与可复现产物

每个 task/group 至少保存：

```text
group_id
policy_checkpoint
original_image_path
question
8 rollout texts
8 final answers
8 rewards
external_analyzer_request
external_analyzer_raw_response
parsed_group_analysis
dino_queries_and_scores
raw_and_expanded_bboxes
crop_paths
teacher_visible_prefix
plain_logprobs
visual_teacher_logprobs
tokenwise_delta
tokenwise_gate
grpo_advantages
opvd_loss
```

所有图像路径、bbox 和 token span 必须可回溯到原始样本。V*Bench GT bbox 单独存放在 evaluation-only
字段，严禁进入 Analyzer、DINO query 构造和训练上下文。

## 16. 预期贡献

如果假设得到验证，本项目的贡献不是简单地“给 Teacher 多一张图”，而是：

1. 将 hindsight skill 从纯语言扩展为可执行的多模态 Crop/Zoom skill；
2. 将 GRPO 的 Group 结构用于成功/失败对比式视觉归因，而不仅用于 reward normalization；
3. 用固定外部 Analyzer 稳定地产生随当前 on-policy trajectory distribution 变化的视觉监督；
4. 证明失败轨迹也能通过高质量视觉 privileged context 提供 dense OPVD signal；
5. 在部署时移除 Analyzer、Grounder 和 Crop，使能力内化到只看原图的 Student。

## 17. 当前下一步

当前应优先完成 Phase A，而不是立即启动 RL：

1. 固定 Group Analyzer prompt 和 JSON schema；
2. 接入 Grounding DINO 和确定性 spatial selector；
3. 在现有 40 样本 pilot 上生成每题 8-rollout Group；
4. 运行 external group、external per-trajectory、random、shuffled 和 oracle 对照；
5. 检查 Crop 定位、答案泄漏和 token-level OPVD 方向；
6. 只有达到第 11.5 节条件后，再进入小规模 GRPO + OPVD。

