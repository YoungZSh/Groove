# 稀疏图文交错证据 OPSD：研究与实现计划

> 工作名称：Sparse Interleaved Visual-Evidence On-Policy Self-Distillation（SIVE-OPSD）  
> 状态：Phase 1 pilot 已完成，训练尚未开始  
> 日期：2026-08-24

Phase 1 的 40 样本无训练 probe、数值结果与结论见
[`phase1_no_training_probe_report.md`](phase1_no_training_probe_report.md)。当前结论是：稳定相对注意力
裁图显著好于随机窗口，但无门控图文交错 Teacher 信号不够稳定；进入训练前必须先实现定位置信与
Teacher utility 双门控。

随后按照 (E_i) 必须位于 (R_i) 前的定义完成了 dense Teacher probe，详见
[`phase1_dense_interleaved_probe_report.md`](phase1_dense_interleaved_probe_report.md)。该实验修正了
上一段关于复杂双门控的优先级：dense oracle 已在错误 rollout 上产生显著信号，当前首要瓶颈是把
40% 场景窗口变成更接近 oracle 的 tight zoom，而不是先增加复杂门控。

## 1. 核心想法

本项目希望把视觉 grounding 作为 Teacher 的特权上下文，并通过 On-Policy
Self-Distillation（OPSD）将这种能力蒸馏回只生成纯文本推理轨迹的 Student。

严格的问题定义如下：

1. Student 接收原始图像和问题，生成不含中间证据图的纯文本 reasoning rollout。
2. rollout 完成后，将其切分为若干 reasoning step。
3. 根据每个 step 的图像注意力，从原图中选择、裁剪并放大可能相关的证据窗口。
4. 只有满足注意力与效用门控条件的 step 才插入证据图，不要求固定间隔插图。
5. 使用相同模型的冻结快照作为 Teacher；Teacher 在 Student rollout 的基础上读取
   图文交错的增强上下文，并 teacher-force 评价同一批 Student token。
6. 对齐 Teacher 与 Student 在相同目标 token 上的分布，只有 Student 分支接收梯度。
7. 更新后的 Student 在下一轮重新 rollout、重新计算注意力和证据图，构成外层
   self-evolution 循环。

这里的“纯文本 Student”表示 Student 的生成轨迹不插入中间裁剪图；Student 的初始
输入仍然包含原始多模态图像。证据图只属于训练期 Teacher 的 privileged context。

## 2. 明确不做什么

- 不要求 Student 在部署推理时调用动态裁剪工具。
- 不把裁剪窗口视为 Student 的显式 action，也不在第一阶段训练一个可微裁剪策略。
- 不让 Teacher 自由生成另一条无法与 Student token 对齐的新轨迹。
- 不把单个相对注意力最大 patch 直接视为可靠证据位置。
- 不使用当前 191 张 V*Bench 测试集进行训练；它只用于评测和可视化诊断。

## 3. 轨迹与上下文

Student rollout 记为：

\[
\tau_S=(r_1,r_2,\ldots,r_T),\qquad
\tau_S\sim p_{\theta_k}(\cdot\mid V,Q).
\]

其中 \(V\) 是原图，\(Q\) 是问题，\(r_t\) 是第 \(t\) 个 reasoning step。

Student 上下文始终是：

```text
[原图 V] [问题 Q]
r1
r2
r3
...
```

如果第 2、5 个 step 通过证据门控，Teacher 上下文是：

```text
[原图 V] [问题 Q]
r1
[证据图 E2]
r2
r3
r4
[证据图 E5]
r5
...
```

Teacher 对原始 Student token 序列做 teacher forcing。插入图像导致两侧绝对 token
位置不同，因此实现时必须按 reasoning span 和目标 token ID 建立显式对齐，不能按
张量位置直接对齐。

## 4. Reasoning step 切分

第一版应优先使用明确的 step 终止标记，例如：

```text
<step>Locate the second blue sail from the left.</step>
<step>Inspect the black silhouette on that sail.</step>
<final>A</final>
```

如果不能修改输出格式，可暂时使用换行和句子边界切分，但必须记录原始字符区间和
token 区间。离线按标点切句可以用于 Teacher 构造，但不能被误认为在线 causal
step boundary。

## 5. 证据图提取

### 5.1 CoFFT 风格相对注意力

对图像 \(V\) 和文本 \(X\)，定义：

\[
A^{rel}(V,X)=
\operatorname{Softmax}
\left(
\frac{A(V,X)}{A(V,D)+\epsilon}
\right),
\]

其中：

- \(D\) 是描述性基线提示，例如 `Describe the image in detail`；
- \(\epsilon=10^{-10}\)；
- 除法逐 patch 进行；
- 注意力先在选定的层、头和目标文本 token 上聚合。

Qwen3.5 是 DeltaNet 与传统 Attention 的混合架构。第一版只从能够提供标准二维
注意力矩阵的 Full-Attention 层提取图像 patch 注意力；DeltaNet 层单独记录为未覆盖，
不能假装它提供了同等含义的空间图。

描述基线 \(A(V,D)\) 对同一原图只需计算一次，并在该 rollout 内缓存。

### 5.2 已知数值问题

严格 CoFFT 比值可能被极小的描述基线分母放大。当前蓝帆样本中，绝对注意力峰值
位于第二张蓝帆，但相对注意力由于蓝天 patch 的分母过小而形成了近似单点伪峰。

因此实验必须同时保留以下版本：

1. Absolute：原始绝对注意力。
2. CoFFT-strict：严格论文公式，用于复现。
3. CoFFT-stable：带分母下限、空间平滑或稳健 log-ratio 的版本。

稳定版本不得替代严格版本而不做标注；两者必须作为独立消融项。

### 5.3 从注意力图到候选窗口

对每个 step 构造裁剪评分图：

\[
A_{crop}^{t}
=
0.5C^{rel}(V,Q,R_{<t})
+
0.5A^{rel}(V,r_t),
\]

其中：

\[
C^{rel}(V,Q,R_{<t})
=
\max
\left(
A^{rel}(V,Q)-\alpha A^{rel}(V,R_{<t}),0
\right).
\]

在原图上搜索覆盖原图尺寸约 40%--90% 的多尺度窗口，使用约 10% 的尺寸间隔和
滑动步长。窗口分数取内部平均注意力密度：

\[
B_t^*
=
\arg\max_{B\in\Omega}
\frac{1}{|B|}
\sum_{(x,y)\in B}A_{crop}^{t}(x,y).
\]

候选窗口通过门控后，始终从原图裁剪，再缩放为模型支持的证据图输入。

## 6. 稀疏证据触发器

固定每一步插图会增加图像 token、改变上下文格式并引入无关视觉噪声。本项目采用
两级门控。

### 6.1 第一级：注意力预筛选

对第 \(t\) 个 step 计算：

#### 图像依赖强度

\[
m_t
=
\operatorname{Mean}_{l,h,q\in r_t}
\sum_{i\in V}A_{l,h}(q,i).
\]

低 \(m_t\) 的逻辑过渡、格式 token 和结论模板通常不需要证据图。

#### 最佳窗口相对全图的优势

\[
z_t
=
\frac{\mu_{B_t^*}-\mu_V}{\sigma_V+\epsilon}.
\]

#### 有效注意力面积

\[
N_{eff,t}=\exp(H(A_t)).
\]

有效面积过大表示注意力过于弥散；有效面积接近一个 patch 则可能是小分母伪峰。

#### 绝对与相对注意力的一致性

\[
a_t
=
\operatorname{IoU}
\left(
\operatorname{Top}_{p}(A_t^{abs}),
\operatorname{Top}_{p}(A_t^{rel})
\right).
\]

如果绝对注意力关注目标、相对注意力却关注天空，应拒绝该候选。

#### 跨层稳定性

比较各个 Full-Attention 层的高注意力区域与聚合窗口之间的 IoU。只有多个层支持
同一区域时才认为 focus 稳定。

#### 新颖性

\[
n_t
=
1-
\max_{j<t}\operatorname{IoU}(B_t^*,B_j^*).
\]

它用于避免连续插入几乎相同的证据图。

一个可解释的预筛选形式是：

\[
g_t^{pre}
=
\mathbb{1}[m_t>\tau_m]
\mathbb{1}[z_t>\tau_z]
\mathbb{1}[a_t>\tau_a]
\mathbb{1}[n_t>\tau_n]
\mathbb{1}[N_{eff,t}\in\mathcal R].
\]

阈值不应直接写死。第一版应根据训练数据分位数或置换/随机裁剪产生的经验零分布
校准，使假阳性率可控。

### 6.2 第二级：Teacher 证据效用门控

对通过预筛选的 step，用冻结 Teacher 分别计算：

\[
p_{T0}^{t,n}
=
p_{\bar\theta_k}
(\cdot\mid V,Q,R_{<t}),
\]

\[
p_{TE}^{t,n}
=
p_{\bar\theta_k}
(\cdot\mid V,Q,R_{<t},E_t).
\]

定义证据效用，例如：

\[
u_t
=
\operatorname{Mean}_{n\in r_t}
\operatorname{JS}(p_{TE}^{t,n},p_{T0}^{t,n}).
\]

也可以结合：

- 正确答案 log-prob 的变化；
- Teacher entropy 的变化；
- step verifier 分数；
- 多个邻近窗口之间的预测一致性；
- 相比随机同面积裁剪的增益。

只有 \(u_t>\tau_u\) 时才正式保留证据图并产生该 step 的 OPD 信号。

### 6.3 Evidence budget

为防止阈值漂移导致所有 step 都插图，每条轨迹设置最大证据预算 \(K\)：

1. 先通过预筛选；
2. 再通过 Teacher 效用门控；
3. 按效用选择 Top-\(K\)；
4. 每轮监控平均插图率和零插图轨迹比例。

第一阶段建议比较 \(K\in\{1,2,4\}\)，而不是直接追求高插图密度。

## 7. Hindsight 与 causal 两种插入协议

### 7.1 Hindsight grounding

使用完整 Student step \(r_t\) 选择证据 \(E_t\)，再把 \(E_t\) 插到 \(r_t\) 前：

```text
r(t-1)
[E(t)]
r(t)
```

这最符合当前设想，也与 OPSD 中 Teacher 可使用 privileged information 的原则一致。
但它利用了即将被监督的 step 来选择图像，必须明确标注为 hindsight，并检查错误
step 是否会引导出自我确认的错误裁剪。

### 7.2 Causal grounding

只使用 \(Q+R_{<t}\) 选择 \(E_t\)，然后插到 \(r_t\) 前。该版本没有 future-step
信息泄漏，但证据定位可能更弱。

两种协议必须作为核心消融，不能混用。

## 8. OPD 目标

第 \(k\) 轮开始时冻结 Teacher 快照：

\[
\bar\theta_k\leftarrow\operatorname{StopGradient}(\theta_k).
\]

Student token 分布为：

\[
p_S^{t,n}
=
p_{\theta}
(\cdot\mid V,Q,R_{<t},r_{t,<n}).
\]

Grounded Teacher 分布为：

\[
p_{TE}^{t,n}
=
p_{\bar\theta_k}
(\cdot\mid V,Q,E_{<t},R_{<t},E_t,r_{t,<n}).
\]

基础损失：

\[
\mathcal L_{OPD}
=
\sum_t g_t
\sum_{n\in r_t}
w_{t,n}
D_{KL}
\left(
\operatorname{sg}[p_{TE}^{t,n}]
\parallel
p_S^{t,n}
\right).
\]

其中：

- \(g_t\) 是 step 级插图门控；
- \(w_{t,n}\) 是 token 级证据权重；
- 图像 token、chat template token 和非目标 prompt token 全部 mask；
- Teacher 分支严格 stop-gradient。

优先实验 full-vocabulary forward KL；显存不足时再比较 top-k teacher logits 或
sampled-token objective。必须使用 per-token KL clipping，避免格式和风格 token
支配梯度。

## 9. 视觉监督提纯

插入图片同时改变了视觉内容、token 数量、位置编码和模板结构。为提取真正由证据
内容带来的分布变化，引入 matched control Teacher：

- \(p_{T0}\)：无证据或使用格式匹配的空白/控制图；
- \(p_{TE}\)：使用真实证据裁剪图。

一种简单 token 权重是：

\[
w_{t,n}
=
\operatorname{clip}
\left(
\operatorname{JS}(p_{TE}^{t,n},p_{T0}^{t,n}),
0,w_{max}
\right).
\]

这样，证据图未改变的普通连接词权重较低；视觉实体、属性、数量和空间关系 token
更容易获得较高权重。

后续可以研究更强的 logit-residual/PMI 目标，但不应阻塞第一版验证。

## 10. Self-evolution 外循环

```text
Round k
  1. θ_k 生成纯文本 Student rollouts
  2. θ_k 为每个 step 提取注意力并生成候选证据
  3. 冻结 θ̄_k 计算无证据与有证据 Teacher logits
  4. 应用注意力门控、效用门控和 evidence budget
  5. 在 Student 自己的 token 轨迹上执行 OPD
  6. 得到 θ_(k+1)
  7. 重新 rollout、重新裁图、重新估计阈值
```

不能在同一优化步里让 Teacher 目标随 Student 同步变化。每个 outer round 应使用冻结
快照或足够滞后的 EMA Teacher，并在 held-out 集上选择是否接受新一轮 checkpoint。

可能的正反馈：

```text
Student reasoning 改善
  -> attention query 更准确
  -> 证据窗口更相关
  -> Grounded Teacher 的视觉修正更可靠
  -> OPD 信号改善
  -> Student 继续改善
```

也必须监控相反的 self-confirmation loop：

```text
错误 reasoning
  -> 错误裁剪
  -> Teacher 在错误证据上合理化
  -> OPD 强化错误
```

## 11. 信息上限

如果裁剪放大使 Teacher 看到了 Student 原始视觉编码中已经丢失的高频细节，Student
无法仅靠蒸馏恢复这些不存在的信息。第一版需要确保：

- Student 原图分辨率足够高；
- 裁剪主要起注意力重分配作用，而不是引入原表示完全没有的信息；
- 比较不同原图视觉 token 预算；
- 单独报告“小目标因分辨率不足而不可蒸馏”的失败样本。

## 12. 实现模块建议

```text
rollout/
  student_rollout.py       # 生成纯文本 on-policy 轨迹
  step_segmenter.py        # 字符/token step 对齐

evidence/
  attention_capture.py     # Full-Attention 图像注意力
  relative_attention.py    # strict/stable 相对注意力
  window_search.py         # 多尺度滑窗
  trigger.py               # 预筛选、效用门控、预算
  crop_cache.py            # 证据图与视觉 embedding 缓存

distill/
  interleaved_builder.py   # Teacher 图文交错上下文
  token_alignment.py       # Teacher/Student 目标 token 对齐
  teacher_scorer.py        # 冻结 Teacher 与控制分支
  opsd_loss.py             # KL、权重、clipping、mask

self_evolve/
  outer_loop.py            # checkpoint -> rollout -> OPD -> checkpoint
  acceptance.py            # held-out 指标与 round 接受规则
```

Qwen3.5 的 Teacher 图文交错模板需要先做兼容性测试。如果模型或 processor 不支持在
单个 assistant 消息中间插图，第一版可按 step 分别执行 Teacher forward：每次使用
原图、累计文本前缀和当前证据图来评价下一个 step。这在条件分布上等价得更干净，
也更容易建立 token 对齐。

## 13. 数据记录格式

每条 rollout 至少保存：

```json
{
  "sample_id": "...",
  "round": 0,
  "model_checkpoint": "...",
  "question": "...",
  "student_completion": "...",
  "steps": [
    {
      "step_id": 1,
      "text": "...",
      "char_span": [0, 42],
      "token_span": [0, 11],
      "absolute_image_mass": 0.0,
      "relative_entropy": 0.0,
      "window_score_z": 0.0,
      "abs_rel_iou": 0.0,
      "layer_stability": 0.0,
      "novelty": 0.0,
      "candidate_box_xyxy": [0.0, 0.0, 1.0, 1.0],
      "pre_gate": false,
      "teacher_utility": 0.0,
      "insert_evidence": false,
      "evidence_path": null
    }
  ],
  "student_answer": "...",
  "ground_truth": "...",
  "correct": false
}
```

同时记录随机种子、图像预处理参数、注意力层、头聚合方式、描述基线、阈值版本和
chat template，保证每轮证据构造可复现。

## 14. 实验矩阵

### 14.1 不训练的 Teacher 增益验证

| 组别 | Teacher 视觉上下文 | 目的 |
|---|---|---|
| T0 | 与 Student 相同，无证据 | 零增益对照 |
| T-random | 随机同面积裁剪 | 控制额外图像/分辨率效应 |
| T-abs | 绝对注意力裁剪 | 判断相对注意力是否必要 |
| T-rel-strict | 严格 CoFFT 裁剪 | 论文公式复现 |
| T-rel-stable | 稳定相对注意力裁剪 | 检查小分母修复 |
| T-oracle | 标注框或人工证据图 | 估计方法上限 |

首要问题不是最终准确率，而是：插入证据后同一个冻结模型是否真的成为更好的
Teacher。

### 14.2 单轮训练消融

| 组别 | 训练方式 |
|---|---|
| SFT | 固定专家轨迹监督 |
| OPSD-base | 无证据 Teacher |
| OPSD-all | 每个 step 插图 |
| SIVE-pre | 仅注意力门控 |
| SIVE-utility | 注意力 + Teacher 效用门控 |
| SIVE-purified | 加 matched control/token 权重 |

### 14.3 关键协议消融

- Hindsight vs causal grounding。
- 固定阈值 vs 分位数阈值。
- 无预算 vs Top-\(K\) evidence budget。
- Strict relative vs stable relative vs absolute。
- 只在答案正确的 rollout 蒸馏 vs 所有 on-policy rollout。
- 每轮刷新 Teacher vs EMA Teacher vs 多轮固定 Teacher。
- Student 原图不同视觉 token 预算。

## 15. 评估指标

### 最终任务

- Pass@1、Pass@K、平均准确率。
- 正确答案 log-prob。
- 推理长度和视觉 token 成本。

### Teacher 质量

- Grounded Teacher 相对 T0 的答案增益。
- Teacher 修正错误 Student rollout 的比例。
- 插图前后 entropy、JS/KL 和正确 token 概率变化。

### Grounding 质量

- 候选窗口与人工/标注区域的 IoU 或 recall。
- 相比随机裁剪的因果增益。
- 跨层、跨采样和跨 round 的窗口稳定性。
- 小分母伪峰率和 fallback 率。

### 稀疏性与演化

- 每条轨迹平均插图数。
- 零插图轨迹比例。
- 重复窗口比例。
- 每轮 Teacher--Student KL 和 Teacher advantage。
- 每轮 accepted/rejected checkpoint。

## 16. 分阶段计划

### Phase 0：定义和兼容性验证

- [ ] 固化 Student/Teacher token 对齐协议。
- [ ] 验证 Qwen3.5 图文交错模板或 per-step Teacher forward。
- [ ] 定义 hindsight 与 causal 两种数据构造模式。
- [ ] 冻结 V*Bench 191 样本为只读评测集。

### Phase 1：稀疏证据数据管线

- [ ] 将完整 rollout 切成可复现 step/token span。
- [ ] 缓存描述基线注意力。
- [ ] 实现 absolute、CoFFT-strict、CoFFT-stable 三种地图。
- [ ] 实现滑窗、跨层稳定性、有效面积和新颖性指标。
- [ ] 输出包含证据图的离线 Teacher 轨迹记录。

### Phase 2：训练前因果验证

- [ ] 对比无证据、随机、absolute、strict-relative、stable-relative。
- [ ] 测量 evidence-induced token distribution shift。
- [ ] 验证 Grounded Teacher 是否更可能纠正 Student 错误。
- [ ] 校准预筛选和效用阈值。
- [ ] 若 Grounded Teacher 无稳定优势，暂停进入 OPD。

### Phase 3：单轮 OPD

- [ ] 冻结 Teacher snapshot。
- [ ] 实现 full-vocabulary KL、mask 和 pointwise clipping。
- [ ] 从无证据 OPSD baseline 开始。
- [ ] 加入稀疏证据 Teacher。
- [ ] 加入 matched control 与 token 权重。
- [ ] 在独立 validation 集选择 checkpoint。

### Phase 4：多轮 self-evolution

- [ ] 用新 checkpoint 重新 rollout 和裁图。
- [ ] 每轮重新估计阈值分布。
- [ ] 设置最大 round 数和回退条件。
- [ ] 监控 self-confirmation、插图率坍缩和 KL 消失。
- [ ] 只接受 held-out 指标改善且 Teacher advantage 未退化的 round。

### Phase 5：规模化与论文实验

- [ ] 扩展到足够大的多模态训练集。
- [ ] 多模型/多尺度验证。
- [ ] 完成全部协议、注意力和预算消融。
- [ ] 报告计算成本与视觉 token 增量。
- [ ] 分析成功、失败和不可蒸馏的小目标案例。

## 17. Go/No-Go 判据

进入单轮 OPD 前，至少应满足：

1. Grounded Teacher 相比无证据 Teacher 在独立训练样本上有稳定增益。
2. 注意力裁剪显著优于随机同面积裁剪。
3. 稳定相对注意力没有被少数单 patch 伪峰支配。
4. Teacher 插图增益集中在视觉实体、属性、数量或空间关系 token。
5. Teacher 能在一部分 Student 错误轨迹上降低错误答案概率，而不只是提高已有 token
   的置信度。

进入多轮 self-evolution 前，至少应满足：

1. 单轮 SIVE-OPSD 优于 OPSD-base。
2. 插图率与计算成本可控。
3. 新 Student 在不插中间图的推理条件下仍然改善。
4. 改善不能由训练集记忆或测试集泄漏解释。

## 18. 主要开放问题

1. Evidence 应插在产生它的 step 前（hindsight）还是只影响下一 step（causal）？
2. Teacher 是否还需要参考答案/参考轨迹，还是只依赖视觉证据 privilege？
3. 证据增益应作为 KL 权重，还是只作为二值 gate？
4. Student 原图视觉分辨率需要多高，才能内化 Teacher 的局部视觉能力？
5. 哪些 Full-Attention 层最适合 Qwen3.5 的空间定位？
6. 是否需要对 attention map 做空间平滑后再滑窗？
7. Teacher 快照每轮刷新、EMA 更新和固定多轮哪种最稳定？
8. 当 Teacher advantage 随 Student 变强而消失时，系统应停止还是提高证据难度？

## 19. 参考资料

- CoFFT: Chain of Foresight-Focus Thought for Visual Language Models  
  <https://arxiv.org/abs/2509.22010>
- Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models  
  <https://arxiv.org/abs/2601.18734>
- OPSD 官方实现  
  <https://github.com/siyan-zhao/OPSD>
- Purified OPSD: On-Policy Self-Distillation Without Losing How to Think  
  <https://arxiv.org/abs/2607.02234>

## 20. 当前仓库已有资产

- `infer_vstar_qwen35.py`：Qwen3.5-4B V*Bench 推理脚本。
- `analyze_vstar_attention.py`：逐句 Full-Attention、相对注意力、热力图和遮挡分析。
- `outputs/qwen3.5-4b-vstar/traces.jsonl`：191 条非 thinking rollout。
- `outputs/attention-analysis/`：当前两个样本的注意力与干预结果。

这些资产适合用于 Phase 0--2 的小规模原型验证，但不能替代正式训练数据和独立测试集。
