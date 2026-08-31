# 第一阶段无训练 Probe 实验报告

> 项目：Sparse Interleaved Visual-Evidence OPSD  
> 模型：Qwen3.5-4B（thinking 关闭）  
> 数据：V*Bench 官方 191 样本中的 40 样本分层 pilot  
> 日期：2026-08-24

> 后续说明：本报告是每样本仅插一张证据图的第一版 probe。按 (E_i\rightarrow R_i) 重建完整
> `[E1]R1 [E2]R2 ...` Teacher 轨迹后的结果见
> [`phase1_dense_interleaved_probe_report.md`](phase1_dense_interleaved_probe_report.md)；dense oracle
> 已显示更强的 Self-Evolution 上界信号，并将当前首要问题定位为 crop 放大质量，而非复杂门控。

## 1. 结论先行

### 问题一：图文交错 Teacher 能否产生更好的 OPD 信号？

**能产生真实且大部分位于 Student 支持集内的分布变化，但在不加门控时，还不能称为稳定的“更好 OPD 信号”。**

- 稳定相对注意力裁图 Teacher 相对“相同插图格式的空白图 Teacher”的直接 KL 为
  `0.277 nat/token`，说明证据图确实改变了 Teacher，而不只是插图格式造成变化。
- Teacher 在原始 Student top-20 token 集合上的平均概率质量为 `97.34%`，因此这个变化大部分仍在
  Student 当前支持集内，具有较好的可蒸馏性。
- 但相对空白图，稳定裁图在全部 40 样本上的正确选项概率只变化 `+0.10` 个百分点，
  95% bootstrap CI 为 `[-3.79, +3.94]` 个百分点；强制选择准确率也没有净提升。
- 稳定裁图对既有 Student 目标句 token 的平均 NLL 比空白图高 `0.167 nat/token`。
  对错误轨迹，这可能表示 Teacher 正在反对错误 token；但正确轨迹也出现了类似变化，因此不能直接
  把这部分 KL 全部用于训练。

最重要的是条件效应：

| 条件 | n | 稳定裁图相对空白图的正确选项概率变化 | 95% bootstrap CI |
|---|---:|---:|---:|
| 原 rollout 错误，且稳定窗口命中全部目标中心 | 15 | **+5.14 pp** | **[+2.42, +8.69] pp** |
| 原 rollout 错误，但稳定窗口未命中 | 5 | **-5.78 pp** | **[-10.11, -1.44] pp** |
| 全部错误 rollout | 20 | +2.41 pp | [-0.85, +5.91] pp |
| 全部正确 rollout | 20 | -2.21 pp | [-9.09, +4.53] pp |

官方目标框构造的 oracle crop 在错误 rollout 上相对空白图提升 `+7.24 pp`
（95% CI `[+2.59, +12.24] pp`）。这说明“正确局部证据能够改善错误轨迹”这一核心假设有初步支持；
当前瓶颈是可靠地判断何时裁对、何时监督方向正确。

因此，**不建议现在直接把所有插图 Teacher KL 用于 OPD**。建议只在定位置信度高且 Teacher
效用通过对照检验的 step 上蒸馏。

### 问题二：CoFFT 风格滑窗采图是否准确？

**显著好于同尺寸随机窗口，但目前只能算中等准确，尚不足以无门控使用。**

| 方法 | 全部目标中心命中率 | 同尺寸随机 | 平均 bbox 覆盖率 | 随机覆盖率 | 热力峰落入 bbox |
|---|---:|---:|---:|---:|---:|
| Absolute | 52.5% | 16.2% | 58.4% | 21.6% | 10.0% |
| CoFFT-strict | 42.5% | 15.5% | 48.1% | 21.2% | 5.0% |
| **CoFFT-stable** | **57.5%** | **15.0%** | **64.1%** | **20.6%** | **20.0%** |

稳定版全部目标中心命中率的 95% bootstrap CI 为 `[42.5%, 72.5%]`；其 `any-target`
命中率为 `72.5%`。窗口平均覆盖原图约 `16.2%`，与 40% × 40% 的最小搜索尺度一致。

严格相对注意力比绝对注意力更差，和蓝帆样本观察一致：描述基线中接近零的 patch 会被比值放大，
导致天空等无关位置形成尖峰。稳定版对分母设置正值 25% 分位数下限，并使用
`softmax(log ratio)` 等价的线性归一化与 3×3 平滑，明显改善了窗口定位。

## 2. 实验设计

### 2.1 样本

从 191 条 V*Bench 中用固定随机种子分层抽取 40 条：

| 题型 | 原 rollout 正确 | 原 rollout 错误 |
|---|---:|---:|
| direct attributes | 10 | 10 |
| relative position | 10 | 10 |

这是为了比较错误轨迹和正确轨迹，不代表完整数据集中两类样本的自然比例。

### 2.2 Teacher/Student 条件

模型权重始终冻结，不进行任何参数更新。对每个样本选择绝对图像注意力质量最高的一个非纯答案句，
在同一 Student rollout 上比较七种上下文：

1. `none`：原图 + 问题 + 纯文本 rollout 前缀；
2. `blank`：在目标句前插入同尺寸空白图，控制插图 token 和位置格式；
3. `random`：插入同尺寸随机窗口；
4. `absolute`：绝对注意力滑窗；
5. `relative_strict`：严格 CoFFT 相对注意力滑窗；
6. `relative_stable`：带小分母抑制的稳定相对注意力滑窗；
7. `oracle`：由官方目标框扩张得到的上界窗口。

Teacher 对同一 Student 目标句做 teacher forcing。由于原句间空格与插图后的换行可能改变目标句
第一个词的 tokenization，分布对齐从目标句第二个 token 开始；其余 token ID 完全一致。

### 2.3 注意力和滑窗

- 只读取 Qwen3.5 混合架构中的 8 个标准 Full-Attention 层；DeltaNet 层不伪装成二维注意力。
- 对问题、目标句和此前 reasoning 句分别聚合图像 patch 注意力。
- 使用描述提示 `Describe the image in detail` 作为相对注意力分母。
- 裁图评分为问题/前缀上下文图与当前句图的 0.5/0.5 混合。
- 在图像 patch 网格上搜索边长为全图 40%–90%、步长约 10% 的多尺度窗口，以窗口内平均注意力
  密度选取最佳窗口。
- 用 V*Bench 官方 bbox 评估中心命中、bbox 覆盖、peak pointing game、中心距离，并用每样本
  200 个同尺寸随机窗口建立基线。

### 2.4 OPD 信号指标

- Teacher → Student token KL 与 JSD；
- Teacher/Student top-20 overlap；
- Teacher 在 Student top-20 支持集上的概率质量；
- Student rollout 目标 token NLL 的变化；
- 强制选择条件下正确选项概率与预测翻转；
- 最关键的格式控制：裁图 Teacher 相对空白图 Teacher 的**直接** KL，而不是两个 KL 的差。

## 3. OPD Probe 详细结果

| 插图条件 | 对空白图直接 KL | 对空白图 top-20 overlap | Teacher 在原 Student top-20 的质量 | 正确选项相对空白图变化 |
|---|---:|---:|---:|---:|
| Random | 0.217 | 81.9% | 96.14% | -1.33 pp |
| Absolute | 0.279 | 79.8% | 97.18% | +0.91 pp |
| CoFFT-strict | 0.241 | 80.9% | 96.74% | -0.64 pp |
| **CoFFT-stable** | **0.277** | **79.8%** | **97.34%** | **+0.10 pp** |
| Oracle | 0.323 | 76.6% | 96.96% | +1.04 pp |

稳定裁图相对空白图的直接 KL 比随机图高约 `0.0595 nat/token`，说明它包含一部分
证据特异信号，但幅度不大。全样本上，稳定裁图与空白图之间发生 2 次 wrong→correct 翻转，
也发生 2 次 correct→wrong 翻转，净值为零。

这组结果支持如下判断：

1. 插图格式本身就是强干预，必须保留 blank control；空白图相对无插图 Student 的 KL 已约为
   `1.024 nat/token`。
2. 视觉内容在格式效应之外确实产生额外信号。
3. 额外信号的方向并不自动正确；OPD loss 需要 step/样本级 utility gate 或权重，而不能全收。
4. 错误 rollout 上 oracle 的改善表明 Teacher 特权视觉上下文仍有可利用上限。

## 4. 典型案例

### 4.1 定位与答案方向均成功：index 154

问题：`Is the pillar box on the left or right side of the telephone booth?`

- 原 rollout：错误，预测 B，gold A。
- 稳定相对窗口覆盖 telephone booth 和 pillar box 两个官方框，热力峰也在目标框内。
- gold 概率：无插图 `43.8%`，空白图 `32.1%`，随机图 `46.9%`，稳定裁图 `56.2%`。
- 稳定裁图把强制选择从 B 翻转为正确的 A。
- 稳定裁图相对空白图直接 KL 为 `0.180 nat/token`，Teacher 在原 Student top-20 上的质量为
  `97.6%`。

可视化：`outputs/interleaved-opd-probe/samples/index-154/localization.jpg`

### 4.2 定位失败且答案方向恶化：index 56

问题：`What is the color of the cyclist's box?`

- 原 rollout：错误，预测 C，gold A。
- 稳定相对窗口完全漏掉官方目标框，bbox coverage 为 0。
- gold 概率：无插图 `16.4%`，空白图 `16.9%`，稳定裁图 `5.3%`；稳定裁图相对空白图下降
  `11.6` 个百分点。
- oracle crop 将 gold 概率提高到 `42.6%`，说明该样本主要是定位失败，而不是局部证据本身无用。

可视化：`outputs/interleaved-opd-probe/samples/index-056/localization.jpg`

## 5. 当前结论的边界

- 这是无训练诊断，只能说明监督信号的存在、方向与可蒸馏性，不能直接证明训练后准确率会提高。
- 每个样本只 probe 一个最具视觉依赖的句子，不代表完整轨迹上的所有 step。
- 裁图来自 rollout 完成后的 hindsight attention；尚未验证在线 causal 触发器。
- 40 样本是刻意平衡正误的 pilot，置信区间仍较宽；完整结论需要扩展到 191 条。
- forced-choice 概率是诊断量，不等同于重新生成完整图文交错 Teacher 轨迹后的最终准确率。
- 官方框只标注问题目标，不一定覆盖所选句中所有被提及对象；中心命中比 peak 指标更适合当前大窗口。
- “稳定相对注意力”是工程消融，不是论文严格公式，报告中始终与 strict 版本分开。

## 6. 下一阶段建议

在进入训练前，建议先做 Phase 1b：

1. 扩展到全部 191 条，保留同样的 blank/random/oracle 对照。
2. 将 gate 设为主要研究对象，而不是直接调 OPD loss：
   - 稳定相对与绝对窗口的一致性；
   - 跨 Full-Attention 层窗口稳定性；
   - 严格相对注意力的有效面积与小分母异常检测；
   - 多个相邻 crop 的 Teacher 方向一致性；
   - Teacher 相对 blank 的直接 KL、熵变化和 Student top-k 支持质量。
3. 在有可验证 reward 的任务上，优先保留“Student 错误或低置信 + 裁图高置信 + Teacher 方向改善”的
   step；正确且高置信的 Student step 默认不插图，避免无谓扰动。
4. 先用固定模型做一次 leave-one-stratum-out 阈值校准，再进入小规模 OPD，以避免在 V* 测试标注上
   直接拟合 gate。

## 7. 产物

- 实验脚本：`probe_interleaved_opd.py`
- 汇总：`outputs/interleaved-opd-probe/summary.json`
- 去重后的每样本记录：`outputs/interleaved-opd-probe/records.jsonl`；两个 `records.shard-*.jsonl`
  保留可续跑的原始 shard 日志
- 每样本裁图与注意力窗口图：`outputs/interleaved-opd-probe/samples/index-*/`
