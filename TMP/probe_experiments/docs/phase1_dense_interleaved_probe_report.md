# Dense `E_i → R_i` 无训练 Probe 报告

> 模型：Qwen3.5-4B，thinking 关闭，权重冻结  
> 数据：与第一版相同的 40 条分层 V*Bench pilot  
> 结构：Student `[V,Q] R1 R2 ...`；Teacher `[V,Q] E1 R1 E2 R2 ...`  
> 日期：2026-08-24

## 1. 直接结论

**这次找到了比较明确的 Self-Evolution 上界信号，但它来自“正确且足够放大的证据图”；当前注意力裁图还没有稳定获得这个信号。**

在原 Student rollout 错误的 20 条样本上：

| Teacher 证据 | `R_1` 前 gold 概率相对 dense blank | 逐 step 平均 gold 概率变化 |
|---|---:|---:|
| Random crop | +1.76 pp | +1.66 pp |
| CoFFT-stable crop | +1.62 pp | +0.34 pp |
| **Oracle crop** | **+14.31 pp** | **+7.92 pp** |

Oracle 的两个结果均有正的 bootstrap 95% CI：

- 错误 rollout 首步：`+14.31 pp`，95% CI `[+5.94, +24.09] pp`；
- 错误 rollout 逐 step 平均：`+7.92 pp`，95% CI `[+2.64, +13.81] pp`。

这比第一版单句 probe 更强，说明用户提出的结构：

```text
[E1] R1 [E2] R2 ... [ET] RT
```

确实可以给失败 Student 轨迹提供有方向的 Teacher 信号。关键不在复杂门控，而在 `E_i` 是否真的把
目标放大到模型看得清的程度。

当前 stable attention crop 在错误 rollout 上与 random crop 基本同量级，置信区间跨零，所以**目前
还不能把 attention crop 的 dense OPD 直接用于训练**。这不是整体想法没有信号，而是当前 crop 距离
oracle 仍然太远。

## 2. 这次严格实现的轨迹

Student 先完成纯文本 rollout：

\[
R_1,\ldots,R_T\sim p_S(\cdot\mid V,Q).
\]

rollout 完成后，根据已经生成的 `R_i` 对原图的 hindsight attention 裁出 `E_i`，再重建：

```text
Teacher:
[原图 V] [问题 Q]
[E1] R1
[E2] R2
...
[ET] RT
```

Teacher 对相同的 `R_i` token 做 teacher forcing：

\[
p_T(R_i\mid V,Q,E_1,R_1,\ldots,E_{i-1},R_{i-1},E_i).
\]

Student 的对应分布不含中间证据图：

\[
p_S(R_i\mid V,Q,R_{<i}).
\]

两侧按每个 `R_i` 的相同 token ID 显式对齐。句首可能因“原始空格”与“插图后换行”产生不同
token，因此只舍弃每句第一个边界敏感 token，其余 token 全部进入 OPD probe。

## 3. 对照条件

每个 `R_i` 前均固定插图，不使用门控：

1. `none`：Student 原始纯文本轨迹；
2. `dense_blank`：每步插同尺寸空白图，控制视觉 token 数量与位置；
3. `dense_random`：每步插同尺寸随机窗口；
4. `dense_absolute`：每步插绝对注意力窗口；
5. `dense_strict`：每步插严格 CoFFT 相对注意力窗口；
6. `dense_stable`：每步插稳定相对注意力窗口；
7. `dense_oracle`：每步插官方目标框扩张后的 tight crop。

所有非空白证据图都 resize 到该 step 的同一 canonical 尺寸，因此不同条件的图像 token 布局可比。

40 条样本平均包含 `5.125` 个 `R_i`，范围为 1–22；每样本平均对齐 `69.5` 个 Student token。

## 4. 全轨迹 OPD 信号密度

以下均为证据 Teacher 相对 `dense_blank` Teacher 的直接距离，而不是两个 KL 的差：

| 条件 | 直接 KL | KL > 0.05 的 token 比例 | 内容 token KL | Teacher 在原 Student top-20 的质量 |
|---|---:|---:|---:|---:|
| Random | 0.129 | 28.5% | 0.143 | 97.63% |
| Absolute | 0.119 | 27.8% | 0.130 | 98.14% |
| CoFFT-strict | 0.122 | 27.2% | 0.135 | 97.86% |
| CoFFT-stable | 0.123 | 27.7% | 0.135 | 98.20% |
| **Oracle** | **0.173** | **31.3%** | **0.192** | **97.75%** |

结论：

- Dense Teacher 的信号覆盖了约三成 token，不再只是单个句子的局部扰动。
- Teacher 变化仍大量位于 Student top-20 支持集内，具有可蒸馏性。
- 但是 KL 密度本身不能判断方向：random 和 stable 的 KL 密度非常接近。
- Oracle 同时具有更大的内容 token KL 和明确的 gold 方向改善，才是“比较好的信号”。

## 5. 为什么首步信号强于后续 step

对错误 rollout，oracle 在 `R_1` 前提高 `14.31 pp`，但跨所有 step 平均为 `7.92 pp`。

原因是 Teacher 仍然 teacher-force 原 Student 的前缀。一旦错误 `R_1` 已写入上下文，后面的
`E_2,E_3` 需要在错误语言前缀下工作，纠错能力会被削弱。

这并不否定 Self-Evolution：第一轮 OPD 可以先改变 `R_1` 的分布；下一轮 Student 重新 rollout 后，
得到较好的 `R_1` 前缀，再为后续 `R_i` 重建新的证据 Teacher。它更像逐轮修正，而不是一次 Teacher
forcing 就把整条错误轨迹全部翻转。

## 6. 典型样本

### 6.1 index 4：oracle 显示非常强的上界信号

问题：`What is the pose of the woman with yellow backpack?`

- 原 rollout 错误。
- `R_1` 前 dense blank 的 gold 概率：`1.2%`。
- Stable crop：`2.2%`，只提升约 `1.0 pp`。
- Oracle crop：约 `66.6%`，提升约 `65.4 pp`。
- Stable 的 40% 边长窗口其实覆盖了官方目标框，热力峰也命中目标；但目标人物在裁图中仍然很小。
- Oracle 将背黄色包的女性真正放大后，模型立即得到强信号。

相关产物：

- `outputs/dense-interleaved-opd-probe/samples/index-004/decision-localization.jpg`
- `outputs/dense-interleaved-opd-probe/samples/index-004/decision-dense_stable.jpg`
- `outputs/dense-interleaved-opd-probe/samples/index-004/decision-dense_oracle.jpg`

### 6.2 index 154：attention crop 偶尔可以直接纠正 `R_1`

问题：`Is the pillar box on the left or right side of the telephone booth?`

- dense blank 在 `R_1` 前的 gold 概率：`32.1%`，预测 B。
- stable crop：`56.2%`，预测翻转为正确 A。
- 但 teacher-force 原错误 `R_1` 后，stable 的后续 gold 概率依次下降到约 `46.9%`、`1.0%`。

该例同时说明：当前 crop 有时能提供强信号，以及错误前缀为何会掩盖后续信号。

## 7. 当前 attention crop 的主要问题

稳定滑窗的边长下限是原图的 40%，即面积约 16%。V* 目标通常远小于原图的 1%。因此即使窗口
“命中 bbox”，模型得到的也常是包含大量背景的场景图，而不是真正的目标放大图。

在错误 rollout 的首步上：

- Stable 窗口命中目标的 15 条：gold 概率平均 `+3.71 pp`，CI 略跨零；
- 未命中的 5 条：平均 `-4.65 pp`，95% CI 约 `[-8.97,-0.33] pp`。

定位命中仍然重要，但仅仅“目标落在 40% 窗口里”不等于证据足够清楚。index 4 正是命中但放大
不足的例子。

## 8. 对 Self-Evolution 假设的判断

现在可以把结论分成两层：

1. **系统机制层：成立。** 对失败 Student rollout，正确的 `E_i → R_i` privileged
   context 能产生大幅、方向正确且可蒸馏的 Teacher 信号。
2. **当前自动证据提取层：尚未成立。** stable attention crop 的平均效果与 random crop 接近，
   还没有稳定逼近 oracle。

因此不需要先设计复杂门控。更直接的下一步是保持“每个 `R_i` 前固定插图”不变，只把证据变得
更接近 oracle：

- 把滑窗尺度从当前 40%–90% 改为 10%–40%；
- 每步固定插两张图：一个 40% context crop，加一个 10%–20% tight zoom；
- 或在当前窗口内部再做一次固定的二阶段 attention crop；
- 继续用 dense blank/random/oracle 判断 tight crop 是否真正接近上界。

这仍然是简单、无门控的系统。只有当 tight attention crop 接近 oracle 后，才值得进入第一次小规模
OPD 训练。

## 9. 产物

- 脚本：`probe_dense_interleaved_opd.py`
- 汇总：`outputs/dense-interleaved-opd-probe/summary.json`
- 40 条去重明细：`outputs/dense-interleaved-opd-probe/records.jsonl`
- 每样本 decision crop 和定位图：`outputs/dense-interleaved-opd-probe/samples/index-*/`
