# 未中心化 GRPO + Signed OPSD Advantage 设计

> 状态：已在 `groove` 联合训练分支实现。
> 目标：在不训练额外 Critic、不使用 trajectory 中心化、也不引入 OCR/DINO 连续可靠性系数的前提下，将现有单向 OPD 改造成有正有负的 sampled-token OPSD advantage。

## 1. 设计结论

本方案把训练信用拆成两部分：

1. **GRPO outcome advantage**：判断整条 rollout 相对同组其他 rollout 更好还是更差；
2. **OPSD evidence advantage**：判断 privileged visual Teacher 相比原始输入 Student，是更支持还是更反对当前已经采样出的 token。

二者先在 token advantage 层合并，再统一进入同一个 PPO-clip policy loss：

\[
A^{\mathrm{total}}_{i,t}
=
A^{\mathrm{GRPO}}_i
+
\lambda_E A^{\mathrm{OPSD}}_{i,t},
\]

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{PPO\text{-}clip}}
\left(A^{\mathrm{total}}\right)
+
\beta_{\mathrm{ref}}\mathcal L_{\mathrm{ref\text{-}KL}}.
\]

这与当前运行时的

\[
\mathcal L_{\mathrm{GRPO}}
+
\lambda\mathcal L_{\mathrm{one\text{-}sided\ OPD}}
\]

不同：新方案不再保留一个绕开 PPO clipping 的独立单向 OPD loss。

## 2. 范围与明确不采用的设计

第一版采用以下约束：

- 不训练 value model、return model 或其他 Critic；
- 不对 OPSD advantage 做 trajectory 内减均值；
- 不使用 OCR、DINO 或 Analyzer 分数构造连续可靠性系数；
- 只保留二值 evidence availability mask；
- 不使用 sigmoid 正值 gate；
- 不使用 \((1-r_i)\) reward-complement mask；
- 正确和错误 rollout 都可以获得正或负 OPSD advantage；
- 继续使用 sampled-token log-prob，不要求完整词表 KL；
- Teacher、Student gap 和最终 advantage target 全部 stop-gradient。

## 3. 符号

| 符号 | 含义 |
| --- | --- |
| \(g\) | 一个 prompt 对应的 rollout group |
| \(i\) | group 中第 \(i\) 条 rollout |
| \(t\) | rollout 中第 \(t\) 个 response token |
| \(y_{i,t}\) | Student 已经采样出的 token |
| \(h_{i,t}\) | 原图、问题和 token prefix 构成的 Student 状态 |
| \(E_g\) | Analyzer/grounding 为整个 group 构造的 privileged visual evidence |
| \(\tilde h_{i,t}\) | 加入 focus text 和 crops 后的 Teacher 状态 |
| \(m_{i,t}\) | response token mask，有效 token 为 1 |
| \(e_i\) | evidence availability mask，可用为 1，否则为 0 |
| \(R_i\) | 第 \(i\) 条 rollout 的 terminal reward |
| \(A_i^{\mathrm{GRPO}}\) | trajectory-level GRPO advantage |
| \(A_{i,t}^{\mathrm{OPSD}}\) | token-level signed evidence advantage |
| \(\lambda_E\) | OPSD advantage 系数 |

同一个 group 的 rollout 共用一份 evidence program，但 Teacher 对每条 rollout 的不同 token prefix 分别打分。

## 4. GRPO Outcome Advantage

### 4.1 Terminal reward

当前任务继续使用：

\[
R_i
=
0.9\,\mathbf 1[\text{答案正确}]
+
0.1\,\mathbf 1[\text{以 FINAL: X 结束}].
\]

OPSD 不修改这个环境 reward，也不向 reward tensor 添加 token reward。

### 4.2 Group-relative normalization

对同一个 prompt 的 \(N\) 条 rollout：

\[
\mu_g=\frac{1}{N}\sum_{i\in g}R_i,
\]

\[
\sigma_g=\operatorname{Std}\{R_i:i\in g\}.
\]

trajectory advantage 为：

\[
A_i^{\mathrm{GRPO}}
=
\begin{cases}
\dfrac{R_i-\mu_g}{\sigma_g+\epsilon}, & \sigma_g>\epsilon,\\[6pt]
0, & \sigma_g\le\epsilon.
\end{cases}
\]

其中 \(\epsilon=10^{-6}\)。同一个 \(A_i^{\mathrm{GRPO}}\) 广播到该 rollout 的所有有效 response token。

因此：

- 高于 group 平均 reward 的 rollout 获得正 advantage；
- 低于 group 平均 reward 的 rollout 获得负 advantage；
- uniform-reward group 的 GRPO advantage 严格为 0。

## 5. Signed OPSD Evidence Advantage

### 5.1 Student 与 Teacher sampled-token log-prob

Student 只看原始输入：

\[
\ell^S_{i,t}
=
\log\pi_\theta
\left(
y_{i,t}\mid h_{i,t}
\right).
\]

Teacher 使用同一份当前 actor 权重，但额外看 privileged visual evidence，并在 `no_grad` 下前向：

\[
\ell^T_{i,t}
=
\log\pi_{\operatorname{sg}(\theta)}
\left(
y_{i,t}\mid \tilde h_{i,t},E_g
\right).
\]

Teacher 不重新生成 response。它在 privileged context 下对 Student 已经采样出的同一个 \(y_{i,t}\) 做 teacher forcing，因此 Teacher 与 Student 的 token 位置一一对齐。

### 5.2 Signed evidence gap

定义：

\[
\Delta_{i,t}
=
\operatorname{sg}
\left[
\ell^T_{i,t}-\ell^S_{i,t}
\right].
\]

它也可以写成 pointwise log-ratio：

\[
\Delta_{i,t}
=
\operatorname{sg}
\left[
\log
\frac{
\pi_T(y_{i,t}\mid\tilde h_{i,t},E_g)
}{
\pi_S(y_{i,t}\mid h_{i,t})
}
\right].
\]

符号含义为：

| 条件 | 解释 | OPSD 方向 |
| --- | --- | --- |
| \(\Delta_{i,t}>0\) | evidence 让 Teacher 更支持 sampled token | 提高该 token 概率 |
| \(\Delta_{i,t}=0\) | evidence 没有改变支持程度 | OPSD 不更新该 token |
| \(\Delta_{i,t}<0\) | evidence 让 Teacher 更不支持 sampled token | 降低该 token 概率 |

第一版不使用 \(\sigma(\beta\Delta)\)。sigmoid 会把权重限制为正数，从而重新产生“只能提高、不能降低”的问题。

### 5.3 Evidence availability mask

不引入连续可靠性系数。只使用：

\[
e_i
=
\begin{cases}
1, & \text{该 group 成功构造 Teacher evidence},\\
0, & \text{Analyzer/grounding/evidence 构造失败}.
\end{cases}
\]

OPSD advantage 定义为：

\[
\boxed{
A^{\mathrm{OPSD}}_{i,t}
=
e_i\Delta_{i,t}
}
\]

不乘 reward，不乘 \((1-R_i)\)，也不根据 rollout 正确与否改变 OPSD 公式。

## 6. Reverse KL 解释

在固定 Teacher、on-policy sampling 和忽略数值 clipping 的条件下，考虑：

\[
D_{\mathrm{KL}}
\left(
\pi_S(\cdot\mid h_t)
\Vert
\pi_T(\cdot\mid\tilde h_t,E)
\right).
\]

其梯度可写为：

\[
\nabla_\theta D_{\mathrm{KL}}(\pi_S\Vert\pi_T)
=
\mathbb E_{y\sim\pi_S}
\left[
(\log\pi_S(y)-\log\pi_T(y))
\nabla_\theta\log\pi_S(y)
\right].
\]

严格展开时括号内还会出现常数项 \(+1\)，但

\[
\mathbb E_{y\sim\pi_S}
[\nabla_\theta\log\pi_S(y)]
=0,
\]

所以该常数项在期望梯度中消失，得到上式。

令 \(\Delta=\log\pi_T-\log\pi_S\)，则：

\[
\nabla_\theta D_{\mathrm{KL}}(\pi_S\Vert\pi_T)
=
\mathbb E_{y\sim\pi_S}
\left[
-\Delta\nabla_\theta\log\pi_S(y)
\right].
\]

所以 sampled-token surrogate 可以写成：

\[
\mathcal L_{\mathrm{sampled\text{-}RKL}}
=
-\operatorname{sg}(\Delta)\log\pi_S(y).
\]

其梯度为：

\[
\frac{\partial\mathcal L_{\mathrm{sampled\text{-}RKL}}}
{\partial\log\pi_S(y)}
=
-\Delta.
\]

因此，本方案的 signed OPSD advantage 不是任意设计的正负 gate，而是 privileged Teacher Reverse KL 的 sampled policy-gradient coefficient。

如果实际 sample 来自 \(\pi_{\mathrm{old}}\) 而不是当前 \(\pi_\theta\)，则由 PPO importance ratio \(\rho=\pi_\theta/\pi_{\mathrm{old}}\) 处理策略差异。

## 7. 合并后的 Token Advantage

对所有有效 response token：

\[
\boxed{
A^{\mathrm{total}}_{i,t}
=
A_i^{\mathrm{GRPO}}
+
\lambda_E e_i\Delta_{i,t}
}
\]

它表示：

- \(A_i^{\mathrm{GRPO}}\) 给出整条 trajectory 的任务结果方向；
- \(\Delta_{i,t}\) 给出当前 token 的 privileged evidence 方向；
- \(\lambda_E\) 控制 evidence credit 相对 outcome credit 的强度。

忽略 PPO clipping 和 reference KL 时：

\[
\frac{\partial\mathcal L}
{\partial\log\pi_S(y_{i,t})}
\approx
-A^{\mathrm{total}}_{i,t}.
\]

因此：

- \(A^{\mathrm{total}}_{i,t}>0\)：提高 sampled token 的概率；
- \(A^{\mathrm{total}}_{i,t}<0\)：降低 sampled token 的概率；
- GRPO 与 OPSD 符号相同时相互增强；
- GRPO 与 OPSD 符号相反时，最终方向由二者加权后的净 advantage 决定。

## 8. “未中心化”的准确含义

本方案明确不计算：

\[
\bar\Delta_i
=
\frac{\sum_t m_{i,t}\Delta_{i,t}}
{\sum_t m_{i,t}}
\]

也不使用 \(\Delta_{i,t}-\bar\Delta_i\)。因此：

\[
\sum_t m_{i,t}A^{\mathrm{total}}_{i,t}
=
T_iA_i^{\mathrm{GRPO}}
+
\lambda_Ee_i
\sum_t m_{i,t}\Delta_{i,t}.
\]

结论是：

- terminal reward \(R_i\) 从未被修改；
- 但 token-level advantage 的总和不保证和加入 OPSD 前一致；
- OPSD 在这版设计中既提供 token-level 方向，也可能增加或减少整条 trajectory 的训练信用；
- 因此它应被称为“附加的 signed evidence advantage”，而不是“严格零和的 trajectory credit redistribution”。

这是本版本的有意选择。后续可以用 trajectory-centered 版本作为独立消融，但不能把两种方案混为同一个目标。

## 9. PPO Policy Loss

定义当前策略相对 rollout/old policy 的 token ratio：

\[
\rho_{i,t}
=
\exp
\left(
\log\pi_\theta(y_{i,t})
-
\log\pi_{\mathrm{old}}(y_{i,t})
\right).
\]

标准 PPO-clip surrogate 为：

\[
\mathcal L_{\mathrm{policy}}
=
-\mathbb E_{i,t}
\left[
\min
\left(
\rho_{i,t}A^{\mathrm{total}}_{i,t},
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)
A^{\mathrm{total}}_{i,t}
\right)
\right].
\]

实际实现应继续调用 verl 已有的 vanilla/dual-clip policy loss，只把输入 `advantages` 从 GRPO advantage 替换成 `total_advantages`，从而保持现有 clipping、rollout importance weight 和 token aggregation 行为。

reference-policy KL 继续作为独立正则项：

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{policy}}
+
\beta_{\mathrm{ref}}
\mathcal L_{\mathrm{ref\text{-}KL}}.
\]

当前配置可继续使用 \(\epsilon=0.2\)、\(\beta_{\mathrm{ref}}=0.001\) 和 entropy coefficient 0。

## 10. 不同 group 情况下的行为

### 10.1 Mixed-reward group，evidence 可用

GRPO 和 signed OPSD 都生效：

\[
A^{\mathrm{total}}=A^{\mathrm{GRPO}}+\lambda_E\Delta.
\]

正确 rollout 中被 evidence 反对的局部 token 可以少奖励甚至被降低；错误 rollout 中被 evidence 支持的局部 token 可以少惩罚甚至被提高。

### 10.2 Uniform-reward group，evidence 可用

\[
A^{\mathrm{GRPO}}=0,
\qquad
A^{\mathrm{total}}=\lambda_E\Delta.
\]

GRPO 没有组内排序信号，但 OPSD 仍根据 privileged evidence 对 token 做有正有负的更新。由于本版本未中心化，该 group 的 OPSD credit 总和可能非零。

### 10.3 Evidence 不可用

\[
e_i=0,
\qquad
A^{\mathrm{total}}=A^{\mathrm{GRPO}}.
\]

必须严格退化为普通 GRPO，不构造伪 Teacher 信号。

## 11. 建议的第一版数值处理

理论基线使用原始 \(\Delta\)。为避免极端 log-prob gap 产生过大 advantage，可以提供可配置的对称 clipping：

\[
\Delta^{\mathrm{clip}}_{i,t}
=
\operatorname{clip}
\left(
\Delta_{i,t},-c_\Delta,c_\Delta
\right).
\]

然后使用：

\[
A^{\mathrm{OPSD}}_{i,t}
=
e_i\Delta^{\mathrm{clip}}_{i,t}.
\]

注意：

- 不 clipping 时具有最直接的 sampled Reverse-KL 解释；
- clipping 会引入受控偏差，但通常更稳定；
- 必须使用对称 clipping，不能只截断负值；
- 第一版不再需要 `opd_gate_beta`；
- 为了与当前实验作最小变化比较，\(\lambda_E\) 可以从现有 `0.01` 起步，但应根据实际 advantage/gradient RMS 再校准，而不能因为数值名字相同就认为梯度强度相同。

## 12. 实现伪代码

```python
# Existing trajectory-level GRPO advantage: [B, T]
grpo_advantages = advantages

# Both targets are detached. Teacher was produced under torch.no_grad().
delta = (teacher_log_prob.detach() - student_log_prob.detach())

if delta_clip is not None:
    delta = delta.clamp(min=-delta_clip, max=delta_clip)

opsd_advantages = delta * self_distillation_mask.unsqueeze(1)

total_advantages = (
    grpo_advantages
    + opsd_advantage_coef * opsd_advantages
)

# Use one shared PPO/dual-clip loss. Do not add a separate OPD scalar loss.
policy_loss, metrics = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=student_log_prob,
    advantages=total_advantages,
    response_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    config=actor_config,
    rollout_is_weights=rollout_is_weights,
)

if use_reference_kl:
    policy_loss = policy_loss + reference_kl_coef * reference_kl
```

禁止继续使用：

```python
gate = sigmoid(beta * delta)
opd_loss = mean(gate * (teacher_log_prob.detach() - student_log_prob))
policy_loss = grpo_loss + opd_coef * opd_loss
```

否则会重新回到单向正权重 OPD，并且 OPSD 不受统一 PPO clipping 约束。

## 13. 必须记录的诊断指标

至少记录：

- `opsd/delta_mean`、`delta_std`、P10/P50/P90；
- `opsd/positive_token_fraction`；
- `opsd/negative_token_fraction`；
- `opsd/zero_token_fraction`；
- `opsd/advantage_rms_raw`；
- `opsd/advantage_rms_weighted`；
- `opsd_to_grpo_advantage_rms_ratio`；
- `total_advantage_mean/std/rms`；
- GRPO 与 OPSD advantage 的 cosine/alignment；
- correct/incorrect rollout 各自的 delta 和 signed credit 分布；
- evidence-ready、missing 和 fallback fraction；
- PPO clipping fraction；
- reference KL 和总 gradient norm。

由于本版本未中心化，还必须监控：

\[
\frac{1}{T_i}\sum_t\Delta_{i,t}
\]

在 correct、incorrect、mixed、all-correct 和 all-wrong group 上的分布。它直接反映 OPSD 正在给各类 trajectory 净增加还是净减少训练信用。

## 14. 必须通过的单元测试

1. **Teacher 等于 Student**：\(\Delta=0\)，OPSD advantage 和梯度都为 0；
2. **Teacher 更支持 token**：\(\Delta>0\)，OPSD 提高该 token 概率；
3. **Teacher 更反对 token**：\(\Delta<0\)，OPSD 降低该 token 概率；
4. **Evidence 缺失**：`total_advantages` 与原始 GRPO advantages 完全一致；
5. **Uniform reward group**：GRPO advantage 为 0，但 signed OPSD 仍可正可负；
6. **正确/错误无特殊 mask**：同样的 \(\Delta\) 使用同样的 OPSD 公式；
7. **Teacher stop-gradient**：Teacher 参数和 Teacher log-prob 不接收梯度；
8. **统一 PPO 路径**：联合模式只执行一次 policy loss，不再额外相加旧 OPD scalar loss；
9. **对称 clipping**：正负极端 gap 都按相同阈值截断；
10. **零差异回归测试**：该版本不能复现旧 sigmoid 在 \(\Delta=0\) 时仍产生 0.5 权重的行为。

## 15. 理论边界

本方案比旧的单向 sigmoid OPD 更完整，但仍需明确以下边界：

1. \(\Delta\) 是 privileged evidence advantage，不是任务 \(Q(s,a)\)；
2. 它表示 evidence 如何改变 Teacher 对 sampled token 的支持，不证明该 token 对最终 reward 的因果贡献；
3. sampled-token OPSD 可以降低已经采样出的坏 token，但不能直接告诉 Student 应该选择哪个未采样的替代 token；
4. Teacher 必须平均比 Student 拥有更有效的信息，否则 signed OPSD 也可能稳定地分配错误信用；
5. Teacher 使用随训练变化的当前 actor，因此 Reverse-KL target 是逐 update 变化的 moving target，而不是全程固定分布；
6. 未中心化意味着 OPSD 是附加 evidence objective，不是严格的 trajectory credit-budget redistribution。

## 16. 推荐消融

为了区分“方向变成双向”和“总信用发生变化”带来的效果，至少比较：

1. GRPO；
2. GRPO + 当前 one-sided sigmoid OPD；
3. GRPO + 未中心化 signed OPSD（本文方案）；
4. GRPO + trajectory-centered signed OPSD（后续零和消融）。

本文方案首先回答：

> 在不训练 Critic 的情况下，把 Teacher–Student sampled-token log-ratio 恢复为有正有负的 Reverse-KL evidence advantage，是否优于当前只能强化 sampled token 的单向 OPD？
