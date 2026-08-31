# Local-Analyzer + Grounding DINO 端到端 Smoke Report

日期：2026-08-24

## 设置

- Student / 临时 Analyzer：本地 Qwen3.5-4B，均关闭显式 thinking mode；
- 每题 rollout 8 次；
- 首轮采样 8 题，严格筛出 5 个同时含成功和失败轨迹的 mixed group；
- 共重评分 40 条轨迹；
- Grounder：官方公开可下载的 GroundingDINO-B / Swin-B；
- Teacher 前缀：原图与问题之后、assistant 推理之前插入无答案泄漏的聚焦说明和 Crop/Zoom；
- 对照：text-only、crop-only、blank crop、random crop、shuffled crop；
- 这一轮使用本地 Qwen 临时替代外部 GPT-5.6 Analyzer，因此只用于 Workflow 和定位能力 smoke test。

## Grounding DINO 定位结果

| V* index | Analyzer 目标 | Crop 面积占原图 | 人工判断 |
|---:|---|---:|---|
| 140 | 黑桶、橙色皮搋子、水槽 | 2.5% | 很准；三个对象完整且 Crop 紧凑 |
| 154 | 红色邮筒、红色电话亭 | 0.4% | 很准；目标很小，定位仍然集中，但分辨率有限 |
| 189 | 三轮车内的红色行李箱/凳子 | 6.1% | 基本准确；抓到相关三轮车内部区域 |
| 56 | 绿色骑行者、过街标志 | 8.6% | 对象均覆盖，但纵向 Crop 偏长 |
| 116 | 警用车辆、小红车、喷泉/纪念柱 | 48.4% | 较差；多对象分散导致 union 过大，放大作用弱 |

结论：Swin-B 对小型、具名、外观明确的物体定位能力不错；主要失败模式不是“完全找不到”，而是 Analyzer 一次给出多个分散参照物后，union Crop 变得过大。定位质量明显受 Analyzer 查询和关系选择器影响。

## 聚合 OPD / OPVD 结果

### 门控后的 sampled OPVD term

| 条件 | 平均 OPVD term |
|---|---:|
| text + crop | **0.03406** |
| crop only | 0.03221 |
| random crop | 0.02789 |
| blank crop | 0.02760 |
| shuffled crop | 0.02680 |
| text only | 0.01816 |

`text + crop` 是该指标上最强的条件：比 shuffled crop 高约 27%，比 blank/random crop 高约 22%–23%。失败轨迹上的 `text + crop` OPVD term 为 0.03622，高于成功轨迹的 0.03190，说明它没有只在本来正确的轨迹上产生信号。

### 正确选项概率

| 条件 | 正确选项平均概率 | 相对 plain |
|---|---:|---:|
| plain | 0.3305 | 0 |
| text + crop | **0.3698** | **+0.0393** |
| shuffled crop | 0.3601 | +0.0296 |
| crop only | 0.3455 | +0.0150 |
| blank crop | 0.3453 | +0.0148 |
| random crop | 0.3313 | +0.0008 |
| text only | 0.3228 | -0.0077 |

完整前缀平均提高正确选项概率，但 shuffled crop 也有较大提升，所以目前的净视觉特异性增益只有约 `+0.0097`（text+crop 减 shuffled）。样本量只有 5 组，不能据此宣称信号已经稳定。

### 逐题关键现象

- index 154：最干净的正例。Crop 仅占 0.4%，`text+crop` 的正确选项概率提升 `+0.1086`，明显高于 blank/random/shuffled。
- index 189：正确选项概率提升 `+0.0312`，但 random/shuffled 也有相近提升，特异性不足。
- index 116：正确选项概率提升 `+0.0603`，但 blank/random crop 同样提升 `+0.0603`，主要是额外图像格式效应。
- index 56：定位看起来合理，但正确选项概率下降 `-0.0036`，shuffled crop 反而更强。
- index 140：定位非常准确，但正确选项概率基本不变；定位正确不必然产生答案级增益。

## 当前判断

Workflow 已经完整可运行，Grounding DINO 的局部物体定位也足以支持 Crop/Zoom。但这轮只得到“存在候选 OPVD 信号”的证据，还没有得到“信号已可靠对齐正确性”的证据：

1. 完整前缀的门控 OPVD term 确实高于所有对照；
2. 平均正确选项概率也提高；
3. 但逐题差异很大，且部分提升可被 blank/shuffled crop 复现；
4. 精确定位与答案增益并非单调相关；
5. 下一次必须用计划中的外部 GPT-5.6 Analyzer 在完全相同的 5 个 group 上复跑，才能分离“Analyzer 太弱”和“视觉前缀机制本身不稳定”。

## 产物

- Analyzer：`outputs/group-visual-opvd-probe/analyses-local-qwen.jsonl`
- Grounding：`outputs/group-visual-opvd-probe/grounded-local-qwen.jsonl`
- 定位框与 Crop：`outputs/group-visual-opvd-probe/crops-local-qwen/`
- GPU0 探针：`outputs/group-visual-opvd-probe/probe-local-qwen-gpu0/records.jsonl`
- GPU1 探针：`outputs/group-visual-opvd-probe/probe-local-qwen-gpu1/records.jsonl`
- 聚合：`outputs/group-visual-opvd-probe/probe-local-qwen-summary.json`
