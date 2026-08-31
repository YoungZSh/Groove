# Group-Contrastive Visual Prefix OPVD：无训练探针运行手册

## 1. 这个探针回答什么

本探针先不更新任何参数，只检验下面这条链路能否产生可用的
Teacher→Student 蒸馏信号：

1. 同一个 V*Bench 问题由当前 Qwen3.5-4B 采样 8 条纯文本可见推理轨迹；
2. 用最终答案是否正确得到二值 reward，组成与 GRPO 相同粒度的 group；
3. 只保留同时包含成功和失败轨迹的 mixed group；
4. 外部多模态 Analyzer 同时查看原图、问题、8 条轨迹和 reward，通过组内对比总结“成功轨迹看对了哪里、失败轨迹漏看或混淆了哪里”；
5. Grounding DINO 只执行对象定位，空间关系由确定性代码执行；Analyzer 返回多个对象时，每个对象分别从原图裁剪并放大，不再用一个大 union 框把它们合并；
6. 将无答案泄漏的聚焦说明和一至多张 Crop 放在推理轨迹最前面，作为 privileged Teacher 前缀；
7. 对完全相同的 rollout token 做 Teacher/Student 配对重评分，比较 OPD/OPVD 信号是否变强。

整个流程中 Qwen 的显式 thinking mode 都关闭。rollout 中的 2–5 句解释是普通、可见的 assistant 文本，不是 `<think>` 内容。

## 2. 文件

- 主程序：`group_visual_opvd_probe.py`
- Conda/动态库启动封装：`run_group_visual_opvd_probe.sh`
- 纯函数测试：`test_group_visual_opvd_probe.py`
- 默认结果目录：`outputs/group-visual-opvd-probe/`

默认本地资源：

- Qwen：`/root/siton-tmp/yzs/ckpts/Qwen3.5-4B`
- V*Bench：`/root/siton-tmp/yzs/datasets/vstar-bench/data/test-00000-of-00001.parquet`
- Grounder：`IDEA-Research/grounding-dino-base`（官方公开可下载的 Swin-B 强版；HF 名称里的 `base` 指 B 规格，不是更弱的 Swin-T）

版本说明：`DINOv3` 是 Meta 的通用视觉特征骨干，并不是 Grounding DINO v3。IDEA
官方 Grounding DINO 系列已到 1.6 Pro，后继的 DINO-X Pro 检测能力更强，但两者均只通过
DeepDataSpace API 托管，没有公开本地权重。本探针因此使用当前最强的官方可下载本地版
GroundingDINO-B/Swin-B。该 Transformers 权重已经通过 MiHoMo 下载到本机 HF cache，并完成了离线加载验证。

启动封装会使用 `/home/yzs/miniconda3/envs/vision-opd`，并优先加载该环境的 `libstdc++`，避免 `causal_conv1d` 的 `CXXABI_1.3.15` 导入错误。

## 3. 分阶段运行

### 3.1 每题采样 8 条轨迹

两张卡可以分别生成 JSONL shard：

```bash
./run_group_visual_opvd_probe.sh rollout \
  --device cuda:0 \
  --indices 1 4 21 56 \
  --samples-per-question 8 \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-new-tokens 256 \
  --output outputs/group-visual-opvd-probe/groups-gpu0.jsonl

./run_group_visual_opvd_probe.sh rollout \
  --device cuda:1 \
  --indices 116 140 154 189 \
  --samples-per-question 8 \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-new-tokens 256 \
  --output outputs/group-visual-opvd-probe/groups-gpu1.jsonl
```

命令可断点续跑；同一个输出文件里已经成功完成的 index 会被跳过。只有显式传入 `--overwrite` 才会重新生成。

### 3.2 选出有 GRPO 信号的 mixed group

```bash
./run_group_visual_opvd_probe.sh select \
  --groups \
    outputs/group-visual-opvd-probe/groups-gpu0.jsonl \
    outputs/group-visual-opvd-probe/groups-gpu1.jsonl \
  --mode mixed \
  --min-valid-rollouts 8 \
  --output outputs/group-visual-opvd-probe/selected-groups.jsonl
```

本次实际产物已经把两张卡和两道格式修复重采样题合并为
`outputs/group-visual-opvd-probe/groups-merged.jsonl`，所以继续本次实验时也可以直接运行：

```bash
./run_group_visual_opvd_probe.sh select \
  --groups outputs/group-visual-opvd-probe/groups-merged.jsonl \
  --mode mixed \
  --min-valid-rollouts 8 \
  --output outputs/group-visual-opvd-probe/selected-groups.jsonl
```

`mixed` 的含义是同一个 8-rollout group 内 reward 既有 1 又有 0。程序按 reward 方差优先、预测熵次优先排序。`nonuniform` 仅要求预测选项不完全一致，适合诊断，但不能替代主实验的 mixed 标准。

本次 2026-08-24 的首轮采样共跑了 8 题，严格筛出 5 个 mixed group：

- index 140：4/8 正确；
- index 189：4/8 正确；
- index 56：3/8 正确；
- index 154：3/8 正确；
- index 116：6/8 正确。

index 1 和 21 为 8/8 正确，index 4 为 0/8 正确，因没有组内 reward 方差而不送入 Analyzer。格式不合法的采样会在有限预算内重试，因此进入主实验的每组都包含 8 条可解析轨迹。

### 3.3 先预览 Analyzer 请求

在没有 API Key 时可以先生成脱敏请求预览：

```bash
./run_group_visual_opvd_probe.sh analyze \
  --dry-run \
  --analyzer-model gpt-5.6
```

预览不会包含图片的 base64 数据，也不会调用网络。

### 3.4 调外部 Analyzer

拿到服务信息后只放进当前 shell 的环境变量，不写入脚本或 JSONL：

```bash
export ANALYZER_BASE_URL='待填写的 OpenAI-compatible Base URL'
export ANALYZER_API_KEY='待填写的 Key'
export ANALYZER_MODEL='gpt-5.6'

./run_group_visual_opvd_probe.sh analyze
```

Analyzer 看不到 ground-truth label，只看到每条轨迹的预测和二值 reward。其可见输出会做答案泄漏检查；Grounding DINO 查询词和空间选择器属于工具侧私有字段，不直接作为 Student 目标。

### 3.5 Grounding DINO 定位与裁剪

```bash
./run_group_visual_opvd_probe.sh ground --device cuda:0
```

当前规则输出一至三张 per-object Crop。Analyzer 返回多个对象时，程序对每个查询分别选择一个 box，按 `context_margin` 独立扩展、裁剪和放大，保存为 `object-crop-01.jpg`、`object-crop-02.jpg` 等；不会再把相距很远的对象合成一张大 union Crop。每题同时保存总览画框图，方便人工核验“对象是否找准”以及“不同查询是否误落在同一对象上”。

旧 Analyzer 结果中的 `selector.type=union` 会按兼容模式解释为“每个 query 各取一个候选”，但输出仍是彼此独立的 Crop，不再执行几何 union。blank、random、shuffled 对照会严格匹配本题的图片张数和每张图片尺寸，避免多图格式本身成为混淆因素。

### 3.6 无训练 OPD/OPVD 探针

```bash
./run_group_visual_opvd_probe.sh probe --device cuda:0
```

主比较是：

- Student：原图 + 原问题；
- Teacher：原图 + 聚焦说明 + Crop/Zoom + 原问题；
- 两边强制评分同一条已采样 completion，不重新 rollout，保证 token 级配对。

同时跑以下消融（图片数与各 Crop 尺寸匹配）：

- `text_only`：只有聚焦语言；
- `crop_only`：只有 Crop；
- `text_crop`：完整 Teacher 前缀；
- `blank_crop`：控制“多一张图”本身的影响；
- `random_crop`：控制一般放大效果；
- `shuffled_crop`：使用别题 Crop，控制非特异视觉上下文。

结果写入：

- 逐题逐轨迹：`outputs/group-visual-opvd-probe/probe/records.jsonl`
- 聚合结果：`outputs/group-visual-opvd-probe/probe/summary.json`

若要按 8 条 rollout 的正确/错误状态分别观察答案概率，使用轨迹条件化的选项探针。它保留每条 rollout 自己的解释，去掉末尾 `FINAL: X`，再比较各前缀下下一步生成 A/B/C/D 的归一化概率：

```bash
./run_group_visual_opvd_probe.sh trajectory-options \
  --grounded outputs/group-visual-opvd-probe/grounded-local-qwen-multicrop.jsonl \
  --output-dir outputs/group-visual-opvd-probe/trajectory-options-multicrop
```

该指标同时报告正确选项概率、该 rollout 原先所选选项的概率，以及两者 margin 的变化。对于错误 rollout，理想方向是正确选项概率上升、原错误选项概率下降。

## 4. 首轮判据

不能只看 Teacher 与 Student 分布“变得不同”。更有意义的候选信号应同时满足：

1. `text_crop` 的 sampled-token log-prob 增量高于 `blank_crop`、`random_crop` 和 `shuffled_crop`；
2. 增量在失败轨迹上仍可观，而不是只把本来正确的轨迹变得更自信；
3. 正确选项概率相对 plain 上升；
4. 人工查看 Grounding 可视化时，Crop 确实覆盖了解题所需对象或关系；
5. 聚焦说明不包含答案字母、答案文本或最终属性值。

如果只有 KL/JSD 上升而正确选项概率、sampled-token log-prob 或对照差值没有改善，它只能说明前缀扰动了模型，不能说明它是可靠的 self-evolution 信号。
