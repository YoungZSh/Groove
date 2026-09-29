# VOPD 的无框 GRPO 数据适配

`scripts/prepare_vision_opd_grpo.py` 将官方 JSONL 转为当前训练使用的 parquet。
Student 只读取 `original_images` 原图、问题和完整选项；图像字节不修改。
优先使用 `extra_info.question`，删除官方红框句和旧字母输出指令，使用项目的
`reasoning_answer` 系统提示词及末尾 `<answer>` 标签。原答案标签保持不变，
语义 Judge 的参考答案为字母加选项全文，例如 `(D) purple`。

官方红框图、裁剪与 bbox 只写入独立的 `lineage.jsonl`，不进入训练 parquet。
不使用官方裁剪代替原图，也不自动生成目标描述或改写答案。验证继续使用独立的
`data/vstar_bench/validation.parquet`，不从训练数据新划分验证集。

传入 `--audit` 时，先核验抽查记录的来源哈希、行号和原图名称，然后隔离
`confirmed_defect` 和 `rewrite_required` 样本；`needs_review` 或
`rewrite_recommended` 本身不会导致删除。题干仍直接引用 highlighted/marked/boxed
area/region/object 或 red bounding box 时也隔离，避免删除提示词后直接猜测目标。
此规则不能证明其余题目已经完成逐图复核。

当前 6,220 条 V*Bench 原图去重版中，已有抽查识别的 7 条明确缺陷、14 条需改写
目标描述，以及另 1 条直接依赖 highlighted area 的题目被隔离，剩余 6,198 条候选。
排除清单、来源索引和原因均单独保存，原始数据不变，已有输出目录禁止覆盖。

本次用户指定随机抽 3,000 条、熵和 KL 都为零：

```bash
PYTHONPATH="$PWD/src" /data/home/yangzesheng/.conda/envs/groove/bin/python \
  scripts/prepare_vision_opd_grpo.py \
  --source data/vision_opd_6k_vstar_disjoint/train.jsonl \
  --output-dir data/vision_opd_grpo_3000_seed20260904 \
  --audit outputs/data-audits/vision-opd-manual100-20260928-162543/audit100.json \
  --sample-size 3000 --seed 20260904
```

抽样为固定种子的均匀无放回抽样，采样顺序和完整源索引写入 manifest；训练仍执行
现有 seed=20260904 的 shuffle。两卡入口设置 `TRAINING_MODE=grpo`、
`DATA_DIR` 指向新目录、`LEARNING_RATE=1e-6`、`ENTROPY_COEFF=0`。
两卡入口原本已关闭奖励 KL 和 actor KL，系数均为零，不启用 OPSD。
32 组/步、8 回答/组、一轮 epoch 保持不变；3,000 条的一轮为 93 次完整更新，
按现有 `drop_last=True`，shuffle 后末尾 24 条当轮不参与更新。

正式运行使用新的实验名和独立的源码快照，`scripts/training_service_handoff.py`
在停止 GPU 0/8000、GPU 3/8003 推理前先启动独立恢复监控。训练正常或异常退出后，
只清理本次带唯一标记的进程并恢复原启动脚本，实际请求通过后记录 `RESTORED`。
GPU 1、2 不参与；恢复需要模型加载时间。

## 冻结 ViT、训练 merger 和 LLM 的无熵对照

2026-09-29 用户指定的新实验保持同一份 3,000 条数据、基础 2B 模型和学习率，
关闭熵奖励与 KL，同时把 PPO clip 上限从 0.2 改为 0.3、下限保持 0.2。
冻结范围与 clip 上限同时变化，结果不能单独归因于冻结 ViT。

两卡启动器通过以下环境覆盖选择这组行为，默认训练行为保持不变：

```text
TRAINING_MODE=grpo
ENTROPY_COEFF=0
FREEZE_VISION_TOWER=true
TRAIN_VISION_MERGER=true
PPO_CLIP_RATIO_HIGH=0.3
```

`FSDPActorConfig` 把两个视觉开关传给实际 FSDP engine。加载模型之后、FSDP 包装之前，
先冻结 `model.visual` 全部参数，再只解冻 `model.visual.merger`；
`model.language_model` 与 `lm_head` 保持可训练。当前实现明确支持 Qwen3.5 dense/MoE
布局的 FSDP1 全参数训练；不支持的布局、LoRA 或引擎组合会报错，避免空开关。
FSDP1 为混合冻结/训练参数使用 `use_orig_params=True`，启动器在启用冻结时默认设置它。
优化器排除冻结参数；视觉前向仍然执行，Student 的输入边界不变。

每个 rank 在初始化时打印 `vision_freeze_parameters`，报告 ViT、merger、LLM 的
总参数量和可训练量。第一次有非零有限梯度的更新打印 `vision_freeze_update_audit`：
验证冻结参数没有梯度、优化前后本 rank 冻结分片的 SHA256 完全相同，并记录 merger/LLM
局部梯度范数和参数样本变化。merger 可能只分布在其中一个 rank，应汇总两卡判断其更新。
后续更新继续依赖 `requires_grad=False` 和优化器参数排除保证冻结。
实现及回归检查见 `src/verl/workers/utils/vision_freezing.py` 和 `tests/test_vision_freezing.py`。
