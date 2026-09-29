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
