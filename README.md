# GROOVE

本项目在视觉问答上训练 Qwen3.5-2B，提供 GRPO、DAPO 和 **GRPO + OPSD** 三种实验模式。
Student 始终只接收原始图片和问题；Analyzer、裁剪、OCR 和额外视觉证据仅用于训练中的 Teacher。

## 独立启动脚本

每份脚本直接列出完整参数并启动 Python，没有相互引用的 shell 调用链：

- [`scripts/train_a800_4gpu.sh`](scripts/train_a800_4gpu.sh)：本机 A800 四卡。
- [`scripts/train_siton_2gpu.sh`](scripts/train_siton_2gpu.sh)：Siton 两卡。

两份都可以选择 `grpo`、`dapo`、`grpo_opsd`。也可以复制脚本，为某台机器、某种训练固定参数。

```bash
# 配置检查，不启动训练
GROOVE_DRY_RUN=true TRAINING_MODE=grpo EXPERIMENT_NAME=check-grpo-config \
  bash scripts/train_a800_4gpu.sh

# 正式实验，每次使用新名称
TRAINING_MODE=grpo EXPERIMENT_NAME=my-grpo-4k-run01 \
  bash scripts/train_a800_4gpu.sh
```

启动器默认从基础模型新训，拒绝无意覆盖同名检查点。完整设置见
[脚本索引](scripts/README.md)、[训练入口说明](docs/TRAINING_LAUNCHERS.md)
和[四卡运行说明](docs/FOUR_GPU_TRAINING.md)。

## 算法和执行路径

| 模式 | Trainer | 关键设置 |
| --- | --- | --- |
| GRPO | VERL V1 同步 Trainer | 16 题 × 8 回答；token-mean；reference KL 0.01 |
| DAPO | 同一 VERL V1 同步 Trainer | 动态过滤并补采；clip 0.2 / 0.28；无 reference KL；关闭超长奖励惩罚 |
| GRPO + OPSD | GrooveRayPPOTrainer | 在 GRPO 优势上叠加未中心化 OPSD token credit |

原生 V1 依赖 `TransferQueue==0.1.10`，对应项目可选依赖 `native-training`。
当前配置与 CPU 行为测试已验证；迁移后的原生 V1 GPU 训练还需要实际运行验证。
历史 GRPO 运行使用的是旧 Trainer 路径，不能把它当作新路径的实测结果。

GRPO + OPSD 的优化保持：

```text
delta_t = stopgrad(log p_teacher(y_t) - log p_student(y_t))
A_OPSD,t = evidence_mask * delta_t
A_total,t = A_GRPO + 0.01 * A_OPSD,t
L_actor = VERL_vanilla_PPO(A_total) + 0.01 * low_var_reference_KL
```

Teacher 使用更新前的当前 Student 权重，对相同的 sampled tokens 评分。
OPSD 不增加独立蒸馏损失、不做中心化、不按正确性门控；同奖励题组仍可贡献 OPSD 信号。
Analyzer 或视觉工具失败时回退到 GRPO。

## 数据和奖励

- 当前默认训练数据：GRPO/DAPO 的 `data/vstar_grpo_4000_seed20260917`，
  GRPO + OPSD 的 `data/vstar_opsd_4000_seed20260917`。通过 `DATA_DIR` 覆盖。
- 验证：`data/vstar_bench/validation.parquet` 的全部 191 题，通过 `VALIDATION_FILE` 独立覆盖。
- 训练使用远程语义 Judge，`score = accuracy + 0.2 * format_penalty`。
  普通推理后接一个终止的 `<answer>...</answer>`；严重重复不能得到正奖励，原始 accuracy 保留。
- 所有模式均关闭超长奖励惩罚；DAPO 按最终优化奖励的组内差异过滤。
- V*Bench 验证只计算确定性选项准确率，不应用格式、重复或长度训练惩罚。

所有方法保留 1024 token 回答上限；训练 temperature 1，验证 temperature 0。
GRPO + OPSD 的远程 Analyzer 与 Judge 为 Qwen3.8-27B；DINO/OCR 仍为单 worker 服务。

## 目录

```text
scripts/          独立实验入口、常用数据准备、评测和运行管理
src/groove/       提示词、奖励、Trainer 路由、Analyzer、证据和 OPSD 优势
src/verl/         本仓库实际使用的 VERL 运行时及必要适配
remote_tools/     远程视觉服务源码
configs/          项目 Hydra 训练配置；独立启动脚本覆盖实验参数
tests/            CPU 行为测试
docs/             架构和运行说明
TMP/scripts/      一次性筛选、探测、旧启动链及旧 README
outputs/          运行日志、rollout、证据与 W&B 本地记录
checkpoints/      训练检查点
```

旧数据、模型、检查点和运行记录保持原位置。历史配置见
[TMP/scripts/README.md](TMP/scripts/README.md)；当前实现以 `src/` 和新启动器为准。

## 测试

本机已建立的环境：

```bash
PYTHONPATH="$PWD/src" /ssd/home/zc/miniconda3/envs/groove/bin/python \
  -m unittest discover -s tests -v
bash -n scripts/train_a800_4gpu.sh
bash -n scripts/train_siton_2gpu.sh
git diff --check
```

Siton 使用 `/home/yzs/miniconda3/envs/vision-opd/bin/python`，无需复制本机 Conda 路径。
