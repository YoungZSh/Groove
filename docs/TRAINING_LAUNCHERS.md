# 扁平训练入口

## 组织方式

`scripts/train_a800_4gpu.sh` 与 `scripts/train_siton_2gpu.sh` 都是完整、独立的实验脚本。
每份文件内直接列出机器参数、算法选择、数据路径、奖励、优化器、采样、验证、日志和 Ray 网络参数，
最后直接调用 `python -m groove.verl_entrypoint`。没有中间 shell 启动器，也不读取共享机器 shell 配置。
需要某台机器、某种方法的固定实验入口时，复制其中一份并修改顶部参数即可。

旧链路 `run_2b_4gpu.sh -> run_grpo*_2b.sh -> run_groove.sh` 已完整归档到
`TMP/scripts/`，用于追溯旧实验，不再作为当前启动路径。

## Trainer 分流

共同入口只负责提示词适配、配置验证、Ray 内存设置和选择 TaskRunner：

| 模式 | 训练器 | 组过滤 | 裁剪下限 / 上限 | Reference KL |
| --- | --- | --- | --- | --- |
| `grpo` | VERL `TaskRunnerV1` / `PPOTrainerSync` | 关闭 | 0.2 / 0.2 | 0.01 |
| `dapo` | VERL `TaskRunnerV1` / `PPOTrainerSync` | 开启 | 0.2 / 0.28 | 关闭 |
| `grpo_opsd`（别名 `groove`） | `GrooveTaskRunner` / `GrooveRayPPOTrainer` | 关闭 | 0.2 / 0.2 | 0.01 |

所有模式均使用 GRPO 优势估计和 vanilla PPO policy loss，聚合方式是 `token-mean`。
GRPO + OPSD 保持未中心化 token credit、系数 0.01、不裁剪；同奖励题组仍进入 Analyzer。
`trainer_routing.py` 拒绝“旧训练器开启组过滤”及“OPSD 同时开启 V1/组过滤”的无效组合，
避免配置通过但算法未执行。

原生 V1 需要 **TransferQueue 0.1.10**。本机 groove 环境已安装；其他机器需要在其训练环境安装：

```bash
python -m pip install 'TransferQueue==0.1.10'
# 项目也提供同一依赖声明：pip install -e '.[native-training]'
```

V1 的实现复用仓库内 `src/verl/trainer/ppo/v1/`；没有另写 DAPO 训练循环。
原生路径保留奖励分项均值及 rollout 中的 accuracy、格式、重复和长度诊断。
rollout 的 `score` 保存实际优化的最终奖励，奖励函数原始返回值另存为 `reward_function_score`。

## 共同实验参数

- 模型 Qwen3.5-2B；训练 seed 20260904；全局 prompt batch / PPO mini-batch 均为 16。
- 每题 8 个回答；TP=1；学习率 1e-6；PPO epoch 1；默认训练 epoch 1。
- prompt / response / 总上下文上限：9216 / 1024 / 10240。
- 普通推理文字 + 末尾 `<answer>`；保留原生空 think prefill，`enable_thinking=false`。
- 训练 temperature 1.0；验证 temperature 0、每题 1 个回答、不采样。
- 验证、保存间隔均为 10 updates；保留最近两份 actor 检查点。
- GRPO/DAPO 默认 W&B online，可显式设置 offline；GRPO + OPSD 默认 offline。
- 默认 `RESUME_MODE=disable`，每次要求新的实验名称；显式续训参数在脚本顶部。

| 机器配置 | 本机 A800 四卡 | Siton 两卡 |
| --- | --- | --- |
| Python | `/ssd/home/zc/miniconda3/envs/groove/bin/python` | `/home/yzs/miniconda3/envs/vision-opd/bin/python` |
| 模型 | `/ssd/home/zc/yzs/models/ckpts/Qwen3.5-2B` | `/root/siton-tmp/yzs/ckpts/Qwen3.5-2B` |
| CUDA devices | 0,1,2,3 | 0,1 |
| Agent / reward workers | 16 / 4 | 8 / 1 |
| Actor/log-prob、vLLM token 预算 | 65536 | 32768 |
| Validation batch | null，完整 191 题 | 8 |
| Ray 固定内存上限 | null，节点内存 95% | 220 GiB，保留 4 GiB headroom |

两份都是单机脚本，NCCL 使用 loopback、关闭 IB，并将网络及 `OMP_NUM_THREADS=4`
设置传给 Ray workers。实际运行会检查所选 CUDA 数量；dry-run 跳过硬件数量检查。

## 数据与服务

GRPO、DAPO 默认 `data/vstar_grpo_4000_seed20260917/train.parquet`；
GRPO + OPSD 默认 `data/vstar_opsd_4000_seed20260917/train.parquet`。
`DATA_DIR` 可独立覆盖。旧 1979 条训练文件没有修改。
另一台机器需要先准备或传输相应数据；OPSD 的 `extra_info.image_path` 必须指向该机器的本地图片。

三种模式均使用独立的 `data/vstar_bench/validation.parquet`，完整 191 题；
可通过 `VALIDATION_FILE` 覆盖。Benchmark 按选项准确率判分，不经过远程 Judge，也不应用训练惩罚。

训练语义 Judge 为 `127.0.0.1:8002/v1` 的 Qwen3.8-27B。
GRPO + OPSD 额外使用同一服务的 Analyzer，以及 8011 DINO、8012 OCR。
DINO/OCR 保持单 worker；启动脚本不会部署或重启远程服务。

## DAPO 的明确语义

- 由原生 V1 ReplayBuffer 过滤组内 **最终训练奖励** 完全相同的题组，并补采到 16 组。
- 过滤字段为 `training_reward`，包含语义、格式、重复和可选长度惩罚；`accuracy` 保持独立。
- 默认最大回答 1024，长度缓冲区 128：896 token 开始线性惩罚，1024 token 时为 -1。
  `DAPO_OVERLONG_ENABLED=false` 可用于关闭长度惩罚的单独对照。
- `VisualQARewardManager` 仅对训练应用该惩罚；验证记录保持原始准确率。
- `max_inflight_gen_batches=1` 限制同时生成的批量，V1 不执行旧的 `max_num_gen_batches` 总重试上限。
  后者固定写为 0，避免假设它能终止补采。
- V1 默认按 `4000 // 16 = 250` 个**优化更新**计算一轮预算；DAPO 补采可能多次遍历数据。
  250 次更新不等于只生成过 4000 道题。比较成本时需看过滤计数、rollout/token 数和墙钟时间。

## 检查与使用

```bash
GROOVE_DRY_RUN=true TRAINING_MODE=dapo EXPERIMENT_NAME=check-dapo-config \
  bash scripts/train_a800_4gpu.sh
```

当前整理通过配置解析、CPU 奖励/过滤/路由测试，以及独立 CPU Ray 上的 TransferQueue
张量、奖励元数据和 PIL 图片往返检查。原生 V1 的 GRPO、DAPO 尚未在本次整理中
启动正式 GPU 训练。原有 GRPO 历史成绩来自旧 Trainer 路径，
新原生路径应先完成实际训练验证，再比较算法效果。

训练结束后的独立 vLLM 监督器仍是 `scripts/serve_2b_after_training.py`，与训练入口分开运行。
已知限制：它等待训练进程退出，W&B 收尾网络重试可能延后 GPU 释放和推理启动。
