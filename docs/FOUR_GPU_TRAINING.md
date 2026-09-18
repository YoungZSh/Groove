# 四卡训练

当前入口是 **`scripts/train_a800_4gpu.sh`**。它在一份文件中写明全部参数，直接调用
Python 训练入口，不再调用 `run_grpo_2b.sh` 或 `run_groove.sh`。
旧的四层调用链保存在 `TMP/scripts/` 供历史追溯。

## 启动

在仓库根目录执行，算法可选 `grpo`、`dapo`、`grpo_opsd`：

```bash
TRAINING_MODE=grpo EXPERIMENT_NAME=my-grpo-4gpu-run01 \
  bash scripts/train_a800_4gpu.sh

TRAINING_MODE=dapo EXPERIMENT_NAME=my-dapo-4gpu-run01 \
  bash scripts/train_a800_4gpu.sh

TRAINING_MODE=grpo_opsd EXPERIMENT_NAME=my-opsd-4gpu-run01 \
  bash scripts/train_a800_4gpu.sh
```

每份脚本都可以复制后固定机器和实验参数。Siton 两卡的独立模板是
`scripts/train_siton_2gpu.sh`，不引用四卡脚本。
算法和 Trainer 对应关系、DAPO 过滤字段及关闭超长奖励惩罚的设置见 [训练入口说明](TRAINING_LAUNCHERS.md)。

## 本机四卡参数

| 参数 | 默认值 |
| --- | --- |
| Python | `/ssd/home/zc/miniconda3/envs/groove/bin/python` |
| 模型 | `/ssd/home/zc/yzs/models/ckpts/Qwen3.5-2B` |
| CUDA devices | 0,1,2,3 |
| 全局 prompt batch / PPO mini-batch | 16 / 16 |
| 每题 rollout 数量 | 8 |
| rollout TP | 1，每卡一个副本 |
| prompt / response / 总上下文 | 9216 / 1024 / 10240 |
| Agent / reward workers | 16 / 4 |
| 每卡 actor、log-prob 与 vLLM token 预算 | 65536 |
| vLLM max_num_seqs / memory utilization | 64 / 0.45 |
| Seed / learning rate | 20260904 / 1e-6 |
| 训练 / 验证 temperature | 1.0 / 0，验证不采样 |
| 验证 / 保存间隔 | 10 / 10 updates |
| 保留 actor 检查点数量 | 最近 2 份 + 独立最佳 1 份 |
| Validation batch | null，完整 191 题 |
| Ray 固定内存上限 | null，节点内存 95% |
| OMP_NUM_THREADS | 4，传递到 Ray workers |

Agent 和 reward workers 是 CPU 调度/评分进程，不是 GPU 推理副本。
全局 batch 为 16 时，每步生成 128 条回答，平均每卡 32 条。
DAPO 会过滤并补采，实际生成量可能超过 128 条，但更新时仍为 16 个有效题组。

Token 预算控制每卡动态批处理容量，不会放大 prompt batch，也不会强制填满显存。
Actor 采用四卡 FSDP；vLLM 在训练阶段休眠，更新后同步权重。

四卡验证整集提交，再按 worker 数补齐；评分时去除补齐行，保留原始 191 题。
可显式覆盖 `VAL_BATCH_SIZE`，该值独立于训练 batch 与 agent worker 数量。
验证集始终独立于训练目录；`DATA_DIR` 默认新 4K 数据，`VALIDATION_FILE` 默认 V*Bench。
验证与训练共用答案提取和语义 Judge，但验证仅报告原始语义准确率。
每个验证点的全部回答保存为 `outputs/validation/<EXPERIMENT_NAME>/<step>.jsonl`，
独立于训练 rollout 和 W&B 展示样例数量；旧选项规则结果只作为诊断保留。

## 网络与服务

单机通信设置：

```bash
NCCL_SOCKET_IFNAME=lo
NCCL_IB_DISABLE=1
```

两种大小写的 `NO_PROXY` 补齐 loopback 和本机 IP，并与 NCCL、OMP 设置一起传给 Ray。
现有 HTTP/SOCKS 代理配置由进程环境继承，脚本不读取凭证或改动代理服务。

训练语义 Judge 需要 `127.0.0.1:8002/v1`。GRPO + OPSD 额外需要
`127.0.0.1:8011` 的 DINO 和 `127.0.0.1:8012` 的 OCR；这两个服务保持单 worker。
本机 `6xA800` SSH 别名现在直接使用 `ProxyJump school-jump`，目标 `10.184.17.171:22`；
服务转发由 `groove-6xa800-tunnel` 管理，不再依赖本机 17824 中转。

## 配置检查与日志

```bash
GROOVE_DRY_RUN=true TRAINING_MODE=grpo EXPERIMENT_NAME=check-four-gpu-grpo \
  bash scripts/train_a800_4gpu.sh
```

Dry-run 可以在较小 GPU 配额下完成，只校验配置，不启动 Ray、不加载模型。
真实启动先验证 CUDA 可见设备数量；不自动停止任何现有服务。
完整日志追加到 `outputs/logs/<EXPERIMENT_NAME>.log`，训练失败退出码通过 tee 保留。
新训要求显式名称，默认关闭 auto-resume；已有同名检查点会被拒绝。

## 训练后的独立推理服务

`scripts/serve_2b_after_training.py --config <任务配置.json>` 保留为独立监督工具。
它确认最终步、最终验证、四卡完整检查点和原训练进程退出，再等待 GPU 释放并启动
GPU 0～3 的四个 Qwen3.5-2B 基础模型 TP=1 服务，端口 8100～8103。
每个实例预留 69 GiB KV 缓存，实际显存至少 73728 MiB，并通过推理请求后才标为 READY。

已知运行限制：W&B 收尾上传若长时间重试，会延迟训练进程退出，进而延迟上述自动切换。
本次脚本整理没有更改正在运行的推理服务或其监督器。
