# 四卡 GRPO 与 GRPO + OPSD

统一入口是 `scripts/run_2b_4gpu.sh`，用于同一个容器中的单机四卡训练。
它根据 `TRAINING_MODE` 调用已有的 2B 实验脚本，并保留其奖励、采样和优化设置。

## 启动方式

在项目根目录执行，每次运行指定一个新的 `EXPERIMENT_NAME`。长时间任务放在
tmux 或既有 supervisor 中运行；脚本会把完整输出追加到
`outputs/logs/<EXPERIMENT_NAME>.log`，并保留训练进程的失败退出码。

```bash
# GRPO
EXPERIMENT_NAME=qwen35-2b-grpo-4gpu-run01 \
TRAINING_MODE=grpo \
bash scripts/run_2b_4gpu.sh

# GRPO + OPSD：使用单独的新实验名
EXPERIMENT_NAME=qwen35-2b-grpo-opsd-4gpu-run01 \
TRAINING_MODE=grpo_opsd \
bash scripts/run_2b_4gpu.sh
```

`TRAINING_MODE` 默认为 `grpo`。两种模式默认都从
`/root/siton-tmp/yzs/ckpts/Qwen3.5-2B` 开始，方便分别比较。
如果后续实验要从 GRPO 训练后的模型开始，显式设置 `MODEL_PATH`，指向已经合并、
导出为 Hugging Face 格式的模型目录，并保留 tokenizer、processor 和原生
`chat_template.jinja`。这会初始化一个新实验；它与恢复原训练的优化器状态不同。

## GPU、数据与共同参数

| 参数 | 四卡入口默认值 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3`，可指定其他四张卡的编号或 UUID |
| `N_GPUS` / `trainer.n_gpus_per_node` | 4 |
| `trainer.nnodes` | 1 |
| `SEED` | 两种模式均为 `20260904`，可显式覆盖 |
| 全局 prompt batch / PPO mini-batch | 16 / 16 |
| 每组 rollout 数量 | 8 |
| rollout tensor parallel size | 1 |
| 最大回答长度 | 1024 |
| 最大 prompt / 模型上下文长度 | 9216 / 10240 |
| Validation 集合 / batch | 完整 V*Bench 191 题 / 16 |
| rollout 调度 worker / 本地 reward worker | 16 / 4 |
| 每卡 actor、reference、Teacher log-prob token 预算 | 65536 |
| 每个 TP=1 vLLM 副本的批处理 token 预算 | 65536 |
| 学习率 / reference KL 系数 | `1e-6` / `0.01` |
| 训练 / 验证 temperature | `1.0` / `0`，验证关闭采样 |
| 回答格式 | 普通推理文字，末尾一个 `<answer>...</answer>` |
| W&B | offline |

扩展到四卡时，全局 batch 保持 16，每步仍采样 128 条回答。
GRPO + OPSD 沿用 `0.01` 的 OPSD advantage 系数、不做 OPSD clipping。
原有 2B 入口仍默认使用两卡；其 seed 默认值仍分别为 22 和 20260904。

### Worker 与 token 预算

四卡入口的 rollout 调度 worker 为 16，本地 reward worker 为 4；两卡入口默认分别为 8 和 1。
这些是 CPU 侧的调度/评分进程；GPU 上仍是四个 TP=1 的 Qwen3.5-2B rollout
副本，每张卡一个。actor 更新使用四卡 FSDP，更新后的权重同步给 rollout 副本。

`ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU` 和 `ROLLOUT_MAX_NUM_BATCHED_TOKENS`
从 32768 提升到 65536。前者是每卡动态微批次中 prompt 和 response 的 token
总预算，reference 与当前策略 Teacher 的 log-prob 计算也继承它；后者是 vLLM
单次调度的批处理预算。它们都是处理容量上限，实际批次较小时不会强行填满预算。
单条回答仍限制为 1024 token，每步仍是 16 组 × 8 条 = 128 条回答。

原有两卡入口继续使用 32768 的预算和 8/1 的 worker 默认值。
四卡入口支持以下独立覆盖：

```bash
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=65536 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 \
ROLLOUT_AGENT_NUM_WORKERS=16 REWARD_NUM_WORKERS=4 \
EXPERIMENT_NAME=qwen35-2b-grpo-4gpu-run02 \
bash scripts/run_2b_4gpu.sh
```

Validation batch 默认跟随 `ROLLOUT_AGENT_NUM_WORKERS`，避免每个完整 batch
为了分发到 worker 而被补成更大的重复批次。191 题的最后一个 batch 仍会在评分
前移除补齐样本。`OMP_NUM_THREADS` 默认为 4，并显式传给 Ray worker，约束多个
CPU 进程的线程数；需要时可以覆盖。

`ROLLOUT_MAX_NUM_SEQS` 保持 64：当前每步平均分配到每张卡的回答只有 32 条。
Analyzer 最大并发保持 16 组，与全局 prompt batch 一致；DINO/OCR 沿用现有
单 worker 服务。增加本地 reward worker 不会改变远程服务进程数。
当前 reward 流式接口通过每个进程的默认线程池调用语义 Judge；本机每个线程池为
32 个线程，4 个 worker 可以承接当前每步的 128 条回答。远端服务会按自身的
`max_num_seqs` 调度和排队，继续增加本地 worker 不一定提高吞吐。
实际吞吐和峰值显存仍需在四卡环境实测，预算翻倍不代表速度必然翻倍。

GRPO 训练数据默认使用 `data/vstar_grpo_2200_seed20260904/train.parquet`；
GRPO + OPSD 默认使用 `data/vstar_opsd_2200_seed20260904/train.parquet`。
可通过 `DATA_DIR` 指定训练数据目录，两者原有的 1979 条训练样本保持不变。

两种模式的 Validation 都使用 `data/vstar_bench/validation.parquet` 中完整的
191 道 V*Bench 题目；`VALIDATION_FILE` 可独立覆盖验证文件路径。首次准备时执行：

```bash
/home/yzs/miniconda3/envs/vision-opd/bin/python scripts/prepare_vstar_validation.py
```

准备脚本保留原始图片字节、选项、答案和顺序，生成哈希清单，并拒绝覆盖已有文件。
旧的 220 条验证集和历史记录保留。V*Bench 按选项匹配统计正确率，格式有效率单独
记录；训练阶段仍使用原有语义 Judge 和格式奖励。

主指标 `val-core/vstar_bench/reward/mean@1` 就是这 191 题的正确率。验证采用
temperature 0、每题一个回答，四卡模式每批 16 题，最后不足 16 题的一批也会保留。
9216 token 的 prompt 空间用于容纳原始高分辨率图片，不缩减训练 batch 或回答长度。

## 本机网络配置

本机另一项目的参考脚本：

- `/root/siton-tmp/yzs/GLaQ/mmcot/scripts/run_rl_glq_grpo_text_only.sh`
- `/root/siton-tmp/yzs/GLaQ/mmcot/scripts/run_rl_glq_2a800.sh`
- `/root/siton-tmp/yzs/GLaQ/mmcot/scripts/run_rl_glq_depo.sh`

这些脚本记录了 NCCL 自动选择 IB/RoCE 接口后初始化卡住的问题，采用以下设置：

```bash
NCCL_SOCKET_IFNAME=lo
NCCL_IB_DISABLE=1
```

四卡入口默认沿用这两个值，也允许显式覆盖。前者选择 loopback 接口，后者关闭
NCCL 的 IB/RoCE 传输；配置含义见
[NCCL 环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。
这套默认值用于同一节点、同一网络命名空间中的进程。

脚本同时补齐 `NO_PROXY` / `no_proxy` 中的 loopback 地址和本机 IP，保留用户已有
的排除项。上述四个环境变量都会通过
`ray_kwargs.ray_init.runtime_env.env_vars` 显式传给 Ray worker。
同一处也转发四卡入口的 `OMP_NUM_THREADS`。
NCCL 的 P2P、共享内存及 cuMem 设置沿用环境和库默认值。

GRPO 仍需要 `127.0.0.1:8002/v1` 的语义 Judge；GRPO + OPSD 还需要 Analyzer、
`127.0.0.1:8011` 的 GroundingDINO 和 `127.0.0.1:8012` 的 OCR。
四卡入口沿用既有服务端点及并发设置。

2026-09-16 已验证 Judge 的连接链路：A100 上的 `127.0.0.1:8002` 经 SSH 转发至
`6xA800`（`10.184.17.171`）的 `127.0.0.1:8000`，服务名称为 `Qwen3.8-27B`。
当时远端配置为 TP=2、`max_num_seqs=64`、`max_num_batched_tokens=32768`。
一次通过实际训练奖励适配器发出的合成判分请求成功返回正确率 1；这只验证连通和
判分协议，不代表持续吞吐测量。

2026-09-16 的本机检查中，容器主机名无法解析，`torchrun --standalone`
停在基于主机名的 rendezvous 阶段。改用 `--master-addr=127.0.0.1` 和单独的
空闲端口后，两张可见 GPU 的 NCCL all-reduce、broadcast 均通过。
项目入口使用 Ray；现有 VERL worker 通过 `ray.util.get_node_ip_address()`
选择通信主地址，四卡脚本继续沿用这条路径。

## 配置验证

```bash
GROOVE_DRY_RUN=true \
EXPERIMENT_NAME=check-grpo-4gpu-run01 \
TRAINING_MODE=grpo \
bash scripts/run_2b_4gpu.sh

GROOVE_DRY_RUN=true \
EXPERIMENT_NAME=check-grpo-opsd-4gpu-run01 \
TRAINING_MODE=grpo_opsd \
bash scripts/run_2b_4gpu.sh
```

Dry-run 可以在只有两张可见 GPU 的容器中完成，不启动 Ray worker、加载权重或
创建训练日志。正常启动则先检查 CUDA 是否实际看得到所选的四张卡，检查失败会
在启动训练前退出。修改 `CUDA_VISIBLE_DEVICES` 无法增加容器获得的物理 GPU。
Dry-run 验证的是配置，四卡通信和训练吞吐仍需在四卡环境中验证。
