# 扁平训练入口

## 组织方式

`scripts/train_a800_4gpu.sh` 与 `scripts/train_siton_2gpu.sh` 都是完整、独立的实验脚本。
每份文件内直接列出机器参数、算法选择、数据路径、奖励、优化器、采样、验证、日志和 Ray 网络参数，
最后直接调用 `python -m groove.verl_entrypoint`。没有中间 shell 启动器，也不读取共享机器 shell 配置。
需要某台机器、某种方法的固定实验入口时，复制其中一份并修改顶部参数即可。

### 本机两卡、batch 32、无 KL

`scripts/train_a800_2gpu_nokl.sh` 是完整独立的本机实验脚本，默认只选择 GPU `0,3`。
默认 Python 为 `/data/home/yangzesheng/.conda/envs/groove/bin/python`，
模型为 `/data/home/yangzesheng/models/ckpts/Qwen3.5-2B`，
仍支持通过 `PYTHON_BIN`、`MODEL_PATH` 覆盖。
本机两卡默认 `NCCL_CUMEM_HOST_ENABLE=0` 并传入 Ray worker，规避本机驱动在
`ncclCuMemHostEnable/cuMemCreate` 的初始化崩溃；此设置仅改变通信内存分配路径。
GPU 0、3 同时默认 `NCCL_P2P_DISABLE=1`，通过 SHM 通信，规避实测的直接 P2P all-reduce 超时。
这两个设置下已通过双卡 NCCL、FSDP 初始化和冻结 Teacher 评分/每 5 步同步测试。
它使用全局 prompt batch 和 PPO mini-batch 均为 `32`，每题 `8` 条 rollout，
回答上限 `1024`、一个 PPO epoch；4,000 条训练数据的一轮为 125 步。
下一次纯 GRPO 实验恢复学习率 `1e-6`、保留熵奖励系数 `0.001`，
使用原生 actor 损失 `L_PPO - 0.001 * H`，不改变 Judge 分数或 GRPO 优势。
GRPO + OPSD 保持学习率 `1e-6`、熵系数 `0`。
两种模式均可用 `LEARNING_RATE`、`ENTROPY_COEFF` 覆盖，命令行 Hydra 参数仍具有最高优先级。
复现旧纯 GRPO 配置时显式使用 `LEARNING_RATE=1e-6 ENTROPY_COEFF=0`。
按本次运行要求，两卡纯 GRPO 默认 `GROOVE_JUDGE_PROVIDER=qwen`，Judge 使用
`http://127.0.0.1:8005/v1`，端点实际登记模型名为 `Qwen3.8-27B`。
`GROOVE_JUDGE_BASE_URL`、`GROOVE_JUDGE_MODEL` 可覆盖，并随配置传给 Ray worker；凭证不写入配置。
Qwen 保留 `enable_thinking=false`。GRPO + OPSD 默认仍使用 `8002/v1`，Analyzer 地址不变。
复现上一轮半学习率/Gemini 配置时显式设置 `LEARNING_RATE=5e-7 GROOVE_JUDGE_PROVIDER=gemini`；
Gemini 从 `GROOVE_JUDGE_ENV_FILE`（默认仓库 `.env`）读取 `OPENAI_BASE_URL`、
`OPENAI_API_KEY` 和 `OPENAI_MODEL`，不把凭证写入 Hydra 配置或日志。
Gemini 请求不传任何 thinking / reasoning_effort / chat_template_kwargs 参数，使用服务默认强度；
Qwen 和 Gemini Judge 的默认温度统一为 0.3；512-token 预算、唯一末尾 `Judgement: 0/1`、
首次加 5 次重试及耗尽归零策略保持不变。
此温度调整适用于后续运行，已启动运行的源码快照保持原协议；跨温度比较须统一重评。
训练及 V*Bench 验证共用这一 Judge，仍分别使用塑形奖励与原始语义准确率。
GRPO + OPSD 默认继续使用 `GROOVE_JUDGE_PROVIDER=qwen`；可显式选择 provider。
本次与上一轮同时改变学习率和 Judge，不能把分数差异全部归给学习率或熵奖励。
每 5 步验证和保存一次，对应处理 160 道训练题，保持与原 batch 16、每 10 步验证相同的题数间隔。
默认 `TRAINING_MODE=grpo_opsd`（别名 `groove`），运行 GRPO + OPSD 的仅正优势 RLSD 模式。
纯 GRPO 对照需显式设置 `TRAINING_MODE=grpo`；每次运行仍须指定新的 `EXPERIMENT_NAME`。
两种模式的损失 KL 和奖励 KL 都关闭，两个系数均为 0，因此不加载参考策略。
现有四卡及 Siton 脚本的默认 KL 设置保持原样。

两卡的 actor/log-prob 和 vLLM 单卡 token 预算均为 `32768`，通过动态 micro-batch
控制峰值显存，不改变全局优化 batch。仍使用 16 个 agent worker、4 个 reward worker，
验证一次提交全部 191 题，验证温度为 0。完整验证回答、最佳检查点及 W&B 日志照常保存。

```bash
EXPERIMENT_NAME=qwen35-2b-rlsd-positive-nokl-2gpu-unique-run \
  bash scripts/train_a800_2gpu_nokl.sh

TRAINING_MODE=grpo EXPERIMENT_NAME=qwen35-2b-grpo-nokl-2gpu-unique-run \
  bash scripts/train_a800_2gpu_nokl.sh
```

纯 GRPO 使用原生 V1 Trainer，默认在线 W&B；GRPO + OPSD 使用项目 Trainer，W&B 离线。
训练集分别默认选择 `vstar_grpo_4000_seed20260917` / `vstar_opsd_4000_seed20260917`，
均支持 `DATA_DIR` 覆盖，验证文件仍独立使用全部 191 题。
本机两卡试验采用 **40 步线性衰减**，Teacher 每 5 步同步：
`lambda_s = 0.5 * max(1 - global_step / 40, 0)`。
第 40 步起权重归一，跳过 Analyzer/Teacher 评分，使用普通 GRPO 更新。
在这一轮 125 步训练中，40 步占 32%。可用 `RLSD_LAMBDA_DECAY_STEPS` 独立调整；
四卡和 Siton 默认仍为 50 步，衰减设置不按 batch 或总训练步数自动缩放。
设置 `OPSD_ADVANTAGE_MODE=opsd` 可以运行相同两卡、batch 32、无 KL 的原加法对照。

当前 GPU 0、3 运行可使用 `scripts/training_service_handoff.py` 管理训练与推理接管。
它读取单次运行的 JSON 配置，核验原服务的 PID/启动 tick、所有者、完整命令、GPU UUID 和 tmux pane，
先启动独立恢复监控，再停止原服务。训练使用独立的本地 Ray 实例、临时目录和唯一进程标记。
训练正常结束、初始化失败、被终止，或管理训练的进程被强制结束后，独立监控只清理本次带标记的进程，
等待 GPU 与端口释放，使用保存的原始启动脚本恢复 GPU 0/8000、GPU 3/8003 的推理服务。
模型、上下文长度、端口及推理参数沿用原服务；恢复后执行实际推理请求确认可用。
这一接管需要用户明确授权停止服务和启动训练；GPU 1、2 不参与。模型重载期间存在短暂启动时间。

用户明确要求续训时，接管配置可设置 `RESUME_MODE=resume_path` 与绝对路径
`RESUME_FROM_PATH=.../global_step_N`，同时使用新的实验名和进程标记，避免覆盖旧日志或回答。
接管前核验各 rank 的模型、优化器、随机状态、world size 和 `data.pt`；只恢复完整检查点。
原最佳快照及其阈值会独立复制到新分段，后续最佳轮换不影响旧运行。
这一步在停止推理服务前完成。`TOTAL_STEPS` 仍指整个训练的目标步数，
例如从 Step 5 恢复、目标 125 时，从 Step 6 继续；可以关闭重复的训练前验证。
接管入口不接受隐式 `RESUME_MODE=auto`，也不复用已有输出实验目录。

历史 GPU 1、2 训练配置中，当 GPU 3 需要持续提供推理时，服务监督器应拆分为 GPU 3 与 GPU 1–2 两个实例。
GPU 3 实例通过 `existing_services` 接管原 PID，不重启服务。训练前只停止 GPU 1–2
服务；训练结束后只恢复这两张卡，其恢复配置使用 `training_world_size=2`。
不要正常终止仍管理 GPU 3 的共享监督器：它的清理逻辑会停止所有受管服务。

用户明确要求训练正常或异常退出后恢复推理时，恢复配置可设置
`restore_on_any_training_exit: true`，覆盖第 1 步之前的初始化失败。
该选项默认关闭，且始终等待所监控进程退出、所选 GPU 和端口释放。
运行守护程序负责清理本次训练的进程，恢复配置只包含 GPU 1–2、端口 8101/8102，
`training_world_size=2`；GPU 3 的独立监督器保持运行。

### 参数数组格式

采用 [VERL 示例](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_4b_fsdp.sh)
的 parameter arrays 组织方式，保留本项目自己的参数值与训练入口：

| 数组 | 内容 |
| --- | --- |
| `DATA` | 数据文件、batch、提示词和长度设置 |
| `MODEL` | 模型路径与模型执行设置 |
| `ACTOR` | 优化器、PPO loss、FSDP 与 KL |
| `ROLLOUT` / `REF` | vLLM 采样、验证生成与参考模型评分 |
| `ALGORITHM` | 优势估计与 DAPO 动态采样 |
| `REWARD` / `OPSD` | 奖励适配、关闭超长奖励惩罚与 OPSD 信用分配 |
| `TRAINER` | 日志、检查点、训练步数和 Trainer 选择 |
| `RAY` | 数据通道资源与 worker 环境变量 |
| `EXTRA` | 实验专用的附加 Hydra 覆盖项 |

`LAUNCH` 保存 Python 命令，`COMMAND` 只负责组合模块数组，供 dry-run 和正式训练共同使用。
数组通过 `"${ARRAY[@]}"` 展开，确保含空格的路径仍然是单个参数。
覆盖优先级为：模块数组 → `EXTRA` → 命令行 `"$@"`。
服务相关环境变量继续在同一脚本内显式设置。

旧链路 `run_2b_4gpu.sh -> run_grpo*_2b.sh -> run_groove.sh` 已完整归档到
`TMP/scripts/`，用于追溯旧实验，不再作为当前启动路径。

## Trainer 分流

共同入口只负责提示词适配、配置验证、Ray 内存设置和选择 TaskRunner。
下表为四卡与 Siton 的默认配置；本机两卡脚本的 GRPO 和 GRPO + OPSD 均不使用 KL。

| 模式 | 训练器 | 组过滤 | 裁剪下限 / 上限 | Reference KL |
| --- | --- | --- | --- | --- |
| `grpo` | VERL `TaskRunnerV1` / `PPOTrainerSync` | 关闭 | 0.2 / 0.2 | 0.01 |
| `dapo` | VERL `TaskRunnerV1` / `PPOTrainerSync` | 开启 | 0.2 / 0.28 | 关闭 |
| `grpo_opsd`（别名 `groove`） | `GrooveTaskRunner` / `GrooveRayPPOTrainer` | 关闭 | 0.2 / 0.2 | 0.01 |

所有模式均使用 GRPO 优势估计和 vanilla PPO policy loss，聚合方式是 `token-mean`。
本分支 GRPO + OPSD 默认使用下文的仅正优势 RLSD 重加权。
`OPSD_ADVANTAGE_MODE=opsd` 恢复未中心化 token credit、系数 0.01、不裁剪的原模式；
该原模式中同奖励题组仍进入 Analyzer。
`trainer_routing.py` 拒绝“旧训练器开启组过滤”及“OPSD 同时开启 V1/组过滤”的无效组合，
避免配置通过但算法未执行。

### Teacher 特权图像：整图聚焦 / Crop 切换

三个独立训练入口在 GRPO + OPSD 下默认使用 `TEACHER_EVIDENCE_MODE=focus`。
`crop` 恢复原来的“原图 + 每个选中区域的放大裁剪”；`focus` 使用“原图 + 一张整图聚焦图”，
把本组所有选中区域放在同一幅原始画面中。纯 GRPO/DAPO 不使用这些特权图像。
此设置独立于 `OPSD_ADVANTAGE_MODE`，原加法 OPSD 和仅正优势 RLSD 都可使用任一图像模式。

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `TEACHER_EVIDENCE_MODE` | `focus` | `focus` 或 `crop` |
| `FOCUS_BLUR_ALPHA` | `0.5` | 框外高斯模糊图的占比，范围 `[0,1]`；原图占比为 `1-alpha` |
| `FOCUS_BLUR_RADIUS` | `12.0` | 高斯模糊半径，以原图像素为单位，必须为有限正数 |

在 `focus` 模式下，框内并集直接保留原始 RGB 像素，框外为
`(1-alpha) * original + alpha * GaussianBlur(original, radius)`。
默认即模糊图 / 原图 = `0.5 : 0.5`。最后给每个框画红色边界，不填充、不标注答案或数量。
重叠区域保持清晰，分离框之间的空隙仍按框外处理，不合成一个包围所有目标的大框。
图像按原尺寸无损保存为 PNG；红框线覆盖的像素除外，其余框内像素不变。
Teacher 的后续图像像素预算及上下文适配仍可能缩小整张图，此模式不提供 Crop 的局部放大效果。

普通区域沿用选中裁剪的 `expanded_box`（包含原有上下文边距），以固定两种模式的关注范围。
若选择了实例框叠加图，则使用其中每个实例框，不使用叠加图的整图外接框；不做 NMS 或 IoU 去重。
所有证据仅进入 Teacher 前缀；Student 的原图、问题、采样回答与评分 token 对齐保持不变。
没有有效证据时仍回退普通 GRPO。

例如，本机两卡新模式（请使用新的实验名）：

```bash
TRAINING_MODE=grpo_opsd TEACHER_EVIDENCE_MODE=focus \
FOCUS_BLUR_ALPHA=0.5 FOCUS_BLUR_RADIUS=12 \
EXPERIMENT_NAME=qwen35-2b-focus-unique-run \
  bash scripts/train_a800_2gpu_nokl.sh
```

原 Crop 对照把 `TEACHER_EVIDENCE_MODE` 改为 `crop`，并使用另一实验名。
四卡和 Siton 使用同名变量。参数写入解析后的 Hydra/W&B 配置：
`groove.teacher_evidence_mode`、`groove.focus_blur_alpha`、`groove.focus_blur_radius`；
命令行参数保持最高优先级，例如在脚本后追加 `groove.teacher_evidence_mode=crop`。
`GROOVE_DRY_RUN=true` 会验证并显示这些配置，不启动训练。

共用配置 `configs/groove.yaml` 和直接构造 `EvidenceBuilderConfig` 的默认仍为 `crop`，
保留旧调用者兼容性；三个独立启动脚本显式选择 `focus`。
Gemini 当前仍为离线 Analyzer 入口，可用同一证据处理层生成聚焦图，见 `docs/GEMINI_ANALYZER.md`；
本次图像模式切换不会自动将正式训练 Analyzer 改为 Gemini。

审计仍保存选中的 `crops`、`focus.tool_regions` 及完整 `tool_trace`；聚焦模式的裁剪仅用于审计，
Teacher 不再接收这些额外裁剪。`evidence.json` 新增 `image_config` 和 `focus_image`（图像路径及实际像素框）。
旧证据缺少这些字段时按 `crop` 解释。聚焦记录存入 `<uid>/focus-<参数哈希>/`，
模式、混合比例或半径改变时不会复用/覆盖其他模式的缓存；Crop 保持原 `<uid>/evidence.json` 路径。

### 本分支：仅正优势 RLSD

`TRAINING_MODE=grpo_opsd` 默认选择 `OPSD_ADVANTAGE_MODE=rlsd_positive`。
它复用现有视觉证据和 PPO 路径，只对 `A_GRPO > 0` 的有效回答 token
使用 Teacher/Student 概率比；负优势和零优势保持普通 GRPO。
证据缺失时权重为 1，不额外使用准确率或连续置信度门控。
不叠加原来的 `0.01 * OPSD` 信号，也不增加独立蒸馏损失。

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `OPSD_ADVANTAGE_MODE` | `rlsd_positive` | 可设为 `opsd` 回到原加法模式 |
| `RLSD_LAMBDA_INITIAL` | `0.5` | 混合系数的 step 0 值 |
| `RLSD_LAMBDA_DECAY_STEPS` | 本机两卡 `40`；四卡/Siton `50` | 按日志 global step 线性衰减至零的步数 |
| `RLSD_CLIP_RANGE` | `0.2` | 原始概率比限制在 `[0.8, 1.2]` |
| `RLSD_TEACHER_SYNC_INTERVAL` | 本机两卡 `5`；四卡/Siton `10` | 每完成多少次外循环更新复制一次 Student 参数 |

本机两卡第 1 步 lambda 为 0.4875，第 40 步起为零；
四卡/Siton 第 1 步 lambda 为 0.49，第 50 步起为零。归零后跳过 Analyzer/Teacher 评分。
本机两卡 Teacher 在第 1–5 步使用初始 Student，第 6–10 步使用完成第 5 步更新的 Student，依此类推。
四卡/Siton 仍在第 1–10 步使用初始 Student，第 11 步开始使用完成 10 次更新的 Student。
Teacher 不使用 EMA，也不进行独立优化。负优势仍参与 PPO 学习。
四卡与 Siton 保留模型 2B、1024 回答长度、batch 16、clip 0.2/0.2 和 reference KL 0.01；
未自动照搬上游 8B、4096 长度或关闭 KL 的设置。4000 题一轮为 250 步，50 步占 20%。

四卡正式运行示例（应在 GPU 可用时使用新的实验名）：

```bash
TRAINING_MODE=grpo_opsd EXPERIMENT_NAME=qwen35-2b-rlsd-positive-unique-run \
  bash scripts/train_a800_4gpu.sh
```

在同一命令前添加 `GROOVE_DRY_RUN=true` 只检查配置，不分配 Ray/GPU worker。
Siton 使用 `scripts/train_siton_2gpu.sh` 和相同的模式及变量。
原方案对照设置 `OPSD_ADVANTAGE_MODE=opsd`；命令行 Hydra 参数仍具有最终优先级。
纯 GRPO/DAPO 不启用 RLSD。本机两卡 `train_a800_2gpu_nokl.sh` 支持同样的
`grpo_opsd` 模式与 RLSD 变量，使用上文的 batch 32、无 KL 设置。

冻结 Teacher 以各 rank 的 CPU 分片驻留，评分前临时加载、评分后恢复 Student。
这增加 CPU 内存和状态复制开销，不额外常驻一份 GPU Teacher。
Teacher 快照随 actor 检查点保存，最佳检查点包含相同文件；当前只支持 FSDP/FSDP2、
同步检查点和本地/共享文件系统。不能在冻结周期中间从缺少 Teacher 分片的旧检查点
静默恢复。衰减结束后不再需要教师快照。

诊断新增 `rlsd/lambda`、`rlsd/teacher_snapshot_step`、正优势/激活 token 比例、
权重均值与范围、上下界裁剪比例和实际修正与 GRPO 的 RMS 比值。
token 审计保留有效权重及实际优势修正，并标记 `advantage_mode=rlsd_positive`。

原生 V1 的纯 GRPO/DAPO 需要 **TransferQueue 0.1.10**；GRPO + OPSD 的项目 Trainer
不需要这一可选依赖。使用原生 V1 前需确认训练环境已安装：

```bash
python -m pip install 'TransferQueue==0.1.10'
# 项目也提供同一依赖声明：pip install -e '.[native-training]'
```

V1 的实现复用仓库内 `src/verl/trainer/ppo/v1/`；没有另写 DAPO 训练循环。
原生路径保留奖励分项均值及 rollout 中的 accuracy、格式、重复和长度诊断。
rollout 的 `score` 保存实际优化的最终奖励，奖励函数原始返回值另存为 `reward_function_score`。

## 共同实验参数

以下是四卡与 Siton 两份入口的默认值；本机两卡无 KL 入口的 batch、验证/保存间隔
和 KL 覆盖值见本文开头。

- 模型 Qwen3.5-2B；训练 seed 20260904；全局 prompt batch / PPO mini-batch 均为 16。
- 每题 8 个回答；TP=1；学习率 1e-6；PPO epoch 1；默认训练 epoch 1。
- prompt / response / 总上下文上限：9216 / 1024 / 10240。
- 普通推理文字 + 末尾 `<answer>`；保留原生空 think prefill，`enable_thinking=false`。
- 训练 temperature 1.0；验证 temperature 0、每题 1 个回答、不采样。
- 验证、保存间隔均为 10 updates；保留最近两份 actor 检查点，并独立保留一份最佳检查点。
- GRPO/DAPO 默认 W&B online，可显式设置 offline；GRPO + OPSD 默认 offline。
- 默认 `RESUME_MODE=disable`，每次要求新的实验名称；显式续训参数在脚本顶部。

### 最佳检查点

两份启动脚本的所有模式默认设置 `SAVE_BEST_CHECKPOINT=true`，按
`BEST_CHECKPOINT_METRIC=val-core/vstar_bench/reward/mean@1` 最大化选择最佳模型。
该指标是完整 V*Bench 的 Judge 语义准确率，不应用训练奖励塑形。只有严格提高才替换，同分保留较早的模型。
训练前验证（step 0）也参与选择；`val_only` 不保存检查点。

最佳快照写入 `checkpoints/<EXPERIMENT_NAME>/best_checkpoint/global_step_<N>/`，
其中包含当次保存的 actor 权重、优化器、额外状态和 `data.pt`；如有 critic 也一并复制。
`best_checkpoint/metadata.json` 记录指标名、分数、步数和快照相对路径。
它是独立文件副本，原始 `global_step_*` 的最近两份轮换不会删除最佳模型。
同一步已有同步保存时直接复制；如果验证和保存周期不同，则为新的最佳步额外保存一次。
额外保存会计入普通检查点的最近两份轮换，因此最近两份也可能包含非周期保存的验证步。
复制成功并原子更新 metadata 后才删除上一份最佳副本；复制失败保留原最佳副本。
稳定状态下额外占用一份完整检查点空间，替换过程中短暂保留新旧两份最佳副本。

续训时从 metadata 恢复最佳分数，避免较差的后续模型覆盖旧最佳。需要从最佳模型显式续训时，
将 `RESUME_FROM_PATH` 指向 metadata 对应的 `best_checkpoint/global_step_<N>`，并设置
`RESUME_MODE=resume_path`。普通自动续训仍跟随外层 `latest_checkpointed_iteration.txt`。
W&B/控制台在验证点记录 `checkpoint/best_step` 和 `checkpoint/best_score`。

可通过 `SAVE_BEST_CHECKPOINT=false` 关闭，或覆盖指标；比较方向可通过最后的 CLI 参数
`trainer.best_checkpoint.mode=min` 改为最小化。开启时要求有验证且使用同步保存，
不支持 `checkpoint.async_save=true` 或 V1 异步训练。最佳副本保存在本地实验目录。
这些设置只影响新启动的进程，无法恢复旧运行已清理的模型权重。

| 机器配置 | 本机 A800 四卡 | Siton 两卡 |
| --- | --- | --- |
| Python | `/ssd/home/zc/miniconda3/envs/groove/bin/python` | `/home/yzs/miniconda3/envs/vision-opd/bin/python` |
| 模型 | `/ssd/home/zc/yzs/models/ckpts/Qwen3.5-2B` | `/root/siton-tmp/yzs/ckpts/Qwen3.5-2B` |
| CUDA devices | 0,1,2,3 | 0,1 |
| Agent / reward workers | 16 / 8（DAPO），16 / 4（其余模式） | 8 / 1 |
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
可通过 `VALIDATION_FILE` 覆盖。训练期间的验证与训练共用 `extract_answer()` 和远程 Judge：
先提取 `<answer>` 内容，无法提取完整标签时使用完整回答；每条回答均进行语义判分。
Judge 接收题目、全部有效选项以及标准答案的字母和文本，因此既能评判选项字母，
也能评判含义相同的自然语言答案。验证 `score=accuracy`，不应用格式、重复或长度惩罚。
实际传入 Judge 的标准答案形如 `(C) purple`，并非只有 parquet 中保存的字母 `C`。
评分适配器会剥离题目末尾的 Student 回复格式指令，防止 Judge 因回答颜色文字而非选项字母
误判；正确字母（大小写均可）、选项文本和等价表达都应接受，字母与文本相冲突仍应判错。
新的验证数据将纯题目/选项与回复指令分开保存。对于已有 parquet，数据集适配器只在内存中
把 Student 指令改为允许字母或答案文本，不改写文件；末尾 answer 标签约定保持不变。
格式和重复仍记录为诊断；旧规则的结果另存为 `rule_accuracy`、`rule_unparsed`，不再决定得分。
Judge 使用与训练相同的“简短依据 + 最终二值判定”协议、超时与重试。
后续运行取消仅允许输出 `0/1` 的 choice 约束，生成预算为 512 tokens，Judge 默认温度为 0.3；
`enable_thinking=false` 下的依据是普通响应文本。解析器只读取唯一的末尾
`Judgement: 0/1` 判定（允许紧接依据出现在同行，兼容旧版纯数字响应），
缺失、冲突或生成截断时重试。
提示词拒绝单答案题中未消解的候选枚举和矛盾结论，同时允许真正的多属性答案。
校准后的简短提示词以所问属性为准，允许普通色差、次要点缀和明确区分对象的附加描述，
不因未询问属性的不确定性而自动判错；固定说明与示例总量控制在 400 英文词以内。
按用户指定策略，首次请求加默认 5 次重试仍无有效判定时，该 rollout 以 `accuracy=0`
按错误答案处理，保留格式扣分并继续训练；请求超时/连接中断耗尽重试也采用该策略。
训练期间验证同样设 `score=accuracy=0`，不使用规则分数替代。
每条结果记录 `judge_attempts`、`judge_retries_exhausted`；重试内成功的耗尽标记为 0。
每个训练 step 在常规日志与 W&B 中记录 `reward/judge_retries_exhausted_count`（rollout 数）、
`reward/judge_retries_exhausted_fraction`、`reward/judge_attempts_mean`、`reward/judge_attempts_max`。
未触发耗尽的 batch 也记录零值。逐条 rollout/验证 JSONL 保留对应字段；验证均值在 `val-aux` 下。
`GROOVE_JUDGE_AUDIT_DIR` 继续保存失败响应，最终失败含 `fallback_accuracy=0`，
另有含请求 ID 的警告日志。与旧版宽松 Judge 比较时须统一协议重评。
修复格式指令污染前后的语义分数也需在同一协议下重评；原始回答、历史日志及 W&B 分数不自动回改。

两份脚本的所有模式默认保存每次验证的全部 rollout（默认 191 条），包括训练前 step 0：
`outputs/validation/<EXPERIMENT_NAME>/<step>.jsonl`。文件包含完整 input/output、标准答案、
语义得分和格式/规则诊断。该保存不受 `log_val_generations` 的 W&B 展示数量限制，
两卡的分批验证也会汇总全部回答。可用 `VALIDATION_DATA_DIR` 覆盖输出目录。
训练 rollout 继续保存在独立的 `outputs/rollouts/<EXPERIMENT_NAME>/`。

上述语义评分是新的验证协议，不能直接替代历史纯规则成绩。历史 parquet、日志和分数不回改；
`scripts/evaluate_vstar.py` 及其比较工具继续保留明确的纯规则独立评测协议。

训练和训练期间验证的语义 Judge 为 `127.0.0.1:8002/v1` 的 Qwen3.8-27B。
GRPO + OPSD 额外使用同一服务的 Analyzer，以及 8011 DINO、8012 OCR。
DINO/OCR 保持单 worker；启动脚本不会部署或重启远程服务。

## DAPO 的明确语义

- 由原生 V1 ReplayBuffer 过滤组内 **最终训练奖励** 完全相同的题组，并补采到 16 组。
- 过滤字段为 `training_reward`，包含语义、格式和重复处理；`accuracy` 保持独立。
- 两个脚本在所有模式下都设置 `reward.reward_kwargs.overlong_buffer_cfg.enable=false`。
  最大回答仍为 1024 token，但生成长度不追加奖励惩罚。
  旧环境变量 `DAPO_OVERLONG_ENABLED`、`DAPO_OVERLONG_BUFFER`、`DAPO_OVERLONG_PENALTY`
  不再参与启动配置。
- `VisualQARewardManager` 保留可选的训练长度塑形能力，当前脚本不启用；验证记录保持原始准确率。
- 四卡脚本默认 `max_inflight_gen_batches=2`，两卡仍为 1；可用
  `DAPO_MAX_INFLIGHT_GEN_BATCHES` 覆盖。该值乘以全局 16 组，得到同时处于生成/评分阶段
  的提示词组上限，不改变最终更新的 16 组。V1 不执行旧的 `max_num_gen_batches` 总重试上限。
  后者固定写为 0，避免假设它能终止补采。
- V1 默认按 `4000 // 16 = 250` 个**优化更新**计算一轮预算；DAPO 补采可能多次遍历数据。
  250 次更新不等于只生成过 4000 道题。比较成本时需看过滤计数、rollout/token 数和墙钟时间。

## DAPO 吞吐短测与计时

`REWARD_NUM_WORKERS` 控制原生 GRPO/DAPO 的评分 worker 数；同步 `compute_score`
通过各 worker 的默认线程池调用远程 Judge。`GROOVE_JUDGE_CONCURRENCY` 仅用于
`compute_score_batched`，不控制原生逐条评分路径。四卡 DAPO 默认 8 个 reward worker，
其他模式仍为 4。`ROLLOUT_ENFORCE_EAGER` 可覆盖，两份脚本的所有模式均默认 true
（关闭 CUDA Graph）。CUDA Graph 已通过四卡 DAPO 连续参数更新短测，但尚未验证
长期训练效果，按用户决定正式运行先关闭；吞吐短测不能替代数值和固定验证集对照。

逐项比较配置为 `(DAPO_MAX_INFLIGHT_GEN_BATCHES, REWARD_NUM_WORKERS, ROLLOUT_ENFORCE_EAGER)`：
`(1, 4, true)` → `(2, 4, true)` → `(2, 8, true)` → `(2, 8, false)`。
每个短测使用新的实验名、同一模型/数据/种子，保持 16×8 和 1024 token 上限。
窗口变大会改变补采样量和可能选中的题组，不能视为逐样本完全相同的对照。

2026-09-18 四卡各 5 步短测，排除首步后平均完整迭代耗时依次为
49.7、42.6、45.2、34.7 秒。最后一组吞吐最好，但用户决定正式运行先关闭 CUDA Graph；
当前四卡 DAPO 默认采用第三组 `(2, 8, true)`。
单独从 4 增至 8 个 reward worker 没有显示收益，4-worker＋CUDA Graph 的组合尚未实测。
这些结果来自基础模型的短测，样本长度和过滤量存在差异，不能直接预测后期高过滤率时的加速幅度。
短测关闭验证和保存，未改变奖励协议；正式运行仍保留原验证和保存频率。
原始报告位于 `outputs/diagnostics/dapo-speed-20260918-004655/`（本地生成状态，不纳入 Git）。

原生 V1 新增 `timing_s/retained_agent/{generate_sequences,compute_score}/{mean,max,p95}`，
仅统计实际保留的非 padding 轨迹，含请求等待时间；并发请求的耗时不能相加当作墙钟时间。
另记录指标计算、rollout 导出、队列清理、标量日志和 DAPO 表格日志的耗时。
前几项进入正常指标；两类日志调用本身的耗时在它们返回后写入
`outputs/timing/<EXPERIMENT_NAME>/steps.jsonl`，`STEP_TIMING_DIR` 可覆盖目录。
每行 `timing_s.iteration` 从本步开始计至两类日志调用结束，包含验证及保存，
不包含该计时文件写入、进度条和训练结束后的清理。它与历史 `timing_s.step` 保持分开。
这些本地记录也包含最后一步，无需额外 W&B 调用或将指标挪到下一步；续训时追加写入。
OPSD Trainer 暂不生成这组 V1 专用计时。

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

监督器的 `gpus` 和 `ports` 可以配置为等长的子集，例如 `[1, 2, 3]` 与
`[8101, 8102, 8103]`；未配置的 GPU 不参与监控或启动。训练与推理卡数不同时，
用 `training_world_size` 保留原训练的检查点分片数，例如四卡训练设置为 `4`。
接管仍在运行的服务时，可显式配置 `existing_services`，以 GPU 编号字符串为键，
记录 `pid`、`start_ticks`（`/proc/<pid>/stat` 的启动 tick）和原 `log` 路径。
接管前会核对进程身份、所有者、进程组、完整启动命令和 CUDA 设备选择；
全部匹配后直接恢复健康检查及自动重启，无需重新等待训练或重载现有模型。
`status.json` 会输出当前进程的 `start_ticks`；再次接管应使用当前身份，旧 PID
记录失效时会拒绝启动。正常终止监督器会停止它启动或接管的全部已配置服务。
