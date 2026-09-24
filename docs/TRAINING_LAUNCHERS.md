# 扁平训练入口

## 组织方式

`scripts/train_a800_4gpu.sh` 与 `scripts/train_siton_2gpu.sh` 都是完整、独立的实验脚本。
每份文件内直接列出机器参数、算法选择、数据路径、奖励、优化器、采样、验证、日志和 Ray 网络参数，
最后直接调用 `python -m groove.verl_entrypoint`。没有中间 shell 启动器，也不读取共享机器 shell 配置。
需要某台机器、某种方法的固定实验入口时，复制其中一份并修改顶部参数即可。

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
后续运行取消仅允许输出 `0/1` 的 choice 约束，生成预算为 512 tokens，温度仍为 0；
`enable_thinking=false` 下的依据是普通响应文本。解析器只读取唯一的末尾
`Judgement: 0/1` 判定（允许紧接依据出现在同行，兼容旧版纯数字响应），
缺失、冲突或生成截断时重试。
提示词拒绝单答案题中未消解的候选枚举和矛盾结论，同时允许真正的多属性答案。
校准后的简短提示词以所问属性为准，允许普通色差、次要点缀和明确区分对象的附加描述，
不因未询问属性的不确定性而自动判错；固定说明与示例总量控制在 400 英文词以内。
重试耗尽会报错，不静默回退成规则分数。与旧版宽松 Judge 比较时须统一协议重评。
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
