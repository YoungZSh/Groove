# 编码代理的仓库指南

## 适用范围与项目目的

本文件适用于仓库根目录及其所有子目录；如果某个子目录存在更具体的
`AGENTS.md`，则以该文件为准。

本项目实现 GROOVE：在常规 GRPO 的基础上，利用特权视觉证据，为采样出的 token
增加未中心化的 OPSD 优势信用分配。方法名称统一使用 `GRPO + OPSD`。
不要把旧实验名称、内部标识或论文中的限定词添加到对外方法名称中。

必须遵守以下部署边界：

- Student 只能看到原始图像和问题。
- 使用当前策略的 Teacher，在包含 Analyzer 所选视觉证据的特权前缀下，
  对 Student rollout 中完全相同的 token 进行评分。
- Analyzer、GroundingDINO、OCR、裁剪图像和 Teacher 专用文本都是训练阶段的特权信息，
  绝不能进入 Student 的推理输入。

## 当前实现的权威依据

以仓库中的实际执行路径为准，按以下顺序阅读：

1. `scripts/train_a800_4gpu.sh` 和 `scripts/train_siton_2gpu.sh` 是完整、
   独立的实验脚本。它们直接启动 Python，不通过 source 引入或调用其他 shell 启动脚本。
   每个脚本可选择 `grpo`、`dapo` 或 `grpo_opsd`
   （`groove` 是 GRPO + OPSD 的别名）。机器设置和所有实验覆盖参数都写在所选脚本中，
   详见 `docs/TRAINING_LAUNCHERS.md`。
   所有当前和新增训练入口均按 parameter arrays 格式组织：按 DATA、MODEL、ACTOR、
   ROLLOUT、REF、ALGORITHM、REWARD、OPSD、TRAINER、RAY 分组，末尾统一展开。
   保留带引号的数组展开，命令行 `"$@"` 放在最后；不要恢复成长串参数或多层 shell 调用链。
   `src/groove/trainer_routing.py` 将纯 GRPO/DAPO 路由到 VERL 原生 V1 同步训练，
   将 GRPO + OPSD 路由到项目自定义 Trainer。旧启动脚本已归档到
   `TMP/scripts/`，不作为当前执行规范。
2. `src/groove/verl_trainer.py::_postprocess_advantages()` 是训练集成的权威实现。
   它在线构建 Teacher 证据，计算参数更新前的 Teacher 对数概率，构造 OPSD token 优势，
   与已经计算好的 GRPO 优势合并，并在 actor 更新前写回
   `batch.batch["advantages"]`。
3. `src/groove/losses.py::groove_opsd_advantages()` 定义未中心化的 OPSD
   逐 token 信用分配。`combine_grpo_opsd_advantages()` 定义优化时唯一使用的
   GRPO + OPSD 优势合并方式。
4. `src/groove/objective.py::validate_objective_config()` 定义允许的目标配置：
   使用 GRPO 优势估计、单一 vanilla PPO 策略损失，在损失中加入参考策略 KL，
   不设置独立的蒸馏目标。
5. `src/groove/semantic_reward.py` 定义语义 Judge 准确率、格式奖励塑形、
   重试行为和重复处理。
6. `src/groove/analyzer.py`、`src/groove/analyzer_tools.py` 和
   `src/groove/evidence.py` 定义 Analyzer 输入、工具使用、裁剪选择、
   泄漏检查和 Teacher 提示词构造。
7. `src/verl/` 是当前实际使用的内置运行时。`tests/` 中的测试是上述行为约定的
   可执行验证依据。

`docs/` 中的文档用于解释历史和设计意图，不是实现的权威依据。
如果文字说明、论文、探测脚本或旧报告与上述代码路径不一致，以当前代码为准，
并单独更新过时文档。

`TMP/scripts/LEGACY_README.md` 描述的是旧版 Vision-OPD/4B 配置。
未经核对 2B 启动脚本，不要把其中的通用 batch size、回答长度、KL 系数、
检查点保存频率或服务拓扑直接用于 2B 视觉问答训练。

`Papers/` 和 `TMP/probe_experiments/` 中的文件仅作为参考资料和归档探测实验，
不作为可执行规范。不要根据其中的术语推导当前算法。
不要修改 `TMP/upstream/verl-v0.9.0`；当前实际使用的内置运行时是 `src/verl`。

## 仓库目录结构

- `src/groove/`：项目目标函数、Analyzer、证据、奖励、Trainer 钩子、
  数据结构和诊断工具。
- `src/verl/`：内置 VERL 0.9 运行时及项目适配。
- `scripts/`：独立实验启动脚本，以及可复用的数据准备、评测和运行工具。
- `TMP/scripts/`：一次性筛选、探测脚本及历史启动链。
- `remote_tools/`：单进程远程 GroundingDINO 和 PaddleOCR 服务端。
- `configs/`：共享 Hydra 默认配置。
- `tests/`：CPU 单元测试。
- `docs/`：说明文档，必须与当前代码保持一致。
- `data/`、`outputs/`、`checkpoints/`：本地生成状态，大部分被 Git 忽略，
  绝不能随意删除。

## Python 与测试环境

使用现有环境，不要使用 `/usr/bin/python3`：

```bash
PYTHON_BIN=/ssd/home/zc/miniconda3/envs/groove/bin/python
PYTHONPATH="$PWD/src" "$PYTHON_BIN" -m unittest discover -s tests -v
```

奖励模块的定向测试：

```bash
PYTHONPATH="$PWD/src" \
  /ssd/home/zc/miniconda3/envs/groove/bin/python \
  -m unittest discover -s tests -p 'test_semantic_reward.py' -v
```

交付启动脚本改动前，还要运行：

```bash
bash -n scripts/train_a800_4gpu.sh
bash -n scripts/train_siton_2gpu.sh
git diff --check
```

Siton 使用 `/home/yzs/miniconda3/envs/vision-opd/bin/python`。原生 V1 需要
`TransferQueue==0.1.10`（`native-training` 可选依赖）。
如果修改了归档 shell 文件的路径，也要对这些文件进行语法检查。

使用 `GROOVE_DRY_RUN=true` 可以验证完整解析后的配置，而不启动 Ray worker
或加载模型权重。检查正式启动脚本时，必须提供唯一的 `EXPERIMENT_NAME`。

## 当前 GRPO + OPSD 优势信用分配

当前实现采用未中心化的 OPSD 优势。OPSD 在优势层面进行逐 token 信用分配，
不作为独立损失添加。实际执行的计算为：

```text
delta_t = stopgrad(log p_teacher(y_t) - log p_student(y_t))
A_OPSD,t = evidence_mask * delta_t
A_total,t = A_GRPO + 0.01 * A_OPSD,t
L_actor = VERL_vanilla_PPO(A_total) + 0.01 * low_var_reference_KL
```

以上公式是对当前代码的简写说明，代码本身才是权威依据。
除非用户明确要求修改算法，否则必须保留以下性质：

- Teacher 和 Student 对完全相同的采样回答 token 进行评分。
- Teacher 评分发生在 actor 优化器更新之前，且不计算梯度。
- OPSD token 信用未中心化：不减去轨迹均值或组均值，
  不添加 sigmoid 门控、`(1-r_i)` 因子或正确性乘数。
- GRPO 作用于每条 rollout。只要组证据可用，OPSD 同样作用于正确和错误的 rollout。
- 奖励完全相同的组仍需分析；即使 GRPO 优势为零，这些组也可能提供 OPSD 信号。
- Analyzer 或 grounding 失败时，回退到普通 GRPO。
- 证据可用性采用组级二值掩码，不使用连续的 DINO/OCR 置信度权重。
- 当前默认值为 `OPSD_ADVANTAGE_COEF=0.01`，且不裁剪 OPSD 优势。

当前 2B GRPO 和 GRPO + OPSD 的参考策略 KL 系数为 `0.01`，
训练采用 `token-mean`，学习率为 `1e-6`，执行一个 PPO epoch，裁剪比例为 `0.2`。

纯 DAPO 使用原生 V1 动态组过滤，裁剪下限/上限为 0.2/0.28，不使用参考策略 KL。
两个启动脚本在所有模式下都关闭超长奖励惩罚，保留 1024 token 生成上限，
不再读取旧的 `DAPO_OVERLONG_*` 环境变量。`src/groove/reward_manager.py`
仍保留可选长度塑形能力，但当前启动配置不启用。
按 `training_reward`（实际用于优化的最终标量奖励）进行过滤，
同时保留原始 `accuracy`。V*Bench 不应用训练奖励塑形。
不要在 GRPO + OPSD 中启用这种过滤，因为奖励完全相同的组也可能提供 OPSD 信号。
V1 名义上的 epoch 用于确定优化器更新次数；DAPO 补充采样可能多次遍历源数据。

## 语义奖励约定

`src/groove/semantic_reward.py` 明确区分语义正确性与输出格式：

- 训练及训练期间的 V*Bench 验证使用下文描述的同一个远程语义 Judge。
  验证包含全部 191 题，通过 `data_source=vstar_bench` 路由到
  `src/groove/semantic_reward.py::_judge_vstar_validation()`，复用训练的 `extract_answer()`：
  提取 answer 内容，提取不到完整标签时使用完整回答。每条验证回答都调用 Judge，
  不是只有规则失败时才调用；Judge 接收题目、全部选项，以及标准选项字母和文本。
  验证 `score=accuracy`，不叠加格式、重复或长度惩罚。
  `rule_accuracy`、`rule_unparsed` 仅用于保留旧选项匹配规则的诊断，不影响语义分数。
  独立 `scripts/evaluate_vstar.py` 仍明确使用历史纯规则协议；
  不要将新语义验证分数与旧规则验证分数直接作为同一评分协议比较。
- 远程 Judge 返回的语义 `accuracy` 只能是 `0` 或 `1`。
- 后续运行的 Judge 先输出一到三句简短依据，再以唯一的 `Judgement: 0/1`
  行结束；请求保留温度 0 和 `enable_thinking=false`，生成预算为 512 tokens，
  不再使用二值 choice 约束解码。解析器只读取唯一的末尾判定，允许其紧接简短依据
  出现在同一行，兼容旧版纯数字响应；
  判定缺失、冲突或生成截断时重试，失败不能静默转为错误标签。
  提示词明确拒绝单答案题中的未消解候选、互相矛盾的结论和未定位目标的整图枚举，
  同时允许题目要求的多属性答案、明确区分对象及排除候选后的确定结论。
  简短提示词只评判题目所问属性：未询问属性的不确定性、普通色差/阴影、主色的
  次要点缀不应自动判错；先排除未消解候选，再应用这些宽容条件。
  保持四条准则和五个简短示例，固定提示词不超过 400 英文词。
  训练和训练期间验证共用此协议；与旧版宽松 Judge 的分数比较时必须统一重评。
- Analyzer 的成功/失败分组使用原始 `accuracy`，绝不能使用塑形后的 `score`。
- 标签格式错误时，语义评判可以回退到完整回答，避免输出格式在无提示的情况下
  改变语义正确性的定义。
- 有效格式必须恰好包含一对小写、非空、位于结尾的 `<answer>...</answer>` 标签。
  标签前允许普通推理文本；标签后只允许空白字符。
  嵌套、重复或不配对的 answer 标签均视为格式无效。
- 接下来的 2B 运行使用 `data.response_format=reasoning_answer`：
  先输出普通推理文本，再给出最终答案。不添加 think 包裹或推理长度门控。
  数据集适配器只在内存中修改指令，不重写历史 parquet 文件。
  Student/Teacher 共用的聊天模板在 `enable_thinking=false` 时保留
  Qwen3.5 原生的空 think 预填充。该预填充属于提示词，不属于采样回答；
  不要为了要求普通推理而将其从模板中删除。
- 接下来运行的奖励塑形方式为：

```text
format_penalty = 0 if format is valid else -1
score = accuracy + 0.2 * format_penalty
```

因此，格式正确且答案正确时得分为 `1.0`；格式错误或无标签但答案正确时为 `0.8`；
格式正确但答案错误时为 `0.0`；格式和答案都错误时为 `-0.2`。

精确连续重复检测器独立运行，默认要求至少重复四次且覆盖至少 80 个字符。
严重重复的轨迹不能获得正奖励，但仍保留原始语义 `accuracy`，
用于诊断和 Analyzer 分组。

未经明确决定，不要因缺少标签就强制设定 `accuracy=0`，也不要跳过 Judge：
这会让呈现格式错误污染 Analyzer 的成功/失败分组。
如果需要更严格的训练约束，应对 `score` 设置门控，同时单独记录语义准确率。

历史注意事项：已完成的 OPSD v4 运行使用 `FORMAT_REWARD_WEIGHT=0.0`。
当前纳入版本管理的启动脚本为下一次运行采用 `0.2`。
比较新旧运行时，既要看塑形后的 `score`，也要看原始 `accuracy`。

## Analyzer 与证据边界

Analyzer 的行为约定定义在 `src/groove/analyzer.py` 中。

- 系统提示词描述的是视觉证据任务，不是自我进化任务。
  不要重新引入“self evolution”措辞或相关产物。
- 输入包含原始图像、问题，以及由程序分别整理的成功/失败推理列表。
- 不得接收标准答案、数值奖励、解析后的预测标签或 rollout ID。
- 不得重新判断结果标签，也不得回答问题。
- 工具描述通过原生工具注册，并由 Qwen 聊天模板注入；
  不要在系统提示词中重复工具描述。
- Grounding 和 OCR 查询必须使用简短的英文视觉目标描述。
- 模型必须检查每张返回的裁剪预览；允许重试，但不得超过配置的工具轮数。
- 每张可用裁剪图都有候选 ID。Analyzer 最终选择一到三个候选。
  不进行 IoU 去重，也不隐式优先选择最新裁剪。
- 单个目标使用一张裁剪图即可。比较、计数或空间关系任务允许使用多张图像。
- 部署多实例 DINO 接口后，可通过 `ANALYZER_ENABLE_INSTANCE_BOXES=true`
  启用计数框叠加图。`ground_instances` 返回候选框；无论框的数量多少，
  一张完整图像的叠加图都只算一个可选证据候选。保留原始图像，只绘制无填充的框
  （不标注 ID 或总数答案），叠加图仅供 Teacher 使用。
  为兼容现有单框服务，该开关默认为 false。
  不得将单框响应静默当作计数结果接受。
- 独立工具 `count_objects(target, region=None)` 通过
  `ANALYZER_ENABLE_COUNTING=true` 显式启用。它使用固定的 DINO 基线配置，
  默认保留相互重叠的检测框：重叠并不能证明两个框表示同一对象。
  可选 NMS 仅用于明确指定的离线比较，绝不能用于 Analyzer 候选图像或现有裁剪选择流程。
  区域坐标和返回框均使用原始图像坐标系。
  不要向 Analyzer 暴露逐样本阈值调节能力或预期数量。
  两个开关同时启用时，该工具优先于原始 `ground_instances` 工具。
- `visible_focus_instruction` 必须保持答案中立，不得泄漏 OCR 文本、
  答案断言、选项字母、奖励或 rollout 结果。

裁剪图像、实例框叠加图和完整工具轨迹只能用于 Teacher/审计路径。
新的证据记录必须持久化完整的 `tool_trace` 数据；历史记录中缺失的轨迹无法重建。
未选中的候选仍只能用于审计。

## 当前 2B 实验设置

独立实验脚本使用以下设置：

- 模型：本机使用 `/ssd/home/zc/yzs/models/ckpts/Qwen3.5-2B`；
  Siton 使用 `/root/siton-tmp/yzs/ckpts/Qwen3.5-2B`。
- 随机种子：`20260904`。
- 本机四卡：`CUDA_VISIBLE_DEVICES=0,1,2,3`；Siton 两卡：`0,1`。
- 训练数据：包含 4000 行的 `vstar_grpo_4000_seed20260917` 或
  `vstar_opsd_4000_seed20260917` 数据划分，可通过 `DATA_DIR` 独立覆盖。
- 提示词批大小：16 组。
- 每组 rollout 数：8。
- 最大回答长度：1024 tokens。
- 训练采样温度：`1.0`。
- 验证温度：`0`，不采样。
- 验证和检查点保存频率：每 10 步。
- 保留的 actor 检查点数：2。
- Thinking 模式：关闭。
- 训练期间的 W&B 模式：GRPO/DAPO 默认在线，支持显式切换为离线；
  GRPO + OPSD 保持离线。

不要仅凭随机种子相同就认为比较公平。必须核对模型、parquet 行顺序、
shuffle 设置、batch size、rollout 数、Judge 协议、奖励权重和验证温度。

独立四卡启动脚本使用全局 16 组提示词、每组 8 条 rollout，
所有模式默认种子均为 20260904，并要求显式指定新的 `EXPERIMENT_NAME`。
其吞吐配置使用 16 个 agent-loop worker、4 个本地 reward worker，
以及每 GPU 65536-token 的 actor/对数概率计算和 vLLM 批次预算。
保持 TP=1（每 GPU 一个 Qwen3.5-2B rollout 副本），回答长度保持 1024。
调节这些预算时，全局 batch 必须保持 16；提高 token 上限不意味着增大 batch。
验证 batch 默认为 null（完整验证集），与 16 个 agent worker 无关；
仍支持显式指定有限的验证 batch。`OMP_NUM_THREADS=4` 会传递给 Ray worker。

四卡启动脚本默认使用 `RAY_NODE_MEMORY_CAP_GIB=null`，
取消历史上固定的 216 GiB 触发阈值。Ray 改为使用其检测到的节点/容器内存的 95%；
这一设置在 Ray 启动时生效，不会更新已经运行的任务。
单节点 NCCL 默认使用 `NCCL_SOCKET_IFNAME=lo` 和 `NCCL_IB_DISABLE=1`，
并与 `NO_PROXY`、`no_proxy` 一起通过 Ray 运行环境传递。
Dry-run 可以在较小的 GPU 配额上验证四卡配置；
正式启动时，会在启动 Ray 前检查 CUDA 是否能看到所选的四张卡。
不要把 dry-run 当作通信测试。

两个独立 2B 启动脚本的所有模式都使用 `data/vstar_bench/validation.parquet`，
包含全部 191 道原始基准题目（115 道直接属性题、76 道相对位置题）。
使用 `scripts/prepare_vstar_validation.py` 准备此文件；
`VALIDATION_FILE` 独立于 `DATA_DIR`。
保留原始图像字节和全部选项。启动脚本使用 9216 个提示词 token、
10240 个总上下文 token。两卡启动脚本的验证 batch 为 8；
四卡启动脚本一次提交完整验证集，让 vLLM 在 token 和序列数量限制内连续调度请求。
两个启动脚本均通过 `trainer.validation_data_dir` 保存每个验证点的全部原始回答，
默认路径为 `outputs/validation/<EXPERIMENT_NAME>/<step>.jsonl`（包含 step 0 验证）。
该落盘独立于 W&B 样例展示数量及验证 batch；保留完整 input/output、标准答案、
语义分数和规则/格式诊断。`VALIDATION_DATA_DIR` 可覆盖目录，默认与训练 rollout 分开。
不要把基准样本追加到训练数据，不要覆盖旧的 220 行验证文件，
也不要把新旧验证分数当作来自同一数据集直接比较。

已完成的参考运行：

- 纯 GRPO，W&B ID 为 `iam35gwn`：最终验证分数 `0.6636`，最佳分数 `0.6773`。
- OPSD v4，W&B ID 为 `5zmaf7ds`：最终验证分数 `0.7364`，最佳分数 `0.7455`。
- 从最终本地日志回填后，两次运行均有 123 个在线历史记录点。

## 服务拓扑

正式 OPSD 启动脚本要求以下三个本地 HTTP 端点，它们对接已部署的服务：

- `127.0.0.1:8002/v1`：Qwen3.8-27B Analyzer 和二值 Judge。
- `127.0.0.1:8011`：单 worker GroundingDINO 服务。
- `127.0.0.1:8012`：单 worker PaddleOCR 服务。

运行前以只读方式检查：

```bash
curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8011/ >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8012/ >/dev/null
```

Analyzer 的组级编排可以使用并发度 16。
除非用户明确批准远程部署变更，否则 DINO 和 OCR 必须各自保持单 worker。
在本地编辑 `remote_tools/*.py` 不会更新远程机器，也不会重启正在运行的服务。

不要打印 API 密钥、`.netrc`、环境中的秘密信息或请求授权头。
`remote-qwen38` 之类的占位值不是用户凭证。

## 训练运行与监控

- 未经用户要求，不得启动、停止、恢复或终止正式训练任务。
- 除非用户明确要求恢复训练，否则优先新开运行，不恢复已中断或已完成的运行。
- 每次正式运行都要设置新的 `EXPERIMENT_NAME`。
  当前独立启动脚本强制要求该参数，且默认 `RESUME_MODE=disable`；
  除非显式选择恢复，否则会拒绝使用已有检查点目录。
  归档启动脚本保留历史默认值，不得用于新的正式实验。
- 不得删除或覆盖检查点、rollout、证据、日志或 W&B 离线文件。
- 运行期间避免修改奖励或训练源码。如果改动用于下一次运行，必须明确说明，
  并在比较结果前核对当前运行在 W&B 中保存的完整解析配置。
- 启动前检查 GPU、内存和服务健康状态。
  长时间任务使用 tmux 或现有进程管理工具，并保留完整日志。

生成状态按 `EXPERIMENT_NAME` 组织：

- `outputs/logs/`
- `outputs/rollouts/`
- `outputs/validation/`
- `outputs/evidence/`
- `outputs/opsd-token-dumps/`
- `outputs/wandb/wandb/`
- `checkpoints/`

至少监控以下指标：

- rollout 的 `reward/answer_reward_mean`，以及验证奖励/准确率；
- 严格的 `has_answer_tag` / `format_valid` 比例；
- 严重重复比例和回答长度；
- `actor/kl_loss`、熵和梯度范数；
- 证据就绪/错误/回退比例，以及 Analyzer 实际耗时；
- OPSD 与 GRPO 优势的 RMS 比值、信用方向一致性和裁剪比例；
- 每步实际耗时，以及检查点保存和验证的额外开销。

训练奖励噪声较大。判断效果时，应优先看固定验证点和多步趋势，
不要根据单个 rollout batch 下结论。
不同运行的奖励塑形不同时，必须单独比较原始语义准确率。

## W&B 操作规范

GRPO 启动脚本默认在线上传指标，并在 `outputs/wandb` 保留本地记录；
`WANDB_MODE=offline` 会显式关闭在线上传。GRPO + OPSD 启动脚本仍使用离线日志。
只有用户要求时才上传历史离线记录。
根据日志中的 `trainer.experiment_name` 定位准确的离线目录，
不要误将所有历史探测运行一起同步。

同步后，通过 W&B API 核验：

- 运行名称和 ID 与目标实验一致；
- 状态为 `finished`；
- 历史记录达到本地最终的 `training/global_step`；
- 最终和最佳验证分数与本地日志一致。

SDK 可能在进程退出时未能写出最后一到两步。
如果在线历史记录不完整，只从对应本地日志回填缺失的数值记录，
记录源日志的哈希，然后再次核验。绝不能编造或插值补齐缺失的训练指标。

当前用于下一次运行的代码快照是 W&B artifact
`mmcot-opsd-source-format-penalty:v0`，关联的代码快照 Run ID 为 `oqsxbci5`。
它对应 Git 提交 `a6c9f6c`；始终用 `git rev-parse HEAD` 确认当前版本，
不要假定该快照仍是最新版本。

## Git 与编辑规范

- 保留与当前任务无关的用户改动，编辑前先检查 `git status`。
- 使用聚焦的补丁，尽量减少对内置 VERL 的修改。
- 每次修改奖励、提示词结构、优势或启动脚本行为，都要新增或更新测试。
- 约束条件发生变化时，更新相关架构文档。
- 提交前运行定向测试、shell 语法检查和 `git diff --check`。
- 按完整、相关的行为变更组织提交。
  不要提交生成数据、模型权重、日志、rollout、证据、W&B 目录或检查点。
- 未经明确要求，不要推送 Git 提交或修改远程服务。
- 汇报提交哈希、运行过的测试，以及工作区是否干净。

向用户汇报状态时，默认使用简洁中文，并区分观察到的事实与分析判断。
