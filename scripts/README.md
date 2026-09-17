# 独立实验脚本

每份训练脚本包含完整的机器路径、资源、算法参数和 Python 启动命令。
可以复制为某台机器、某次实验专用的脚本；脚本之间不 `source`、不相互调用，
也不依赖 `TMP/scripts/` 的历史启动器。

训练参数按 VERL 示例的 **parameter arrays** 风格组织：`DATA`、`MODEL`、`ACTOR`、
`ROLLOUT`、`REF`、`ALGORITHM`、`REWARD`、`OPSD`、`TRAINER`、`RAY`。
每个数组元素对应一个完整的 Hydra 覆盖项；末尾统一展开数组组成启动命令。
`EXTRA` 留给当前实验的附加参数，命令行 `"$@"` 放在最后，优先级最高。

| 独立脚本 | 默认机器参数 |
| --- | --- |
| `train_a800_4gpu.sh` | 本机 A800，GPU 0～3，groove 环境，65536 token 预算 |
| `train_siton_2gpu.sh` | Siton，GPU 0～1，vision-opd 环境，32768 token 预算 |

两份脚本内部均可设置 `TRAINING_MODE=grpo`、`dapo` 或 `grpo_opsd`。
`groove` 是 `grpo_opsd` 的命令行别名，方法名称统一写作 **GRPO + OPSD**。
复制脚本后可修改顶部常用设置或对应模块的参数数组，不需要维护另一层机器 wrapper。

```bash
TRAINING_MODE=grpo EXPERIMENT_NAME=my-grpo-4k-run01 \
  bash scripts/train_a800_4gpu.sh

TRAINING_MODE=dapo EXPERIMENT_NAME=my-dapo-4k-run01 \
  bash scripts/train_a800_4gpu.sh

TRAINING_MODE=grpo_opsd EXPERIMENT_NAME=my-opsd-4k-run01 \
  bash scripts/train_siton_2gpu.sh
```

每次正式运行设置新的 `EXPERIMENT_NAME`。默认从基础模型新训，
`RESUME_MODE=disable`；已有检查点的同名目录会被拒绝。
显式续训使用 `RESUME_MODE=resume_path`、`RESUME_FROM_PATH=/path/to/global_step_N`。

只检查配置时加 `GROOVE_DRY_RUN=true`。它不启动 Ray 或加载模型权重，不能替代实际 GPU 训练验证。
Siton 脚本在本机 dry-run 时需要覆盖 `PYTHON_BIN` 和 `MODEL_PATH`。

完整参数、Trainer 分流及 DAPO 语义见 [训练入口说明](../docs/TRAINING_LAUNCHERS.md)。

## 保留的常用工具

- `prepare_vstar_grpo.py`：可复现的筛选数据准备；其原 CLI 的 sample-size 包含 train/validation，不能直接当作 4000 条训练数据的命令。
- `materialize_opsd_images.py`：为 OPSD 建立本机图片路径；迁移机器后重新生成到新目录。
- `prepare_vstar_validation.py`：准备完整 191 题 V*Bench。
- `evaluate_vstar.py`、`compare_vstar_evaluations.py`：独立模型评测及比较。
- `analyze_opd_tokens.py`、`report_trajectory_reasoning.py`、`summarize_training_progress.py`：日常训练诊断。
- `paddle_ocr_tool.py`：项目视觉工具调用的运行时组件。
- `serve_2b_after_training.py`：训练结束后的独立 vLLM 服务监督器。

一次性筛选、旧探测、重检查和历史启动链已迁至 [TMP/scripts](../TMP/scripts/README.md)。
