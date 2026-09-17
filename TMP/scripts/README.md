# 临时与历史脚本

这里保存一次性筛选、探测、重检查、旧模型兼容转换及历史训练入口。
它们不是当前正式训练入口；运行前核对脚本内的数据、权重和输出路径。

当前独立训练入口见 ../../scripts/README.md。

迁移时只调整仓库根目录定位和脚本之间的路径引用，保留旧实验参数。
历史启动器仍保留旧的调用链，用于追溯；新启动器不引用本目录。
已有数据、检查点、日志、模型以及 outputs 下的历史记录均未移动。

## 分类

- 数据筛选：probe_vstar_vllm.py、judge_vstar_rollouts.py、run_vstar_1000.sh、run_vstar_remaining.sh、summarize_vstar_screening.py。
- 历史准备：prepare_vision_opd.py、prepare_vstar.py、download_qwen35_analyzer.py、normalize_qwen35_export.py。
- 诊断与重检查：probe_vstar_advantages.py、compare_vstar_teacher_views.py、run_vstar_advantage_probe.sh、run_analyzer_tool_probe.py、recheck_auto_analyzer.py、recheck_english_grounding.py、make_grounding_contact_sheet.py、render_vllm_rollout_report.py、report_baseline_validation.py。
- 历史训练与服务：run_groove.sh、run_grpo_2b.sh、run_grpo_opsd_2b.sh、run_2b_4gpu.sh、run_grpo_ablation.sh、run_baseline_training.sh、launch_local_analyzer_training.sh、serve_qwen35_analyzer.sh、supervise_formal_training.sh。
