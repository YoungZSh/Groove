# Gemini API Analyzer（离线对照入口）

`src/groove/gemini_analyzer.py::GeminiAPIAnalyzer` 实现独立的 Gemini 3.8 Flash
Analyzer。基于 Ground Truth 的轨迹复核、共用核验规则提炼、目标定位和候选选择
都由同一个 Gemini 模型完成；
本地仅校验坐标、裁剪图像并返回预览，不调用 DINO、OCR、Judge 或 Student。
现有训练入口仍使用原 Analyzer；此实现没有接入训练路由或改动训练参数。

## 官方协议依据

核对日期：2026-09-26。

- [图像理解与目标检测](https://ai.google.dev/gemini-api/docs/image-understanding)：
  模型直接预测 `box_2d=[ymin,xmin,ymax,xmax]`，坐标按原图归一化到 0–1000。
  Bounding box 不是一个由 Google 托管的检测工具。本实现把模型预测出的坐标
  作为自定义 `crop_image` 函数参数，由程序执行裁剪后送回模型检查。
- [OpenAI 兼容接口](https://ai.google.dev/gemini-api/docs/openai)：
  使用 `OpenAI(base_url=..., api_key=...).chat.completions.create(...)`，
  通过 `reasoning_effort="high"` 请求高推理档；不同时发送 `thinking_config`、
  `thinking_budget` 或 Qwen 的 `chat_template_kwargs`。
- [模型说明](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash/)：
  Gemini 3.8 Flash 支持函数调用、图像输入和 high 推理档。
- [函数调用](https://ai.google.dev/gemini-api/docs/function-calling)及
  [OpenAI 兼容接口的推理说明](https://ai.google.dev/gemini-api/docs/openai#thinking)：
  完整保留返回的 assistant 消息、tool call ID 和附加字段，包括中转站转发的
  thought signature。多个并行调用先追加全部 tool 响应，再追加裁剪预览。

这些文档说明官方接口行为。中转站是否实际转发 high、保留签名，以及它返回的模型名
是否对应预期版本，仍取决于中转站；代码不会在不兼容时静默降档或切换模型。

## 配置与依赖

可选依赖组为 `gemini-analyzer`（`openai`、`python-dotenv`）。当前项目环境已安装。

从指定 `.env` 读取以下变量，已有进程环境中的同名变量优先：

```dotenv
OPENAI_BASE_URL=https://your-relay.example/v1
OPENAI_API_KEY=your-private-key
OPENAI_MODEL=gemini-3.8-flash
```

不修改 `.env` 或进程环境，不执行 shell，不展开 `${...}`。API Key 不参与配置的
字符串表示，不写入清单或审计文件。Base URL 按 SDK 约定直接使用；也兼容完整
`/chat/completions` 地址，不自动补 `/v1`，避免破坏自定义路径和官方
`/v1beta/openai/` 路径。Base URL 中不接受内嵌凭证或查询参数。

固定 high；默认温度 1.0，单次生成预算 16384 tokens（包含思考开销），请求超时
300 秒，SDK 最多重试 2 次。默认最多 3 轮工具调用，每轮最多执行 3 个裁剪；
额外调用返回预算错误。工具结束后若需要再进行一次最终选择；最终 JSON/选择不合法时
最多修复一次。截断或拒绝的回答不会当作有效结果。

## 证据流程

1. 系统提示词完整写在 `src/groove/gemini_analyzer.py::GEMINI_SYSTEM_PROMPT`
   的单个多行字符串中，可直接审阅；不导入旧 `SYSTEM_PROMPT`，不通过其他提示词变量
   拼接、切片或替换生成。任务、预览检查、候选选择和最终 JSON 格式均在该文本中。
   `build_gemini_analysis_text()` 发送问题、`ground_truth` 和原有成功/失败推理列表，
   原图作为同一请求的图像输入。标准答案是答案正确性的主要参考；原标签可能存在
   false positive/false negative，不能把答案正确等同于推理正确。图像与 GT 的未解冲突
   写入私有诊断，不强行编造证据，也不修改原 `is_correct` 或训练奖励。
   `uid`、`rollout_id`、`predicted_label` 和数值奖励不会单独发送。
   原 Qwen 的 `build_group_analysis_text()` 仍不发送 GT。
2. Gemini 调用 `crop_image(query, box_2d)`。所有坐标均相对原图，即使已经看过裁剪。
   校验四个整数、范围和正面积；拒绝越界、倒置或非整数坐标，不猜测坐标顺序或单位。
3. 按原图宽高转换为像素 `XYXY`，添加与现有方法一致的 12% 上下文边距，返回裁剪。
   预览最长边默认 1024，使用与原 Analyzer 相同的裁剪编码函数。
4. Gemini 检查预览，可修正坐标；最终仅返回三个字段：

   ```json
   {
     "group_summary": "One-sentence diagnosis or unresolved conflict",
     "visible_focus_instruction": "Short answer-neutral shared verification rule",
     "selected_candidate_ids": ["candidate_1"]
   }
   ```

   `group_summary` 用一句话合并分歧诊断和必要的证据说明，只供审计。
   `visible_focus_instruction` 提炼本组共用的可执行核验步骤和易错点，传给 Teacher；
   不写答案结论、选项字母、OCR 转录、奖励或轨迹标签。
   最终明确选择 1–3 个可用候选。
   候选不进行 IoU 去重，不自动选择最新候选，不接受模型在最终 JSON 中伪造的框。
   非法、空白或多余字段会触发一次格式修复，不接受旧的八字段模型输出。
5. 程序将三字段结果适配为共用 `FocusProgram`，固定 `tool_route="gemini"`，
   `grounding_queries` 取自所选工具记录，区域来源为 `gemini_native_bbox`。
   不再让模型输出 `crucial_evidence`、`crucial_evidence_type`、`tool_route`、
   `grounding_queries` 或 `confidence`。为兼容旧证据记录，内部保留字段：
   `crucial_evidence=""`、`crucial_evidence_type="unknown"`、`confidence=null`。
   统计单列 unknown，不把未分类的 Gemini 证据默认为 visual 或 text。
   区域 `score=1.0` 仅表示有效候选，不是校准的检测置信度，不参与连续权重计算。
   原始预测框和归一化坐标保留在 `tool_trace`，Teacher 的实际边距框保存在区域记录中。
6. 复用 `TeacherEvidenceBuilder`、`crop_tool_regions()` 和答案泄漏检查，生成共用
   `TeacherEvidence`。没有证据时记录 error；不回退到 DINO。所有轨迹、坐标与工具
   审计和 Ground Truth 留在 Analyzer/审计侧，Teacher 只得到允许的共用核验规则及
   所选图像，Student 输入不变。现有规则式泄漏检查继续使用，但不保证所有语义泄漏
   都能被识别；规则与证据质量仍需人工抽查或独立评估。

文本区域也由 Gemini 定位和检查，不调用独立 OCR，也不向 Teacher 注入转录文本。
计数可以选择包含相关对象的区域；此入口不生成现有 DINO 的实例框叠加图。

## 离线调用

Python 接口与现有 Analyzer 一样：

```python
from groove.gemini_analyzer import GeminiAPIAnalyzer, GeminiAnalyzerConfig

analyzer = GeminiAPIAnalyzer(GeminiAnalyzerConfig.from_env(".env"))
try:
    focus = analyzer.analyze(group)  # GroupRollout 必须含非空 ground_truth
    boxes = [region.expanded_box for region in focus.tool_regions]
finally:
    analyzer.close()
```

CLI 输入为 GroupRollout JSONL，每行一组，例如：

```json
{"uid":"step-5-sample-17","question":"What color is the small sign?","ground_truth":"red","image_path":"/absolute/path/image.jpg","rollouts":[{"rollout_id":0,"completion":"The sign appears red. <answer>red</answer>","predicted_label":null,"is_correct":true},{"rollout_id":1,"completion":"It seems blue. <answer>blue</answer>","predicted_label":null,"is_correct":false}]}
```

相对图像路径相对 groups 文件的所在目录解析。所有成功或所有失败的组也可以分析。
`is_correct` 必须由已有语义 `accuracy` 得到，不能按塑形 `score` 判断或重新调用 Judge。
这些是待复核的原始标签，不是 Analyzer 必须认同的结论。
`ground_truth` 必须是来自数据记录的非空答案字符串，不能从 Judge 标签或模型回答猜测。
缺失或空白时，Python 接口在发请求前报错，CLI 在写输出前报错。
共用 `GroupRollout` 的字段仍可为空，以兼容不读取 GT 的原 Analyzer 和历史记录。
如果准备当前训练的回放输入，沿用训练中 `repetition_start_character` 的截断规则。

**现有原始 rollout JSONL 不能直接当作此输入。** 当前落盘通常只有 `input/output`、
`accuracy`、`gts` 等字段，缺少原图路径及可靠的组标识。对照前需与对应训练 parquet
或可核实的样本映射对齐，生成上述 groups；不要按题目字符串或简单每 8 行自动猜图。
对齐后将真实 `gts` 写入 `ground_truth`；选择题保留答案文本，不能仅凭选项字母猜含义。
历史不含 GT 的 groups 需要先补齐来源可核验的答案，再写入新的输入文件，
不要覆盖历史回放记录。GT 辅助版本与旧版无 GT 的 Gemini 结果须区分协议比较。

先只校验，不发 API 请求、不写结果：

```bash
PYTHONPATH="$PWD/src" /data/home/yangzesheng/.conda/envs/groove/bin/python \
  scripts/analyze_gemini_rollouts.py \
  --groups /absolute/path/groups.jsonl \
  --output-dir outputs/gemini-analyzer/unique-run-name \
  --limit 2 --dry-run
```

去掉 `--dry-run` 才会调用中转站；支持 `--env-file`、`--max-tool-rounds` 和
`--max-completion-tokens`。任何已有输出目录均拒绝使用，不覆盖历史记录。
组内失败落盘后继续后续组，若存在失败最终退出码为 1；全部 ready 则为 0。

输出包括：

- `manifest.json`：不含 API Key 的配置、源文件路径及 SHA-256，
  `analysis_protocol="ground-truth-shared-rule-v1"` 和系统提示词 SHA-256。
- `group-000001/group.json`：完整本地组记录，含 Ground Truth 和原始标签；
  ID 等仅供本地追溯，模型请求字段以上述输入构造为准。
- `group-000001/<uid>/evidence.json`：最终选择、像素框、Teacher 提示词、完整工具轨迹。
- 同目录的 `tool-crop-*.jpg`：所选证据图像。
- `group-000001/api_trace.json`：逐轮请求结构、完整返回、usage、耗时；请求中的图片
  data URL 以 SHA-256 引用替代，可通过原图和工具框重建，不重复保存大量 base64。
  中转站请求异常只记录异常类型和 HTTP 状态，不保存可能回显凭证的原始报错正文。
- `summary.json`：ready/error/skipped 数量。

同一 Analyzer 实例按组顺序调用会清空上一组轨迹。并发时每个 worker 使用独立实例。
对照测试固定轨迹顺序、原始 accuracy、原图、预览预算和图像后处理，并使用不同的新输出目录。
记录各方法是否读取 GT；当前 Gemini 接收 GT、原 Qwen 不接收，不能把差异完全归因于模型能力。
