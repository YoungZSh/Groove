#!/usr/bin/env python3
"""Extract CoFFT-style relative sentence-to-image attention maps from Qwen3.5.

The generated completion is teacher-forced exactly as recorded in traces.jsonl.
For every visible sentence token, the query position is shifted by one token so
the captured attention is the attention actually used to predict that token.
The same image is paired with a generic descriptive prompt to form the
denominator: softmax(A(image, sentence) / (A(image, description) + epsilon)).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


DEFAULT_MODEL = Path("/root/siton-tmp/yzs/ckpts/Qwen3.5-4B")
DEFAULT_DATA = Path(
    "/root/siton-tmp/yzs/datasets/vstar-bench/data/test-00000-of-00001.parquet"
)
DEFAULT_TRACES = Path(
    "/root/siton-tmp/yzs/mmcot_opsd/outputs/qwen3.5-4b-vstar/traces.jsonl"
)
DEFAULT_OUTPUT = Path("/root/siton-tmp/yzs/mmcot_opsd/outputs/attention-analysis")
DEFAULT_VISUAL = Path(
    "/root/siton-tmp/yzs/.codex/visualizations/2026/08/23/"
    "01a02f60-6fc9-7c42-b245-4c0ed1d73db6/qwen35-vstar-attention.html"
)
DESCRIPTIVE_PROMPT = "Describe the image in detail"
RELATIVE_EPSILON = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visual-output", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--indices", type=int, nargs="+", default=[21, 189])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--occlusion-fraction", type=float, default=0.10)
    return parser.parse_args()


def load_trace_records(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(record["index"]): record
        for record in (json.loads(line) for line in path.open(encoding="utf-8"))
    }


def sentence_char_spans(text: str) -> list[tuple[int, int, str]]:
    """Split visible output into sentence-like spans while preserving offsets."""
    spans: list[tuple[int, int, str]] = []
    for line_match in re.finditer(r"[^\n]+", text):
        line = line_match.group(0)
        line_start = line_match.start()
        cursor = 0
        # A line can contain multiple prose sentences; answer-only lines stay whole.
        for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", line):
            fragment = match.group(0)
            left = len(fragment) - len(fragment.lstrip())
            right = len(fragment.rstrip())
            if right <= left:
                continue
            start = line_start + match.start() + left
            end = line_start + match.start() + right
            value = text[start:end]
            spans.append((start, end, value))
            cursor = match.end()
        if cursor < len(line) and line[cursor:].strip():
            start = line_start + cursor + len(line[cursor:]) - len(line[cursor:].lstrip())
            spans.append((start, line_match.end(), text[start : line_match.end()]))
    return spans


def prepare_teacher_forced_inputs(
    processor: Any,
    image: Image.Image,
    record: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, list[int], tuple[int, int]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": record["question"]},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if prompt != record["prompt"]:
        raise ValueError(f"Reconstructed prompt differs for index {record['index']}")

    batch = processor(text=[prompt], images=[image], return_tensors="pt")
    prompt_length = int(batch["input_ids"].shape[1])
    generated_ids = [int(value) for value in record["generated_token_ids"]]
    generated = torch.tensor([generated_ids], dtype=batch["input_ids"].dtype)
    batch["input_ids"] = torch.cat((batch["input_ids"], generated), dim=1)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    generated_types = torch.zeros_like(generated, dtype=batch["mm_token_type_ids"].dtype)
    batch["mm_token_type_ids"] = torch.cat(
        (batch["mm_token_type_ids"], generated_types), dim=1
    )

    image_positions = torch.where(batch["mm_token_type_ids"][0] == 1)[0].tolist()
    grid_t, grid_h, grid_w = [int(v) for v in batch["image_grid_thw"][0].tolist()]
    if grid_t != 1:
        raise ValueError("This analysis expects one still image")
    merge = 2
    llm_grid = (grid_h // merge, grid_w // merge)
    if len(image_positions) != math.prod(llm_grid):
        raise ValueError(
            f"Image token/grid mismatch: {len(image_positions)} versus {llm_grid}"
        )

    encoded = processor.tokenizer(
        record["completion"],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    special_ids = set(processor.tokenizer.all_special_ids)
    visible_positions = [
        index for index, token_id in enumerate(generated_ids) if token_id not in special_ids
    ]
    visible_ids = [generated_ids[index] for index in visible_positions]
    if visible_ids != encoded["input_ids"]:
        raise ValueError(f"Completion token round-trip mismatch for index {record['index']}")

    sentence_specs: list[dict[str, Any]] = []
    for sentence_index, (start, end, text) in enumerate(
        sentence_char_spans(record["completion"]), start=1
    ):
        completion_token_indices = [
            token_index
            for token_index, (token_start, token_end) in enumerate(
                encoded["offset_mapping"]
            )
            if token_end > start and token_start < end
        ]
        if not completion_token_indices:
            continue
        generated_positions = [visible_positions[i] for i in completion_token_indices]
        # Query q predicts target q+1, so shift each target token back by one.
        query_positions = [prompt_length + position - 1 for position in generated_positions]
        sentence_specs.append(
            {
                "sentence_index": sentence_index,
                "text": text,
                "char_start": start,
                "char_end": end,
                "completion_token_indices": completion_token_indices,
                "generated_positions": generated_positions,
                "query_positions": query_positions,
            }
        )

    return (
        {key: value.to(device) for key, value in batch.items()},
        sentence_specs,
        prompt_length,
        image_positions,
        llm_grid,
    )


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    matches = [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    ]
    if not matches:
        raise ValueError("Descriptive prompt tokens were not found in model input")
    return matches[-1]


def prepare_descriptive_inputs(
    processor: Any,
    image: Image.Image,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[int], tuple[int, int]]:
    """Build the same-image generic-description baseline used by relative attention."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DESCRIPTIVE_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    batch = processor(text=[prompt], images=[image], return_tensors="pt")
    input_ids = batch["input_ids"][0].tolist()
    description_ids = processor.tokenizer(
        DESCRIPTIVE_PROMPT, add_special_tokens=False
    )["input_ids"]
    description_start = find_subsequence(input_ids, description_ids)
    # Match the sentence analysis semantics: query q is attributed to target q+1.
    query_positions = [
        description_start + offset - 1 for offset in range(len(description_ids))
    ]

    image_positions = torch.where(batch["mm_token_type_ids"][0] == 1)[0].tolist()
    grid_t, grid_h, grid_w = [int(v) for v in batch["image_grid_thw"][0].tolist()]
    if grid_t != 1:
        raise ValueError("This analysis expects one still image")
    llm_grid = (grid_h // 2, grid_w // 2)
    if len(image_positions) != math.prod(llm_grid):
        raise ValueError(
            f"Description image token/grid mismatch: {len(image_positions)} versus {llm_grid}"
        )
    spec = [
        {
            "sentence_index": 0,
            "text": DESCRIPTIVE_PROMPT,
            "query_positions": query_positions,
        }
    ]
    return (
        {key: value.to(device) for key, value in batch.items()},
        spec,
        image_positions,
        llm_grid,
    )


def capture_full_attention(
    model: Qwen3_5ForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    sentences: list[dict[str, Any]],
    image_positions: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Capture raw full-attention maps and image attention mass per sentence/layer."""
    full_layers = [
        (index, layer.self_attn)
        for index, layer in enumerate(model.model.language_model.layers)
        if layer.layer_type == "full_attention"
    ]
    image_index = torch.tensor(image_positions, dtype=torch.long, device=inputs["input_ids"].device)
    layer_maps: dict[int, list[np.ndarray]] = {s["sentence_index"]: [] for s in sentences}
    layer_masses: dict[int, list[float]] = {s["sentence_index"]: [] for s in sentences}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            attention = output[1]
            if attention is None:
                raise RuntimeError(
                    f"Layer {layer_index + 1} returned no attention weights; eager attention is required"
                )
            for sentence in sentences:
                query_index = torch.tensor(
                    sentence["query_positions"], dtype=torch.long, device=attention.device
                )
                selected = attention[0].index_select(1, query_index).index_select(2, image_index)
                mean_map = selected.float().mean(dim=(0, 1))
                mass = selected.float().sum(dim=-1).mean()
                layer_maps[sentence["sentence_index"]].append(
                    mean_map.detach().cpu().numpy()
                )
                layer_masses[sentence["sentence_index"]].append(float(mass.item()))

        return hook

    for layer_index, attention_module in full_layers:
        handles.append(attention_module.register_forward_hook(make_hook(layer_index)))

    model.config.text_config._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            outputs = model.model(**inputs, use_cache=False)
        del outputs
    finally:
        for handle in handles:
            handle.remove()
        model.config.text_config._attn_implementation = "flash_attention_2"

    return (
        {key: np.stack(value) for key, value in layer_maps.items()},
        {key: np.asarray(value, dtype=np.float32) for key, value in layer_masses.items()},
    )


def normalize_heatmap(raw_map: np.ndarray) -> tuple[np.ndarray, float]:
    conditional = raw_map.astype(np.float64)
    total = float(conditional.sum())
    if total > 0:
        conditional /= total
    positive = conditional[conditional > 0]
    if positive.size == 0:
        return np.zeros_like(conditional, dtype=np.float32), 0.0
    ceiling = float(np.quantile(positive, 0.99))
    visual = np.clip(conditional / max(ceiling, np.finfo(float).eps), 0, 1)
    visual = np.sqrt(visual).astype(np.float32)
    entropy = float(-(conditional * np.log(conditional + 1e-30)).sum())
    entropy /= math.log(conditional.size) if conditional.size > 1 else 1
    return visual, entropy


def relative_attention(
    task_attention: np.ndarray,
    descriptive_attention: np.ndarray,
    epsilon: float = RELATIVE_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CoFFT Eq. 1 element-wise over image patches with stable softmax."""
    ratio = task_attention.astype(np.float64) / (
        descriptive_attention.astype(np.float64) + epsilon
    )
    weights = np.exp(ratio - float(ratio.max()))
    weights /= float(weights.sum())
    return ratio.astype(np.float32), weights.astype(np.float32)


def occlude_cells(
    image: Image.Image,
    grid_shape: tuple[int, int],
    selected: np.ndarray,
) -> Image.Image:
    result = image.copy()
    array = np.asarray(image, dtype=np.float32)
    fill = tuple(int(value) for value in array.reshape(-1, 3).mean(axis=0))
    draw = ImageDraw.Draw(result)
    grid_h, grid_w = grid_shape
    width, height = result.size
    for flat_index in selected.tolist():
        row, col = divmod(int(flat_index), grid_w)
        left = round(col * width / grid_w)
        right = round((col + 1) * width / grid_w)
        top = round(row * height / grid_h)
        bottom = round((row + 1) * height / grid_h)
        draw.rectangle((left, top, right, bottom), fill=fill)
    return result


def score_options(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    image: Image.Image,
    record: dict[str, Any],
    options: list[str],
    device: str,
) -> tuple[dict[str, float], dict[str, int]]:
    forced_prefix = "The correct answer is ("
    prompt_batch = processor(
        text=[record["prompt"] + forced_prefix], images=[image], return_tensors="pt"
    )
    candidate_ids: dict[str, int] = {}
    for option in options:
        token_ids = processor.tokenizer(option, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"Forced-choice option {option} is not one token")
        candidate_ids[option] = int(token_ids[0])
    prompt_batch = {key: value.to(device) for key, value in prompt_batch.items()}
    model.config.text_config._attn_implementation = "flash_attention_2"
    with torch.inference_mode():
        logits = model(
            **prompt_batch,
            use_cache=False,
            logits_to_keep=1,
        ).logits[0, -1]
    option_logits = torch.stack([logits[candidate_ids[option]] for option in options])
    probabilities = torch.softmax(option_logits.float(), dim=0).cpu().tolist()
    return (
        {option: float(probability) for option, probability in zip(options, probabilities)},
        candidate_ids,
    )


def heat_overlay(image: Image.Image, heat: np.ndarray, grid: tuple[int, int]) -> Image.Image:
    grid_h, grid_w = grid
    heat_uint8 = np.clip(heat.reshape(grid_h, grid_w) * 255, 0, 255).astype(np.uint8)
    heat_image = Image.fromarray(heat_uint8, mode="L").resize(
        image.size, Image.Resampling.BILINEAR
    )
    values = np.asarray(heat_image, dtype=np.float32) / 255.0
    # Conventional attention palette: yellow -> orange -> red.
    low = np.array([255, 225, 55], dtype=np.float32)
    mid = np.array([255, 125, 15], dtype=np.float32)
    high = np.array([210, 25, 20], dtype=np.float32)
    colors = np.empty((*values.shape, 3), dtype=np.float32)
    lower = values <= 0.5
    mix = np.clip(values * 2, 0, 1)[..., None]
    colors[lower] = (low + (mid - low) * mix)[lower]
    upper_mix = np.clip((values - 0.5) * 2, 0, 1)[..., None]
    colors[~lower] = (mid + (high - mid) * upper_mix)[~lower]
    original = np.asarray(image, dtype=np.float32)
    alpha = (0.72 * np.power(values, 0.7))[..., None]
    composed = original * (1 - alpha) + colors * alpha
    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


def wrap_pixels(draw: ImageDraw.ImageDraw, text: str, font: Any, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def save_montage(
    sample: dict[str, Any],
    image: Image.Image,
    output: Path,
) -> None:
    panel_width = 1100
    resized = image.copy()
    resized.thumbnail((panel_width, 620), Image.Resampling.LANCZOS)
    title_font = load_font(27, bold=True)
    body_font = load_font(21)
    small_font = load_font(18)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    header_lines = wrap_pixels(probe, sample["question"].replace("\n", " "), title_font, panel_width - 48)
    header_height = 44 + len(header_lines) * 35 + 58
    sentence_blocks = []
    for sentence in sample["sentences"]:
        lines = wrap_pixels(probe, sentence["text"], body_font, panel_width - 48)
        caption_height = 28 + len(lines) * 28 + 34
        sentence_blocks.append((sentence, lines, caption_height + resized.height))
    total_height = header_height + sum(height + 22 for _, _, height in sentence_blocks)
    canvas = Image.new("RGB", (panel_width, total_height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    status = "CORRECT" if sample["correct"] else "WRONG"
    draw.text((24, 18), f"V*Bench #{sample['index']} · {status}", font=title_font, fill=(24, 24, 24))
    y = 60
    for line in header_lines:
        draw.text((24, y), line, font=body_font, fill=(24, 24, 24))
        y += 31
    draw.text(
        (24, y + 8),
        f"gold={sample['gold']}   predicted={sample['predicted']}   attention=relative   full layers={sample['layers']}",
        font=small_font,
        fill=(70, 70, 70),
    )
    y = header_height
    for sentence, lines, block_height in sentence_blocks:
        overlay = heat_overlay(image, np.asarray(sentence["heat"], dtype=np.float32) / 255, tuple(sample["grid"]))
        overlay.thumbnail((panel_width, 620), Image.Resampling.LANCZOS)
        canvas.paste(overlay, ((panel_width - overlay.width) // 2, y))
        y += overlay.height + 10
        draw.text(
            (24, y),
            f"S{sentence['sentence_index']} · absolute image mass {sentence['absolute_image_mass'] * 100:.2f}% · relative attention",
            font=small_font,
            fill=(70, 70, 70),
        )
        y += 28
        for line in lines:
            draw.text((24, y), line, font=body_font, fill=(24, 24, 24))
            y += 28
        y += 18
    canvas.save(output, quality=91, optimize=True)


def image_data_url(image: Image.Image) -> str:
    display = image.copy()
    display.thumbnail((1100, 760), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    display.save(buffer, format="JPEG", quality=76, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_visual_fragment(samples: list[dict[str, Any]], path: Path) -> None:
    visual_samples = []
    for sample in samples:
        visual_samples.append(
            {
                "index": sample["index"],
                "correct": sample["correct"],
                "question": sample["question"],
                "gold": sample["gold"],
                "predicted": sample["predicted"],
                "completion": sample["completion"],
                "image": sample["image_data_url"],
                "imageWidth": sample["image_width"],
                "imageHeight": sample["image_height"],
                "grid": sample["grid"],
                "baselinePrompt": sample["descriptive_baseline_prompt"],
                "sentences": [
                    {
                        "sentenceIndex": sentence["sentence_index"],
                        "text": sentence["text"],
                        "heat": sentence["heat"],
                        "absoluteImageMass": sentence["absolute_image_mass"],
                        "entropy": sentence["relative_spatial_entropy"],
                        "peak": sentence["peak"],
                    }
                    for sentence in sample["sentences"]
                ],
                "causal": sample["causal"],
            }
        )
    data_json = json.dumps(visual_samples, ensure_ascii=False, separators=(",", ":"))
    fragment = f'''<div id="q35-vstar-attention-v2">
  <h2>Qwen3.5-4B · V*Bench 逐句相对图像注意力</h2>
  <div class="q35-method text-small">分子：当前句 · 分母：同图描述提示 “Describe the image in detail” · 逐 patch 比值后 Softmax</div>
  <div class="viz-controls" aria-label="样本与句子选择">
    <div id="q35-sample-buttons" class="viz-row"></div>
    <label class="form-label" for="q35-sentence-select">句子
      <select id="q35-sentence-select" class="form-select"></select>
    </label>
    <label class="form-label" for="q35-opacity">热力图透明度 <span id="q35-opacity-value">68%</span>
      <input id="q35-opacity" class="form-range" type="range" min="0" max="100" value="68">
    </label>
  </div>
  <div id="q35-sentence-detail" class="card" aria-live="polite"></div>
  <canvas id="q35-attention-canvas" role="img" aria-label="所选句子到原图 patch 的相对注意力热力图"></canvas>
  <div class="q35-legend"><span>低</span><span class="q35-gradient" aria-hidden="true"></span><span>高</span></div>
  <div id="q35-occlusion" aria-live="polite"></div>
</div>
<style>
  #q35-vstar-attention-v2 {{ color: var(--foreground); width: 100%; }}
  #q35-vstar-attention-v2 .q35-method {{ color: var(--muted-foreground); margin: -4px 0 12px; }}
  #q35-vstar-attention-v2 .viz-controls {{ align-items: end; }}
  #q35-vstar-attention-v2 .form-label {{ min-width: min(100%, 260px); }}
  #q35-vstar-attention-v2 .card {{ margin: 12px 0; padding: 12px; }}
  #q35-attention-canvas {{ display: block; width: 100%; border: 1px solid var(--border); }}
  #q35-vstar-attention-v2 .q35-legend {{ display: grid; grid-template-columns: auto minmax(120px, 280px) auto; gap: 8px; align-items: center; margin: 8px 0 16px; color: var(--muted-foreground); }}
  #q35-vstar-attention-v2 .q35-gradient {{ height: 10px; background: linear-gradient(to right, var(--yellow), var(--orange), var(--red)); }}
  #q35-vstar-attention-v2 .q35-meta {{ color: var(--muted-foreground); margin-top: 6px; }}
  #q35-vstar-attention-v2 .q35-bars {{ display: grid; gap: 8px; margin-top: 10px; }}
  #q35-vstar-attention-v2 .q35-bar-row {{ display: grid; grid-template-columns: minmax(145px, 220px) 1fr 56px; gap: 10px; align-items: center; }}
  #q35-vstar-attention-v2 .q35-track {{ height: 10px; background: var(--muted); }}
  #q35-vstar-attention-v2 .q35-fill {{ height: 100%; background: var(--orange); }}
  @media (max-width: 520px) {{
    #q35-vstar-attention-v2 .q35-bar-row {{ grid-template-columns: 1fr 48px; }}
    #q35-vstar-attention-v2 .q35-bar-row > span:first-child {{ grid-column: 1 / -1; }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById('q35-vstar-attention-v2');
  const samples = {data_json};
  const buttons = root.querySelector('#q35-sample-buttons');
  const sentenceSelect = root.querySelector('#q35-sentence-select');
  const detail = root.querySelector('#q35-sentence-detail');
  const canvas = root.querySelector('#q35-attention-canvas');
  const opacity = root.querySelector('#q35-opacity');
  const opacityValue = root.querySelector('#q35-opacity-value');
  const occlusion = root.querySelector('#q35-occlusion');
  let sampleIndex = 0;
  let sentenceIndex = 0;
  let loadedImage = null;

  samples.forEach((sample, index) => {{
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn' + (index === 0 ? ' btn-primary' : '');
    button.setAttribute('aria-pressed', index === 0 ? 'true' : 'false');
    button.textContent = `${{sample.correct ? '正确' : '错误'}}样本 · #${{sample.index}}`;
    button.addEventListener('click', () => selectSample(index));
    buttons.appendChild(button);
  }});

  function selectSample(index) {{
    sampleIndex = index;
    sentenceIndex = 0;
    [...buttons.children].forEach((button, i) => {{
      button.className = 'btn' + (i === index ? ' btn-primary' : '');
      button.setAttribute('aria-pressed', i === index ? 'true' : 'false');
    }});
    sentenceSelect.replaceChildren();
    samples[index].sentences.forEach((sentence, i) => {{
      const option = document.createElement('option');
      option.value = String(i);
      const shortText = sentence.text.length > 72 ? sentence.text.slice(0, 69) + '…' : sentence.text;
      option.textContent = `S${{sentence.sentenceIndex}} · ${{shortText}}`;
      sentenceSelect.appendChild(option);
    }});
    loadedImage = new Image();
    loadedImage.onload = () => draw();
    loadedImage.src = samples[index].image;
    renderOcclusion();
    updateDetail();
  }}

  function updateDetail() {{
    const sample = samples[sampleIndex];
    const sentence = sample.sentences[sentenceIndex];
    detail.innerHTML = '';
    const main = document.createElement('div');
    main.textContent = `S${{sentence.sentenceIndex}} · ${{sentence.text}}`;
    const meta = document.createElement('div');
    meta.className = 'q35-meta text-small';
    meta.textContent = `绝对图像注意力占比 ${{(sentence.absoluteImageMass * 100).toFixed(2)}}% · 相对空间熵 ${{sentence.entropy.toFixed(3)}} · 相对峰值坐标 (${{sentence.peak[0].toFixed(2)}}, ${{sentence.peak[1].toFixed(2)}})`;
    detail.append(main, meta);
  }}

  function draw() {{
    if (!loadedImage || !loadedImage.complete) return;
    const sample = samples[sampleIndex];
    const sentence = sample.sentences[sentenceIndex];
    const width = Math.max(320, canvas.clientWidth || root.clientWidth);
    const height = width * sample.imageHeight / sample.imageWidth;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.aspectRatio = `${{sample.imageWidth}} / ${{sample.imageHeight}}`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(loadedImage, 0, 0, width, height);
    const styles = getComputedStyle(root);
    const paletteCanvas = document.createElement('canvas');
    paletteCanvas.width = 256;
    paletteCanvas.height = 1;
    const paletteContext = paletteCanvas.getContext('2d');
    const gradient = paletteContext.createLinearGradient(0, 0, 256, 0);
    gradient.addColorStop(0, styles.getPropertyValue('--yellow').trim());
    gradient.addColorStop(0.55, styles.getPropertyValue('--orange').trim());
    gradient.addColorStop(1, styles.getPropertyValue('--red').trim());
    paletteContext.fillStyle = gradient;
    paletteContext.fillRect(0, 0, 256, 1);
    const palette = paletteContext.getImageData(0, 0, 256, 1).data;
    const rows = sample.grid[0];
    const cols = sample.grid[1];
    const alphaScale = Number(opacity.value) / 100;
    sentence.heat.forEach((value, flatIndex) => {{
      if (value <= 0) return;
      const row = Math.floor(flatIndex / cols);
      const col = flatIndex % cols;
      const x0 = col * width / cols;
      const y0 = row * height / rows;
      const x1 = (col + 1) * width / cols;
      const y1 = (row + 1) * height / rows;
      const paletteOffset = Math.min(255, Math.max(0, value)) * 4;
      ctx.fillStyle = `rgb(${{palette[paletteOffset]}}, ${{palette[paletteOffset + 1]}}, ${{palette[paletteOffset + 2]}})`;
      ctx.globalAlpha = alphaScale * Math.pow(value / 255, 0.72);
      ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
    }});
    ctx.globalAlpha = 1;
  }}

  function renderOcclusion() {{
    const sample = samples[sampleIndex];
    const label = sample.predicted;
    const values = [
      ['原图', sample.causal.original[label]],
      ['遮挡相对注意力最高 10% patch', sample.causal.top_attention_occlusion[label]],
      ['随机遮挡同等数量 patch', sample.causal.random_occlusion[label]]
    ];
    occlusion.innerHTML = '<h3>强制选择干预（高相对注意力区域来自 S' + sample.causal.attention_source_sentence + '）· P(' + label + ' | ' + sample.causal.options.join('–') + ')</h3>';
    const bars = document.createElement('div');
    bars.className = 'q35-bars';
    values.forEach(([name, value]) => {{
      const row = document.createElement('div');
      row.className = 'q35-bar-row';
      const labelNode = document.createElement('span');
      labelNode.textContent = name;
      const track = document.createElement('div');
      track.className = 'q35-track';
      const fill = document.createElement('div');
      fill.className = 'q35-fill';
      fill.style.width = `${{Math.max(0, Math.min(100, value * 100))}}%`;
      track.appendChild(fill);
      const amount = document.createElement('span');
      amount.textContent = `${{(value * 100).toFixed(1)}}%`;
      row.append(labelNode, track, amount);
      bars.appendChild(row);
    }});
    occlusion.appendChild(bars);
  }}

  sentenceSelect.addEventListener('change', () => {{
    sentenceIndex = Number(sentenceSelect.value);
    updateDetail();
    draw();
  }});
  opacity.addEventListener('input', () => {{
    opacityValue.textContent = `${{opacity.value}}%`;
    draw();
  }});
  new ResizeObserver(draw).observe(canvas);
  new MutationObserver(draw).observe(document.documentElement, {{ attributes: true, attributeFilter: ['class', 'style'] }});
  selectSample(0);
}})();
</script>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def analyze_sample(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    row: dict[str, Any],
    record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
    sample_dir = args.output_dir / f"index-{record['index']}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    image.save(sample_dir / "original.jpg", quality=95)

    inputs, sentence_specs, prompt_length, image_positions, grid = prepare_teacher_forced_inputs(
        processor, image, record, args.device
    )
    layer_maps, layer_masses = capture_full_attention(
        model, inputs, sentence_specs, image_positions
    )
    del inputs
    baseline_inputs, baseline_specs, baseline_image_positions, baseline_grid = (
        prepare_descriptive_inputs(processor, image, args.device)
    )
    if baseline_grid != grid or baseline_image_positions != image_positions:
        raise ValueError(
            f"Task/description image layout mismatch: {grid} versus {baseline_grid}"
        )
    baseline_layer_maps, baseline_layer_masses = capture_full_attention(
        model, baseline_inputs, baseline_specs, baseline_image_positions
    )
    del baseline_inputs
    descriptive_by_layer = baseline_layer_maps[0]
    descriptive_attention = descriptive_by_layer.mean(axis=0)
    descriptive_mass_by_layer = baseline_layer_masses[0]
    layer_numbers = [
        index + 1
        for index, kind in enumerate(model.config.text_config.layer_types)
        if kind == "full_attention"
    ]

    sentences: list[dict[str, Any]] = []
    relative_maps: dict[int, np.ndarray] = {}
    npz_values: dict[str, np.ndarray] = {
        "descriptive_baseline_raw_by_layer": descriptive_by_layer,
        "descriptive_baseline_raw": descriptive_attention,
        "descriptive_baseline_mass_by_layer": descriptive_mass_by_layer,
    }
    for sentence in sentence_specs:
        number = sentence["sentence_index"]
        maps = layer_maps[number]
        masses = layer_masses[number]
        absolute_attention = maps.mean(axis=0)
        ratio, relative = relative_attention(
            absolute_attention, descriptive_attention
        )
        relative_maps[number] = relative
        visual, entropy = normalize_heatmap(relative)
        peak_flat = int(np.argmax(relative))
        peak_row, peak_col = divmod(peak_flat, grid[1])
        heat_uint8 = np.clip(visual * 255, 0, 255).astype(np.uint8)
        npz_values[f"sentence_{number}_absolute_raw_by_layer"] = maps
        npz_values[f"sentence_{number}_absolute_raw"] = absolute_attention
        npz_values[f"sentence_{number}_absolute_mass_by_layer"] = masses
        npz_values[f"sentence_{number}_relative_ratio"] = ratio
        npz_values[f"sentence_{number}_relative_attention"] = relative
        sentences.append(
            {
                **sentence,
                "heat": heat_uint8.tolist(),
                "absolute_image_mass": float(masses.mean()),
                "absolute_image_mass_by_layer": masses.tolist(),
                "relative_spatial_entropy": entropy,
                "peak": [
                    (peak_col + 0.5) / grid[1],
                    (peak_row + 0.5) / grid[0],
                ],
            }
        )
    np.savez_compressed(sample_dir / "attention-raw.npz", **npz_values)

    grounding_sentence = max(
        sentences, key=lambda sentence: sentence["absolute_image_mass"]
    )
    causal_heat = relative_maps[grounding_sentence["sentence_index"]]
    patch_count = causal_heat.size
    selected_count = max(1, round(args.occlusion_fraction * patch_count))
    top_indices = np.argpartition(causal_heat, -selected_count)[-selected_count:]
    rng = np.random.default_rng(20260824 + int(record["index"]))
    complement = np.setdiff1d(np.arange(patch_count), top_indices)
    random_indices = rng.choice(complement, size=selected_count, replace=False)
    top_image = occlude_cells(image, grid, top_indices)
    random_image = occlude_cells(image, grid, random_indices)
    top_image.save(sample_dir / "occluded-top-attention.jpg", quality=93)
    random_image.save(sample_dir / "occluded-random.jpg", quality=93)

    options = re.findall(r"^\(([A-D])\)", record["question"], flags=re.MULTILINE)
    if not options:
        raise ValueError(f"No answer options found for index {record['index']}")
    original_scores, candidate_ids = score_options(
        model, processor, image, record, options, args.device
    )
    top_scores, _ = score_options(
        model, processor, top_image, record, options, args.device
    )
    random_scores, _ = score_options(
        model, processor, random_image, record, options, args.device
    )
    causal = {
        "method": "forced-choice next-token probability after 'The correct answer is ('",
        "attention_type": "CoFFT-style relative attention",
        "attention_source_sentence": grounding_sentence["sentence_index"],
        "occlusion_fraction": args.occlusion_fraction,
        "selected_patch_count": selected_count,
        "options": options,
        "candidate_token_ids": candidate_ids,
        "original": original_scores,
        "top_attention_occlusion": top_scores,
        "random_occlusion": random_scores,
    }

    sample = {
        "index": int(record["index"]),
        "question_id": record["question_id"],
        "category": record["category"],
        "correct": bool(record["correct"]),
        "question": record["question"],
        "gold": record["ground_truth_label"],
        "predicted": record["predicted_label"],
        "completion": record["completion"],
        "image_width": image.width,
        "image_height": image.height,
        "grid": list(grid),
        "image_token_count": len(image_positions),
        "layers": layer_numbers,
        "heads_per_layer": model.config.text_config.num_attention_heads,
        "aggregation": "mean over next-token queries, 16 heads, and 8 full-attention layers before ratio",
        "attention_type": "CoFFT-style relative attention",
        "attention_formula": "softmax(A(V,S) / (A(V,D) + 1e-10))",
        "descriptive_baseline_prompt": DESCRIPTIVE_PROMPT,
        "descriptive_baseline_image_mass": float(descriptive_mass_by_layer.mean()),
        "descriptive_baseline_image_mass_by_layer": descriptive_mass_by_layer.tolist(),
        "attention_semantics": "same image; sentence attention divided patch-wise by generic-description attention, then spatial softmax",
        "sentences": sentences,
        "causal": causal,
        "image_data_url": image_data_url(image),
    }
    save_montage(sample, image, sample_dir / "sentence-attention-montage.jpg")
    return sample


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_trace_records(args.traces)
    rows = pq.read_table(args.data).to_pylist()

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).eval().to(args.device)

    results = []
    for index in args.indices:
        print(f"analyzing index={index}", flush=True)
        results.append(analyze_sample(model, processor, rows[index], traces[index], args))
        torch.cuda.empty_cache()

    serializable = []
    for sample in results:
        cleaned = {key: value for key, value in sample.items() if key != "image_data_url"}
        serializable.append(cleaned)
        sample_path = args.output_dir / f"index-{sample['index']}" / "analysis.json"
        sample_path.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "model": str(args.model),
        "dataset": str(args.data),
        "trace_source": str(args.traces),
        "thinking": False,
        "samples": serializable,
    }
    (args.output_dir / "attention-analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    build_visual_fragment(results, args.visual_output)
    print(f"analysis={args.output_dir / 'attention-analysis.json'}")
    print(f"visual={args.visual_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
